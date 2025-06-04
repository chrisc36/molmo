"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from functools import partial
from pathlib import Path
from typing import (
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
)
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .config import (
    ImagePooling2DType,
    ImageProjectType, 
)


from .config import (
    VisionBackboneConfig
)
from olmo.nn.llm import Activation

log = logging.getLogger(__name__)


class OLMoConfigurationError(Exception):
    pass

def activation_checkpoint_function(cfg):
    preserve_rng_state = not (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and
        (cfg.residual_dropout == 0.0) and (cfg.response_residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=True,
        use_reentrant=False,
    )


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)



class LlamaSwiGLU(Activation):
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return F.silu(x1) * x2

    @property
    def output_multiplier(self) -> float:
        return 0.5


class QuickGELU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

    @property
    def output_multiplier(self) -> float:
        return 1.0


class ViTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        v_cfg = config

        self.w1 = nn.Linear(
            v_cfg.image_emb_dim,
            v_cfg.image_mlp_dim,
            bias=True,
            device=config.init_device,
        )
        # Activation function.
        cfg = deepcopy(config)
        cfg.activation_type = v_cfg.image_mlp_activations
        self.act = QuickGELU(cfg)
        self.w2 = nn.Linear(
            v_cfg.image_mlp_dim,
            v_cfg.image_emb_dim,
            bias=True,
            device=config.init_device,
        )
    
    def reset_parameters(self):
        v_cfg = self.config
        nn.init.trunc_normal_(self.w1.weight, std=math.sqrt(1 / v_cfg.image_emb_dim), a=-2.0, b=2.0)
        nn.init.trunc_normal_(self.w2.weight, std=math.sqrt(1 / v_cfg.image_mlp_dim), a=-2.0, b=2.0)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class ImageProjectorMLP(nn.Module):
    """MLP used for the image projector"""

    def __init__(self, config, input_dim: int, dropout: float = 0.0, device=None):
        super().__init__()
        self.hidden_size = config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        self.initializer_range = config.initializer_range

        self.w1 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=device,
        )
        self.w2 = nn.Linear(
            self.hidden_size // 2,
            config.d_model,
            bias=False,
            device=device,
            )
        self.w3 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=device,
        )
        # Activation function.
        self.act = Activation.build(config.activation_type, split_inputs=True)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.normal_(self.w1.weight, std=self.initializer_range)
        nn.init.normal_(self.w2.weight, std=self.initializer_range)
        nn.init.normal_(self.w3.weight, std=self.initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(self.act(self.w1(x), self.w3(x)))
        x = self.dropout(x)
        return x

class MolmoVisionBackbone(nn.Module):
    def __init__(self, config: VisionBackboneConfig):
        super().__init__()
        self.config = config
        vit_cfg = config
        input_dim: int = None
        pool_dim = vit_cfg.image_emb_dim * len(config.vit_layers)

        from .image_vit import ViTMultiHeadDotProductAttention

        if config.image_pooling_2d in {ImagePooling2DType.attention, ImagePooling2DType.attention_meanq}:
            self.image_pooling_2d = ViTMultiHeadDotProductAttention(config.vit, input_dim=pool_dim)
            input_dim = vit_cfg.image_emb_dim
        elif config.image_pooling_2d == ImagePooling2DType.attention_2wide:
            cfg = deepcopy(config.vit)
            vit_cfg.image_emb_dim *= 2
            vit_cfg.image_head_dim *= 2
            self.image_pooling_2d = ViTMultiHeadDotProductAttention(cfg, input_dim=pool_dim)
            input_dim = vit_cfg.image_emb_dim
        elif config.image_pooling_2d in [ImagePooling2DType.none, ImagePooling2DType.stack]:
            self.image_pooling_2d = None
            nlayers = 1 if config.vit_layers is None else len(config.vit_layers)
            input_dim = nlayers * vit_cfg.image_emb_dim
            if config.image_pooling_2d == ImagePooling2DType.stack:
                input_dim *= 4
        else:
            raise NotImplementedError(f"Unknown image pooling 2D method: {config.image_pooling_2d}")

        self.input_dim = input_dim

        if config.image_projector == ImageProjectType.mlp:
            self.image_projector = ImageProjectorMLP(config, input_dim, device=config.init_device)
        elif config.image_projector == ImageProjectType.linear:
            self.image_projector = nn.Linear(input_dim, config.d_model, bias=False, device=config.init_device)
        else:
            raise NotImplementedError(f"Unknown image projector: {config.image_projector}")

        self.image_feature_dropout = nn.Dropout(config.image_feature_dropout)

    @classmethod
    def build(cls, config):
        assert config is not None
        return MolmoPretrainedVisionBackbone(config)

    def reset_parameters(self):
        if self.image_pooling_2d is not None:
            self.image_pooling_2d.reset_parameters()
        if self.config.image_projector == "2mlp":
            for module in self.image_projector:
                module.reset_parameters()
        elif self.config.image_projector == "linear":
            nn.init.xavier_uniform_(self.image_projector.weight)
        else:
            self.image_projector.reset_parameters()

    @abstractmethod
    def forward(self, images: torch.Tensor, image_masks: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError
        
from .image_vit import *
from dataclasses import replace
from olmo.torch_util import freeze_module

class MolmoPretrainedVisionBackbone(MolmoVisionBackbone):
    def __init__(self, config):
        super().__init__(config)
        device = config.init_device

        self.vit_layers = []
        for layer in config.vit_layers:
            if layer >= 0:
                self.vit_layers.append(layer)
            else:
                self.vit_layers.append(config.image_num_layers + layer)
        last_layer_needed = (max(self.vit_layers)+1)

        vit_cfg = self.config.vit
        if last_layer_needed < config.image_num_layers:
            if self.config.skip_unused_layers:
                vit_cfg = replace(vit_cfg, image_num_layers=last_layer_needed)
                self.image_vit: VisionTransformer = vit_cfg.build(device)
            else:
                # We might need to keep the layers for checkpoint compatibility, but we
                # freeze them since unfrozen layers with no gradient confuses torch's distributed
                # optimizer checkpointer
                self.image_vit: VisionTransformer = vit_cfg.build(device)
                for block in self.image_vit.transformer.resblocks[last_layer_needed-1:]:
                    freeze_module(block)
        else:
            self.image_vit: VisionTransformer = vit_cfg.build(device)

        self.num_prefix_tokens = self.image_vit.num_prefix_tokens
        assert self.num_prefix_tokens in {0, 1}, "Only 0 or 1 prefix tokens are supported"

        self.pad_embed = None
        if config.image_padding_embed:
            image_dim = vit_cfg.image_emb_dim*len(self.config.vit_layers)
            if config.image_padding_embed in ["pad_embed", "regress"]:
                self.pad_embed = nn.Parameter(
                    torch.zeros((image_dim,), device=device))
            elif config.image_padding_embed == "pad_and_partial_pad":
                self.pad_embed = nn.Parameter(
                    torch.zeros((2, image_dim), device=device))
            else:
                raise ValueError(config.image_padding_embed)

    def reset_with_pretrained_weights(self):
        super().reset_parameters()  # resets the connector
        self.image_vit.reset_with_pretrained_weights()

    def reset_parameters(self):
        super().reset_parameters()
        self.image_vit.reset_parameters()
        if self.config.use_cls_feature:
            nn.init.xavier_uniform_(self.cls_projector.weight)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        : param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        cfg = self.config
        B, T, N, D = images.shape
        images = images.view(B * T, N, D)
        image_features = self.image_vit(images)

        features = []
        for layer in self.vit_layers:
            features.append(image_features[layer])
        image_features = torch.cat(features, dim=-1)

        if self.num_prefix_tokens > 0:
            image_features = image_features[:, 1:]
        image_features = image_features.view(B, T, N, -1)
        return image_features
    
    def forward(self, images: torch.Tensor, image_masks: torch.Tensor,
                pooled_patches_idx: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.config

        # image_features: (batch_size, num_crops(=num_image), num_patch, nximage_emb_dim)
        batch_size, num_image = images.shape[:2]
        image_features = self.encode_image(images)

        if cfg.image_padding_embed:
            assert image_masks is not None
            if cfg.image_padding_embed == "pad_embed":
                all_pad = (image_masks == 0)
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(all_pad, -1)
            elif cfg.image_padding_embed == "regress":
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(torch.maximum(image_masks, torch.zeros_like(image_masks)), -1)
            elif cfg.image_padding_embed == "pad_and_partial_pad":
                pad_embed = self.pad_embed[:, None, None, None, :]
                all_pad = image_masks == 0
                partial_pad = torch.logical_and(image_masks < 1, torch.logical_not(all_pad))
                image_features = image_features + pad_embed[0] * torch.unsqueeze(all_pad, -1)
                image_features = image_features + pad_embed[1] * torch.unsqueeze(partial_pad, -1)
            else:
                raise ValueError(cfg.image_padding_embed)

        image_features = self.image_feature_dropout(image_features)
        dim = image_features.shape[-1]

        multiple_pooling = isinstance(pooled_patches_idx, (tuple, list))
        if not multiple_pooling:
            pooled_patches_idxs = [pooled_patches_idx]
        else:
            pooled_patches_idxs = pooled_patches_idx

        all_pooled_features = []
        for pooled_patches_idx in pooled_patches_idxs:
            valid = pooled_patches_idx >= 0
            valid_token = torch.any(valid, -1)

            # Use `pooled_patches_idx` to arange the features for image pooling
            batch_idx = torch.arange(pooled_patches_idx.shape[0], dtype=torch.long, device=pooled_patches_idx.device)
            batch_idx = torch.tile(batch_idx.view(batch_size, 1, 1), [1, pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]])

            # Now [batch, num_high_res_features, pool_dim, dim]
            to_pool = image_features.reshape(batch_size, -1, dim)[batch_idx, torch.clip(pooled_patches_idx, 0)]
            to_pool = to_pool * valid[:, :, :, None]
            to_pool = to_pool.reshape([-1, pooled_patches_idx.shape[-1], dim])
            if self.config.pooling_attention_mask:
                attn_mask = valid.reshape([-1, 1, 1, valid.shape[-1]])
            else:
                attn_mask = None

            if cfg.image_pooling_2d == ImagePooling2DType.attention_meanq:
                if self.config.pooling_attention_mask:
                    denom = valid.view(-1, to_pool.shape[-2]).sum(-1)
                    denom = torch.where(denom == 0, 1, denom)
                    query = to_pool.sum(-2, keepdim=True) / denom[:, None, None]
                else:
                    query = to_pool.mean(-2, keepdim=True)
                pooled_features = self.image_pooling_2d(query, to_pool, attn_mask=attn_mask)
            elif cfg.image_pooling_2d not in {ImagePooling2DType.none, ImagePooling2DType.stack}:
                pooled_features = self.image_pooling_2d(to_pool[:, :1, :], to_pool, attn_mask=attn_mask)
            else:
                pooled_features = to_pool

            pooled_features = pooled_features.reshape([batch_size, -1, pooled_features.shape[-1]])

            # MLP layer to map the feature.
            if cfg.image_projector == ImageProjectType.mlpx2:
                for module in self.image_projector:
                    pooled_features = module(pooled_features)
            else:
                pooled_features = self.image_projector(pooled_features)
            all_pooled_features.append((pooled_features, valid_token))

        if multiple_pooling:
            return all_pooled_features
        else:
            image_features, valid_token = all_pooled_features[0]
            return image_features.view(-1, image_features.shape[-1])[valid_token.flatten()]
