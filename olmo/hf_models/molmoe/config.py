from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from glob import glob
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from .image_vit import VitConfig


C = TypeVar("C", bound="BaseConfig")
D = TypeVar("D", bound="DictConfig|ListConfig")


PathOrStr = Union[str, PathLike]


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"



class AttentionType(StrEnum):
    sdpa = "sdpa"
    direct = "direct"
    flash = "flash"


class LayerNormType(StrEnum):
    default = "default"
    """
    The default LayerNorm implementation, equivalent to PyTorch's built-in version.
    """

    low_precision = "low_precision"
    """
    A low-precision version of the default LayerNorm.
    """

    rms = "rms"
    """
    An RMSNorm implementation. When using ``torch.compile`` this is
    probably the fastest implementation.
    """

    gemma_rms = "gemma_rms"
    """
    A GemmaRMSNorm implementation. When using ``torch.compile`` this is
    probably the fastest implementation.
    """


class ActivationType(StrEnum):
    quick_gelu = "quick_gelu"
    gelu = "gelu"
    gelu_tanh = "gelu_tanh"
    relu = "relu"
    silu = "silu"
    llama_geglu = "llama_geglu"
    llama_geglu_tanh = "llama_geglu_tanh"
    llama_swiglu = "llama_swiglu"
    swiglu = "swiglu"


class BlockType(StrEnum):
    sequential = "sequential"

    llama = "llama"
    """
    A block similar to the sequential block with slightly different
    implementations of operations like attention to imitate the behavior of Llama.
    """

    gemma = "gemma"
    """
    A block similar to the sequential block with slightly different
    implementations of operations like attention to imitate the behavior of Gemma.
    """

    moe = "moe"


class InitFnType(StrEnum):
    mitchell = "mitchell"
    """
    The strategy suggested to us by Mitchell Wortsman from UW.
    This uses a truncated normal distribution with an adaptive standard deviation that depends
    on the size of the weights as well as the depth of the layer.
    """

    normal = "normal"
    """
    All weights are initialized from the same normal distribution.
    """

    kaiming_normal = "kaiming_normal"
    """
    All weights are initialized with the Kaiming method from a normal distribution.
    Note this currently won't work with FSDP.
    """

    fan_in = "fan_in"
    """
    "Fan-in variance scaling", i.e. normal with a standard deviation of ``1/sqrt(d_in)`` where ``d_in``
    is the input dimensionality of the kernel.
    """

    full_megatron = "full_megatron"
    """
    This is what metaseq calls "full megatron init". It is the init used for Llama 2.
    """


class VisionBackboneType(StrEnum):
    openai = "openai"


class ImagePaddingEmbed(StrEnum):
    pad_and_partial_pad = "pad_and_partial_pad"
    pad_embed = "pad_embed"
    regress = "regress"


class ImagePooling2DType(StrEnum):
    attention = "attention"
    attention_meanq = "attention-meanq"
    attention_2wide = "attention_2wide"
    attention_v2 = "attention-v2"
    none = "none"
    stack = "stack"


class ImageProjectType(StrEnum):
    mlp = "mlp"
    mlpx2 = "2mlp"
    linear = "linear"

@dataclass
class VisionBackboneConfig:
    vit: str = "siglip2"
    fix_image_input_idx: int = 2
    gin_bindings: Optional[Any] = None
    image_feature_dropout: float = 0.0
    image_padding_embed: str = "pad_and_partial_pad"
    image_pooling_2d: str = "attention-meanq"
    image_projector: str = "mlp"
    include_bias: bool = False
    init_device: str = "cpu"
    mlp_hidden_size: Optional[int] = None
    mlp_ratio: float = 1
    overlap_margins: Tuple[int, int] = (4, 4)
    pad_to: Optional[int] = None
    pad_token_id: int = 1
    pad_tokenizer: bool = False
    use_position_ids: bool = True
    vit_layers: Tuple[int] = (-2, -9)
    vit_load_path: Optional[str] = None
    skip_unused_layers: bool = True

    d_model: int = 2048
    image_emb_dim: int = 1024
    image_mlp_dim: int = 4096
    initializer_range: float = 0.02
    residual_dropout: float = 0.0
    resize_mode: str = "default"
    float32_attention: bool = False
    attention_type: AttentionType = AttentionType.sdpa
    pooling_attention_mask: bool = False
    activation_type: str = "swiglu"
    image_num_layers: int = 23

    @property
    def llm_patches_per_crop(self):
        h, w = self.image_num_patch
        # Round up in case we need to pad the image features for pooling
        h = (h + self.image_pooling_h - 1) // self.image_pooling_h
        w = (w + self.image_pooling_w - 1) // self.image_pooling_w
        return h, w

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        ### drop keys from data that are not in the config
        data = {k: v for k, v in data.items() if k in cls.__annotations__}
        cfg = cls(**data)
        cfg.vit = VitConfig(**data.get("vit"))
        return cfg


class TruncationDirection(StrEnum):
    right = "right"
    left = "left"


class MolmoeConfig(PretrainedConfig):
    model_type = "molmoe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=2048,
        intermediate_size=2048,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        clip_qkv=None,
        num_experts_per_tok=8,
        num_experts=64,
        output_router_logits=False,
        router_aux_loss_coef=0.01,
        norm_topk_prob=False,
        vision_backbone: VisionBackboneConfig = None,
        z_loss_scale=1e-4,
        **kwargs,
    ):
        
        if isinstance(vision_backbone, dict):
            self.vision_backbone = VisionBackboneConfig.from_dict(vision_backbone)
        else:
            self.vision_backbone = vision_backbone
            
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.clip_qkv = clip_qkv
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.norm_topk_prob = norm_topk_prob
        self.z_loss_scale = z_loss_scale
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def to_dict(self):
        output = super().to_dict()
        output["vision_backbone"] = (
            self.vision_backbone.to_dict()
            if self.vision_backbone else None
        )
        return output

from transformers import AutoConfig
AutoConfig.register("molmoe", MolmoeConfig)