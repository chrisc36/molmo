"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import os
import logging
import math
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union, Any,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig, Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import (
    ModelOutput,
    logging,
)

from olmo.hf_models.molmoe import MolmoeConfig
from olmo.hf_models.olmoe.modeling import load_balancing_loss_func
from olmo.hf_models.molmoe.vision import MolmoVisionBackbone
from olmo.hf_models.molmoe.config import VisionBackboneConfig
from olmo.hf_models.modeling_olmoe import (
    OlmoeModel,
    OlmoeForCausalLM
)
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
)
from torch.nn import CrossEntropyLoss

log = logging.get_logger(__name__)
    

class MolmoeModel(OlmoeModel):
    def __init__(self, config: MolmoeConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.vision_backbone: Optional[MolmoVisionBackbone] = None
        if config.vision_backbone is not None:
            self.vision_backbone = MolmoVisionBackbone.build(config.vision_backbone)
            self._image_patch_id = config.image_patch_id

    def init_vision_backbone(self, config: MolmoeConfig):
        if config.vision_backbone is not None:
            self.vision_backbone = MolmoVisionBackbone.build(config.vision_backbone)
            self.vision_backbone.reset_with_pretrained_weights()
            self._image_patch_id = config.image_patch_id

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        pooled_patches_idx: Optional[torch.Tensor] = None,
    ):
        has_image = images is not None

        assert not (has_image and inputs_embeds is not None), "Cannot provide both images and inputs_embeds."
        assert not (has_image and (past_key_values is not None and len(past_key_values) > 0)), "Cached key and values should not be used with images."

        batch_size, seq_len = input_ids.size() if inputs_embeds is None else inputs_embeds.size()[:2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if images is not None:
            image_features = self.vision_backbone(images, image_masks, pooled_patches_idx)
            is_image_patch = input_ids.view(-1) == self._image_patch_id
            assert is_image_patch.sum() == len(image_features)
            inputs_embeds.view(-1, inputs_embeds.shape[-1])[is_image_patch] = image_features.to(inputs_embeds.dtype)

        
        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    def num_params(self, include_embedding: bool = True) -> int:
        """
        Get the total number of parameters.
        """
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".embed" not in np[0],
                params,
            )
        return sum(p.numel() for _, p in params)


class MolmoeForCausalLM(OlmoeForCausalLM):
    config_class = MolmoeConfig

    def __init__(
        self, 
        config: MolmoeConfig
    ):
        super(OlmoeForCausalLM, self).__init__(config)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.z_loss_scale = config.z_loss_scale
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        config.init_device = "cpu"
        self.model = MolmoeModel(config)

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        pooled_patches_idx: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pad_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_masks: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=self.router_aux_loss_coef > 0,
            return_dict=return_dict,
            cache_position=cache_position,
            images=images,
            image_masks=image_masks,
            pooled_patches_idx=pooled_patches_idx,
        )
        
        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
        z_loss = None
        if self.z_loss_scale > 0:
            z_squared = logits.logsumexp(-1).pow(2)
            z_squared = (z_squared * (labels != -100)).mean()
            z_loss = self.z_loss_scale * z_squared
            if labels is not None:
                loss = loss + z_loss.to(loss.device)

        aux_loss = None
        if self.router_aux_loss_coef > 0:
            if labels is not None:
                aux_loss = load_balancing_loss_func(
                    outputs.router_logits if return_dict else outputs[-1],
                    self.num_experts,
                    self.num_experts_per_tok,
                    pad_attention_mask,
                ) * self.router_aux_loss_coef
                loss += aux_loss.to(loss.device)  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            output = ((z_loss, aux_loss),) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=(z_loss, aux_loss),
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def can_generate(self) -> bool:
        return True

    @torch.no_grad()
    def generate_from_batch(
        self,
        batch: Dict[str, Any],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        if generation_config is not None:
            assert generation_config.use_cache
        
        images = batch.get("images")
        image_masks = batch.get("image_masks")
        pooled_patches_idx = batch.get("pooled_patches_idx")

        # Validate inputs.
        input_ids = batch["input_ids"]
        batch_size, seq_len = input_ids.shape
        attention_mask = batch.get("attention_mask", None)
        max_new_tokens = generation_config.max_new_tokens
        assert max_new_tokens is not None
        
        out = super().generate(
            batch["input_ids"].long(),
            generation_config,
            attention_mask=attention_mask,
            images=images,
            image_masks=image_masks,
            pooled_patches_idx=pooled_patches_idx,
            **kwargs,
        )

        return out

    @torch.no_grad()
    def get_expert_scores(
        self,
        batch,
        **kwargs,
    ):
        outputs = self.forward(
                **batch,
                output_router_logits=True
            )
        routed_experts = outputs.router_logits
        outputs["routed_experts"] = {}
        for idx, expert_logits in enumerate(routed_experts):
            routing_weights = F.softmax(expert_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            outputs["routed_experts"][idx] = {
                "routing_weights": routing_weights.float().detach().cpu().numpy().tolist(),
                "selected_experts": selected_experts.float().detach().cpu().numpy().tolist(),
            }
        return outputs
        
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None,
        images=None, image_masks=None, pooled_patches_idx=None,
        **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
                "image_masks": image_masks,
                "pooled_patches_idx": pooled_patches_idx,
            }
        )

        if len(past_key_values) > 0:
            model_inputs.pop("images", None)
            model_inputs.pop("image_masks", None)
            model_inputs.pop("pooled_patches_idx", None)

        return model_inputs
    

from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("molmoe", MolmoeConfig)
AutoModelForCausalLM.register(MolmoeConfig, MolmoeForCausalLM)