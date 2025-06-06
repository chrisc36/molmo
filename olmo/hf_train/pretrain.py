# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
from dataclasses import replace
import pathlib
from typing import Optional, List

import torch

import transformers
import tokenizers

from olmo.hf_train.collator import HF_MMCollator
from olmo.hf_train.molmo_trainer import MolmoTrainer

from olmo.hf_models import *
from olmo.hf_models.processing import *
from olmo.hf_models.molmoe.vision import VisionBackboneConfig
from olmo.hf_models.molmoe.vision_utils import VISION_BACKBONES

from olmo.hf_train.data_loader import HFDataLoaderConfig
from olmo.torch_util import get_world_size
from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig
from olmo.nn.vision_backbone import ImagePaddingEmbed
from olmo.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType
from transformers import set_seed

set_seed(42)

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    z_loss_scale: float = field(
        default=1e-4,
        metadata={"help": "Scale for z loss."}
    )
    router_loss_scale: float = field(
        default=0.0,
        metadata={"help": "Scale for router loss."}
    )
    vision_backbone: Optional[str] = field(
        default="siglip2",
        metadata={"help": f"Vision backbone to use. Must be one of {list(VISION_BACKBONES.keys())}."}
    )
    do_normalize: bool = field(
        default=True,
        metadata={"help": "Whether to normalize image inputs."}
    )
    image_padding_mask: bool = field(
        default=True,
        metadata={"help": "Whether to use padding mask for images."}
    )
    overlap_margins: List[int] = field(
        default_factory=lambda: [4, 4],
        metadata={"help": "Overlap margins for image crops."}
    )

@dataclass
class DataArguments:
    dataset: str = field(
        default="pixmo_cap_with_transcripts",
        metadata={"help": "Dataset to use for training."}
    )
    seq_len: int = field(
        default=2304,
        metadata={"help": "Maximum sequence length for training."}
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    n_eval_examples: int = field(
        default=2048,
        metadata={"help": "Number of examples to use for evaluation."}
    )
    eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for evaluation."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=True)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    group_by_modality_length: bool = field(default=False)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
            
def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    bnb_model_from_pretrained_args = {}

    if "olmoe" in model_args.model_name_or_path.lower():
        is_moe = True
        model = MolmoeForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
        model.config.use_cache = False
        model.config.num_frozen_experts = model_args.num_frozen_experts
    elif "olmo" in model_args.model_name_or_path.lower():
        is_moe = False
        model = MolmoOForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
        model.config.use_cache = False
    elif "qwen" in model_args.model_name_or_path.lower():
        is_moe = False
        model = MolmoDForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    else:
        raise ValueError(
            f"Model {model_args.model_name_or_path} not supported. "
            "Please use a model from the olmoe or olmo or olmo2 or qwen family."
        )

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    from olmo.hf_models.processing.processing_molmo import EXTRA_TOKENS
    ids = tokenizer.encode("".join(EXTRA_TOKENS), add_special_tokens=False)
    if len(ids) != len(EXTRA_TOKENS):
        tokenizer.add_tokens(list(EXTRA_TOKENS))
        ids = tokenizer.encode("".join(EXTRA_TOKENS), add_special_tokens=False)
    assert len(ids) == len(EXTRA_TOKENS), f"Tokenizer did not add all special tokens. Expected {len(EXTRA_TOKENS)} but got {len(ids)}"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    

    eval_examples = data_args.n_eval_examples
    global_batch_size = training_args.per_device_train_batch_size * get_world_size() * training_args.gradient_accumulation_steps
    vit_layers = [-2, -9] if model_args.vision_backbone == "openai" else [-3, -9]
    optimizer_cfg = OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=2e-4,
            vit_learning_rate=6e-6,
            llm_learning_rate=2e-5,
            connector_weight_decay=0.0,
            vit_weight_decay=0.0,
            llm_weight_decay=0.0,
            connector_betas=[0.9, 0.95],
            vit_betas=[0.9, 0.95],
            llm_betas=[0.9, 0.95],
            connector_eps=1e-6,
            vit_eps=1e-6,
            llm_eps=1e-6,
            metrics_log_interval=-1
        )
    
    training_args.optim_cfg = optimizer_cfg

    training_args.scheduler_cfg = SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200,
            vit_t_warmup=2000,
            llm_t_warmup=2000,
            alpha_f=0.1,
            warmup_min_lr=0.0
        )

    image_vit = VISION_BACKBONES[model_args.vision_backbone]

    model.config.vision_backbone = VisionBackboneConfig(
        vit=image_vit,
        vit_layers=vit_layers,
        d_model=model.config.hidden_size,
        image_emb_dim=image_vit.image_emb_dim,
        image_padding_embed=ImagePaddingEmbed.pad_and_partial_pad if model_args.vision_backbone == "openai" else None,
        image_num_layers=image_vit.image_num_layers,
        image_mlp_dim=image_vit.image_mlp_dim,
        resize_mode=image_vit.resize_mode,
    )

    # sync the vision backbone config with values from image_vit (whatever sam attributes are present)
    for attr in dir(image_vit):
        if not attr.startswith('_'):
            if hasattr(model.config.vision_backbone, attr):
                setattr(model.config.vision_backbone, attr, getattr(image_vit, attr))

    image_processor_cfg = dict(
        max_crops=8 if model_args.vision_backbone in ["siglip", "siglip2"] else 12,
        overlap_margins=model_args.overlap_margins,
        base_image_input_size=image_vit.image_default_input_size,
        image_token_length_w=14 if model_args.vision_backbone in ["siglip", "siglip2"] else 12,
        image_token_length_h=14 if model_args.vision_backbone in ["siglip", "siglip2"] else 12,
        image_patch_size=image_vit.image_patch_size,
        image_padding_mask=model_args.image_padding_mask,
        do_normalize=model_args.do_normalize,
        resize=image_vit.resize_mode if model_args.vision_backbone != "openai" else "default",
        normalize=model_args.vision_backbone,
    )

    evaluator_cfg = LossDatasetEvaluatorConfig(
        label="val",
        max_examples=eval_examples,
        device_batch_size=data_args.eval_batch_size,
        console_log_interval=20,
        data=HFDataLoaderConfig(
            seed=training_args.seed,
            dataset=data_args.dataset,
            shuffle=False,
            split="validation",
            drop_last=True,
            sequence_length=data_args.seq_len,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        ),
    )

    image_processor = MolmoImageProcessor(**image_processor_cfg)
    
    data_formatter_cfg = dict(
        system_prompt='style_and_length',
        is_hf_model=True,
        is_training=True,
        always_start_with_space=True,
    )
    processor = MolmoProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        data_formatter_cfg=data_formatter_cfg,
    )

    if model.model.vision_backbone is None:
        from olmo.hf_models.processing.processing_molmo import DEFAULT_IMAGE_PATCH_TOKEN
        model.config.image_patch_id = processor.special_token_ids[DEFAULT_IMAGE_PATCH_TOKEN]
        model.model.init_vision_backbone(
        model.config
    )

    evaluators = [
            evaluator_cfg,
            replace(evaluator_cfg, data=replace(evaluator_cfg.data, dataset="pixmo_cap"), label="caption_val")
    ]

    train_dataset = HFDataLoaderConfig(
            dataset=data_args.dataset,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=data_args.seq_len,
            seed=95818,
            num_workers=2,
            pad="to_max",
            pin_memory=True,
        ).build_train_dataset(
            preprocessor=processor,
            global_batch_size=global_batch_size,
    )

    dataset_args = dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=HF_MMCollator(tokenizer=tokenizer)
    )

    if is_moe:
        model.config.router_aux_loss_coef = model_args.router_loss_scale
        model.router_aux_loss_coef = model_args.router_loss_scale
        
    model.config.z_loss_scale = model_args.z_loss_scale
    model.z_loss_scale = model_args.z_loss_scale
    
    training_args.run_name = training_args.output_dir.split("/")[-1]

    trainer = MolmoTrainer(model=model,
                    tokenizer=processor,
                    args=training_args,
                    evaluators=evaluators,
                    **dataset_args)
    
    print('starting training...', local_rank)

    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    from deepspeed.runtime.zero.config import ZeroStageEnum
    from deepspeed.utils.tensor_fragment import fragment_address
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    from numpy.core.multiarray import _reconstruct
    import numpy as np
    from numpy.dtypes import UInt32DType

    torch.serialization.add_safe_globals([
        DeepSpeedZeroOptimizer,
        ZeroStageEnum,
        fragment_address,
        LossScaler,
        _reconstruct,
        np.ndarray,
        np.dtype,
        np.generic,
        np.bool_, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
        np.float16, np.float32, np.float64,
        UInt32DType
        # *[getattr(np, t) for t in dir(np) if isinstance(getattr(np, t), type)]
    ])

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True, evaluators=evaluators)
    else:
        trainer.train(evaluators=evaluators)
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer,
                                    output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="sdpa")