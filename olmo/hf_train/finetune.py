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

from olmo.hf_train.molmo_trainer import MolmoTrainer

from olmo.hf_models import *
from olmo.hf_models.processing import *
from launch_scripts.utils import get_hf_evaluation

from olmo.data.data_loader import RootSizeMixture
from olmo.hf_train.data_loader import HFDataLoaderConfig
from olmo.torch_util import get_world_size
from olmo.eval.loss_evaluator import LossDatasetEvaluatorConfig
from launch_scripts.utils import VISION_BACKBONES
from olmo.data.pixmo_datasets import PixMoCap
from olmo.nn.vision_backbone import ImagePaddingEmbed
from olmo.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType
from olmo.models.molmo.collator import HF_MMCollator
from transformers import set_seed

set_seed(42)

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


AUX_EXCEPT_DOCS = [
    # Supervised datasets we want eval on
    "coco_2014_vqa_multi",
    "text_vqa",
    "okvqa",
    "chart_qa_weighted",
    "doc_qa",
    "info_qa",
    "ai2_diagram_v2_mix_transparent",
    "a_okvqa_mc",
    "a_okvqa_da",
    "android_control",

    # Some other datasets we might want to eval on
    "science_qa_img",
    "tabwmp_da",
    "st_qa",
    "tally_qa",

    ("pixmo_clocks", 250000),  # Downsample since it is huge

    # # Other synthetic data, also downsampled since they are huge
    ("dv_qa", 10000),
    ("figure_qa", 10000),
    ("plot_qa", 20000),
]


AUX = AUX_EXCEPT_DOCS + [
    "pixmo_docs_charts",
    "pixmo_docs_tables",
    "pixmo_docs_other",
    "pixmo_docs_diagrams",
]


AUX_COSYN_V1 = AUX_EXCEPT_DOCS + [
    "cosyn_chart_exp",
    "cosyn_chemical_exp",
    # "cosyn_circuit_exp", # quality not good
    "cosyn_diagram_exp",
    "cosyn_document",
    # "cosyn_graphic_exp", # quality not good
    "cosyn_math_exp",
    "cosyn_music_exp",
    # "cosyn_nutrition_exp", # zero-shot evaluation dataset
    "cosyn_table_exp",
]


def get_training_mixture(submixture):
    resolved_weights = {}
    for task_name in submixture:
        mix = {}
        if isinstance(task_name, tuple):
            task_name, size = task_name
        else:
            size = None
        resolved_weights[task_name] = size
    return resolved_weights


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
    base_image_input_size: List[int] = field(
        default_factory=lambda: [336, 336],
        metadata={"help": "Base input size for images."}
    )
    do_normalize: bool = field(
        default=True,
        metadata={"help": "Whether to normalize image inputs."}
    )
    image_mean: List[float] = field(
        default_factory=lambda: [0.48145466, 0.4578275, 0.40821073],
        metadata={"help": "Mean values for image normalization."}
    )
    image_padding_mask: bool = field(
        default=True,
        metadata={"help": "Whether to use padding mask for images."}
    )
    image_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size for image processing."}
    )
    image_processor_type: str = field(
        default="MolmoImageProcessor",
        metadata={"help": "Type of image processor to use."}
    )
    image_std: List[float] = field(
        default_factory=lambda: [0.26862954, 0.26130258, 0.27577711],
        metadata={"help": "Standard deviation values for image normalization."}
    )
    image_token_length_h: int = field(
        default=12,
        metadata={"help": "Token length for image height."}
    )
    image_token_length_w: int = field(
        default=12,
        metadata={"help": "Token length for image width."}
    )
    max_crops: int = field(
        default=12,
        metadata={"help": "Maximum number of crops for images."}
    )
    overlap_margins: List[int] = field(
        default_factory=lambda: [4, 4],
        metadata={"help": "Overlap margins for image crops."}
    )

@dataclass
class DataArguments:
    mixture: str = field(
        default="3.2-synthetic",
        metadata={"help": "Name of datset mixture to train on"}
    )
    seq_len: int = field(
        default=2304,
        metadata={"help": "Maximum sequence length for training."}
    )
    inf_seq_len: int = field(
        default=1792,
        metadata={"help": "Maximum sequence length for inference."}
    )
    max_inf_examples: int = field(
        default=2048,
        metadata={"help": "Maximum number of examples for inference."}
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    n_eval_examples: int = field(
        default=2048,
        metadata={"help": "Number of examples to use for evaluation."}
    )
    eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for evaluation."}
    )
    inf_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for inference."}
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

    if data_args.mixture.startswith("single"):
        task_name = data_args.mixture.split("_", 1)[1]
        eval_tasks = [task_name,]
        tasks = [["eval", eval_tasks, 1.0]]
    elif data_args.mixture == "android":
        eval_tasks = ["android_control_ll"]
        tasks = [["eval", ["android_control"], 1.0]]
    elif data_args.mixture in ["small1", "debug"]:
        eval_tasks = ["chart_qa", "doc_qa"]
        tasks = [["aux", ["chart_qa", "doc_qa"], 1.0]]
    elif data_args.mixture in ["pointing"]:
        eval_tasks = ["pointing_eval:test"]
        tasks = [["pointing", [
            "pixmo_points",
            "pixmo_count",
            "pixmo_points_high_freq",
            "pixmo_points_counting",
            "pixmo_points_high_freq_counting",
            "pixmo_count_counting",
        ], 1.0]]

    elif data_args.mixture == "small2":
        eval_tasks = ["chart_qa", "doc_qa", "info_qa"]
        tasks = [["aux", [("chart_qa", 4*4),
                          ("doc_qa", 2*2), ("info_qa", 1)], 1.0]]
    elif data_args.mixture in ["3.2-synthetic"]:
        aux = list(AUX)
        eval_tasks = [
            "chart_qa",
            "info_qa",
            "doc_qa",
            "ai2_diagram_v2_mix_transparent",
            "coco_2014_vqa_multi",
            "pixmo_clocks",
            "android_control_ll",
            "pointing_eval:test",
        ]
        tasks = [
            ["demo", [
                "pixmo_ask_model_anything",
                ("pixmo_cap", 50000),
                "pixmo_cap_qa_as_user_qa",
                "pixmo_pointing_explanations"
            ], 0.15],
            ["aux", aux, 0.50],
            ["pointing", [
                "pixmo_points_train",
                "pixmo_count_train",
                "pixmo_points_high_freq_train",
            ], 0.35]
        ]
    elif data_args.mixture in ["3.3-synthetic"]:
        aux = list(AUX_COSYN_V1)
        eval_tasks = [
            "chart_qa",
            "chart_qa_exp",
            "info_qa",
            "doc_qa",
            "ai2_diagram_v2_mix_transparent",
            "coco_2014_vqa_multi",
            "pixmo_clocks",
            "android_control_ll",
            "pointing_eval:test",
        ]
        tasks = [
            ["demo", [
                "pixmo_ask_model_anything",
                ("pixmo_cap", 50000),
                "pixmo_cap_qa_as_user_qa",
                "pixmo_pointing_explanations"
            ], 0.15],
            ["aux", aux, 0.50],
            ["pointing", [
                "pixmo_points",
                "cosyn_point",
                "pixmo_count",
                "pixmo_points_high_freq",
                "pixmo_points_counting",
                "pixmo_points_high_freq_counting",
                "pixmo_count_counting",
            ], 0.35]
        ]
    else:
        raise NotImplementedError(data_args.mixture)
    
    
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
    elif "olmo-2" in model_args.model_name_or_path.lower() or "olmo-o2" in model_args.model_name_or_path.lower():
        is_moe = False
        model = MolmoO2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
        model.config.use_cache = False
    else:
        raise ValueError(
            f"Model {model_args.model_name_or_path} not supported. "
            "Please use a model from the olmoe or olmo2 family."
        )

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    

    processor = MolmoProcessor.from_pretrained(model_args.model_name_or_path)
    data_formatter_cfg = dict(
        prompt_templates="uber_model",
        message_format="role",
        system_prompt="demo_or_style",
        always_start_with_space=True,
        is_hf_model=True,
        is_training=True,
    )
    processor.data_formatter_cfg = data_formatter_cfg
    processor.is_training = True

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.unk_token
    
    from olmo.hf_models.processing.processing_molmo import EXTRA_TOKENS, DEFAULT_IMAGE_PATCH_TOKEN
    ids = processor.tokenizer.encode("".join(EXTRA_TOKENS), add_special_tokens=False)
    if len(ids) != len(EXTRA_TOKENS):
        processor.tokenizer.add_tokens(list(EXTRA_TOKENS))
        ids = processor.tokenizer.encode("".join(EXTRA_TOKENS), add_special_tokens=False)
    assert len(ids) == len(EXTRA_TOKENS), f"Tokenizer did not add all special tokens. Expected {len(EXTRA_TOKENS)} but got {len(ids)}"
    model.config.image_patch_id = processor.special_token_ids[DEFAULT_IMAGE_PATCH_TOKEN]
    model.model._image_patch_id = processor.special_token_ids[DEFAULT_IMAGE_PATCH_TOKEN]
    
    eval_examples = 2048
    max_inf_examples = data_args.max_inf_examples
    global_batch_size = training_args.per_device_train_batch_size * get_world_size() * training_args.gradient_accumulation_steps
    eval_subset_batches = eval_examples//(data_args.eval_batch_size*get_world_size())
    assert eval_subset_batches > 0

    optimizer_cfg = OptimizerConfig(
            name=OptimizerType.adamw,
            connector_learning_rate=5e-6,
            vit_learning_rate=5e-6,
            llm_learning_rate=2e-5 if "olmoe" in model_args.model_name_or_path.lower() else 1e-5,
            connector_weight_decay=0.0,
            vit_weight_decay=0.0,
            llm_weight_decay=0.0,
            connector_betas=[0.9, 0.95],
            vit_betas=[0.9, 0.95],
            llm_betas=[0.9, 0.95],
            connector_eps=1e-6,
            vit_eps=1e-6,
            llm_eps=1e-6,
        )
    
    training_args.optim_cfg = optimizer_cfg

    training_args.scheduler_cfg = SchedulerConfig(
            name=SchedulerType.multimodal,
            connector_t_warmup=200,
            vit_t_warmup=200,
            llm_t_warmup=200,
            alpha_f=0.1,
            warmup_min_lr=0.0
        )

    assert model.model.vision_backbone is not None, "Vision backbone not found"
    root_size_mixture: List[RootSizeMixture] = []
    for name, submixture, rate in tasks:
        submixture = get_training_mixture(submixture)
        root_size_mixture.append(RootSizeMixture(rate, submixture))

    num_workers = 2
    evaluators = []
    for task in eval_tasks:
        evaluation = get_hf_evaluation(
            task,
            data_args.inf_seq_len,
            device_batch_size=data_args.inf_batch_size,
            max_examples=max_inf_examples,
            num_workers=num_workers,
        )
        evaluation.data.persistent_workers = True
        evaluators.append(evaluation)

    train_dataset = HFDataLoaderConfig(
            root_size_mixture=root_size_mixture,
            shuffle=True,
            split="train",
            drop_last=True,
            sequence_length=data_args.seq_len,
            num_workers=num_workers,
            pad="to_max",
            pin_memory=True,
            seed=50189
        ).build_train_dataset(
            preprocessor=processor,
            global_batch_size=global_batch_size,
    )

    dataset_args = dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=HF_MMCollator(tokenizer=processor.tokenizer)
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

    import torch.distributed as dist
    import os

    # Store the original barrier function
    original_barrier = dist.barrier

    def safe_barrier(group=None, device_ids=None, async_op=False):
        """A safer barrier that explicitly specifies device IDs"""
        # If device_ids not provided, use local_rank
        if device_ids is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device_ids = [local_rank]
        
        # Call original with explicit device IDs
        return original_barrier(group=group, device_ids=device_ids, async_op=async_op)

    # Apply the patch
    dist.barrier = safe_barrier

    # Also ensure we're using the right device
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)


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