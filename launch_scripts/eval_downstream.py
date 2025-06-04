"""Evals a checkpoint on multiple tasks, run this script with 'torchrun'."""
import argparse
import logging
from dataclasses import replace
from typing import cast

from omegaconf import OmegaConf

from launch_scripts.utils import get_evaluation, get_hf_evaluation
from olmo.train.trainer_config import FSDPConfig, FSDPPrecision
from olmo.models.model import FSDPWrapStrategy
from olmo.util import (
    clean_opt,
    prepare_torchrun_environment, select_checkpoint, )
from scripts.mm_eval import ModelEvaluator, DatasetEvaluatorConfig, EvalConfig

log = logging.getLogger(__name__)


def main():
    # enable run on a single GPU
    try:
        prepare_torchrun_environment()
    except:
        log.warning("Failed to prepare torchrun environment, continuing anyway")

    parser = argparse.ArgumentParser(prog="Evaluate a model on downstream tasks")
    parser.add_argument("checkpoint",
                        help="Checkpoint to evaluate, should contain a config file and unshared model file")
    parser.add_argument("tasks", nargs="+", help="Tasks to evaluate")
    parser.add_argument("--high_res", action="store_true",
                        help="User default high rese setting: max crops=36, seq len=4096 and eval_name=36crop")
    parser.add_argument("--max_examples", type=int, default=-1,
                        help="Maximum number of examples to evaluate")
    parser.add_argument("--max_crops", type=int, default=None,
                        help="Override models default number of crops")
    parser.add_argument("--seq_len", default=1536, type=int,
                        help="Max sequence length to use")
    parser.add_argument("--device_batch_size", default=4, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fsdp", action="store_true",
                        help="Load with FSDP, can be used to avoid OOMs")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Override max new tokens, otherwise use task-specific default")
    parser.add_argument("--is_hf_model", action="store_true",
                        help="Use this flag if the model is a huggingface model")
    parser.add_argument("--include_image", action="store_true",
                        help="Include image in the evaluation outputs")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save the evaluation results")
    args, other_args = parser.parse_known_args()

    if args.high_res:
        if "siglip" in args.checkpoint:
            args.max_crops = 24 if args.max_crops is None else args.max_crops
        else:
            args.max_crops = 36 if args.max_crops is None else args.max_crops
        args.seq_len = 8192
        args.eval_name = f"{args.max_crops}crop"

    tasks = []
    for task in args.tasks:
        if task == "low-res":
            # low-res validation tasks
            tasks += [
                "android_control_ll",
                "countbench_qa:huggingface",
                "pixmo_count_counting:validation",
                "pointing_eval:test",
            ]
        elif task in ["high-res", "high-res-exp"]:
            # high-res validation tasks
            tasks += [
                "coco_2014_vqa_multi",
                "text_vqa",
                "okvqa",
                "chart_qa",
                "doc_qa",
                "info_qa",
                "science_qa_img",
                "ai2_diagram_v2_mix_transparent",
                "a_okvqa_mc",
                "a_okvqa_da",
                "mmmu_test",
                "real_world_qa_no_instruction:test",
                "math_vista_v2",
                "pixmo_clocks:validation"
            ]
            if task == "high-res-exp":
                tasks += ["chart_qa_exp"]
        elif task == "test-high-res":
            # high-res test tasks
            tasks = [
                "info_qa:test",  # No metrics, submit to eval server
                "doc_qa:test",  # No metrics, submit to eval server
                "chart_qa:test",
                "text_vqa",  # test server is down, so just have to use val
                "ai2_diagram_v2_mix_transparent:test",
                "math_vista_v2",  # Just use val as is common in the literature
                "real_world_qa_no_instruction:test",
                "a_okvqa_mc",
                "a_okvqa_da",
                "mmmu_test",  # standard practice is to use val
                "a_okvqa_mc:test",
                "a_okvqa_da:test",
            ]

        elif task == "test-low-res":
            # low-res test tasks
            tasks += [
                "countbench_qa:huggingface",
                "pixmo_count_counting:test",
                "android_control_hl_ll:test",
                "android_control_hl:test",
                # Do this last and low-res since it is HUGE
                "vqa_v2_test:test2015", # No metrics
            ]
        elif "," in task:
            tasks += task.split(",")   # support comma seperator just because the jax code does
        else:
            tasks.append(task)
    tasks = list({k: None for k in tasks})  # de-duplicate but keep order

    inf_evaluators = []
    for task in tasks:
        if args.is_hf_model:
            base_config = get_hf_evaluation(name=task, seq_len=args.seq_len, max_examples=args.max_examples)
        else:
            base_config = get_evaluation(name=task, seq_len=args.seq_len, max_examples=args.max_examples)
        eval_config = DatasetEvaluatorConfig(
            label=base_config.label,
            data=replace(base_config.data, pad="to_max" if args.fsdp else None),
            generative_evaluator=replace(
                base_config.evaluator,
                n_to_log=4,
                num_wandb_examples=300,
                save_predictions="_default",
            ),
            device_batch_size=args.device_batch_size,
            subset_num_batches=None,
            max_examples=args.max_examples,
            max_new_tokens=args.max_new_tokens or base_config.max_new_tokens,
        )
        inf_evaluators.append(eval_config)

    if args.is_hf_model:
        checkpoint_dir = args.checkpoint
    else:
        checkpoint_dir = "debug" if args.checkpoint == "debug" else select_checkpoint(args.checkpoint)

    cfg = EvalConfig(
        is_hf_model=args.is_hf_model,
        max_crops_override=args.max_crops,
        model_max_length_override=args.seq_len,
        evaluations=inf_evaluators,
        load_path=checkpoint_dir,
        console_log_interval=10,
        precision="amp_bf16",
        pbar=False,
        eval_name=f"{args.max_crops}crop" if args.high_res else None,
        fsdp=FSDPConfig(
            wrapping_strategy=FSDPWrapStrategy.by_block_and_size,
            precision=FSDPPrecision.float,
            fsdp2=True
        ) if args.fsdp else None,
        skip_if_metrics_cached=not args.overwrite,
        include_image=args.include_image,
        save_dir=args.save_dir,
    )

    config = OmegaConf.create(cfg)
    config.merge_with_dotlist([clean_opt(arg) for arg in other_args])
    cfg = cast(EvalConfig, OmegaConf.to_object(config))
    if args.is_hf_model:
        data_formatter_cfg = dict(
                    prompt_templates="uber_model",
                    message_format="role",
                    system_prompt="demo_or_style",
                    always_start_with_space=True,
                    is_hf_model=True,
                    is_training=False,
                )
        cfg.build().run_hf(data_formater_cfg=data_formatter_cfg)
    else:
        cfg.build().run()


if __name__ == "__main__":
    main()
