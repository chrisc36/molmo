#!/bin/bash


torchrun --nproc-per-node 2 olmo/hf_train/pretrain.py \
    --deepspeed ./scripts/hf/deepspeed/zero2_offload.json \
    --model_name_or_path Qwen/Qwen2-7B \
    --dataset pixmo_cap_with_transcripts \
    --vision_backbone siglip2 \
    --bf16 True \
    --output_dir outputs/siglip2_molmo_o-7b-captioner \
    --num_train_epochs 4 \
    --eval_batch_size 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2304 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb