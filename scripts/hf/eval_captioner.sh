#!/bin/sh


torchrun --nproc-per-node 8 -m launch_scripts.eval_captioner \
    outputs/siglip2_molmo_o-7b-captioner \
    --seq_len=1792 --task=dense_caption_eval \
    --split=test --is_hf_model --save_dir=outputs/siglip2_molmo_o-7b-captioner