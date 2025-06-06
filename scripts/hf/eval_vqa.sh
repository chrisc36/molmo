#!/bin/sh

torchrun --nproc-per-node 8 -m launch_scripts.eval_downstream \
    outputs/siglip2_molmo_o-7b \
    text_vqa  --save_dir outputs/text_vqa --is_hf_model