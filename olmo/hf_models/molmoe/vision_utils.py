import logging
from typing import Dict
from dataclasses import replace
from .image_vit import VitConfig

log = logging.getLogger(__name__)

import os

MOLMO_DATA_DIR=os.getenv("MOLMO_DATA_DIR")

DEBUG_VISION_BACKBONE = VitConfig(
    init_path=None,
    resize_mode="siglip",
    image_model_type="openai",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=128,
    image_num_heads=2,
    image_num_key_value_heads=2,
    image_num_layers=2,
    image_head_dim=64,
    image_mlp_dim=256,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
)


DEFAULT_VISION_BACKBONE = VitConfig(
    init_path=f"{MOLMO_DATA_DIR}/pretrained_image_encoders/vit-l-14-336.pt",
    image_model_type="openai",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=23,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
)


SIGLIP_VISION_BACKBONE = VitConfig(
    init_path=f"{MOLMO_DATA_DIR}/pretrained_image_encoders/siglip-so400m-14-384.pt",
    image_model_type="siglip",
    image_default_input_size=(378, 378),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1152,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=27,
    image_head_dim=72,
    image_mlp_dim=4304,
    image_mlp_activations="gelu_pytorch_tanh",
    image_dropout_rate=0.0,
    image_num_pos=729, # no CLS token
    image_norm_eps=1e-6,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="siglip",
    normalize="siglip",
)


SIGLIP2_VISION_BACKBONE = replace(
    SIGLIP_VISION_BACKBONE,
    init_path=f"{MOLMO_DATA_DIR}/pretrained_image_encoders/siglip2-so400m-14-384.pt",
)


DINOV2_LARGE_336_VISION_BACKBONE = VitConfig(
    init_path=f"{MOLMO_DATA_DIR}/pretrained_image_encoders/dinov2-large-336.pt",
    image_model_type="dino",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=24,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-6,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="dino",
)


METACLIP_L14_336_VISION_BACKBONE = VitConfig(
    init_path=f"{MOLMO_DATA_DIR}/pretrained_image_encoders/metaclip-l14-336.pt",
    image_model_type="openai",
    image_default_input_size=(336, 336),
    image_patch_size=14,
    image_pos_patch_size=14,
    image_emb_dim=1024,
    image_num_heads=16,
    image_num_key_value_heads=16,
    image_num_layers=24,
    image_head_dim=64,
    image_mlp_dim=4096,
    image_mlp_activations="quick_gelu",
    image_dropout_rate=0.0,
    image_num_pos=577,
    image_norm_eps=1e-5,
    attention_dropout=0.0,
    residual_dropout=0.0,
    initializer_range=0.02,
    resize_mode="metaclip",
)


DEFAULT_LOAD_PATHS = {
    "openai": f"{MOLMO_DATA_DIR}/pretrained_image_encoders/vit-l-14-336.pt",
    "siglip": f"{MOLMO_DATA_DIR}/pretrained_image_encoders/siglip-so400m-14-384.pt",
    "dinov2_large_336": f"{MOLMO_DATA_DIR}/pretrained_image_encoders/dinov2-large-336.pt",
    "metaclip_l14_336": f"{MOLMO_DATA_DIR}/pretrained_image_encoders/metaclip-l14-336.pt",
}


VISION_BACKBONES: Dict[str, VitConfig] = {
    "debug": DEBUG_VISION_BACKBONE,
    "openai": DEFAULT_VISION_BACKBONE,
    "siglip": SIGLIP_VISION_BACKBONE,
    "siglip2": SIGLIP2_VISION_BACKBONE,
    "dinov2_large_336": DINOV2_LARGE_336_VISION_BACKBONE,
    "metaclip_l14_336": METACLIP_L14_336_VISION_BACKBONE,
}
