"""Image processor class for Molmo"""
from typing import List, Optional, Union, Mapping, Tuple

import numpy as np
import einops
import torch
import torchvision.transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import convert_image_dtype

from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    is_valid_image,
)
from transformers.processing_utils import ImagesKwargs
from transformers.image_processing_utils import BaseImageProcessor
from transformers.utils import logging
from olmo.hf_models.molmoe import MolmoeConfig


logger = logging.get_logger(__name__)


def batch_pixels_to_patches(array, patch_size):
    """Reshape images of [n_images, h, w, 3] -> [n_images, n_patches, pixels_per_patch]"""
    if len(array.shape) == 3:
        n_crops, w, h = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size])
        array = np.transpose(array, [0, 1, 3, 2, 4])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size])
        return array
    else:
        n_crops, w, h, c = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size, c])
        array = np.transpose(array, [0, 1, 3, 2, 4, 5])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size*c])
        return array


def pad_to_bounding_box(
    image, offset_height, offset_width, target_height,
    target_width, value=0
):
    height, width = image.shape[:2]
    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height
    return np.pad(image, [
        [offset_height, after_padding_height],
        [offset_width, after_padding_width],
        [0, 0]
    ], constant_values=value)


def resize_and_pad(
    image,
    desired_output_size,
    is_training=False,
    resize_method="torch-bilinear",
    pad_value=0,
    rng=np.random
):
    """Resize an image while padding to preserve uts aspect ratio."""
    desired_height, desired_width = desired_output_size
    height, width = image.shape[:2]

    # Cast into float32 since the training code did this in float32 and it (very rarely) effects
    # the results after rounding.
    image_scale_y = np.array(desired_height, np.float32) / np.array(height, np.float32)
    image_scale_x = np.array(desired_width, np.float32) / np.array(width, np.float32)
    image_scale = min(image_scale_x, image_scale_y)
    scaled_height = int(np.array(height, np.float32) * image_scale)
    scaled_width = int(np.array(width, np.float32) * image_scale)

    if resize_method in ["tensorflow", "tensorflow-random"]:
        # This how the original training code did resizing, it can produce slightly different
        # results then using torch resize so we keep it just in case
        import tensorflow as tf
        if resize_method == "tensorflow-random" and is_training:
            resize_methods = sorted([k for k in tf.image.ResizeMethod.__dict__.keys() if k.isupper()])
            mode = resize_methods[rng.randint(len(resize_methods))]
            mode = getattr(tf.image.ResizeMethod, mode)
        else:
            mode = tf.image.ResizeMethod.BILINEAR
        image = tf.image.convert_image_dtype(tf.constant(image), dtype=tf.float32)
        image = tf.image.resize(
            image,
            [scaled_height, scaled_width],
            method=mode,
            antialias=True,
        )
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = image.numpy()
    elif resize_method in ["torch-bilinear", "torch-rng"]:
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
        image = convert_image_dtype(image)  # resize in float32 to match the training code
        if resize_method == "torch-rng"  and is_training:
            options = [InterpolationMode.BILINEAR, InterpolationMode.NEAREST_EXACT,
                       InterpolationMode.BICUBIC, InterpolationMode.LANCZOS, InterpolationMode.HAMMING]
            mode = options[rng.randint(len(options))]
        else:
            mode = InterpolationMode.BILINEAR
        image = torchvision.transforms.Resize([scaled_height, scaled_width], mode, antialias=True)(image)
        image = torch.clip(image, 0.0, 1.0)
        image = torch.permute(image, [1, 2, 0]).numpy()
    else:
        raise NotImplementedError(resize_method)

    top_pad = (desired_height - scaled_height) // 2
    left_pad = (desired_width - scaled_width) // 2
    padding = [
        [top_pad, desired_height - scaled_height - top_pad],
        [left_pad, desired_width - scaled_width - left_pad],
        [0, 0]
    ]
    image_mask = np.pad(np.ones_like(image[:, :, 0], dtype=bool), padding[:2])
    image = np.pad(image, padding, constant_values=pad_value)
    return image, image_mask


def metaclip_resize(image, desired_output_size):
    image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    if torch.is_floating_point(image):
        image = torchvision.transforms.Resize(
            desired_output_size, InterpolationMode.BICUBIC, antialias=True)(image)
        image = torch.clip(image, 0.0, 1.0)
    else:
        assert image.dtype == torch.uint8, "Expected float images or uint8 images, but got {}".format(image.dtype)
        image = torchvision.transforms.Resize(
            desired_output_size, InterpolationMode.BICUBIC, antialias=True)(image)
        image = image.to(torch.float32)
        image = torch.clip(image, 0, 255)
        image = image / 255.0
    resized = torch.permute(image, [1, 2, 0]).numpy()
    image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)
    return resized, image_mask


def siglip_resize_and_pad(
    image: np.ndarray,
    desired_output_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    if len(image.shape) == 3:
        is_video = False
        image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    else:
        is_video = True
        image = torch.permute(torch.from_numpy(image), [0, 3, 1, 2])
    dtype = image.dtype
    if torch.is_floating_point(image):
        in_min = 0.0
        in_max = 1.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BILINEAR,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0.0, 1.0).to(dtype)
    else:
        assert image.dtype == torch.uint8, "SigLIP expects float images or uint8 images, but got {}".format(image.dtype)
        in_min = 0.0
        in_max = 255.0
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BILINEAR,
            antialias=False,
        )(image)
        resized = torch.clip(resized, 0, 255).to(dtype)

    resized = resized.to(torch.float32)
    resized = (resized - in_min) / (in_max - in_min)

    if is_video:
        resized = torch.permute(resized, [0, 2, 3, 1]).numpy()
        image_mask = None
    else:
        resized = torch.permute(resized, [1, 2, 0]).numpy()
        image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)

    return resized, image_mask


def dino_resize_and_pad(
    image: np.ndarray,
    desired_output_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    image = torch.permute(torch.from_numpy(image), [2, 0, 1])
    dtype = image.dtype
    if torch.is_floating_point(image):
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BICUBIC,
            antialias=True,
        )(image)
        resized = torch.clip(resized, 0.0, 1.0).to(torch.float32)
    else:
        assert image.dtype == torch.uint8, "DINOv2 expects float images or uint8 images, but got {}".format(image.dtype)
        resized = torchvision.transforms.Resize(
            desired_output_size,
            InterpolationMode.BICUBIC,
            antialias=True,
        )(image)
        resized = torch.clip(resized, 0, 255).to(torch.float32)
        resized = resized / 255.0

    resized = torch.permute(resized, [1, 2, 0]).numpy()
    image_mask = np.ones_like(resized[:, :, 0], dtype=np.bool_)

    return resized, image_mask


def select_tiling(h, w, patch_size, max_num_patches):
    """Decide how best to divide in image of size [w, h] in up to max_num_patches of size patch_size"""
    original_size = np.stack([h, w])  # [1, 2]
    original_res = h * w
    tilings = []
    for i in range(1, max_num_patches+1):
        for j in range(1, max_num_patches+1):
            if i*j <= max_num_patches:
                tilings.append((i, j))
    # sort so argmin and argmax favour smaller tilings in the event of a tie
    tilings.sort(key=lambda x: (x[0]*x[1], x[0]))
    candidate_tilings = np.array(tilings, dtype=np.int32)  # [n_resolutions, 2]
    candidate_resolutions = candidate_tilings * patch_size  # [n_resolutions, 2]

    # How much we would need to scale the image to fit exactly in each tiling
    original_size = np.stack([h, w], dtype=np.float32)  # [1, 2]
    required_scale_d = candidate_resolutions.astype(np.float32) / original_size
    required_scale = np.min(required_scale_d, axis=-1, keepdims=True)  # [n_resolutions, 1]
    if np.all(required_scale < 1):
        # We are forced to downscale, so try to minimize the amount of downscaling
        ix = np.argmax(required_scale)
    else:
        # Pick the resolution that required the least upscaling so that it most closely fits the image
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        ix = np.argmin(required_scale)
    return candidate_tilings[ix]


def arange_for_pooling(idx_arr, pool_h, pool_w):
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(idx_arr, [[h_pad//2, (h_pad+1)//2], [w_pad//2, (w_pad+1)//2]],
                     mode='constant',constant_values=-1)
    return einops.rearrange(
        idx_arr, "(h dh) (w dw) -> h w (dh dw)", dh=pool_h, dw=pool_w)


class MolmoImagesKwargs(ImagesKwargs, total=False):
    max_crops: Optional[int]
    overlap_margins: Optional[List[int]]
    base_image_input_size: Optional[List[int]]
    image_token_length_w: Optional[int]
    image_token_length_h: Optional[int]
    image_patch_size: Optional[int]
    image_padding_mask: Optional[bool]


class MolmoImageProcessor(BaseImageProcessor):
    """Preprocess images and multi-model inputs"""

    def __init__(
        self,
        max_crops: int = 12,
        overlap_margins: List[int] = (4, 4),
        base_image_input_size: List[int] = (336, 336),
        image_token_length_w: int = 12,
        image_token_length_h: int = 12,
        image_patch_size: int = 14,
        image_padding_mask: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        resize: str = "default",
        normalize: str = "openai",
        crop_mode: str = "overlap-and-resize-c2",
        use_col_tokens: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_crops = max_crops
        self.overlap_margins = overlap_margins
        self.base_image_input_size = base_image_input_size
        self.image_token_length_w = image_token_length_w
        self.image_token_length_h = image_token_length_h
        self.image_patch_size = image_patch_size
        self.image_padding_mask = image_padding_mask
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.resize = resize
        self.normalize_mode = normalize
        if resize == "siglip":
            self.normalize_mode = "siglip"
        elif resize == "dino":
            self.normalize_mode = "dino"
        elif resize == "metaclip":
            self.normalize_mode = "metaclip"
        self.crop_mode = crop_mode
        self.use_col_tokens = use_col_tokens
        self.image_pooling_h = 2
        self.image_pooling_w = 2
    
    def get_image_padding_lens(self):
        """Max numbers of image tokens can be built for one image"""
        padding_lens = dict(
            images=self.max_crops
        )
        padding_lens["image_masks"] = self.max_crops
        return padding_lens

    def normalize_image(self, image):
        if self.normalize_mode == "openai":
            image -= np.array(OPENAI_CLIP_MEAN, dtype=np.float32)[None, None, :]
            image /= np.array(OPENAI_CLIP_STD, dtype=np.float32)[None, None, :]
        elif self.normalize_mode == "siglip" or self.normalize_mode == "siglip2":
            image = np.asarray(-1.0, dtype=np.float32) + image * np.asarray(2.0, dtype=np.float32)
        elif self.normalize_mode == "dino":
            image -= np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
            image /= np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
        else:
            raise NotImplementedError(self.normalize_mode)
        return image
    
    def resize_image(self, image, output_size, is_training, rng):
        if self.resize == "siglip":
            return siglip_resize_and_pad(image, output_size)
        elif self.resize == "dino":
            return dino_resize_and_pad(image, output_size)
        elif self.resize == "metaclip":
            return metaclip_resize(image, output_size)
        else:
            resize = "torch-bilinear" if self.resize == "default" else self.resize
            return resize_and_pad(
                image, output_size, rng=rng, is_training=is_training,
                resize_method=resize)
    
    def build_resized_image(self, image, is_training, rng):
        resized, resized_mask = self.resize_image(image, self.base_image_input_size, is_training, rng)
        resized = self.normalize_image(resized)
        if len(resized.shape) == 3:
            resized = np.expand_dims(resized, 0)
        resized_mask = np.expand_dims(resized_mask, 0)
        crop_patch_w = self.base_image_input_size[1] // self.image_patch_size
        crop_patch_h = self.base_image_input_size[0] // self.image_patch_size
        resize_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
        return resized, resized_mask, resize_idx
    
    def build_overlapping_crops(self, image, is_training, rng):
        """Decompose an image into a set of overlapping crops

        :return crop_arr: [n_crops, h, w, 3] The crops
        :return mask_arr: [n_crops, h, w] The padding masks
        :return patch_idx: [overlap_patch_h, overlap_patch_w] For each patch in the resized image
                           the crops were extracted from, what patch in `crop_arr` it corresponds to
        """
        original_image_h, original_image_w = image.shape[:2]
        max_crops = self.max_crops
        overlap_margins = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        image_patch_size = self.image_patch_size
        crop_size = base_image_input_size[0]
        assert base_image_input_size[0] == base_image_input_size[1]

        left_margin, right_margin = overlap_margins
        total_margin_pixels = image_patch_size*(right_margin + left_margin)  # pixels removed per dim
        crop_patches = base_image_input_size[0] // image_patch_size  # patches per crop dim
        crop_window_patches = crop_patches - (right_margin + left_margin)  # usable patches
        crop_window_size = crop_window_patches * image_patch_size
        crop_patch_w = base_image_input_size[1] // image_patch_size
        crop_patch_h = base_image_input_size[0] // image_patch_size
        original_image_h, original_image_w = image.shape[:2]
        crop_size = base_image_input_size[0]

        # Decide how to tile the image, to account for the overlap margins we compute the tiling
        # as if we had an image without the margins and were using a crop size without the margins
        tiling = select_tiling(
            original_image_h - total_margin_pixels,
            original_image_w - total_margin_pixels,
            crop_window_size,
            max_crops
        )
        src, img_mask = self.resize_image(
            image,
            [tiling[0]*crop_window_size+total_margin_pixels, tiling[1]*crop_window_size+total_margin_pixels],
            is_training,
            rng
        )
        src = self.normalize_image(src)

        # Now we have to split the image into crops, and track what patches came from
        # where in `patch_idx_arr`
        n_crops = tiling[0] * tiling[1]
        crop_arr = np.zeros([n_crops, crop_size, crop_size, 3], dtype=src.dtype)
        mask_arr = np.zeros([n_crops, crop_size, crop_size], dtype=img_mask.dtype)
        patch_idx_arr = np.zeros([n_crops, crop_patch_h, crop_patch_w], dtype=np.int32)
        on = 0
        on_crop = 0
        for i in range(tiling[0]):
            # Slide over `src` by `crop_window_size` steps, but extract crops of size `crops_size`
            # which results in overlapping crop windows
            y0 = i*crop_window_size
            for j in range(tiling[1]):
                x0 = j*crop_window_size
                crop_arr[on_crop] = src[y0:y0+crop_size, x0:x0+crop_size]
                mask_arr[on_crop] = img_mask[y0:y0+crop_size, x0:x0+crop_size]
                patch_idx = np.arange(crop_patch_w*crop_patch_h).reshape(crop_patch_h, crop_patch_w)
                patch_idx += on_crop * crop_patch_h * crop_patch_w

                # Mask out idx that are in the overlap region
                if i != 0:
                    patch_idx[:left_margin, :] = -1
                if j != 0:
                    patch_idx[:, :left_margin] = -1
                if i != tiling[0]-1:
                    patch_idx[-right_margin:, :] = -1
                if j != tiling[1]-1:
                    patch_idx[:, -right_margin:] = -1
                patch_idx_arr[on_crop] = patch_idx
                on_crop += 1

        # `patch_idx_arr` is ordered crop-by-crop, here we transpose `patch_idx_arr`
        # so it is ordered left-to-right order
        patch_idx_arr = np.reshape(
            patch_idx_arr,
            [tiling[0], tiling[1], crop_patch_h, crop_patch_w]
        )
        patch_idx_arr = np.transpose(patch_idx_arr, [0, 2, 1, 3])
        patch_idx_arr = np.reshape(patch_idx_arr, [-1])

        # Now get the parts not in the overlap region, so it should map each patch in `src`
        # to the correct patch it should come from in `crop_arr`
        patch_idx_arr = patch_idx_arr[patch_idx_arr >= 0].reshape(
            src.shape[0]//image_patch_size,
            src.shape[1]//image_patch_size,
        )
        return crop_arr, mask_arr, patch_idx_arr

    def compute_overlapping_crops_size(self, image_h, image_w) -> Tuple[int, int]:
        """Returns the number of patches the multi-crop image would have"""
        image_patch_size = self.image_patch_size
        crop_patch_w = self.base_image_input_size[1] // image_patch_size
        crop_patch_h = self.base_image_input_size[0] // image_patch_size
        margin_patches = sum(self.overlap_margins)
        margin_pixels = image_patch_size*margin_patches  # pixels removed per dim
        assert crop_patch_w == crop_patch_h
        crop_window_patches = crop_patch_w - margin_patches
        crop_window_size = crop_window_patches * image_patch_size
        tiling = select_tiling(
            image_h - margin_pixels,
            image_w - margin_pixels,
            crop_window_size,
            self.max_crops
        )
        h, w = [tiling[0]*crop_window_size+margin_pixels, tiling[1]*crop_window_size+margin_pixels]
        return h//image_patch_size, w//image_patch_size

    
    def image_to_patches_and_tokens(
        self,
        image: ImageInput,
        pooling_h: int,
        pooling_w: int,
        image_patch_token_id: int,
        image_col_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
        is_training=False,
        rng=None,
    ):
        """
        :return image_tokens, the token IDS for this image, including special tokens
        :return crops, the image crops to processes with the ViT
        :return mask, the padding mask for each crop
        :return pooled_patch_idx, for each patch_id tokens in `image_tokens`, the indices of the
                                  patches in `crops` to pool for that token, masked with -1
        """
        max_crops = self.max_crops
        overlap_margins = self.overlap_margins
        base_image_input_size = self.base_image_input_size
        image_patch_size = self.image_patch_size

        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        base_image_input_d = image_patch_size
        crop_patch_w = base_image_input_size[1] // base_image_input_d
        crop_patch_h = base_image_input_size[0] // base_image_input_d

        original_image_h, original_image_w = image.shape[:2]
        crop_size = base_image_input_size[0]

        if self.crop_mode == "resize":
            resized, resized_mask, resize_idx = self.build_resized_image(image, is_training=is_training, rng=rng)
            resize_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
            pooling_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
            h, w = pooling_idx.shape[:2]
            pooling_idx = pooling_idx.reshape([-1, pooling_h*pooling_w])
            per_row = np.full(
                (w,),
                image_patch_token_id,
                dtype=np.int32
            )
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [h])
            joint = [
                [image_start_token_id],
                extra_tokens,
                [image_end_token_id],
            ]
            return (np.concatenate(joint, 0), batch_pixels_to_patches(resized, image_patch_size),
                    batch_pixels_to_patches(resized_mask, image_patch_size).mean(-1), pooling_idx)

        if self.crop_mode in ["overlap-and-resize-c2", "overlap-and-resize"]:
            crop_arr, mask_arr, patch_idx_arr = self.build_overlapping_crops(image, is_training=is_training, rng=rng)
            pooling_idx = arange_for_pooling(patch_idx_arr, pooling_h, pooling_w)
            h, w = pooling_idx.shape[:2]
            pooling_idx = pooling_idx.reshape([-1, pooling_h*pooling_w])

            # Now build the output tokens
            per_row = np.full(w, image_patch_token_id, dtype=np.int32)
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [image_col_token_id]], 0)
            joint = np.tile(per_row, [h])
            joint = [
                [image_start_token_id],
                joint,
                [image_end_token_id]
            ]

            if self.crop_mode == "overlap-and-resize":
                crop_arr = batch_pixels_to_patches(crop_arr, image_patch_size)
                mask_arr = batch_pixels_to_patches(mask_arr, image_patch_size).astype(np.float32).mean(axis=-1)
                return np.concatenate(joint, 0), crop_arr, mask_arr, pooling_idx

            # Finally do the same for the global image
            resized, resized_mask, resize_idx = self.build_resized_image(image, is_training=is_training, rng=rng)
            crop_arr = np.concatenate([resized, crop_arr], 0)

            mask_arr = np.concatenate([resized_mask, mask_arr], 0)

            resize_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
            h, w = resize_idx.shape[:2]
            resize_idx = resize_idx.reshape([-1, pooling_h*pooling_w])

            # Global image goes first, so the order of patches in previous crops gets increased
            pooling_idx = np.where(
                pooling_idx >= 0,
                pooling_idx + crop_patch_h*crop_patch_w,
                -1
            )
            pooling_idx = np.concatenate([resize_idx, pooling_idx])

            per_row = np.full(
                (w,),
                image_patch_token_id,
                dtype=np.int32
            )
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [h])
            joint = [
                        [image_start_token_id],
                        extra_tokens,
                        [image_end_token_id],
                    ] + joint
            mask_arr = batch_pixels_to_patches(mask_arr, image_patch_size).astype(np.float32).mean(axis=-1)
            return (np.concatenate(joint, 0), batch_pixels_to_patches(crop_arr, image_patch_size),
                    mask_arr, pooling_idx)
        else:
            raise NotImplementedError(self.crop_mode)

    def multimodal_preprocess(
        self,
        images: np.ndarray,
        tokens: List[int],
        loss_masks: List[int],
        image_idx: np.ndarray,
        image_patch_token_id: int,
        image_col_token_id: int,
        image_start_token_id: int,
        image_end_token_id: int,
        alter_attn_mask: bool = False,
        is_training: bool = False,
        rng: np.random.RandomState = np.random,
        **kwargs,
    ):
        """Merge images and text tokens into multi-modal features for the model
        :param images: images to use as input
        :param tokens: input text tokens
        :param image_idx: where to insert the images into `tokens`
        :params image_patch_token_id: id to use of tokens that will contain image features
        :params image_col_token_id: token id for image column special tokens
        :params image_start_token_id: token id for image start special tokens
        :params image_end_token_id: token id for image end special tokens
        :params kwargs: override preprocessor default args
        """
        image_padding_mask = kwargs.get("image_padding_mask") or self.image_padding_mask

        if images is None:
            if len(tokens) > 1:
                tokens = np.concatenate(tokens, 0)
            else:
                tokens = np.array(tokens[0])
            return {
                "input_ids": tokens,
                "loss_masks": np.ones_like(tokens),
            }
        else:
            n = len(images)
            n_msgs = len(tokens)

            all_crop_masks = []
            all_crops = []
            pooled_patches_idx = []
            all_text_part_indices = []
            out_tokens = []
            all_loss_masks = []

            for ix in range(n):
                token_ix = image_idx[ix]
                image_tokens, crops, img_mask, pooled_idx = self.image_to_patches_and_tokens(
                    images[ix], self.image_pooling_h, self.image_pooling_w, 
                    image_patch_token_id,
                    image_col_token_id,
                    image_start_token_id,
                    image_end_token_id, 
                    is_training=is_training,
                    rng=rng)

                if token_ix == -1:  # -1 is an image inserted at the very start
                    start = 0
                    token_ix = 0
                    end = 0
                else:
                    start = 0 if ix == 0 else image_idx[ix-1] + 1
                    end = token_ix + 1

                pooled_patches_idx.append(pooled_idx + sum(np.prod(x.shape[:2]) for x in all_crops))
                all_crops.append(crops)
                out_tokens.append(tokens[ix][start:token_ix])
                all_loss_masks += [0] * (token_ix - start)
                img_start_idx = len(out_tokens[-1]) if len(out_tokens) > 0 else 0
                out_tokens.append(image_tokens)
                img_tokens_end = len(out_tokens[-1]) + img_start_idx
                all_loss_masks += [0] * len(image_tokens)
                
                if ix == (n - 1):
                    out_tokens.append(tokens[ix][end:])
                    all_loss_masks += loss_masks[ix][end:]
                    last_text_end = 1 + img_tokens_end + len(out_tokens[-1])
                    text_part_indices = (img_start_idx, img_tokens_end + 1, last_text_end) # plus 1 to take care of bos added later
                    all_text_part_indices.append(text_part_indices)
                if image_padding_mask:
                    all_crop_masks.append(img_mask)

            
            for i in range(n, n_msgs):
                out_tokens.append(tokens[i])
                all_loss_masks += loss_masks[i]
                last_text_end += len(out_tokens[-1])
                if i < n_msgs - 1:
                    text_part_indices = (img_start_idx, img_tokens_end + 1, last_text_end) # plus 1 to take care of bos added later
                    all_text_part_indices.append(text_part_indices) 
            
            input_ids = np.concatenate(out_tokens, 0)
            images = np.concatenate(all_crops, 0)
            loss_masks = np.array(all_loss_masks, dtype=np.float32)
            
            pooled_patches_idx = np.concatenate(pooled_patches_idx, 0)
            if image_padding_mask:
                image_masks = np.concatenate(all_crop_masks, 0)
            else:
                image_masks = None
        
        out = {
            "input_ids": input_ids,
            "loss_masks": loss_masks,
            "images": images,
            "pooled_patches_idx": pooled_patches_idx,
            "image_token_indices": [0, image_tokens.shape[0]],
        }
        if len(all_text_part_indices) > 0 and alter_attn_mask:
            out["text_part_indices"] = all_text_part_indices
        if image_masks is not None:
            out["image_masks"] = image_masks
        return out


from transformers import AutoImageProcessor
AutoImageProcessor.register(MolmoeConfig, MolmoImageProcessor)
