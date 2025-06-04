"""
Processor class for Molmo.
"""

from typing import Optional

import PIL
from PIL import ImageOps
from PIL.Image import Image

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

import numpy as np
import torch

from transformers.image_utils import ImageInput
from transformers.processing_utils import (
    TextKwargs,
    ProcessingKwargs,
    ProcessorMixin,
)

from transformers.tokenization_utils_base import TextInput, PreTokenizedInput
from transformers.utils import logging

from transformers import AutoTokenizer
from olmo.hf_models.processing.image_processing_molmo import MolmoImagesKwargs, MolmoImageProcessor
from olmo.hf_models.molmoe import MolmoeConfig

logger = logging.get_logger(__name__)


DEFAULT_IMAGE_PATCH_TOKEN = f"<im_patch>"
DEFAULT_IM_START_TOKEN = f"<im_start>"
DEFAULT_IM_END_TOKEN = f"<im_end>"
DEFAULT_IM_COL_TOKEN = f"<im_col>"
IMAGE_PROMPT = "<|image|>"

EXTRA_TOKENS = (DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_COL_TOKEN, IMAGE_PROMPT)


def get_special_token_ids(tokenizer):
    ids = tokenizer.encode("".join(EXTRA_TOKENS), add_special_tokens=False)
    if len(ids) != len(EXTRA_TOKENS):
        tokenizer.add_tokens(list(EXTRA_TOKENS))
        ids = tokenizer.encode("".join(EXTRA_TOKENS), add_special_tokens=False)
    assert len(ids) == len(EXTRA_TOKENS)
    return {k: i for k, i in zip(EXTRA_TOKENS, ids)}


class MolmoTextKwargs(TextKwargs, total=False):
    style: Optional[str]
    system_prompt: Optional[str]
    message_format: Optional[str]
    always_start_with_space: Optional[bool]
    sequence_length: Optional[int]


class MolmoProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: MolmoTextKwargs
    images_kwargs: MolmoImagesKwargs
    _defaults = {
        "text_kwargs": {
            "style": "long_caption",
            "system_prompt": "none",
            "message_format": "role",
            "always_start_with_space": True,
            "padding": False,
        },
    }


class MolmoProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("GPT2Tokenizer", "GPT2TokenizerFast", "GPTNeoXTokenizerFast", "Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor: MolmoImageProcessor = None, tokenizer : AutoTokenizer = None, data_formatter_cfg: dict = None,  **kwargs):
        # self.image_processor = image_processor
        # self.tokenizer = tokenizer
        super().__init__(image_processor, tokenizer)
        self._special_tokens = None
        if data_formatter_cfg is None:
            self.data_formatter_cfg = dict(
                    prompt_templates='uber_model',
                    message_format='role',
                    system_prompt='demo_or_style',
                    always_start_with_space=True,
                    default_inference_len=65,
                    select_answer='best',
                    debug=False,
                    image_last=False,
                    is_hf_model=True,
                    is_training=False,
                )
        else:
            self.data_formatter_cfg = data_formatter_cfg
        self.is_training = self.data_formatter_cfg.get("is_training", False)
            
    @property
    def special_token_ids(self):
        if self._special_tokens is None:
            self._special_tokens = get_special_token_ids(self.tokenizer)
        return self._special_tokens

    def get_tokens_input(self, prompt, message_format, is_train):

        loss_masks = []
        tokens = []

        if len(prompt) == 1 and isinstance(prompt, (list, tuple)):
            prompt = prompt[0]
        
        if isinstance(prompt, str):
            prompt = [prompt]
        
        assert isinstance(prompt, list), f"Prompt should be a list, got {type(prompt)}"
        
        for idx, p in enumerate(prompt):
            toks = self.tokenizer.encode(p, add_special_tokens=False)
            tokens += toks
            if idx % 2 == 1:
                loss_masks += [1] * len(toks)
            else:
                loss_masks += [0] * len(toks)

        if is_train:
            if tokens[-1] != self.tokenizer.eos_token_id:
                tokens.append(self.tokenizer.eos_token_id)
                loss_masks.append(1)
        return tokens, loss_masks

    def process(
        self,
        text: TextInput = None,
        images: ImageInput = None,
        is_train: bool = False,
        rng: Optional[np.random.Generator] = None,
        *,
        tokens: Optional[PreTokenizedInput] = None,
        **kwargs: Unpack[MolmoProcessorKwargs],
    ):
        
        assert isinstance(text, (str, list, tuple)), f"Text input should be str or list of str, got {type(text)}"
        if isinstance(text, str):
            text = [text]
        
        output_kwargs = self._merge_kwargs(
            MolmoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if tokens is None:
            tokens = []
            loss_masks = []
            if len(text) == 2 and isinstance(text[0], (str)):
                text = [text]
            for msg in text:
                msg_tokens, msg_loss_masks = self.get_tokens_input(
                    msg,
                    output_kwargs["text_kwargs"]["message_format"],
                    is_train,
                )
                tokens.append(msg_tokens)
                loss_masks.append(msg_loss_masks)
        else:
            loss_masks = [0] * len(tokens)
        
        if images is not None:
            if not isinstance(images, (list, tuple)):
                images = [images]
            image_arrays = []
            for image in images:
                if isinstance(image, Image):
                    image = image.convert("RGB")
                    # Handle images with EXIF orientation tags, which PIL will ignore by default
                    # https://github.com/python-pillow/Pillow/issues/4703
                    # img = ImageOps.exif_transpose(image)
                    image_arrays.append(np.array(image))
                else:
                    assert len(image.shape) == 3 and image.shape[-1] == 3
                    image_arrays.append(image.astype(np.uint8))
            images = image_arrays
            # For now only support inserting images at the start
            image_idx = [-1]*len(images)
        else:
            image_idx = None

        image_patch_token_id = self.special_token_ids[DEFAULT_IMAGE_PATCH_TOKEN]
        image_col_token_id = self.special_token_ids[DEFAULT_IM_COL_TOKEN]
        image_start_token_id = self.special_token_ids[DEFAULT_IM_START_TOKEN]
        image_end_token_id = self.special_token_ids[DEFAULT_IM_END_TOKEN]
        out = self.image_processor.multimodal_preprocess(
            images=images,
            image_idx=image_idx,
            tokens=tokens,
            loss_masks=loss_masks,
            image_patch_token_id=image_patch_token_id,
            image_col_token_id=image_col_token_id,
            image_start_token_id=image_start_token_id,
            image_end_token_id=image_end_token_id,
            alter_attn_mask=self.is_training,
            is_training=is_train,
            rng=rng,
            **output_kwargs["images_kwargs"]
        )

        # Prepend BOS
        # qwen2 and olmo do not have a BOS, and instead use EOS as a generic seperator token.
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        decoder_input_tokens = np.pad(out["input_ids"], [[1, 0]], constant_values=bos)
        out["loss_masks"] = np.pad(out["loss_masks"], [[1, 0]], constant_values=0)
        out["input_ids"] = decoder_input_tokens
        if is_train:
            out["labels"] = decoder_input_tokens.copy()
            out["labels"][0] = -100
            out["labels"] = np.where(out["loss_masks"] == 1, out["labels"], -100)
            assert out["labels"][-1] != -100, "Last token should not be masked"
            if out.get("text_part_indices", None) is not None:
                attn_indices = out["text_part_indices"]
                for a_idxs in attn_indices:
                    assert out["labels"][a_idxs[2]-1] != -100, f"Last token should not be masked, got {out['labels'][a_idxs[2]-1]} at {a_idxs[2]-1}"
        else:
            assert out["input_ids"][-1] != self.tokenizer.eos_token_id

        for k, v in out.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v)

        return out


from transformers import AutoProcessor
AutoProcessor.register(MolmoeConfig, MolmoProcessor)