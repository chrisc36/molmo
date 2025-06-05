import os
import warnings
from os.path import join
import warnings
from io import BytesIO
import PIL
from PIL import ImageFile, ImageOps
from PIL import ImageOps
from olmo.io import get_bytes_range
import copy


def setup_pil():
    PIL.Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
from olmo.models.molmo.data_formatter import DataFormatter

DEFAULT_IMAGE_PATH = "/weka/oe-training-default/mm-olmo/torch_datasets"

def load_image(image_path):
    setup_pil()  # Call here so the setting is applied in multi-processing contexts
    if isinstance(image_path, PIL.Image.Image):
        # Avoid annoying palette transparency warnings filling up the logs
        with warnings.catch_warnings(record=True) as w:
            image = image_path.convert("RGB")
        try:
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            pass
        return image
    elif isinstance(image_path, np.ndarray):
        assert len(image_path.shape) == 3, "Image should have 3 dimensions"
        assert image_path.shape[2] == 3, "Image should have 3 channels"
        assert image_path.dtype == np.uint8, "Image should have uint8 type"
        return PIL.Image.fromarray(image_path)
    else:
        # This a bit of hack to handle cases where the image path was hard-coded
        # into the dataset to the weka path
        if DATA_HOME != DEFAULT_IMAGE_PATH and DEFAULT_IMAGE_PATH in image_path:
            image_path = image_path.replace(DEFAULT_IMAGE_PATH, DATA_HOME)

        # Ignore image loading warning
        with warnings.catch_warnings(record=True) as w:
            if image_path.startswith("gs://"):
                image_bytes = get_bytes_range(image_path, 0, None)
                return PIL.Image.open(BytesIO(image_bytes))
            else:
                return PIL.Image.open(image_path)

if "MOLMO_DATA_DIR" in os.environ:
    DATA_HOME = join(os.environ["MOLMO_DATA_DIR"], "torch_datasets")
    VIDEO_DATA_HOME = join(os.environ["MOLMO_DATA_DIR"], "video_datasets")
else:
    warnings.warn("MOLMO_DATA_DIR is not set, data loading might fail")
    DATA_HOME = None
    VIDEO_DATA_HOME = None


class HFDeterministicDataset:
    """Dataset wrapper that supports padding and control the random seed based on the epoch"""

    def __init__(self, dataset, preprocessor, seed, n_pad=0, for_inference=None):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.seed = seed
        self.n_pad = n_pad

        # formatting for hf models
        if not hasattr(self.preprocessor, "formater"):
            if not hasattr(self.preprocessor, "data_formatter_cfg"):
                self.formater = DataFormatter(prompt_templates='uber_model',
                                message_format='role',
                                system_prompt='demo_or_style',
                                always_start_with_space=True,
                                default_inference_len=65,
                                select_answer='best',
                                debug=False,
                                image_last=False)
                self.is_training = False
                self.for_inference = True
            else:
                data_formatter_cfg = copy.deepcopy(self.preprocessor.data_formatter_cfg)
                self.is_hf_model = data_formatter_cfg.pop("is_hf_model", True)
                assert self.is_hf_model, "are you sure this is not an HF model?"
                self.is_training = data_formatter_cfg.pop("is_training", False)
                if self.is_training:
                    self.for_inference = False
                else:
                    self.for_inference = True
                self.formater = DataFormatter(**data_formatter_cfg)

        if for_inference is not None:
            self.for_inference = for_inference
            self.is_training = not for_inference
    
    def __len__(self):
        return len(self.dataset) + self.n_pad

    def __getitem__(self, idx):
        return self.get(idx, 0)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get(self, idx, epoch=0):
        rng = np.random.RandomState(
            (self.seed * 195172 + idx + len(self.dataset)*epoch) % (2 ** 32 - 1))
        
        if idx >= len(self.dataset):
            # Padding example
            item = self.dataset.get(0, rng)
            if "metadata" not in item:
                item["metadata"] = {}
            item["metadata"]["valid"] = False
        else:
            item = self.dataset.get(idx, rng) 
        
        if "image" in item:
            try:
                image = load_image(item["image"])
            except Exception as e:
                raise ValueError(f"Could not load image: {item['image']}")
            else:
                item["image"] = image
        else:
            image = None
        
        metadata = item.get("metadata")
        if metadata is None:
            metadata = {}
        if "image_size" not in metadata and image is not None:
            metadata["image_size"] = image.size
        
        messages, formatter_metadata = self.formater(item, self.is_training, self.for_inference, rng)

        if isinstance(messages[0], list):
            # If there are multiple conversations for this example, shuffle their order
            # This might matter if we truncate the tokens to a max sequence length
            rng.shuffle(messages)

        if formatter_metadata:
            metadata.update(formatter_metadata)
     
        item = self.preprocessor.process(
                images=[image],
                text=messages,
                is_train=self.is_training,
                message_format=self.formater.message_format,
                always_start_with_space=self.formater.always_start_with_space,
                rng=rng,
            )
            
        if "image_token_indices" in item:
            metadata["image_token_indices"] = item.pop("image_token_indices")
            
        item["metadata"] = metadata
        return [item]