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


class Dataset:
    @classmethod
    def download(cls, n_procs=1):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        return self.get(item, np.random)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get(self, item, rng):
        # `rng` is used to support deterministic data augmentation for tasks that require it.
        # Used to avoid the hazards of relying on the global rng state for determinism
        raise NotImplementedError()


class DeterministicDataset:
    """Dataset wrapper that supports padding and control the random seed based on the epoch"""

    def __init__(self, dataset: Dataset, preprocessor, seed, n_pad=0, for_inference=None):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.seed = seed
        self.n_pad = n_pad
    
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
        if self.preprocessor:
            item = self.preprocessor(item, rng)
        return item

class DatasetBase(Dataset):
    def __init__(self, split, sample: int=None):
        super().__init__()
        self.split = split
        self.sample = sample
        self.data = self.load()[:self.sample]

    def load(self):
        raise NotImplementedError()

    def __len__(self):
        if self.data is None:
            raise ValueError("Dataset not loaded")
        return len(self.data)

    def __getitem__(self, item):
        return self.get(item, np.random)

    def get(self, item, rng):
        raise NotImplementedError()