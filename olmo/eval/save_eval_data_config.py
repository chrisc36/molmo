import dataclasses

from olmo.config import BaseConfig


@dataclasses.dataclass
class SaveEvalDataConfig(BaseConfig):
    """Configures how to save low-level data from evaluation"""
    post_processed_inputs: bool = True
    example_metadata: bool = True
    model_internal_data: bool = True
    save_token_losses: bool = True
    save_example_losses: bool = True

