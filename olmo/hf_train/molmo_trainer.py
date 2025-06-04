import torch
import torch.nn as nn

from torch.utils.data import Sampler

from typing import List, Optional
from transformers.trainer_pt_utils import (
    LengthGroupedSampler,
)
from .trainer import (
    has_length,
    logger,
    Trainer,
    is_datasets_available,
    DataLoader,
    datasets
)

from transformers import get_scheduler

class MultiGroupScheduler:
    """
    A wrapper scheduler that applies different warmup steps to different parameter groups
    and is compatible with multi-GPU training.
    """
    def __init__(self, optimizer, config, num_training_steps, scheduler_type="cosine", **kwargs):
        self.optimizer = optimizer
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Create a base scheduler with default warmup
        self.base_scheduler = get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=1,  # This value doesn't matter; we'll override it
            num_training_steps=num_training_steps,
            **kwargs
        )
        
        # Store the warmup steps for each group
        self.warmup_steps = []
        self.num_training_steps = num_training_steps
        
        for i, group in enumerate(optimizer.param_groups):
            if group.get("name") == "connector":
                warmup_steps = int(config.connector_t_warmup)
                print(f"Setting connector warmup steps to {warmup_steps}")
            elif group.get("name") == "vit":
                warmup_steps = int(config.vit_t_warmup)
                print(f"Setting vit warmup steps to {warmup_steps}")
            elif group.get("name") == "llm":
                warmup_steps = int(config.llm_t_warmup)
                print(f"Setting llm warmup steps to {warmup_steps}")
            else:
                raise ValueError(f"Unknown parameter group: {group.get('name', i)}")
            
            self.warmup_steps.append(warmup_steps)
        
        self.current_step = 0
        self.scheduler_type = scheduler_type
        
        # Make sure we're compatible with DeepSpeed
        self._last_lr = self.get_last_lr()
    
    def step(self):
        """
        Advance the scheduler and update the learning rates with appropriate warmup
        for each parameter group.
        """
        # First update base scheduler step count
        self.current_step += 1
        
        # Let the base scheduler step (this is a no-op for our custom scheduling)
        self.base_scheduler.step()
        
        # For each parameter group, calculate its own specific LR
        for i, (group, initial_lr, warmup_steps) in enumerate(zip(
                self.optimizer.param_groups, self.initial_lrs, self.warmup_steps)):
            
            # Apply appropriate lr for this group's warmup stage
            if self.current_step < warmup_steps:
                # Linear warmup
                lr_scale = float(self.current_step) / float(max(1, warmup_steps))
                new_lr = initial_lr * lr_scale
            else:
                # Get the decay factor based on scheduler type
                if self.scheduler_type == "cosine" or self.scheduler_type == "cosine_with_warmup":
                    # Calculate cosine decay factor
                    progress = float(self.current_step - warmup_steps) / float(
                        max(1, self.num_training_steps - warmup_steps))
                    decay_factor = max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item()))
                    new_lr = initial_lr * decay_factor
                elif self.scheduler_type == "linear" or self.scheduler_type == "linear_with_warmup":
                    # Calculate linear decay factor
                    progress = float(self.current_step - warmup_steps) / float(
                        max(1, self.num_training_steps - warmup_steps))
                    decay_factor = max(0.0, 1.0 - progress)
                    new_lr = initial_lr * decay_factor
                else:
                    # For other scheduler types, use a scaling approach
                    decay_factor = self.base_scheduler.get_last_lr()[i] / self.initial_lrs[i]
                    new_lr = initial_lr * decay_factor
            
            # Update the learning rate
            self.optimizer.param_groups[i]['lr'] = new_lr
            
        # Update last_lr for compatibility
        self._last_lr = self.get_last_lr()
    
    def get_last_lr(self):
        """Return the current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Return the state dict for serialization."""
        return {
            "base_scheduler": self.base_scheduler.state_dict(),
            "current_step": self.current_step,
            "initial_lrs": self.initial_lrs,
            "warmup_steps": self.warmup_steps
        }
    
    def load_state_dict(self, state_dict):
        """Load from state dict."""
        self.current_step = state_dict.get("current_step", 0)
        self.initial_lrs = state_dict.get("initial_lrs", self.initial_lrs)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        if "base_scheduler" in state_dict:
            self.base_scheduler.load_state_dict(state_dict["base_scheduler"])
        self._last_lr = state_dict["base_scheduler"]["_last_lr"]
        # update the lr inside the optimizer
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self._last_lr[i]


def create_multi_warmup_scheduler(optimizer, config, num_training_steps, scheduler_type="cosine_with_warmup", **kwargs):
    """
    Create a scheduler that applies different warmup steps to different parameter groups.
    Compatible with DeepSpeed and multi-GPU training.
    
    Args:
        optimizer: PyTorch optimizer with named parameter groups
        config: Config object with connector_t_warmup, vit_t_warmup, and llm_t_warmup attributes
        num_training_steps: Total number of training steps
        scheduler_type: The type of scheduler to use 
                       ("linear_with_warmup", "cosine_with_warmup", etc.)
        
    Returns:
        A MultiGroupScheduler instance
    """
    return MultiGroupScheduler(
        optimizer=optimizer,
        config=config,
        num_training_steps=num_training_steps,
        scheduler_type=scheduler_type,
        **kwargs
    )


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

class  MolmoTrainer(Trainer):

    def get_train_dataloader(self):
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return DataLoader(train_dataset, **dataloader_params)

    def _get_learning_rate(self):
        if hasattr(self, "optimizer") and self.optimizer is not None:
            return {
                f"train/lr/{group.get('name', f'group_{i}')}": group["lr"]
                for i, group in enumerate(self.optimizer.param_groups)
            }
        return {}
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        # You must pass `optimizer` explicitly
        if optimizer is None:
            optimizer = self.optimizer
        if self.lr_scheduler is None:
            self.lr_scheduler = create_multi_warmup_scheduler(
                optimizer=optimizer,
                config=self.args.scheduler_cfg,
                num_training_steps=num_training_steps,
                scheduler_type=self.args.lr_scheduler_type,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            # decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            # decay_parameters = [name for name in decay_parameters if "bias" not in name]

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ("vision_backbone.image_projector" in n and p.requires_grad)
                    ],
                    "weight_decay": self.args.optim_cfg.connector_weight_decay,
                    "lr": self.args.optim_cfg.connector_learning_rate,
                    "betas": self.args.optim_cfg.connector_betas,
                    "eps": self.args.optim_cfg.connector_eps,
                    "name": "connector",
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ("vision_backbone" in n and p.requires_grad and "vision_backbone.image_projector" not in n)
                    ],
                    "weight_decay": self.args.optim_cfg.vit_weight_decay,
                    "lr": self.args.optim_cfg.vit_learning_rate,
                    "betas": self.args.optim_cfg.vit_betas,
                    "eps": self.args.optim_cfg.vit_eps,
                    "name": "vit",
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ("vision_backbone" not in n and p.requires_grad)
                    ],
                    "weight_decay": self.args.optim_cfg.llm_weight_decay,
                    "lr": self.args.optim_cfg.llm_learning_rate,
                    "betas": self.args.optim_cfg.llm_betas,
                    "eps": self.args.optim_cfg.llm_eps,
                    "name": "llm",
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()