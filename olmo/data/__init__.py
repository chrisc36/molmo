# import logging
#
# import numpy as np
# from torch.utils.data import DataLoader, DistributedSampler
#
# from olmo.config import DataConfig, TrainConfig, ModelConfig
# from olmo.data.academic_datasets import ChartQa, ScienceQAImageOnly, TextVqa, OkVqa, DocQa, \
#     InfoQa, AOkVqa, Vqa2, PlotQa, FigureQa, DvQa, SceneTextQa, TabWMPDirectAnswer, \
#     AndroidControl, TallyQa, AI2D, CountBenchQa, RealWorldQa, MathVista, MMMU, ClockBench
# from olmo.data.collator import MMCollator
# from olmo.data.data_formatter import DataFormatter
# from olmo.data.dataset import DeterministicDataset
# from olmo.data.iterable_dataset_mixture import IterableDatasetMixture
# from olmo.data.model_preprocessor import Preprocessor, MultiModalPreprocessor
# from olmo.data.pixmo_datasets import PixMoPointExplanations as PixMoPointExplanationHF, \
#     PixMoDocs, PixMoCount, PixMoPoints, PixMoCapQa, PixMoCap, PixMoPointExplanations, \
#     PixMoAskModelAnything, PixMoPointsEval
# from olmo.torch_util import get_global_rank, get_world_size
#
# log = logging.getLogger(__name__)
#
#
# def build_mm_preprocessor(
#     model_config: ModelConfig,
#     for_inference=False,
#     shuffle_messages=True,
#     is_training=False,
#     require_image_features=False
# ):
#     v_cfg = model_config.vision_backbone
#     h, w = model_config.llm_patches_per_crop()
#     if not model_config.image_padding_embed:
#         image_padding_mask = None
#     elif model_config.fix_image_padding:
#         image_padding_mask = 2
#     else:
#         image_padding_mask = 1
#
#     return Preprocessor(
#         DataFormatter(
#             prompt_templates=model_config.prompt_type,
#             message_format=model_config.message_formatting,
#             system_prompt=model_config.system_prompt_kind,
#             always_start_with_space=model_config.always_start_with_space,
#             default_inference_len=model_config.default_inference_len
#         ),
#         MultiModalPreprocessor(
#             tokenizer=model_config.get_tokenizer(),
#             normalize=str(v_cfg.image_model_type),
#             crop_mode=model_config.crop_mode,
#             max_crops=model_config.max_crops,
#             overlap_margins=model_config.overlap_margins,
#             resize=v_cfg.resize_mode,
#             use_col_tokens=model_config.use_col_tokens,
#             base_image_input_size=v_cfg.image_default_input_size,
#             image_pooling_w=model_config.image_pooling_w,
#             image_pooling_h=model_config.image_pooling_h,
#             image_token_length_w=w,
#             image_token_length_h=h,
#             image_patch_size=v_cfg.image_patch_size,
#             image_padding_mask=image_padding_mask,
#             pad_value=model_config.pad_value,
#             loss_token_weighting=model_config.multi_annotation_weighting,
#         ),
#         for_inference=for_inference,
#         shuffle_messages=shuffle_messages,
#         is_training=is_training,
#         require_image_features=require_image_features,
#     )
#
#
# def build_torch_mm_eval_dataloader(
#     batch_size, seed, model_config, data_config, pad_batches, max_steps=None
# ):
#     preprocessor = build_mm_preprocessor(
#         model_config, for_inference=data_config.for_inference, shuffle_messages=data_config.shuffle_messages,
#         require_image_features=pad_batches
#     )
#     logging.info(f"Loading eval dataset: {data_config.dataset}/{data_config.split}")
#     dataset = get_dataset_by_name(data_config.dataset, data_config.split)
#     n_pad = 0
#     if pad_batches:
#         global_batch_size = batch_size*get_world_size()
#         n_steps = (len(dataset) + global_batch_size - 1) // global_batch_size
#         if max_steps:
#             n_steps = min(n_steps, max_steps)
#         if n_steps*global_batch_size > len(dataset):
#             # Pad the dataset so that it can produce enough batches of `global_batch_size` size
#             # to cover the entire dataset without dropping any examples
#             # We need this if evaluating FSDP models since they will need all devices to get
#             # exactly the same number of batches
#             n_pad = (n_steps*global_batch_size) - len(dataset)
#
#     dataset = DeterministicDataset(
#         dataset=dataset,
#         seed=seed,
#         preprocessor=preprocessor,
#         n_pad=n_pad
#     )
#
#     sampler = DistributedSampler(
#         dataset,
#         drop_last=data_config.drop_last,
#         shuffle=data_config.shuffle,
#         num_replicas=get_world_size(),
#         rank=get_global_rank(),
#         seed=seed,
#     )
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         collate_fn=MMCollator(
#             data_config.sequence_length,
#             max_crops=model_config.get_max_crops(),
#             include_metadata=True,
#             pad=data_config.pad,
#         ),
#         num_workers=data_config.num_workers,
#         sampler=sampler,
#         pin_memory=data_config.pin_memory,
#         prefetch_factor=None if data_config.num_workers == 0 else data_config.prefetch_factor,
#         persistent_workers=False if data_config.num_workers == 0 else data_config.persistent_workers,
#         timeout=data_config.timeout,
#     )
#
#
# def build_eval_dataloader(
#     train_config: TrainConfig,
#     data_config: DataConfig,
#     batch_size: int,
#     max_steps=None
# ) -> DataLoader:
#     seed = data_config.seed if data_config.seed is not None else train_config.seed
#     if data_config.multi_modal in ["torch"]:
#         return build_torch_mm_eval_dataloader(
#             batch_size, seed, train_config.model, data_config,
#             pad_batches=train_config.fsdp is not None and not data_config.drop_last,
#             max_steps=max_steps
#         )
#     else:
#         raise NotImplementedError(data_config.multi_modal)
#
#
# def build_train_dataloader(train_config: TrainConfig, device=None) -> DataLoader:
#     if device is None:
#         device = "cpu"
#     assert train_config.device_train_batch_size is not None
#     seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
#     data_config = train_config.data
#     if train_config.data.multi_modal in ["torch", "torch_hf"]:
#         preprocessor = build_mm_preprocessor(
#             train_config.model, shuffle_messages=data_config.shuffle, is_training=True, require_image_features=True)
#         if data_config.dataset:
#             datasets = [get_dataset_by_name(
#                 data_config.dataset, data_config.split)]
#             rates = [1]
#         else:
#             if data_config.mixture:
#                 mixture = {}
#                 for name, rate in data_config.mixture.items():
#                     logging.info(f"Loading train dataset {name}/{data_config.split}")
#                     mixture[name] = (get_dataset_by_name(name, data_config.split), rate)
#             else:
#                 mixture = {}
#                 for root_size_mixture in data_config.root_size_mixture:
#                     group_datasets = {}
#                     for name, as_size in root_size_mixture.mixture.items():
#                         logging.info(f"Loading train dataset {name}/{data_config.split}")
#                         dataset = get_dataset_by_name(name, data_config.split)
#                         if as_size is not None:
#                             size = as_size
#                         else:
#                             size = len(dataset)
#                         group_datasets[name] = (dataset, np.sqrt(size))
#                     total_rate = sum(x[1] for x in group_datasets.values())
#                     mixture.update({name: (ds, r/total_rate*root_size_mixture.rate)
#                                      for name, (ds, r) in group_datasets.items()})
#
#             total_rate = sum(x[1] for x in mixture.values())
#             mixture = sorted(mixture.items(), key=lambda x: x[0])
#             rates = [rate/total_rate for (_, (_, rate)) in mixture]
#             datasets = [ds for (_, (ds, _)) in mixture]
#             logging.info("Sampling rates:")
#             names = list(x[0] for x in mixture)
#             for ix in np.argsort(rates)[::-1]:
#                 logging.info(f"{names[ix]}: {100*rates[ix]:0.2f}")
#         datasets = [DeterministicDataset(ds, preprocessor, data_config.seed) for ds in datasets]
#         assert train_config.epoch == 0 or train_config.epoch is None
#
#         dataset = IterableDatasetMixture(
#             datasets=datasets,
#             mixture_rates=rates,
#             global_batch_size=train_config.global_train_batch_size,
#             seed=data_config.seed,
#             shuffle=data_config.shuffle,
#         )
#         return DataLoader(
#             dataset,
#             batch_size=train_config.device_train_batch_size,
#             drop_last=train_config.data.drop_last,
#             collate_fn=MMCollator(
#                 train_config.data.sequence_length, False,
#                 pad=data_config.pad, max_crops=train_config.model.get_max_crops()),
#             num_workers=train_config.data.num_workers,
#             pin_memory=train_config.data.pin_memory,
#             prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
#             persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
#             timeout=train_config.data.timeout,
#         )
#     else:
#         raise NotImplementedError(train_config.data.multi_modal)

