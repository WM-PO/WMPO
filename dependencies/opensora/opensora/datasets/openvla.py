"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type
import imageio
import numpy as np
import torch
import imageio
import os
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import PreTrainedTokenizerBase
from prismatic.util import set_global_seed

# from prismatic.models.backbones.llm.prompting import PromptBuilder
# from prismatic.models.backbones.vision import ImageTransform
# from prismatic.util.data_utils import tree_map
# from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset # , make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

from opensora.registry import DATASETS

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

@DATASETS.register_module()
class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        self.context_length = 12
        self.image_size = (256, 256)
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix = data_root_dir, data_mix

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=False,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=12,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=1,                        # For action chunking
                skip_unlabeled=False,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            if not all(rlds_batch['observation']['pad_mask'][3:]):
                continue
            video = rlds_batch['observation']['image_primary']

            video = torch.from_numpy(video.copy()).float() / 127.5 - 1

            video = video.permute(3, 0, 1, 2)
            action = rlds_batch['action'][0:-1][-8:]
            debug_meta = {
                "dataset_name": rlds_batch['dataset_name'],
            }
            yield {
                "video": video,
                "action": action,
                "fps": 30,
                "num_frames": video.shape[1],
                "height": video.shape[2],
                "width": video.shape[3],
                "meta": debug_meta,
            }
            # yield rlds_batch # self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")



if __name__ == "__main__":
    data_mix = "oxe_magic_soup_plus_minus"
    data_root_dir = "/mnt/hdfs/zhufangqi/datasets/open_x/openvla"
    default_image_resolution = (3, 256 ,256)
    shuffle_buffer_size = 256000
    dataset = RLDSDataset(
        data_root_dir,
        data_mix,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=True,
        image_aug=False,
    )
    per_device_batch_size=8
    seed=42
    worker_init_fn = set_global_seed(seed, get_worker_init_fn=True)
    dataloader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        sampler=None,
        # collate_fn=collator,
        num_workers=0,
        worker_init_fn=worker_init_fn,
    )
    from tqdm import tqdm
    for i, batch in enumerate(tqdm(dataloader, total=10000)):
        # torch.Size([64, 3, 12, 256, 256])
        print()
        pass
        # os.makedirs(f"debug/{i}", exist_ok=True)  # 修正了拼写错误
        # for j, name in enumerate(batch['dataset_name']):
        #     imageio.mimwrite(f'debug/{i}/{name}_{j}.mp4', batch['observation']['image_primary'][j], fps=1)
        # if i >= 10:
        #     break
    print()