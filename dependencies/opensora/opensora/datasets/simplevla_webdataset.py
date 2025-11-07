import io
import json
import random
import glob
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from torch.utils.data import IterableDataset, DataLoader
from opensora.registry import DATASETS

import imageio

@DATASETS.register_module()
class SimpleVLAWebDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        shards_pattern: str,
        stats_path: str,
        *,
        Ta: int = 8,
        To: int = 4,
        stride: int = 1,
        action_dim: int = 7,
        image_size: Tuple[int, int] = (256, 256),
        not_repeat = False,
        episode_buf_size: int = 30,
        sample_buf_size: int = 10000,
    ):
        super().__init__()
        self.Ta, self.To, self.stride = Ta, To, stride
        self.context_length = self.Ta + self.To
        self.image_size = image_size

        try:
            shards: List[str] = sorted(glob.glob(shards_pattern, recursive=True))
        except:
            shards = []
            for pattern in shards_pattern:
                shards += sorted(glob.glob(pattern, recursive=True))
            shards_pattern = " ".join(shards_pattern)
        print(f"[SimpleVLA] 从 pattern '{shards_pattern}' 找到 {len(shards)} 个 shard")
        
        if not shards:
            raise FileNotFoundError(f"[SimpleVLA] pattern '{shards_pattern}' 没匹配到任何 .tar 文件")

        # ---------- 2. 读取归一化常量 ----------
        stats = json.load(open(stats_path, "r"))
        if "aloha" in shards_pattern:
            try:
                self.q01 = np.asarray(stats["action"]["min"], np.float32)
                self.q99 = np.asarray(stats["action"]["max"], np.float32)
            except:
                key = list(stats.keys())[0]
                self.q01 = np.asarray(stats[key]["action"]["min"], np.float32)
                self.q99 = np.asarray(stats[key]["action"]["max"], np.float32)
            self.finish_step_shift = -1
        else:
            self.finish_step_shift = 0
            try:
                self.q01 = np.asarray(stats["action"]["q01"], np.float32)
                self.q99 = np.asarray(stats["action"]["q99"], np.float32)
            except:
                key = list(stats.keys())[0]
                self.q01 = np.asarray(stats[key]["action"]["q01"], np.float32)
                self.q99 = np.asarray(stats[key]["action"]["q99"], np.float32)

        # # ---------- 3. 分布式 rank ----------
        # world_size = dist.get_world_size() if dist.is_initialized() else 1

        # # ---------- 4. 估算 epoch size ----------
        # estimated_windows_per_shard = episode_per_shard * 350
        # self.epoch_size = estimated_windows_per_shard * len(shards) // world_size

        # ---------- 5. 构建 WebDataset Pipeline ----------
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        use_resample = ((world_size * 4) >= len(shards)) and (not not_repeat)# 4 is number of workers
        print(f"World size: {world_size}, Use resample: {use_resample}, Shard num: {len(shards)}")
        seed = random.randint(0, 10000)
        self.ds = wds.DataPipeline(
            wds.ResampledShards(shards, seed=seed) if use_resample else wds.SimpleShardList(shards, seed=seed),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(episode_buf_size, initial=episode_buf_size),
            wds.to_tuple("video.npy", "action.npy", "meta.json"),
            self._split_to_windows,
            wds.shuffle(sample_buf_size, initial=sample_buf_size),
        )

    def __iter__(self):
        return iter(self.ds)

    def _split_to_windows(
        self, src: Iterable[Tuple[bytes, bytes, bytes]]
    ) -> Iterable[Dict]:
        for v_bytes, a_bytes, m_bytes in src:
            video_np = np.load(io.BytesIO(v_bytes), allow_pickle=False)  # (T, C, H, W)
            action_np = np.load(io.BytesIO(a_bytes), allow_pickle=False)  # (T, action_dim)
            meta = json.loads(m_bytes.decode())
            finish_step = int(meta.get("finish_step")) + self.finish_step_shift
            for start in range(0, finish_step - self.Ta + 1, self.stride):
                vs, ve = start - self.To + 1, start + self.Ta + 1
                if vs < 0:
                    pad = np.repeat(video_np[0:1], -vs, axis=0)
                    vid_win = np.concatenate([pad, video_np[:ve]], axis=0)
                else:
                    vid_win = video_np[vs:ve]

                video = torch.from_numpy(vid_win).float() / 127.5 - 1
                video = video.permute(3, 0, 1, 2)

                act_np = action_np[start : start + self.Ta]
                action = 2 * ((act_np - self.q01) / (self.q99 - self.q01)) - 1
                action = torch.from_numpy(action).float()

                debug_meta = {
                    "fpath": meta.get("fpath", ""),
                    "unique_id": meta.get("unique_id", ""),
                    "episode_name": meta.get("episode_name", ""),
                    "video_start": vs,
                    "video_end": ve,
                    "action_start": start,
                    "action_end": start + self.Ta,
                }

                yield {
                    "video": video,
                    "action": action,
                    "fps": 30,
                    "num_frames": video.shape[1],
                    "height": self.image_size[0],
                    "width": self.image_size[1],
                    "meta": debug_meta,
                }

@DATASETS.register_module()
class MixedSimpleVLAWebDataset(IterableDataset):
    """Sample from multiple *already‑constructed* datasets with given ratios.

    This wrapper keeps separate iterators for each underlying dataset and on
    every `__next__` randomly chooses which iterator to draw from, according to
    the supplied *weights*.
    """

    def __init__(
        self,
        shards_pattern1: str,
        shards_pattern2: str,
        stats_path: str,
        *,
        weights: Tuple[float, float] = (0.7, 0.3),
        Ta: int = 8,
        To: int = 4,
        stride: int = 1,
        action_dim: int = 7,
        image_size: Tuple[int, int] = (256, 256),
        episode_per_shard: int = 10,
        episode_buf_size: int = 30,
        sample_buf_size: int = 10000,
    ):
        super().__init__()
        ds1 = SimpleVLAWebDataset(shards_pattern1, stats_path=stats_path, Ta=Ta, To=To, stride=stride, action_dim=action_dim, image_size=image_size, episode_buf_size=episode_buf_size, sample_buf_size=sample_buf_size)
        ds2 = SimpleVLAWebDataset(shards_pattern2, stats_path=stats_path, Ta=Ta, To=To, stride=stride, action_dim=action_dim, image_size=image_size, episode_buf_size=episode_buf_size, sample_buf_size=sample_buf_size)
        datasets = [ds1, ds2]
        self.context_length = Ta + To
        self.image_size = image_size

        if len(datasets) != len(weights):
            raise ValueError("`datasets` and `weights` must have the same length")
        if any(w < 0 for w in weights):
            raise ValueError("`weights` must be non‑negative")

        total = float(sum(weights))
        if total == 0:
            raise ValueError("`weights` must sum to a positive value")

        self.datasets = datasets
        # normalize so that sum(weights) == 1.0
        self.weights = [w / total for w in weights]

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _choose_dataset_idx(self) -> int:
        """Return an index in [0, len(datasets)) sampled wrt `self.weights`."""
        r = random.random()
        cum = 0.0
        for idx, w in enumerate(self.weights):
            cum += w
            if r <= cum:
                return idx
        return len(self.weights) - 1  # fallback due to FP precision

    # ------------------------------------------------------------------
    # IterableDataset interface
    # ------------------------------------------------------------------
    def __iter__(self):
        # create one independent iterator per underlying dataset
        iterators = [iter(ds) for ds in self.datasets]
        while True:
            idx = self._choose_dataset_idx()
            try:
                yield next(iterators[idx])
            except StopIteration:
                iterators[idx] = iter(self.datasets[idx])
                yield next(iterators[idx])

