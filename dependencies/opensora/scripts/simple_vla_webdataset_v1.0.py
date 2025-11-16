#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


class SimpleVLAWebDataset(IterableDataset):
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
        episode_per_shard = 10, 
        shuffle_buf_size: int = 10000,
    ):
        super().__init__()
        self.Ta, self.To, self.stride = Ta, To, stride
        self.image_size = image_size

        # ---------- 1. 找到所有 shard ----------
        shards: List[str] = sorted(glob.glob(shards_pattern))
        if not shards:
            raise FileNotFoundError(f"[SimpleVLA] pattern '{shards_pattern}' 没匹配到任何 .tar 文件")
        print(f"[SimpleVLA] 发现 {len(shards)} 个 shard")

        # ---------- 2. 读取归一化常量 ----------
        stats = json.load(open(stats_path, "r"))
        task = "libero_10_no_noops"  # 修改为你自己的 task key
        self.q01 = np.asarray(stats[task]["action"]["q01"], np.float32)
        self.q99 = np.asarray(stats[task]["action"]["q99"], np.float32)

        # ---------- 3. 分布式配置 ----------
        # rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # ---------- 4. 估算每个 epoch 样本数 ----------
        estimated_windows_per_shard = episode_per_shard * 350  # 根据实际情况估算 一个episode约有350个sample
        self.epoch_size = estimated_windows_per_shard * len(shards) // world_size

        # ---------- 5. 构建 WebDataset 管线 ----------
        self.ds = (
            wds.WebDataset(
                shards,
                shardshuffle=1,
                handler=wds.warn_and_continue,
            )
            .repeat()  # 无限遍历（顺序、无放回）
            .shuffle(episode_per_shard, initial=episode_per_shard) # 打乱每个shard中的episode顺序
            .to_tuple("video.npy", "action.npy", "meta.json")
            .compose(self._split_to_windows)
            .shuffle(shuffle_buf_size, initial=shuffle_buf_size)
            .with_epoch(self.epoch_size)
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
            finish_step = int(meta.get("finish_step"))

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
                    "video_start": vs,
                    "video_end": ve,
                    "action_start": start,
                    "action_end": start + self.Ta,
                }

                yield {
                    "video": video,
                    "action": action,
                    "fps": 30,
                    "num_frames": video.shape[0],
                    "height": self.image_size[0],
                    "width": self.image_size[1],
                    "meta": debug_meta,
                }


# ======================================================
#                   快速自测
# ======================================================
if __name__ == "__main__":
    from tqdm import tqdm

    pattern = "/mnt/hdfs/zhufangqi/datasets/simplevla_rl/webdataset_shards/06/16/*.tar"
    stats = "/mnt/hdfs/zhufangqi/datasets/libero/dataset_statistics.json"

    dataset = SimpleVLAWebDataset(pattern, stats, Ta=8, To=4, stride=1)
    # dataset = wds.split_by_node().with_epoch(dataset.epoch_size)
    # dataset = wds.split_by_node(dataset.ds)   # 先拿出内部 DataPipeline 再按节点切分

    dataloader = wds.WebLoader(dataset,
                           batch_size=4,
                           num_workers=4,
                           pin_memory=True)
    all_data = []

    for batch in tqdm(dataloader):
        batch_data = []
        for i in range(batch["video"].shape[0]):
            sample_info = {
                "fpath": batch["meta"]["fpath"][i],
                "video_start": batch["meta"]["video_start"][i].item(),
                "video_end": batch["meta"]["video_end"][i].item(),
                "action_start": batch["meta"]["action_start"][i].item(),
                "action_end": batch["meta"]["action_end"][i].item(),
            }
            batch_data.append(sample_info)

        all_data.extend(batch_data)
    with open("webdataset_data_info.json", "w") as f:
        json.dump(all_data, f, indent=4)
# ¥ print("Saved metadata of first 100 samples to webdataset_data_info.json")
