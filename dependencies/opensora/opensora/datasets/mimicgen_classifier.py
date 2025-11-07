#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VideoMAE success/failure classifier (DDP, step-based training/eval, infinite train stream)

Launch (single node 8 GPUs):
    torchrun --standalone --nproc_per_node=8 videomae_ddp_fixed.py
"""

import io, os, glob, json, random
from typing import Iterable, List, Tuple, Dict, Any
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import webdataset as wds
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import (
    VideoMAEConfig,
    VideoMAEForVideoClassification,
)

# HF feature extractor / image processor (API 兼容)
try:
    from transformers import VideoMAEImageProcessor as VideoMAEFeatureExtractor
except Exception:
    from transformers import VideoMAEFeatureExtractor

# =========================
# CONFIG
# =========================
CFG = dict(
    TRAIN_PATTERN=".datasets/simplevla_rl/mimicgen/square_1280_demos/square_d0/**/*.tar",
    VAL_PATTERN="./datasets/simplevla_rl/mimicgen/square_128_demos_val/**/*.tar",
    IMG_SIZE=224,
    WINDOW=8,
    STRIDE_TRAIN=8,
    STRIDE_VAL=1,
    BATCH_SIZE=4,                # per-GPU
    VAL_BATCH_SIZE=64,           # per-GPU
    NUM_WORKERS=4,
    PERSISTENT_WORKERS=True,
    PREFETCH_FACTOR=2,
    LR=1e-4,
    WEIGHT_DECAY=1e-4,
    # —— 训练完全用 step 控制，不再依赖 epoch —— #
    MAX_STEPS=200_000,           # 你可按需改
    EVAL_STEPS=500,             # 训练多少步验证一次 & 保存一次
    CKPT_DIR="ckpts_videomae_1280",
    SEED=42,
    MODEL_NAME="MCG-NJU/videomae-base",
    NUM_LABELS=2,
    THRESH_MIN=0.3,
    THRESH_MAX=1.0,
    THRESH_STEPS=20,
    USE_RESAMPLE_TRAIN=True,     # 训练使用 ResampledShards(无限数据流)
    DROP_LAST=True,              # 训练时丢弃最后一小批，batch 尺寸稳定
)

# =========================
# Helpers
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dist_env():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return world_size, rank, local_rank, device

def collate_fn(batch):
    vids = torch.stack([b[0] for b in batch])             # (B, C, T, H, W)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    meta_keys = batch[0][2].keys()
    meta: Dict[str, Any] = {k: [b[2][k] for b in batch] for k in meta_keys}
    return vids, ys, meta

# =========================
# Datasets
# =========================
class SuccessWindowDataset(IterableDataset):
    """
    从 WebDataset tar shard 中提取滑动窗口剪辑。
    期望样本 (video.npy, meta.json)
      - meta 里应包含: "finish_step" (T), "complete" (bool)
    训练：对每条 episode 取 (T-W,T] 的正样本 1 个，再随机采样一个负样本。
    验证：从 T 开始往前每 stride 取窗口，T 处为正样本（若 complete），其余为负样本。
    """
    def __init__(
        self,
        shard_globs: List[str],
        window: int = 8,
        stride: int = 8,
        img_size: int = 224,
        mode: str = "train",
        use_resample: bool = False,
        shuffle_buf: int = 1000,
    ):
        super().__init__()
        assert mode in {"train", "val"}
        self.window, self.stride, self.mode = window, stride, mode
        self.fe = VideoMAEFeatureExtractor(size=img_size)

        if mode == "train":
            shard_source = (wds.ResampledShards(shard_globs, seed=42)
                            if use_resample else wds.SimpleShardList(shard_globs))
            pipeline = [
                shard_source,
                wds.split_by_node,
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.to_tuple("video.npy", "meta.json"),
                self._windows_train,                      # 修正：原代码误写 _windows_train
                wds.shuffle(shuffle_buf, initial=shuffle_buf),
            ]
        else:
            pipeline = [
                wds.SimpleShardList(shard_globs),
                wds.split_by_node,
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.to_tuple("video.npy", "meta.json"),
                self._windows_val,
            ]
        self.pipeline = wds.DataPipeline(*pipeline)

    def __iter__(self):
        return iter(self.pipeline)

    # ---------- window generators ----------
    def _windows_val(self, stream: Iterable[Tuple[bytes, bytes]]):
        W, S = self.window, self.stride
        for v_bytes, m_bytes in stream:
            video = np.load(io.BytesIO(v_bytes))  # (T, H, W, C)
            meta = json.loads(m_bytes.decode())
            T = int(meta["finish_step"])
            complete = bool(meta.get("complete", True))
            if T < W:
                continue  # 太短跳过

            # 正样本：以 T 结尾
            end = T
            yield self._to_tensor(video[end - W : end]), int(complete), {
                "video_start": end - W, "video_end": end, "label": int(complete), "complete": complete
            }

            # 负样本：从末端往前每 S 一个窗口
            for end in range(T - S, W - 1, -S):
                yield self._to_tensor(video[end - W : end]), 0, {
                    "video_start": end - W, "video_end": end, "label": 0, "complete": complete
                }

    def _windows_train(self, stream: Iterable[Tuple[bytes, bytes]]):
        W, S = self.window, self.stride
        for v_bytes, m_bytes in stream:
            video = np.load(io.BytesIO(v_bytes))  # (T, H, W, C)
            meta = json.loads(m_bytes.decode())
            T = int(meta["finish_step"])
            complete = bool(meta.get("complete", True))
            if T < W:
                continue

            # 正样本：以 T 结尾
            end = T
            yield self._to_tensor(video[end - W : end]), int(complete), {
                "video_start": end - W, "video_end": end, "label": int(complete), "complete": complete
            }

            # 负样本：在 [W, T-S] 范围随机取一个窗口末端
            if T - S >= W:
                neg_candidates = list(range(T - S, W - 1, -1))
                end = random.choice(neg_candidates)
                yield self._to_tensor(video[end - W : end]), 0, {
                    "video_start": end - W, "video_end": end, "label": 0, "complete": complete
                }

    def _to_tensor(self, clip: np.ndarray) -> torch.Tensor:
        frames = [Image.fromarray(f.astype(np.uint8)) for f in clip]
        # Returns (1, C, T, H, W); take [0]
        return self.fe(frames, return_tensors="pt")["pixel_values"][0]

# =========================
# Evaluation (DDP gather on rank0)
# =========================
@torch.no_grad()
def evaluate_ddp(model: nn.Module, loader: DataLoader, device: torch.device, rank: int, world_size: int):
    model.eval()
    logits_local, trues_local = [], []

    for vids, ys, _ in tqdm(loader, desc="Val", disable=(rank != 0)):
        vids = vids.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        logits = model(pixel_values=vids).logits  # (B,2)
        logits_local.extend(logits.cpu().tolist())
        trues_local.extend(ys.cpu().tolist())

    # gather variable-length lists
    logits_gather, trues_gather = [None] * world_size, [None] * world_size
    dist.all_gather_object(logits_gather, logits_local)
    dist.all_gather_object(trues_gather, trues_local)

    if rank != 0:
        return None

    logits = [x for part in logits_gather for x in part]
    trues = [x for part in trues_gather for x in part]
    logits_t = torch.tensor(logits)  # (N,2)
    probs = torch.softmax(logits_t, dim=-1)[:, 1].numpy()

    thresholds = np.linspace(CFG["THRESH_MIN"], CFG["THRESH_MAX"], CFG["THRESH_STEPS"])
    all_metrics = {}
    best = {"f1": -1.0, "thresh": thresholds[0]}

    for th in thresholds:
        preds = (probs >= th).astype(np.int32).tolist()
        acc = accuracy_score(trues, preds)
        prec = precision_score(trues, preds, zero_division=0)
        rec = recall_score(trues, preds, zero_division=0)
        f1 = f1_score(trues, preds, zero_division=0)
        TP = sum((p == 1 and t == 1) for p, t in zip(preds, trues))
        TN = sum((p == 0 and t == 0) for p, t in zip(preds, trues))
        FP = sum((p == 1 and t == 0) for p, t in zip(preds, trues))
        FN = sum((p == 0 and t == 1) for p, t in zip(preds, trues))

        all_metrics[f"thresh_{th:.2f}"] = OrderedDict(
            acc=acc, precision=prec, recall=rec, f1=f1,
            TP=TP, TN=TN, FP=FP, FN=FN,
            pred_pos=int(sum(preds)), pred_neg=int(len(preds) - sum(preds)),
            true_pos=int(sum(trues)), true_neg=int(len(trues) - sum(trues)),
        )
        if f1 > best["f1"]:
            best["f1"], best["thresh"] = f1, th

    return all_metrics, best

# =========================
# Training
# =========================
def main():
    set_seed(CFG["SEED"])
    torch.backends.cudnn.benchmark = True

    world_size, rank, local_rank, device = get_dist_env()
    if rank == 0:
        print(f"[DDP] world_size={world_size}")

    # --------- Build shard lists ---------
    train_shards = sorted(glob.glob(CFG["TRAIN_PATTERN"], recursive=True))
    val_shards   = sorted(glob.glob(CFG["VAL_PATTERN"], recursive=True))

    if rank == 0:
        print(f"Train shards: {len(train_shards)}")
        print(f"Val   shards: {len(val_shards)}")

    if len(train_shards) == 0 or len(val_shards) == 0:
        if rank == 0:
            raise RuntimeError("Train/Val shard patterns yielded no files.")

    # --------- Datasets ---------
    tr_ds = SuccessWindowDataset(
        shard_globs=train_shards,
        window=CFG["WINDOW"],
        stride=CFG["STRIDE_TRAIN"],
        img_size=CFG["IMG_SIZE"],
        mode="train",
        use_resample=CFG["USE_RESAMPLE_TRAIN"],   # 关键：无限数据流，避免每个 epoch 卡顿
    )
    va_ds = SuccessWindowDataset(
        shard_globs=val_shards,
        window=CFG["WINDOW"],
        stride=CFG["STRIDE_VAL"],
        img_size=CFG["IMG_SIZE"],
        mode="val",
        use_resample=False,
    )

    # --------- DataLoaders ---------
    tr_ld = DataLoader(
        tr_ds,
        batch_size=CFG["BATCH_SIZE"],
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=CFG["PERSISTENT_WORKERS"],
        prefetch_factor=CFG["PREFETCH_FACTOR"],
        drop_last=CFG["DROP_LAST"],
    )
    va_ld = DataLoader(
        va_ds,
        batch_size=CFG["VAL_BATCH_SIZE"],
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=CFG["PERSISTENT_WORKERS"],
        prefetch_factor=CFG["PREFETCH_FACTOR"],
        drop_last=False,
    )

    # --------- Model / Optim ---------
    cfg = VideoMAEConfig.from_pretrained(
        CFG["MODEL_NAME"],
        num_frames=CFG["WINDOW"],
        num_labels=CFG["NUM_LABELS"],
    )
    model = VideoMAEForVideoClassification.from_pretrained(CFG["MODEL_NAME"], config=cfg).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WEIGHT_DECAY"])

    # --------- Step-based Train loop ---------
    os.makedirs(CFG["CKPT_DIR"], exist_ok=True) if rank == 0 else None
    global_step, best_f1 = 0, -1.0

    # 用一个“无尽”迭代器（ResampledShards 下不会触发 StopIteration）
    tr_iter = iter(tr_ld)

    while global_step < CFG["MAX_STEPS"]:
        try:
            vids, ys, _ = next(tr_iter)
        except StopIteration:
            # 如果你把 USE_RESAMPLE_TRAIN=False，这里安全重建迭代器
            tr_iter = iter(tr_ld)
            continue

        model.train()
        vids = vids.to(device, non_blocking=True)
        ys   = ys.to(device, non_blocking=True)

        logits = model(pixel_values=vids).logits
        loss = criterion(logits, ys)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        global_step += 1
        if rank == 0 and global_step % 10 == 0:
            tqdm.write(f"[step {global_step}] loss={loss.item():.4f}")

        # ---- step-based eval & checkpoint ----
        if global_step % CFG["EVAL_STEPS"] == 0:
            out = evaluate_ddp(model, va_ld, device, rank, world_size)
            if rank == 0 and out is not None:
                all_metrics, best = out
                print(f"\n[Val @ step {global_step}]")
                for k, v in all_metrics.items():
                    acc = v["acc"]; prec = v["precision"]; rec = v["recall"]; f1 = v["f1"]
                    print(f"{k}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
                print(f"Best F1={best['f1']:.4f} @ thresh={best['thresh']:.2f}")

                # ---- 保存 step checkpoint ----
                step_pth = os.path.join(
                    CFG["CKPT_DIR"],
                    f"videomae_step{global_step}_f1{best['f1']:.4f}_th{best['thresh']:.2f}.pth"
                )
                torch.save(model.module.state_dict(), step_pth)
                print(f"[Checkpoint] saved → {step_pth}")

                # ---- 保存 best checkpoint ----
                if best["f1"] > best_f1:
                    best_f1 = best["f1"]
                    best_thresh = best["thresh"]
                    best_pth = os.path.join(
                        CFG["CKPT_DIR"],
                        f"best_videomae_f1{best_f1:.4f}_th{best_thresh:.2f}.pth"
                    )
                    torch.save(model.module.state_dict(), best_pth)
                    print(f"[Best Checkpoint] saved → {best_pth}")


    if rank == 0:
        print(f"[Done] total steps={global_step}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
