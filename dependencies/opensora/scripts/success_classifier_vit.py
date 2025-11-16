#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a 2-class success/failure classifier on robot-episode vídeos
using HuggingFace VideoMAE.

Key specs
---------
window_size = 8  (frames)
stride       = 8
model_frames = 16  (VideoMAE default)
The 8-frame clip is *resampled* to 16 before feeding to the model.
"""

import argparse
import glob
import io
import json
import os
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import webdataset as wds
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEFeatureExtractor,
)

# -----------------------------------------------------------------------------#
# Dataset                                                                       #
# -----------------------------------------------------------------------------#


class SuccessWindowDataset(IterableDataset):
    """Yield (video_tensor, label) where label=1 for final window of episode."""

    def __init__(
        self,
        shards: List[str],
        window_size: int = 8,
        stride: int = 8,
        model_frames: int = 16,
        img_size: int = 224,
    ):
        super().__init__()
        self.window = window_size
        self.stride = stride
        self.model_frames = model_frames
        self.fe = VideoMAEFeatureExtractor(size=img_size)

        self.pipeline = wds.DataPipeline(
            wds.SimpleShardList(shards),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.to_tuple("video.npy", "meta.json"),
            self._window_generator,
        )

    # ------------------------------ iter ------------------------------------#
    def __iter__(self):
        return iter(self.pipeline)

    # ---------------------------- internals ---------------------------------#
    def _window_generator(self, stream: Iterable[Tuple[bytes, bytes]]):
        W, S = self.window, self.stride
        for v_bytes, m_bytes in stream:
            video = np.load(io.BytesIO(v_bytes))  # (T, H, W, C)  channel-last
            T = video.shape[0]
            if T < W:
                continue  # too short

            for end in range(T, W - 1, -S):  # tail → head
                start = end - W
                clip = video[start:end]  # (8, H, W, C)
                label = 1 if end == T else 0
                clip = self._resample_clip(clip, self.model_frames)
                yield self._to_tensor(clip), label

    def _resample_clip(self, clip: np.ndarray, target_len: int) -> np.ndarray:
        """Linearly sample / repeat to target length."""
        idx = np.linspace(
            0, clip.shape[0] - 1, num=target_len, dtype=np.int64
        )
        return clip[idx]

    def _to_tensor(self, clip: np.ndarray) -> torch.Tensor:
        pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in clip]
        out = self.fe(pil_frames, return_tensors="pt")["pixel_values"][0]
        # shape: (3, target_len, H, W)
        return out


# -----------------------------------------------------------------------------#
# Metrics & helpers                                                             #
# -----------------------------------------------------------------------------#


def collate_fn(batch):
    vids, labels = zip(*batch)
    vids = torch.stack(vids)  # (B, 3, T, H, W)
    labels = torch.tensor(labels, dtype=torch.long)
    return vids, labels


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for vids, ys in loader:
        vids, ys = vids.to(device), ys.to(device)
        logits = model(pixel_values=vids).logits  # (B, 2)
        preds.extend(logits.argmax(1).cpu().tolist())
        trues.extend(ys.cpu().tolist())
    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)
    print(f"[Val] Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    return f1


# -----------------------------------------------------------------------------#
# Training                                                                      #
# -----------------------------------------------------------------------------#


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- shard split --------------------------------------------------#
    shards = sorted(glob.glob(args.pattern, recursive=True))
    if not shards:
        raise FileNotFoundError(f"No shard matched: {args.pattern}")
    n_val = max(1, int(len(shards) * args.val_split))
    val_shards, train_shards = shards[:n_val], shards[n_val:]
    print(f"[Shards] train={len(train_shards)}  val={len(val_shards)}")

    # ---------- datasets / loaders ------------------------------------------#
    tr_ds = SuccessWindowDataset(
        train_shards,
        window_size=8,
        stride=8,
        model_frames=args.model_frames,
        img_size=args.img_size,
    )
    va_ds = SuccessWindowDataset(
        val_shards,
        window_size=8,
        stride=8,
        model_frames=args.model_frames,
        img_size=args.img_size,
    )

    tr_ld = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    va_ld = DataLoader(
        va_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ---------- model --------------------------------------------------------#
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base",
        num_labels=2,
        ignore_mismatched_sizes=True,  # 自动适配分类头
    ).to(device)

    # ---------- optim / loss -------------------------------------------------#
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, args.pos_weight], device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(tr_ld, desc=f"[Epoch {epoch}/{args.epochs}]")
        for vids, ys in pbar:
            vids, ys = vids.to(device), ys.to(device)
            loss = criterion(model(pixel_values=vids).logits, ys)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        f1 = evaluate(model, va_ld, device)
        if f1 > best_f1:
            best_f1 = f1
            if args.ckpt_dir:
                os.makedirs(args.ckpt_dir, exist_ok=True)
                path = os.path.join(args.ckpt_dir, "best_videomae.pth")
                torch.save(model.state_dict(), path)
                print(f"[Checkpoint] saved to {path}  (F1={best_f1:.4f})")


# -----------------------------------------------------------------------------#
# CLI                                                                          #
# -----------------------------------------------------------------------------#


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="/mnt/hdfs/zhufangqi/datasets/simplevla_rl/webdataset_shards/06/16_mp/**/*.tar", help="glob for *.tar shards")
    ap.add_argument("--img_size", type=int, default=224, help="resize H,W")
    ap.add_argument("--model_frames", type=int, default=16, help="frames fed to model")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--pos_weight", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ckpt_dir", default="ckpts_videomae")
    return ap.parse_args()


if __name__ == "__main__":
    train(get_args())
