#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VideoMAE success/failure classifier
- window = 8, stride = 8
- NO frame resampling
"""

import argparse, glob, io, json, os
from typing import Iterable, List, Tuple

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import webdataset as wds
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import (
    VideoMAEFeatureExtractor,
    VideoMAEConfig,
    VideoMAEForVideoClassification,
)

# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #

class SuccessWindowDataset(IterableDataset):
    def __init__(
        self,
        shards: List[str],
        window: int = 8,
        stride: int = 8,
        img_size: int = 224,
    ):
        super().__init__()
        self.window, self.stride = window, stride
        self.fe = VideoMAEFeatureExtractor(size=img_size)

        self.pipeline = wds.DataPipeline(
            wds.SimpleShardList(shards),
            wds.split_by_node, wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.to_tuple("video.npy", "meta.json"),
            self._windows,
        )

    def __iter__(self):
        return iter(self.pipeline)

    # ------- internal ------------------------------------------------------- #
    def _windows(self, stream: Iterable[Tuple[bytes, bytes]]):
        W, S = self.window, self.stride
        for v_bytes, m_bytes in stream:
            video = np.load(io.BytesIO(v_bytes))            # (T,H,W,C)
            T = video.shape[0]
            if T < W:                                      # skip too-short episodes
                continue
            for end in range(T, W - 1, -S):                # tail → head
                clip = video[end - W:end]
                label = 1 if end == T else 0
                yield self._to_tensor(clip), label

    def _to_tensor(self, clip: np.ndarray) -> torch.Tensor:
        # clip: (8,H,W,C)  → (3,8,H,W)
        frames = [Image.fromarray(f.astype(np.uint8)) for f in clip]
        return self.fe(frames, return_tensors="pt")["pixel_values"][0]

# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #

def collate_fn(batch):
    vids, ys = zip(*batch)
    return torch.stack(vids), torch.tensor(ys, dtype=torch.long)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for vids, ys in loader:
        vids, ys = vids.to(device), ys.to(device)
        logits = model(pixel_values=vids).logits
        preds.extend(logits.argmax(1).cpu().tolist())
        trues.extend(ys.cpu().tolist())
    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)
    print(f"[Val] Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    return f1

# -------- optional: interpolate temporal pos-embed ------------------------- #
def interpolate_temporal_pos_embed(model, new_frames: int):
    emb = model.videomae.embeddings
    possible_names = ["temporal_position_embeddings",
                      "temporal_pos_embed",
                      "pos_embed_temporal"]
    for name in possible_names:
        if hasattr(emb, name):
            tpe: torch.nn.Parameter = getattr(emb, name)
            break
    else:
        print("[Warn] temporal position embedding not found, skip interpolation")
        return
    old_len = tpe.shape[1]      # (1, T_old, D)
    if old_len == new_frames:
        return
    print(f"[Info] Interpolating TPE {old_len} → {new_frames}")
    new_tpe = F.interpolate(
        tpe.permute(0, 2, 1), size=new_frames,
        mode="linear", align_corners=False
    ).permute(0, 2, 1)
    setattr(emb, name, nn.Parameter(new_tpe))
    model.config.num_frames = new_frames

# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- split shards ----------------------------------------------------- #
    shards = sorted(glob.glob(args.pattern, recursive=True))
    if not shards:
        raise FileNotFoundError(f"No shards match: {args.pattern}")
    val_n = max(1, int(len(shards) * args.val_split))
    val_shards, train_shards = shards[:val_n], shards
    print(f"[Shards] train={len(train_shards)}  val={len(val_shards)}")

    # ----- datasets / loaders ---------------------------------------------- #
    tr_ds = SuccessWindowDataset(train_shards, 8, 8, args.img_size)
    va_ds = SuccessWindowDataset(val_shards,   8, 8, args.img_size)
    tr_ld = DataLoader(tr_ds, args.batch_size, collate_fn=collate_fn,
                       num_workers=args.num_workers, pin_memory=True)
    va_ld = DataLoader(va_ds, args.batch_size, collate_fn=collate_fn,
                       num_workers=args.num_workers, pin_memory=True)

    # ----- model ------------------------------------------------------------ #
    cfg = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base")
    cfg.num_frames = 8   
    cfg.num_labels = 2                                   # 告诉模型我们只用 8 帧
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base",
        config=cfg,
    ).to(device)

    # 可选：把 16-帧位置编码线性插值到 8 帧，提升迁移效果
    # interpolate_temporal_pos_embed(model, 8)

    # ----- optim ------------------------------------------------------------ #
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, args.pos_weight], device=device)
    )
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1 = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        bar = tqdm(tr_ld, desc=f"[Ep {ep}/{args.epochs}]")
        for vids, ys in bar:
            vids, ys = vids.to(device), ys.to(device)
            loss = criterion(model(pixel_values=vids).logits, ys)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        f1 = evaluate(model, va_ld, device)
        if f1 > best_f1 and args.ckpt_dir:
            best_f1 = f1
            os.makedirs(args.ckpt_dir, exist_ok=True)
            p = os.path.join(args.ckpt_dir, "best_videomae.pth")
            torch.save(model.state_dict(), p)
            print(f"[Checkpoint] saved to {p}  (F1={best_f1:.4f})")

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="/mnt/hdfs/zhufangqi/datasets/simplevla_rl/webdataset_shards/06/16_mp/**/*.tar", help="glob like **/*.tar")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--pos_weight", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ckpt_dir", default="ckpts_videomae")
    return ap.parse_args()

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train(get_args())
