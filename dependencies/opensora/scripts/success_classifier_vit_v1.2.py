#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VideoMAE success/failure classifier (Distributed with torchrun, supports Focal Loss)

启动示例（单机 8 卡）::

    torchrun --standalone --nproc_per_node=8 videomae_ddp.py \
        --pattern "/mnt/hdfs/zhufangqi/datasets/simplevla_rl/webdataset_shards/06/16_mp/**/*.tar" \
        --batch_size 4 --eval_steps 1000 --ckpt_dir ckpts_videomae

* 训练与评估均使用 DistributedDataParallel (DDP)。
* 评估阶段各 rank 先本地计算，再通过 `dist.all_gather_object` 收集到 rank0 汇总计算指标，
  并仅在 rank0 打印与保存最佳模型。
"""

import argparse, glob, io, os, random
from typing import Iterable, List, Tuple
from collections import OrderedDict
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import webdataset as wds
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import (
    VideoMAEConfig,
    VideoMAEFeatureExtractor,
    VideoMAEForVideoClassification,
)

# ----------------------------- Dataset ----------------------------- #
class SuccessWindowDataset(IterableDataset):
    """从 WebDataset tar shard 中提取滑动窗口剪辑。

    * ``window``:   每个 clip 的帧数
    * ``stride``:   滑动窗口步长
    * ``pos_ratio``:正样本（成功 clip）比例，用于负采样预算
    * ``mode``:    train / val
    """

    def __init__(
        self,
        shards: List[str],
        window: int = 8,
        stride: int = 8,
        img_size: int = 224,
        mode: str = "train",
    ) -> None:
        super().__init__()
        assert mode in {"train", "val"}
        self.window, self.stride, self.mode = window, stride, mode

        self.fe = VideoMAEFeatureExtractor(size=img_size)

        if self.mode == 'train':
            self.pipeline = wds.DataPipeline(
                wds.SimpleShardList(shards),
                wds.split_by_node,   # 按节点切分
                wds.split_by_worker, # 按 DataLoader worker 切分
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.to_tuple("video.npy", "meta.json"),
                self._windows,
                wds.shuffle(1000, initial=1000),
            )
        else:
            self.pipeline = wds.DataPipeline(
                wds.SimpleShardList(shards),
                wds.split_by_node,   # 按节点切分
                wds.split_by_worker, # 按 DataLoader worker 切分
                wds.tarfile_to_samples(handler=wds.warn_and_continue),
                wds.to_tuple("video.npy", "meta.json"),
                self._windows,
            )

    def __iter__(self):
        return iter(self.pipeline)

    def _windows(self, stream: Iterable[Tuple[bytes, bytes]]):
        W, S = self.window, self.stride
        for v_bytes, m_bytes in stream:
            video = np.load(io.BytesIO(v_bytes))  # (T, H, W, C)
            meta = json.loads(m_bytes.decode())
            T = meta["finish_step"]
            complete = meta["complete"]
            for end in range(T, W - 1, -S):
                label = end == T and complete
                label = int(label)
                debug_meta = {
                        "fpath": meta.get("fpath", ""),
                        "episode_name": meta.get("episode_name", ""),
                        "video_start": end-W,
                        "video_end": end,
                        "label": label,
                        "complete": complete
                    }
                yield self._to_tensor(video[end - W : end]), label, debug_meta


    def _to_tensor(self, clip: np.ndarray) -> torch.Tensor:
        frames = [Image.fromarray(f.astype(np.uint8)) for f in clip]
        return self.fe(frames, return_tensors="pt")["pixel_values"][0]


# ----------------------------- Focal Loss ----------------------------- #
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "sum"):
        super().__init__()
        # 二分类 => 先把 [neg, pos] 存成 buffer，DDP 自动同步
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # 交叉熵自带 softmax + log；权重直接传 α 向量就行
        alpha = torch.tensor([1 - self.alpha, self.alpha]).to(logits.device)
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=alpha,          # 关键：类别权重
            reduction="none"
        )
        pt = torch.exp(-ce_loss)        # = softmax(logits)[range(N), targets]
        loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss



import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import OrderedDict
import torch.distributed as dist
from tqdm import tqdm
import numpy as np

@torch.no_grad()
def evaluate_ddp(model: nn.Module, loader: DataLoader, device: torch.device, rank: int, world_size: int):
    """各 rank 本地推断后 all_gather 收集到 rank0 计算指标（支持多阈值）"""
    model.eval()
    logits_local, trues_local = [], []

    for vids, ys, meta in tqdm(loader, desc="Processing batches", leave=False):
        vids = vids.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        logits = model(pixel_values=vids).logits
        logits_local.extend(logits.squeeze(1).cpu().tolist())  # 假设是二分类且 logits 是 (B, 1)
        trues_local.extend(ys.cpu().tolist())

    # 使用 all_gather_object 收集不同长度 list
    logits_gather, trues_gather = [None] * world_size, [None] * world_size
    dist.all_gather_object(logits_gather, logits_local)
    dist.all_gather_object(trues_gather, trues_local)
    torch.cuda.empty_cache()

    if rank == 0:
        logits = [logit for sub in logits_gather for logit in sub]
        trues = [t for sub in trues_gather for t in sub]

        thresholds = np.linspace(0.5, 0.99, 5)
        all_metrics = {}

        probs = torch.sigmoid(torch.tensor(logits)).numpy()

        for thresh in thresholds:
            preds = [1 if p[1] >= thresh else 0 for p in probs]
            TP = sum((p == 1 and t == 1) for p, t in zip(preds, trues))
            TN = sum((p == 0 and t == 0) for p, t in zip(preds, trues))
            FP = sum((p == 1 and t == 0) for p, t in zip(preds, trues))
            FN = sum((p == 0 and t == 1) for p, t in zip(preds, trues))
            pred_pos = sum(preds)
            pred_neg = len(preds) - pred_pos
            true_pos = sum(trues)
            true_neg = len(trues) - true_pos
            acc = accuracy_score(trues, preds)
            prec = precision_score(trues, preds, zero_division=0)
            rec = recall_score(trues, preds, zero_division=0)
            f1 = f1_score(trues, preds, zero_division=0)

            metrics = OrderedDict([
                ("acc",        acc),
                ("precision",  prec),
                ("recall",     rec),
                ("f1",         f1),
                ("TP",         TP),
                ("TN",         TN),
                ("FP",         FP),
                ("FN",         FN),
                ("pred_pos",   pred_pos),
                ("pred_neg",   pred_neg),
                ("true_pos",   true_pos),
                ("true_neg",   true_neg),
            ])
            all_metrics[f"thresh_{thresh:.2f}"] = metrics

        return all_metrics
    else:
        return None


# ----------------------------- Evaluation (DDP) ----------------------------- #
# @torch.no_grad()
# def evaluate_ddp(model: nn.Module, loader: DataLoader, device: torch.device, rank: int, world_size: int):
#     """各 rank 本地推断后 all_gather 收集到 rank0 计算指标。"""
#     model.eval()
#     preds_local, trues_local = [], []
#     # with open("./debug.log", "a") as log_file:
#     for vids, ys, meta in tqdm(loader, desc="Processing batches", leave=False):
#         # for i in range(vids.shape[0]):
#         #     info = {
#         #         "fpath":        meta["fpath"][i],
#         #         "episode_name": meta["episode_name"][i],
#         #         "video_start":  int(meta["video_start"][i]),
#         #         "video_end":    int(meta["video_end"][i]),
#         #         "label":        int(ys[i]),
#         #     }
#         #     log_file.write(f"{info}\n")
#         vids = vids.to(device, non_blocking=True)
#         ys = ys.to(device, non_blocking=True)
#         logits = model(pixel_values=vids).logits
#         preds_local.extend(logits.argmax(1).cpu().tolist())
#         trues_local.extend(ys.cpu().tolist())

#     # 使用 all_gather_object 收集不同长度 list
#     preds_gather, trues_gather = [None] * world_size, [None] * world_size
#     dist.all_gather_object(preds_gather, preds_local)
#     dist.all_gather_object(trues_gather, trues_local)
#     torch.cuda.empty_cache()
#     if rank == 0:
#         preds = [p for sub in preds_gather for p in sub]
#         trues = [t for sub in trues_gather for t in sub]
#         TP = sum((p == 1 and t == 1) for p, t in zip(preds, trues))
#         TN = sum((p == 0 and t == 0) for p, t in zip(preds, trues))
#         FP = sum((p == 1 and t == 0) for p, t in zip(preds, trues))
#         FN = sum((p == 0 and t == 1) for p, t in zip(preds, trues))
#         pred_pos = sum(preds)
#         pred_neg = len(preds) - pred_pos
#         true_pos = sum(trues)
#         true_neg = len(trues) - true_pos
#         acc = accuracy_score(trues, preds)
#         prec = precision_score(trues, preds, zero_division=0)
#         rec = recall_score(trues, preds, zero_division=0)
#         f1 = f1_score(trues, preds, zero_division=0)
#         metrics = OrderedDict([
#             ("acc",        acc),
#             ("precision",  prec),
#             ("recall",     rec),
#             ("f1",         f1),
#             ("TP",         TP),
#             ("TN",         TN),
#             ("FP",         FP),
#             ("FN",         FN),
#             ("pred_pos",   pred_pos),
#             ("pred_neg",   pred_neg),
#             ("true_pos",   true_pos),
#             ("true_neg",   true_neg),
#         ])
#         return metrics
#     else:
#         return None


# ----------------------------- Training ----------------------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    """分布式训练入口。"""

    # import tarfile
    # with tarfile.open("/mnt/hdfs/zhufangqi/datasets/simplevla_rl/webdataset_shards/06/17_mp/part_00/000000.tar", "r") as tar:
    #     for member in tar.getmembers():
    #         print(member.name)


    # ---------- Distributed init ---------- #
    dist.init_process_group(backend=args.backend, init_method="env://")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = args.local_rank

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"[DDP] world_size={world_size}")

    # ---------- Dataset & DataLoader ---------- #
    shards = sorted(glob.glob(args.pattern, recursive=True))
    # shards = sorted(glob.glob(args.pattern, recursive=True))
    if len(shards) < args.val_shards + 1 and rank == 0:
        raise ValueError("Not enough shards for validation")

    val_shards, train_shards = shards[: args.val_shards], shards[args.val_shards :]

    tr_ds = SuccessWindowDataset(
        train_shards, 8, 8, args.img_size, mode="train"
    )
    va_ds = SuccessWindowDataset(
        val_shards, 8, 8, args.img_size, mode="val"
    )

    # IterableDataset 无法使用 DistributedSampler，因此直接依赖 webdataset 的 split_by_* 去重，
    # 每个 rank 仍然搭建独立 DataLoader。
    tr_ld = DataLoader(
        tr_ds,
        args.batch_size,
        # collate_fn=lambda b: (
        #     torch.stack([v for v, _ in b]),
        #     torch.tensor([y for _, y in b]),
        # ),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    va_ld = DataLoader(
        va_ds,
        args.val_batch_size,
        # collate_fn=lambda b: (
        #     torch.stack([v for v, _ in b]),
        #     torch.tensor([y for _, y in b]),
        # ),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---------- Model ---------- #
    cfg = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base", num_frames=8, num_labels=2)
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base", config=cfg
    ).to(device)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ---------- Training Loop ---------- #
    set_seed()
    best_f1, step = 0.0, 0

    metrics = evaluate_ddp(model, va_ld, device, rank, world_size)
    if rank == 0:
        print("\n[Val @ step {}]".format(step))
        print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    # assert False

    epoch = 0
    for epoch in range(args.epochs):
        model.train()
        if rank == 0:
            iterable = tqdm(tr_ld, desc=f"[Epoch {epoch}]", dynamic_ncols=True)
        else:
            iterable = tr_ld  # 非 rank 0 不需要 tqdm

        for vids, ys, meta in iterable:
            vids, ys = vids.to(device, non_blocking=True), ys.to(device, non_blocking=True)
            logits = model(pixel_values=vids).logits
            loss = criterion(logits, ys)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            step += 1
            if rank == 0:
                iterable.set_postfix(loss=f"{loss.item():.4f}")

            if step % args.eval_steps == 0:
                metrics_dict = evaluate_ddp(model, va_ld, device, rank, world_size)
                if rank == 0:
                    f1 = 0
                    print("\n[Val @ step {}]".format(step))
                    for thresh, metrics in metrics_dict.items():
                        print(f"=== Threshold: {thresh} ===")
                        for k, v in metrics.items():
                            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
                        print()
                        if metrics['f1'] > f1:
                            f1 = metrics['f1']
                    if f1 > best_f1 and args.ckpt_dir:
                        best_f1 = f1
                        os.makedirs(args.ckpt_dir, exist_ok=True)
                        pth = os.path.join(args.ckpt_dir, "best_videomae.pth")
                        torch.save(model.module.state_dict(), pth)
                        print(f"[Checkpoint] saved → {pth} (F1={best_f1:.4f})", flush=True)
            dist.barrier()

    # ---------- Cleanup ---------- #
    dist.destroy_process_group()


# ----------------------------- CLI ----------------------------- #

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="/mnt/hdfs/zhufangqi/datasets/simplevla_rl/webdataset_shards/06/17_mp/**/*.tar", help="glob pattern of WebDataset shards")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64, help="per‑GPU batch size")
    ap.add_argument("--val_batch_size", type=int, default=256, help="per‑GPU batch size")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_shards", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--focal_alpha", type=float, default=0.75)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=100000)
    ap.add_argument("--ckpt_dir", default="ckpts_videomae")
    # Distributed
    ap.add_argument("--backend", default="nccl", choices=["nccl", "gloo", "mpi"])
    ap.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Automatically passed by torchrun",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)

# torchrun --standalone --nproc_per_node=8 scripts/success_classifier_vit_v1.2.py --pattern "/mnt/hdfs/zhufangqi/datasets/simplevla_rl/webdataset_shards/06/17_mp/**/*.tar"