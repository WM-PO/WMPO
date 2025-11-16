import os, argparse, json, uuid, warnings, random
from typing import Any
import h5py, numpy as np
from webdataset import ShardWriter
from tqdm import tqdm  # 用于进度条


# ---------- 工具 ----------
def jsonable(v: Any):
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def hdf5_to_sample(ep_idx: int, f: h5py.File, h5_path):
    key = f"{ep_idx:09d}"
    video = f["video"][:]
    action = f["action"][:]
    meta = {k: jsonable(v) for k, v in f.attrs.items()}
    if "finish_step" not in meta:
        warnings.warn(f"'finish_step' missing in attrs; fallback to len(action)")
        meta["finish_step"] = int(action.shape[0])
    meta["fpath"] = h5_path

    return {
        "__key__": key,
        "video.npy": video.astype(np.uint8),
        "action.npy": action.astype(np.float32),
        "meta.json": json.dumps(meta).encode("utf-8"),
    }


def iter_hdf5(root):
    hdf5_files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".hdf5"):
                hdf5_files.append(os.path.join(dp, fn))
    return hdf5_files


# ---------- 主函数 ----------
def convert(root, out_dir, episodes_per_shard=10):
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "%06d.tar")

    hdf5_paths = iter_hdf5(root)
    random.shuffle(hdf5_paths)  # 完全随机打乱

    with ShardWriter(pattern, maxcount=episodes_per_shard) as sink:
        ep_idx = 0
        for h5_path in tqdm(hdf5_paths, desc="Processing HDF5 files"):
            try:
                with h5py.File(h5_path, "r") as f:
                    sink.write(hdf5_to_sample(ep_idx, f, h5_path))
                    ep_idx += 1
            except Exception as exc:
                print(f"[WARN] 跳过 {h5_path}: {exc}")
    print(f"✅  已写入 {ep_idx} 个 episode → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/mnt/hdfs/zhufangqi/datasets/simplevla_rl/rollout_base_dir/06/16",
                        help="递归遍历的 hdf5 根目录")
    parser.add_argument("--output_dir", default="/mnt/hdfs/zhufangqi/datasets/simplevla_rl/webdataset_shards/06/16",
                        help="保存 .tar 的目录")
    parser.add_argument("--episodes_per_shard", type=int, default=10)
    args = parser.parse_args()
    convert(args.data_path, args.output_dir, args.episodes_per_shard)