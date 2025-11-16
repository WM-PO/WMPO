import os, argparse, json, warnings, random, multiprocessing as mp
from typing import Any, List, Tuple
import h5py, numpy as np
from webdataset import ShardWriter
from tqdm import tqdm

# ---------- 工具 ----------
def jsonable(v: Any):
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v

def hdf5_to_sample(f: h5py.File, h5_path: str):
    video = f["video"][:]
    action = f["action"][:]
    meta = {k: jsonable(v) for k, v in f.attrs.items()}
    if "finish_step" not in meta:
        warnings.warn("'finish_step' missing in attrs; fallback to len(action)")
        meta["finish_step"] = int(action.shape[0])
    meta["fpath"] = h5_path

    return {
        "video.npy": video.astype(np.uint8),
        "action.npy": action.astype(np.float32),
        "meta.json": json.dumps(meta).encode("utf-8"),
    }

def iter_hdf5(root: str) -> List[str]:
    files = []
    for dp, _, fns in os.walk(root):
        files += [os.path.join(dp, fn) for fn in fns if fn.endswith(".hdf5")]
    return files

def chunk_list(lst: List[Any], n: int) -> List[List[Any]]:
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


# ---------- 子进程 ----------
def worker(proc_idx: int, idx_paths: List[Tuple[int, str]], out_dir: str, episodes_per_shard: int):
    proc_dir = os.path.join(out_dir, f"part_{proc_idx:02d}")
    os.makedirs(proc_dir, exist_ok=True)
    pattern = os.path.join(proc_dir, "%06d.tar")

    processed = 0

    with ShardWriter(pattern, maxcount=episodes_per_shard) as sink:
        for global_idx, h5_path in tqdm(idx_paths,
                                        desc=f"proc-{proc_idx}",
                                        position=proc_idx,
                                        leave=False,
                                        dynamic_ncols=True):
            try:
                with h5py.File(h5_path, "r") as f:
                    sample = hdf5_to_sample(f, h5_path)
                    sample["__key__"] = f"{global_idx:09d}"  # 全局唯一 key
                    sink.write(sample)
                    processed += 1
            except Exception as e:
                print(f"[WARN] 跳过 {h5_path}: {e}")
    return processed


# ---------- 多进程调度 ----------
def convert_parallel(root, out_dir, episodes_per_shard=50, nproc=4):
    os.makedirs(out_dir, exist_ok=True)
    h5_list = sorted(set(iter_hdf5(root)))
    random.shuffle(h5_list)

    idx_paths = list(enumerate(h5_list))  # (global_idx, h5_path)
    chunks = chunk_list(idx_paths, nproc)

    ctx = mp.get_context("spawn")
    with ctx.Pool(nproc) as pool:
        results = [
            pool.apply_async(worker, (i, chunks[i], out_dir, episodes_per_shard))
            for i in range(nproc)
        ]
        pool.close()
        pool.join()

    total_eps = sum(r.get() for r in results)
    print(f"✅ 共写入 {total_eps} 个 episodes → {out_dir}")


# ---------- CLI ----------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/mnt/hdfs/zhufangqi/datasets/simplevla_rl/rollout_base_dir/06/17",
        help="递归遍历的 hdf5 根目录"
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/hdfs/zhufangqi/datasets/simplevla_rl/webdataset_shards/06/17_mp",
        help="保存 .tar 的目录"
    )
    parser.add_argument(
        "--episodes_per_shard",
        type=int,
        default=128,
        help="每个 .tar 包含的 episode 数"
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=32,
        help="并行进程数（建议 ≤ CPU 核心数）"
    )
    args = parser.parse_args()

    convert_parallel(args.data_path, args.output_dir, args.episodes_per_shard, args.nproc)
