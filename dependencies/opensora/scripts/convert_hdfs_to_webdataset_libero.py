import os, argparse, json, uuid, warnings, random
from typing import Any
import h5py, numpy as np
from webdataset import ShardWriter
from tqdm import tqdm  # 用于进度条
from PIL import Image

# ---------- 工具 ----------
def jsonable(v: Any):
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def episode_to_web_sample(ep_idx, root, episode_name):
    key = f"{ep_idx:09d}"
    image_dir = os.path.join(root, "images", episode_name)
    action_path = os.path.join(root, f"actions/{episode_name}.npy")
    image_paths = [os.path.join(image_dir, file_name) for file_name in sorted(os.listdir(image_dir))]
    action = np.load(action_path)[1:]
    images = [np.array(Image.open(image_path).convert("RGB")) for image_path in image_paths]
    video = np.stack(images)
    # video = f["video"][:]
    # action = f["action"][:]
    meta = {
        "finish_step": int(action.shape[0]),
        "episode_name": episode_name
        }

    return {
        "__key__": key,
        "video.npy": video.astype(np.uint8),
        "action.npy": action.astype(np.float32),
        "meta.json": json.dumps(meta).encode("utf-8"),
    }


# ---------- 主函数 ----------
def convert(root, out_dir, episodes_per_shard=10):
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "%06d.tar")

    episode_names = os.listdir(os.path.join(root,'images'))
    random.shuffle(episode_names)  # 完全随机打乱

    with ShardWriter(pattern, maxcount=episodes_per_shard) as sink:
        ep_idx = 0
        for episode_name in tqdm(episode_names, desc="Processing LIBERO files"):
            try:
                sink.write(episode_to_web_sample(ep_idx, root, episode_name))
                ep_idx += 1
            except Exception as exc:
                print(f"[WARN] 跳过 {episode_name}: {exc}")
    print(f"✅  已写入 {ep_idx} 个 episode → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/mnt/hdfs/zhufangqi/datasets/libero/libero_10_no_noops",
                        help="递归遍历的 hdf5 根目录")
    parser.add_argument("--output_dir", default="/mnt/hdfs/zhufangqi/datasets/simplevla_rl/webdataset_shards/libero/libero_10_no_noops",
                        help="保存 .tar 的目录")
    parser.add_argument("--episodes_per_shard", type=int, default=128)
    args = parser.parse_args()
    convert(args.data_path, args.output_dir, args.episodes_per_shard)