import os
import h5py
import numpy as np
import torch
import json
import tensorflow as tf
from torch.utils.data import Dataset

class SimpleVLALIBERODataset(Dataset):
    def __init__(self, 
                data_path,
                stats_path, 
                Ta=8, 
                To=4, 
                action_dim = 7,
                image_size = (256, 256),
                stride=1):
        """
        参数说明：
            data_path: 根目录，递归查找 .hdf5 文件
            Ta: 动作序列长度 (T)
            To: 视频帧前缀帧数 (比如 4，即 0帧重复4次)
            stride: 滑动步长
        """
        self.data_path = data_path
        self.Ta = Ta
        self.To = To
        self.context_length = Ta + To
        self.image_size = image_size
        self.action_dim = action_dim
        self.stride = stride
        self.task_name = "libero_10_no_noops"
        self.sample_index = []  # (filepath, start_index)
        with open(stats_path, 'r') as f: self.dataset_statistics = json.load(f)
        self.q01, self.q99 = {}, {}
        for t_name in self.dataset_statistics:
            self.q01[t_name] = np.array(self.dataset_statistics[t_name]['action']['q01'])
            self.q99[t_name] = np.array(self.dataset_statistics[t_name]['action']['q99'])
        self._build_index()


    def _build_index(self):
        """先按文件名排序，再逐个读取文件构建 sample_index。"""
        # 1. 收集所有 .hdf5 文件完整路径
        hdf5_paths = []
        for dirpath, _, filenames in os.walk(self.data_path):
            for fname in filenames:
                if fname.endswith(".hdf5"):
                    hdf5_paths.append(os.path.join(dirpath, fname))

        # 2. 统一排序（按路径字符串字典序；可替换为自定义 key）
        for fpath in sorted(hdf5_paths):
            try:
                with h5py.File(fpath, "r") as f:
                    max_step = f.attrs["finish_step"]
                    # 3. 根据步长/窗口生成 (file_path, start_step) 索引
                    for start in range(0, max_step - self.Ta + 1, self.stride):
                        self.sample_index.append((fpath, start))
            except Exception as e:
                print(f"Error reading file {fpath}: {e}")
                        

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        fpath, start = self.sample_index[idx]
        with h5py.File(fpath, "r") as f:
            action = f["action"][start : start + self.Ta]  # (T,)

            video_start = start - self.To + 1
            video_end = start + self.Ta + 1  # T+1

            if video_start < 0:
                pad_len = -video_start
                first_frame = f["video"][0:1]  # (1, C, H, W)
                pad_frames = np.repeat(first_frame, pad_len, axis=0)
                real_frames = f["video"][0 : video_end]  # (T+1,)
                video = np.concatenate([pad_frames, real_frames], axis=0)
            else:
                video = f["video"][video_start : video_end]  # (To + T + 1,)

        video = torch.from_numpy(video)
        assert not torch.all(video[-1] == 0)
        video = video.permute(3, 0, 1, 2)
        video = ((video / 255.0) * 2.0) - 1.0
        video = video.float()

        action = (2 * ((action - self.q01[self.task_name]) / (self.q99[self.task_name] - self.q01[self.task_name]))) - 1
        action = action.reshape(-1, self.action_dim)
        action = torch.from_numpy(action).float()
        meta = {
            "fpath" : fpath,
            "video_start" : video_start,
            "video_end" : video_end,
            "action_start" : start,
            "action_end" : start + self.Ta,
        }

        ret = {
            "video": video, "action": action, "fps": 30,
            "num_frames": video.shape[1], "height": self.image_size[0], "width": self.image_size[1],
            "meta": meta
        }
        return ret

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataset = SimpleVLALIBERODataset(
        data_path="/mnt/hdfs/zhufangqi/datasets/simplevla_rl/rollout_base_dir/06/16",
        stats_path="/mnt/hdfs/zhufangqi/datasets/libero/dataset_statistics.json",
        Ta=8,
        To=4,
        stride=1
    )
    

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    all_data = []

    # 遍历数据加载器
    for batch in tqdm(loader, total=len(loader)):
        batch_data = []

        for i in range(batch['video'].shape[0]):
            sample_info = {
                "fpath": batch['meta']['fpath'][i],
                "video_start": batch['meta']['video_start'][i].item(),
                "video_end": batch['meta']['video_end'][i].item(),
                "action_start": batch['meta']['action_start'][i].item(),
                "action_end": batch['meta']['action_end'][i].item(),
            }
            batch_data.append(sample_info)

        all_data.extend(batch_data)  # 也可以按batch存为嵌套列表

    # 保存为 JSON 文件
    with open("hdf5_data_info.json", "w") as f:
        json.dump(all_data, f, indent=4)
