import os
import json
import tensorflow as tf
from glob import glob
import re
import av
import numpy as np
import torch
from PIL import ImageFile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from PIL import Image
import torchvision.transforms as transforms

from opensora.registry import DATASETS

from opensora.datasets.read_video import read_video
from opensora.datasets.utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop
from opensora.datasets.action_tokenizer import ActionTokenizer

from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from typing import Optional, Tuple, Union

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
from PIL import Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_FPS = 120

def rel2abs_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute actions (0 = closed; 1 = open).

    Assumes that the first relative gripper is not redundant (i.e. close when already closed)!
    """
    # Note =>> -1 for closing, 1 for opening, 0 for no change
    opening_mask, closing_mask = actions < -0.1, actions > 0.1
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(thresholded_actions[i] == 0, lambda: carry, lambda: thresholded_actions[i])

    # If no relative grasp, assumes open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0).numpy()]
    start = tf.cond(start == 0, lambda: 1, lambda: start)

    # Note =>> -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)
    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5

    return new_actions

def resize_image(image: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
    """Resizes an image using Lanczos3 interpolation. Expects & returns uint8."""
    assert image.dtype == tf.uint8
    image = tf.image.resize(image, size, method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image



def read_specific_frames(video_path: str, indices: list) -> list:
    """高效地从视频文件中只读取指定索引的帧。"""
    frames = []
    # 补零的索引(0)可能会重复出现，我们需要保留它们
    # 因此，我们直接使用传入的indices，并在读取时处理
    unique_indices = sorted(list(set(i for i in indices if i >= 0)))
    
    if not unique_indices: # 如果所有索引都是0
        # 假设我们需要一个黑色的帧
        # 从视频中读取第一帧作为模板来获取 H, W, C
        try:
            with av.open(video_path) as container:
                first_frame = next(container.decode(video=0))
                h, w, _ = first_frame.to_ndarray(format='rgb24').shape
            # 返回一个全零的黑色帧列表
            return [np.zeros((h, w, 3), dtype=np.uint8)] * len(indices)
        except (StopIteration, av.AVError): # 视频为空或损坏
            return [] # 返回空列表，让getitem处理

    frame_map = {}
    target_idx_ptr = 0
    
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            frame_count = 0
            for frame in container.decode(stream):
                if target_idx_ptr >= len(unique_indices):
                    break
                if frame_count == unique_indices[target_idx_ptr]:
                    frame_map[frame_count] = frame.to_ndarray(format='rgb24')
                    target_idx_ptr += 1
                frame_count += 1
        
        # 根据原始的indices（包含重复和0）构建最终的帧列表
        # 使用第一帧作为0索引的填充
        first_frame_data = frame_map[unique_indices[0]]
        for i in indices:
            frames.append(frame_map.get(i, first_frame_data))

    except av.AVError as e:
        print(f"处理视频 {video_path} 时发生 PyAV 错误: {e}")
        return []

    return frames

@DATASETS.register_module()
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
        for dirpath, _, filenames in os.walk(self.data_path):
            for fname in filenames:
                if fname.endswith(".hdf5"):
                    fpath = os.path.join(dirpath, fname)
                    try:
                        with h5py.File(fpath, "r") as f:
                            max_step = f.attrs['finish_step']
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
        video = video.permute(3, 0, 1, 2)
        video = ((video / 255.0) * 2.0) - 1.0
        video = video.float()

        action = (2 * ((action - self.q01[self.task_name]) / (self.q99[self.task_name] - self.q01[self.task_name]))) - 1
        action = action.reshape(-1, self.action_dim)
        action = torch.from_numpy(action).float()

        ret = {
            "video": video, "action": action, "fps": 30,
            "num_frames": video.shape[1], "height": self.image_size[0], "width": self.image_size[1]
        }
        return ret

@DATASETS.register_module()
class LiberoContinueDataset(torch.utils.data.Dataset):
    """
    数据集类的最终修正版，正确实现了补零逻辑。
    """
    def __init__(
        self,
        data_path,
        stats_path,
        To=4,
        Ta=8,
        image_size=(224, 224),
        action_dim=7,
    ):
        self.data_path = data_path
        self.To = To
        self.Ta = Ta
        self.image_size = image_size
        self.action_dim = action_dim
        self.context_length = To + Ta
        self.task_name = "libero_10_no_noops"
        
        videos_dir = os.path.join(data_path, 'videos')
        actions_dir = os.path.join(data_path, 'actions')

        if not os.path.isdir(videos_dir) or not os.path.isdir(actions_dir):
            raise FileNotFoundError(f"请确保 'videos' 和 'actions' 目录存在于 '{data_path}'")

        self.samples = []
        for video_filename in tqdm(os.listdir(videos_dir), desc="扫描数据集中"):
            if not video_filename.endswith('.mp4'):
                continue
            match = re.match(r'(.*)_finish_step_(\d+)\.mp4', video_filename)
            if not match: continue
            
            base_name, finish_step = match.groups()
            finish_step = int(finish_step)
            video_path = os.path.join(videos_dir, video_filename)
            action_path = os.path.join(actions_dir, f"{base_name[0:-14]}.npz.npy")

            if not os.path.exists(action_path): continue

            # --- 核心修正点 ---
            # 1. 修正循环范围，使其能产生需要补零的早期窗口
            # 窗口结束帧'i'的最大值必须是有效的动作索引，即 finish_step - 2
            # `range`函数的终点是`finish_step - 1`，所以'i'的最大值是 finish_step - 2
            # 循环的起点恢复为您原始逻辑中的 Ta
            for i in range(Ta, finish_step - 1):
                # 2. 恢复 `max(0, ...)` 逻辑以实现补零
                indices = [max(0, i - self.context_length + 1 + j) for j in range(self.context_length)]
                self.samples.append((video_path, action_path, indices))

        print(f"共找到 {len(self.samples)} 个有效样本。")

        # ... (加载统计数据的部分保持不变) ...
        if not os.path.exists(stats_path): raise FileNotFoundError(f"统计文件未找到: {stats_path}")
        with open(stats_path, 'r') as f: self.dataset_statistics = json.load(f)
        self.q01, self.q99 = {}, {}
        for t_name in self.dataset_statistics:
            self.q01[t_name] = np.array(self.dataset_statistics[t_name]['action']['q01'])
            self.q99[t_name] = np.array(self.dataset_statistics[t_name]['action']['q99'])
        print("动作归一化统计数据加载完毕。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 这个方法无需修改，它现在会接收到正确的、包含0的 indices 列表
        video_path, action_path, frame_indices = self.samples[index]

        try:
            frame_list_np = read_specific_frames(video_path, frame_indices)

            if len(frame_list_np) != len(frame_indices):
                 print(f"警告: 样本 {index} ({video_path}) 帧数不足，将随机替换。")
                 return self.__getitem__(np.random.randint(len(self)))

            processed_frames = []
            for frame_np in frame_list_np:
                frame_tf = tf.convert_to_tensor(frame_np, dtype=tf.uint8)
                frame_tf = tf.image.resize(frame_tf, size=self.image_size, method="lanczos3", antialias=True)
                frame_tf = tf.cast(tf.clip_by_value(tf.round(frame_tf), 0, 255), tf.uint8)
                processed_frames.append(frame_tf.numpy())
            
            video_np = np.stack(processed_frames)
            video = torch.from_numpy(video_np)

            with open(action_path, "rb") as f:
                actions = np.load(f)
            
            # 补零的动作也用第0个动作填充
            action_indices_to_fetch = [max(0, idx) for idx in frame_indices[self.To:]]
            action_list = [actions[idx] for idx in action_indices_to_fetch]
            action = np.stack(action_list)
            
            action = (2 * ((action - self.q01[self.task_name]) / (self.q99[self.task_name] - self.q01[self.task_name]))) - 1
            action = action.reshape(-1, self.action_dim)

            video = video.permute(3, 0, 1, 2)
            video = ((video / 255.0) * 2.0) - 1.0

            ret = {
                "video": video.float(), "action": torch.from_numpy(action).float(), "fps": 30,
                "num_frames": self.context_length, "height": self.image_size[0], "width": self.image_size[1]
            }
            return ret

        except Exception as e:
            print(f"在索引 {index} 加载数据时发生严重错误，路径 {video_path}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

@DATASETS.register_module()
class LiberoDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data_paths=None,
        To = 4,
        Ta = 8,
        image_size=(224, 224),
        action_dim=7, 
    ):
        if not isinstance(data_paths, list):
            data_paths = [data_paths]
        self.data_paths = data_paths
        self.To = To
        self.Ta = Ta
        # List to hold all samples
        self.context_length = To + Ta 
        samples = []
        for data_path_idx, data_path in tqdm(enumerate(data_paths), total = len(data_paths), desc = "loading dataset"):
            if data_path_idx != 0:
                for task_name in os.listdir(data_path):
                    for task_id in os.listdir(os.path.join(data_path, task_name)):
                        for images_name in os.listdir(os.path.join(data_path, task_name, task_id, 'images')):
                            if int(images_name) <= 2:
                                continue
                            images_path = os.path.join(data_path, task_name, task_id, 'images', images_name)
                            actions_path = images_path.replace('images','actions') + '.npy'
                            episode_num = np.load(actions_path).shape[0]
                            for i in range(Ta, episode_num):
                                samples.append((task_name+"_no_noops", images_path, actions_path, [max(0, i - self.context_length + 1 + j) for j in range(self.context_length)]))
            else:
                for task_name in os.listdir(data_path):
                    if "json" in task_name:
                        continue
                    images_path = os.path.join(data_path, task_name, 'images')
                    episodes = os.listdir(images_path)

                    def process_episode(episode, context_length, task_name):
                        """Process a single episode to get valid image paths."""
                        episode_num = sum(1 for _ in os.scandir(os.path.join(images_path, episode)))  # Sort to ensure consistent order
                        samples = []

                        tmp_images_path = os.path.join(data_path, task_name, f"images/{episode}")
                        tmp_actions_path = os.path.join(data_path, task_name, f"actions/{episode}.npy")

                        for i in range(Ta, episode_num):
                            samples.append((task_name, tmp_images_path, tmp_actions_path, [max(0, i - context_length + 1 + j) for j in range(context_length)]))
    
                        return samples
                    
                    # Use ThreadPoolExecutor for parallel processing
                    with ThreadPoolExecutor(32) as executor:
                        # Submit tasks to executor
                        results = list(tqdm(
                            executor.map(lambda ep: process_episode(ep, self.context_length, task_name), episodes),
                            total=len(episodes),
                        ))
                    for res in results:
                        samples.extend(res)

        self.data = samples
        # self.get_text = "text" in self.data.columns
        self.image_size = image_size
        self.action_dim = action_dim
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        with open(os.path.join(self.data_paths[0], 'dataset_statistics.json'), 'r') as f:
            self.dataset_statistics = json.load(f)
        self.q01 = {}
        self.q99 = {}
        for task_name in os.listdir(data_paths[0]):
            if "json" in task_name:
                continue
            self.q01[task_name] = np.array(self.dataset_statistics[task_name]['action']['q01'])
            self.q99[task_name] = np.array(self.dataset_statistics[task_name]['action']['q99'])
        
        self.processor = AutoProcessor.from_pretrained("/mnt/hdfs/zhufangqi/pretrained_models/Haozhan72/Openvla-oft-SFT-libero10-traj1", trust_remote_code=True)
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)

    def convert_actions_to_tokens(self, action: np.ndarray):
        action = np.clip(action, a_min=float(self.action_tokenizer.min_action), a_max=float(self.action_tokenizer.max_action))
        discretized_action = np.digitize(action, self.action_tokenizer.bins)
        return discretized_action

    def convert_tokens_to_actions(self, token: np.ndarray):
        discretized_actions = np.clip(token - 1, a_min=0, a_max=self.action_tokenizer.bin_centers.shape[0] - 1)
        return self.action_tokenizer.bin_centers[discretized_actions]


    def _print_data_number(self):
        num_videos = 0
        num_images = 0
        for path in self.data["path"]:
            if self.get_type(path) == "video":
                num_videos += 1
            else:
                num_images += 1
        print(f"Dataset contains {num_videos} videos and {num_images} images.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def getitem(self, index):
        task_name, images_path, actions_path, idxs = self.data[index]
        # episode_id = "000001"
        # idxs = [8, 9, 10, 11, 12, 13, 14, 15]
        if self.data_paths[0] not in images_path:
            images_path = [os.path.join(images_path, f"{idx}.png") for idx in idxs]
        else:
            images_path = [os.path.join(images_path, f"{(idx):03d}.png") for idx in idxs]
        

        with open(actions_path, "rb") as f:
            actions = np.load(f)

        action_list = []
        for i in range(self.To, self.context_length):
            action_list.append(actions[idxs[i]])

        action = np.stack(action_list)

        action = (2 * ((action-self.q01[task_name]) / (self.q99[task_name]-self.q01[task_name]))) - 1

        # action = self.convert_actions_to_tokens(action)
        # action = self.convert_tokens_to_actions(action)
        action = action.reshape(-1, self.action_dim)

        video = []
        for img_path in images_path:
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
            img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
            img = tf.image.resize(img, size=self.image_size, method="lanczos3", antialias=True)
            img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
            img = img.numpy()
            img = torch.tensor(img)
            video.append(img)

        # 将处理后的图片堆叠成一个视频张量 (N, C, H, W)
        video = torch.stack(video)

        # TCHW -> CTHW
        video = video.permute(3, 0, 1, 2)
        video = ((video / 255.0) * 2.0) - 1

        # import imageio
        # rec_video = (
        #     (video.permute(1, 2, 3, 0).numpy() * 0.5 + 0.5) * 255
        # ).clip(0, 255).astype(np.uint8)
        # imageio.mimwrite('output.mp4', rec_video, fps=3)


        ret = {"video": video, 
               "fps": 3, 
               "action": action,
               "num_frames": video.shape[1],
               "height": video.shape[2],
               "width": video.shape[3]}

        return ret

    def __getitem__(self, index):
        return self.getitem(index)
        # for _ in range(10):
        #     try:
        #         return self.getitem(index)
        #     except Exception as e:
        #         data = self.data[index]
        #         print(f"data {data}: {e}")
        #         index = np.random.randint(len(self))
        # raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)


class FixMinMaxNormalizer:
    
    """
        normalizes data through maximum and minimum expansion.
    """

    def __init__(self):
        # X = X.reshape(-1, X.shape[-1]).astype(np.float32)
        self.min = np.array([13.456424, 32.938293], dtype=np.float32)
        self.max = np.array([496.14618, 510.9579], dtype=np.float32)

        self.range = self.max - self.min
        if np.any(self.range == 0):
            self.range = self.max - self.min
            print("Warning: Some features have the same min and max value. These will be set to 0.")
            self.range[self.range == 0] = 1

    def normalize(self, x):
        x = x.astype(np.float32)
        # nomalize to [0,1]
        nx = (x - self.min) / self.range
        # normalize to [-1, 1]
        nx = nx * 2 - 1
        return nx

    def unnormalize(self, x):
        x = x.astype(np.float32)
        nx = (x + 1) / 2
        x = nx * self.range + self.min
        return x

@DATASETS.register_module()
class PushTDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data_path=None,
        num_frames=18,
        image_size=(96, 96),
    ):
        # List to hold all samples
        samples = []
        for data_name in tqdm(os.listdir(data_path), total = len(os.listdir(data_path)), desc = "loading dataset"):
            for episode_idx in os.listdir(os.path.join(data_path, data_name, 'images')):
                images_path = os.path.join(data_path, data_name, f'images/{episode_idx}')
                actions_path = images_path.replace('images','actions') + '.npy'
                episode_num = len(os.listdir(images_path))
                # 从第 num_frames-1 帧开始采样，保证每个样本都有 num_frames 个连续的帧
                for i in range(num_frames - 1, episode_num):
                    indices = list(range(i - num_frames + 1, i + 1))
                    samples.append((images_path, actions_path, indices))
                # print()


        self.data = samples
        # self.get_text = "text" in self.data.columns
        self.num_frames = num_frames
        self.image_size = image_size
        self.action_dim = 2
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        self.action_normalizer = FixMinMaxNormalizer()

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def getitem(self, index):
        images_path, actions_path, idxs = self.data[index]
        images_path = [os.path.join(images_path, f"{(idx):04d}.png") for idx in idxs]
    
        with open(actions_path, "rb") as f:
            actions = np.load(f)
        action_list = []
        for i in range(2, self.num_frames):
            action_list.append(actions[idxs[i]])
        action = np.stack(action_list)
        action = self.action_normalizer.normalize(action)

        video = []
        for img_path in images_path:
            # 打开图片，调整大小，然后应用 Transform
            image = Image.open(img_path).convert("RGB")
            # image = tf.convert_to_tensor(image)
            # image = resize_image(image, self.image_size)  # 调整大小
            image = np.array(image)
            processed_image = self.transform(image)  # 应用 Transform
            video.append(processed_image)

        # 将处理后的图片堆叠成一个视频张量 (N, C, H, W)
        video = torch.stack(video)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        # import imageio
        # rec_video = (
        #     (video.permute(1, 2, 3, 0).numpy() * 0.5 + 0.5) * 255
        # ).clip(0, 255).astype(np.uint8)
        # imageio.mimwrite('output.mp4', rec_video, fps=3)


        ret = {"video": video, 
               "fps": 3, 
               "action": action,
               "num_frames": video.shape[1],
               "height": video.shape[2],
               "width": video.shape[3]}

        return ret

    def __getitem__(self, index):
        return self.getitem(index)
        # for _ in range(10):
        #     try:
        #         return self.getitem(index)
        #     except Exception as e:
        #         data = self.data[index]
        #         print(f"data {data}: {e}")
        #         index = np.random.randint(len(self))
        # raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)


@DATASETS.register_module()
class RT1Dataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data_path=None,
        num_frames=8,
        image_size=(224, 224),
        transform_name="center",
    ):
        self.data_path = data_path
        # self.data = read_file(data_path)
        images_path = os.path.join(data_path, 'images')
        episodes = os.listdir(images_path)

        def process_episode(episode, num_frames):
            """Process a single episode to get valid image paths."""
            episode_num = sum(1 for _ in os.scandir(os.path.join(images_path, episode)))  # Sort to ensure consistent order
            samples = []
            for i in range(1, episode_num):
                samples.append((episode, [max(0, i - num_frames + 1 + j) for j in range(num_frames)]))
            return samples

        # List to hold all samples
        samples = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(32) as executor:
            # Submit tasks to executor
            results = list(tqdm(
                executor.map(lambda ep: process_episode(ep, num_frames), episodes),
                total=len(episodes),
            ))

        # Combine all results into the samples list
        for res in results:
            samples.extend(res)
        # samples = []
        # for episode in tqdm(episodes, total = len(episodes)):
        #     images = [os.path.join(episode, image) for image in os.listdir(episode)[:-num_frames]]
        #     samples.extend(images)
        self.data = samples
        # self.get_text = "text" in self.data.columns
        self.num_frames = num_frames
        self.image_size = image_size
        # self.transforms = {
        #     "image": get_transforms_image(transform_name, image_size),
        #     "video": get_transforms_video(transform_name, image_size),
        # }
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        with open('opensora/datasets/dataset_statistics.json', 'r') as f:
            self.dataset_statistics = json.load(f)
        self.q01 = np.array(self.dataset_statistics['fractal20220817_data']['action']['q01'])
        self.q99 = np.array(self.dataset_statistics['fractal20220817_data']['action']['q99'])
        
        tokenizer = AutoTokenizer.from_pretrained(
            '/mnt/bn/zhufangqi-lq-c2ec0f30/zhufangqi/world-model/openvla/meta-llama/Llama-2-7b-hf', model_max_length=2048, token="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", padding_side="right"
        )
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        # get dataset
        action_tokenizer = ActionTokenizer(tokenizer)

        self.tokenizer = tokenizer
        self.action_tokenizer = action_tokenizer

    def _print_data_number(self):
        num_videos = 0
        num_images = 0
        for path in self.data["path"]:
            if self.get_type(path) == "video":
                num_videos += 1
            else:
                num_images += 1
        print(f"Dataset contains {num_videos} videos and {num_images} images.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def getitem(self, index):
        sample = self.data[index]
        episode_id, idxs = sample
        # episode_id = "000001"
        # idxs = [8, 9, 10, 11, 12, 13, 14, 15]
        images_path = [os.path.join(self.data_path,f"images/{episode_id}/{(idx):03d}.png") for idx in idxs]

        # lang_path = os.path.join(self.data_path, f"lang/{episode_id}.txt")
        # with open(lang_path, "r") as f:
        #     lang = f.read()
        
        actions_path = os.path.join(self.data_path, f"actions/{episode_id}.npy")

        with open(actions_path, "rb") as f:
            actions = np.load(f)

        gripper_actions_continuous = tf.convert_to_tensor(actions[:, -1])
        gripper_action_discretized = rel2abs_gripper_actions(gripper_actions_continuous)
        gripper_action_discretized = gripper_action_discretized.numpy()
        
        action = actions[idxs[-1]-1]
        action[-1] = gripper_action_discretized[idxs[-1]-1]

        action = (2 * ((action-self.q01) / (self.q99-self.q01))) - 1

        action = self.action_tokenizer.convert_actions_to_tokens(action)

        # 143, 127, 132,  96, 139, 126,   1
        video = []
        for img_path in images_path:
            # 打开图片，调整大小，然后应用 Transform
            image = Image.open(img_path).convert("RGB")
            image = tf.convert_to_tensor(image)
            image = resize_image(image, self.image_size)  # 调整大小
            image = np.array(image)
            processed_image = self.transform(image)  # 应用 Transform
            video.append(processed_image)

        # 将处理后的图片堆叠成一个视频张量 (N, C, H, W)
        video = torch.stack(video)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        # import imageio
        # rec_video = (
        #     (video.permute(1, 2, 3, 0).numpy() * 0.5 + 0.5) * 255
        # ).clip(0, 255).astype(np.uint8)
        # imageio.mimwrite('output.mp4', rec_video, fps=3)


        ret = {"video": video, 
               "fps": 3, 
               "action": action,
               "num_frames": video.shape[1],
               "height": video.shape[2],
               "width": video.shape[3]}

        return ret

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                data = self.data[index]
                print(f"data {data}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)

@DATASETS.register_module()
class VideoTextDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data_path=None,
        num_frames=16,
        frame_interval=1,
        image_size=(256, 256),
        transform_name="center",
    ):
        self.data_path = data_path
        self.data = read_file(data_path)
        self.get_text = "text" in self.data.columns
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }

    def _print_data_number(self):
        num_videos = 0
        num_images = 0
        for path in self.data["path"]:
            if self.get_type(path) == "video":
                num_videos += 1
            else:
                num_images += 1
        print(f"Dataset contains {num_videos} videos and {num_images} images.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        file_type = self.get_type(path)

        if file_type == "video":
            # loading
            vframes, vinfo = read_video(path, backend="av")
            video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            # Sampling video frames
            video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)

            # transform
            transform = self.transforms["video"]
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = self.transforms["image"]
            image = transform(image)

            # repeat
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        ret = {"video": video, "fps": video_fps}
        if self.get_text:
            ret["text"] = sample["text"]
        return ret

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                path = self.data.iloc[index]["path"]
                print(f"data {path}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)


@DATASETS.register_module()
class VariableVideoTextDataset(VideoTextDataset):
    def __init__(
        self,
        data_path=None,
        num_frames=None,
        frame_interval=1,
        image_size=(None, None),
        transform_name=None,
        dummy_text_feature=False,
    ):
        super().__init__(data_path, num_frames, frame_interval, image_size, transform_name=None)
        self.transform_name = transform_name
        self.data["id"] = np.arange(len(self.data))
        self.dummy_text_feature = dummy_text_feature

    def get_data_info(self, index):
        T = self.data.iloc[index]["num_frames"]
        H = self.data.iloc[index]["height"]
        W = self.data.iloc[index]["width"]
        return T, H, W

    def getitem(self, index):
        # a hack to pass in the (time, height, width) info from sampler
        index, num_frames, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]
        path = sample["path"]
        file_type = self.get_type(path)
        ar = height / width

        video_fps = 24  # default fps
        if file_type == "video":
            # loading
            vframes, vinfo = read_video(path, backend="av")
            video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            # Sampling video frames
            video = temporal_random_crop(vframes, num_frames, self.frame_interval)
            video = video.clone()
            del vframes

            video_fps = video_fps // self.frame_interval

            # transform
            transform = get_transforms_video(self.transform_name, (height, width))
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)

            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        ret = {
            "video": video,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
        }
        if self.get_text:
            ret["text"] = sample["text"]
        if self.dummy_text_feature:
            text_len = 50
            ret["text"] = torch.zeros((1, text_len, 1152))
            ret["mask"] = text_len
        return ret

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except:
            return None


@DATASETS.register_module()
class BatchFeatureDataset(torch.utils.data.Dataset):
    """
    The dataset is composed of multiple .bin files.
    Each .bin file is a list of batch data (like a buffer). All .bin files have the same length.
    In each training iteration, one batch is fetched from the current buffer.
    Once a buffer is consumed, load another one.
    Avoid loading the same .bin on two difference GPUs, i.e., one .bin is assigned to one GPU only.
    """

    def __init__(self, data_path=None):
        self.path_list = sorted(glob(data_path + "/**/*.bin"))

        self._len_buffer = len(torch.load(self.path_list[0]))
        self._num_buffers = len(self.path_list)
        self.num_samples = self.len_buffer * len(self.path_list)

        self.cur_file_idx = -1
        self.cur_buffer = None

    @property
    def num_buffers(self):
        return self._num_buffers

    @property
    def len_buffer(self):
        return self._len_buffer

    def _load_buffer(self, idx):
        file_idx = idx // self.len_buffer
        if file_idx != self.cur_file_idx:
            self.cur_file_idx = file_idx
            self.cur_buffer = torch.load(self.path_list[file_idx])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self._load_buffer(idx)

        batch = self.cur_buffer[idx % self.len_buffer]  # dict; keys are {'x', 'fps'} and text related

        ret = {
            "video": batch["x"],
            "text": batch["y"],
            "mask": batch["mask"],
            "fps": batch["fps"],
            "height": batch["height"],
            "width": batch["width"],
            "num_frames": batch["num_frames"],
        }
        return ret

class MinMaxNormalizer:
    """
        normalizes data through maximum and minimum expansion.
    """

    def __init__(self, X):
        X = X.reshape(-1, X.shape[-1]).astype(np.float32)
        self.min, self.max = np.min(X, axis=0), np.max(X, axis=0)
        self.range = self.max - self.min
        if np.any(self.range == 0):
            self.range = self.max - self.min
            print("Warning: Some features have the same min and max value. These will be set to 0.")
            self.range[self.range == 0] = 1

    def normalize(self, x):
        x = x.astype(np.float32)
        # nomalize to [0,1]
        nx = (x - self.min) / self.range
        # normalize to [-1, 1]
        nx = nx * 2 - 1
        return nx

    def unnormalize(self, x):
        x = x.astype(np.float32)
        nx = (x + 1) / 2
        x = nx * self.range + self.min
        return x

@DATASETS.register_module()
class RealPushTDataset(Dataset):
    def __init__(self, data_paths='', num_frames=18, image_size = [96, 72]):
        """
        data_path: HDF5 文件路径
        num_frames: 动作序列长度，本例为 16
        camera: 使用哪个摄像头的 rgb 图像，本例默认 'D405'
        """
        self.data_paths = data_paths
        self.num_frames = num_frames
        self.image_size = image_size
        # 打开 HDF5 文件，注意这里以只读模式打开文件，文件会一直保持打开状态
        self.samples = []
        self.files = []
        for file_idx, data_path in enumerate(self.data_paths):
            file = h5py.File(data_path, 'r')
            self.files.append(file)
            # 存储 (group_name, start_index) 的列表，每个样本对应一个动作段

            replay_buffer = {
                # 'state': [],
                'action': [],}

            # 遍历 HDF5 中的所有组（每个组对应一个轨迹）
            for group_name in file:
                group = file[group_name]
                # rgb_dataset = group[f'cameras/{self.camera}/rgb']
                actions_dataset = group['action']
                # states_dataset = group['state']
                actions = actions_dataset[:, 0:2]
                # states = states_dataset
                replay_buffer['action'].extend(actions)
                # replay_buffer['state'].extend(states)
                # 假设机器人动作与相机帧对齐
                length = actions_dataset.shape[0]
                # 为了取到一段连续的16个动作，需要保证索引 start_idx + num_frames 不越界
                for start_idx in range(length - num_frames):
                    self.samples.append((file_idx, group_name, start_idx))
    
        # agent_pos_normalizer = MinMaxNormalizer(np.array(replay_buffer['state']))
        # image_normalizer = ImageNormalizer()
        self.action_normalizer = MinMaxNormalizer(np.array(replay_buffer['action']))
                
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_idx, group_name, start_idx = self.samples[idx]
        group = self.files[file_idx][group_name]
        
        # 加载相机 RGB 图像数据（忽略深度图）
        rgb_dataset = group[f'video']
        # state_dataset = group[f'state']
        # 加载机器人动作数据
        action_dataset = group['action']
        
        # 取两帧观测：起始帧和经过 num_frames 个动作之后的目标帧
        obs = rgb_dataset[start_idx: start_idx + self.num_frames]       # 起始观测帧
        obs = np.ascontiguousarray(obs)
        # states = state_dataset[start_idx: start_idx+2][:]

        # 取出连续的16个动作，形状 (16, 6)
        actions = action_dataset[start_idx + 1: start_idx + self.num_frames - 1, 0:2]

        frames = []

        for i in range(obs.shape[0]):
            frame = obs[i]  # shape: (3, 240, 320)
            # 用 PIL resize 到 (80, 60)
            pil_image = Image.fromarray(frame)
            pil_image = pil_image.resize(self.image_size, Image.BILINEAR)
            resized_np = np.asarray(pil_image).astype(np.float32) / 255.0  # → [0,1]
            normalized = resized_np * 2.0 - 1.0
            frames.append(normalized)
            # frames.append(((normalized + 1.0) / 2.0 * 255).astype(np.uint8))

        # imageio.mimwrite('resized_normalized_video.mp4', frames, fps=10, codec='libx264')
        video = np.stack(frames, axis=0)  # (T, H, W, C)
        video = video.transpose(3, 0, 1, 2)  # (T, C, H, W)
        # video = self.normalizer['obs']['image'].normalize(obs)
        # agent_pos = states.astype(np.float32)  # (T, 2)
        # agent_pos = self.normalizer['obs']['agent_pos'].normalize(agent_pos)
        action = actions.astype(np.float32)  # (T, 2)
        action = self.action_normalizer.normalize(action)
        
        # 返回的结果为一个字典，包含两帧图像和对应的动作序列
        
        ret = {"video": video, 
               "fps": 3, 
               "action": action,
               "num_frames": video.shape[1],
               "height": video.shape[2],
               "width": video.shape[3]}
        return ret
    
    def __del__(self):
        # 当对象销毁时，关闭 HDF5 文件
        if hasattr(self, 'file'):
            self.file.close()



# # 示例：如何使用该 Dataset
# if __name__ == "__main__":
#     import imageio
#     dataset = RealPushTDataset('data/0406.h5')
#     for sample in dataset:
#         # img = (dataset.normalizer['obs']['image'].unnormalize(image[0])*255.0).transpose(1,2,0).astype(np.uint8)
#         # imageio.imwrite('debug.png', img)
#         print(sample['video'].shape)
#         print(sample['action'].shape)


if __name__ == "__main__":
    dataset = PushTDataset(
        data_path = "/opt/tiger/CleanDiffuser/data/world_model",
        num_frames=18
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True
    )

    for batch in tqdm(dataloader, total = len(dataloader)):
        print(batch["video"].shape)
        print(batch["action"].shape)


    for i in tqdm(range(len(dataset)), total=len(dataset)):
        data = dataset[i]
        print(data["video"].shape)
        print(data["fps"])
        print(data["height"])
        print(data["width"])
        print(data["num_frames"])