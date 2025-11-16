import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
from PIL import Image
from opensora.registry import DATASETS

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
if __name__ == "__main__":
    import imageio
    dataset = RealPushTDataset(['/opt/tiger/opensora/data/0412.h5', '/opt/tiger/opensora/data/0416_inference.h5'])
    # dataset = RealPushTDataset(['/opt/tiger/opensora/data/0412.h5'])
    # dataset = RealPushTDataset(['/opt/tiger/opensora/data/0416_inference.h5'])
    for sample in dataset:
        img = ((sample['video'][:, 0]/2 + 0.5)*255.0).astype(np.uint8).transpose(1,2,0)
        imageio.imwrite('debug.png', img)
        print(sample['video'].shape)
        print(sample['action'].shape)