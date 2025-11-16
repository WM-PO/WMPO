import os
import json
import tensorflow as tf
from glob import glob

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

from transformers import AutoConfig, AutoTokenizer
from typing import Optional, Tuple, Union

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

class LiberoDataset(torch.utils.data.Dataset):
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
        skip_middle=True,
        image_size=(256, 256),
    ):
        self.data_path = data_path
        # List to hold all samples
        samples = []
        for task_name in os.listdir(data_path):
            if "json" in task_name:
                continue
            images_path = os.path.join(data_path, task_name, 'images')
            episodes = os.listdir(images_path)

            def process_episode(episode, num_frames, task_name):
                """Process a single episode to get valid image paths."""
                episode_num = sum(1 for _ in os.scandir(os.path.join(images_path, episode)))  # Sort to ensure consistent order
                samples = []
                for i in range(1, episode_num):
                    samples.append((task_name, episode, [max(0, i - num_frames + 1 + j) for j in range(num_frames)]))
                return samples
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(32) as executor:
                # Submit tasks to executor
                results = list(tqdm(
                    executor.map(lambda ep: process_episode(ep, num_frames, task_name), episodes),
                    total=len(episodes),
                ))
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
        self.skip_middle = skip_middle
        self.action_dim = 7
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
        with open(os.path.join(self.data_path, 'dataset_statistics.json'), 'r') as f:
            self.dataset_statistics = json.load(f)
        self.q01 = {}
        self.q99 = {}
        for task_name in os.listdir(data_path):
            if "json" in task_name:
                continue
            self.q01[task_name] = np.array(self.dataset_statistics[task_name]['action']['q01'])
            self.q99[task_name] = np.array(self.dataset_statistics[task_name]['action']['q99'])
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                '/mnt/bn/zhufangqi-lq-c2ec0f30/zhufangqi/world-model/openvla/meta-llama/Llama-2-7b-hf', model_max_length=2048, token="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", padding_side="right"
            )
        except:
            tokenizer = AutoTokenizer.from_pretrained(
                '/mnt/hdfs/zhufangqi/pretrained_models/meta-llama/Llama-2-7b-hf', model_max_length=2048, token="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", padding_side="right"
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
        task_name, episode_id, idxs = self.data[index]
        # episode_id = "000001"
        # idxs = [8, 9, 10, 11, 12, 13, 14, 15]

        if self.skip_middle:
            frame_idxs = [idxs[0], idxs[-1]]
        else:
            frame_idxs = idxs
        images_path = [os.path.join(self.data_path, task_name, f"images/{episode_id}/{(idx):03d}.png") for idx in frame_idxs]

        # lang_path = os.path.join(self.data_path, f"lang/{episode_id}.txt")
        # with open(lang_path, "r") as f:
        #     lang = f.read()
        
        actions_path = os.path.join(self.data_path, task_name, f"actions/{episode_id}.npy")

        with open(actions_path, "rb") as f:
            actions = np.load(f)

        action_list = []
        for i in range(self.num_frames-1):
            if idxs[i] == idxs[i+1]:
                action_list.append(np.concatenate([np.zeros(self.action_dim-1), np.ones(1)], axis=-1))
            else:
                action_list.append(actions[idxs[i]])

        action = np.stack(action_list)

        action = (2 * ((action-self.q01[task_name]) / (self.q99[task_name]-self.q01[task_name]))) - 1

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

if __name__ == "__main__":
    dataset = LiberoDataset(
        data_path="/mnt/hdfs/zhufangqi/datasets/libero",
        num_frames=4,
        skip_middle=True
    )

    for i in tqdm(range(len(dataset)), total=len(dataset)):
        data = dataset[i]
        print(data["video"].shape)
        print(data["fps"])
        print(data["height"])
        print(data["width"])
        print(data["num_frames"])