import os
import imageio
import tensorflow_datasets as tfds

# 数据集加载
dataset, info = tfds.load(
    "square_d0_300_demos",
    split="train",
    data_dir="/mnt/hdfs/zhufangqi/datasets/mimicgen/iclr2025/tensorflow_datasets",  # 指定 TFDS 数据集根目录
    with_info=True
)

output_dir = "videos_from_tfds"
os.makedirs(output_dir, exist_ok=True)

max_step = 0

for idx, example in enumerate(dataset):
    # 取 steps 中的所有帧
    steps = example["steps"]
    max_step = max(max_step, len(steps))
    # frames = []
    # for step in steps:
    #     img = step["observation"]["image"].numpy()  # HWC, uint8
    #     frames.append(img)
    #     print(step['action'][-1])
    # break

    # 保存为 mp4 视频
    # video_path = os.path.join(output_dir, f"episode_{idx}.mp4")
    # imageio.mimwrite(video_path, frames, fps=30)
    # print(f"已保存: {video_path}")
print(max_step)
print("所有视频已保存完成！")

# square 环境里面 -1是开 1是关
# openvla-oft训练时 -1被转化为1，1被转化为0
