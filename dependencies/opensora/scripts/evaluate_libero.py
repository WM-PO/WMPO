import os
import os
import time
import json
import imageio
from pprint import pformat

import numpy as np
import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from mmengine.runner import set_random_seed
from tqdm import tqdm
from collections import deque
from PIL import Image
from transformers import AutoConfig, AutoTokenizer
import torchvision.transforms as transforms
import tensorflow as tf

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.datasets.action_tokenizer import ActionTokenizer
from opensora.datasets.datasets import resize_image
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype


def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    # text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        .to(device, dtype)
        .eval()
    )
    # text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # inference
    # ======================================================

    # == prepare arguments ==
    fps = cfg.fps
    multi_resolution = cfg.get("multi_resolution", None)
    condition_frame_length = cfg.get("condition_frame_length", 5)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    model_args = prepare_multi_resolution_info(
        multi_resolution, 1, image_size, num_frames, fps, device, dtype
    )
    task_name_list = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    
    for task_id in range(10):
        for task_name in task_name_list:
            data_path = os.path.join(cfg.data_path, task_name)
            task_name = f"{task_name}_no_noops"

            with open('opensora/datasets/libero_dataset_statistics.json', 'r') as f:
                dataset_statistics = json.load(f)
            q01 = np.array(dataset_statistics[task_name]['action']['min'])
            q99 = np.array(dataset_statistics[task_name]['action']['max'])
            ckpt_name = "/".join(cfg.model.from_pretrained.split('/')[9:12])
            replay_videos_dir = f"./video_results/{ckpt_name}/{task_name}"
            for episode_id in range(3):
                actions = np.load(os.path.join(data_path, str(task_id), "actions", f"{episode_id}.npy")) 

                actions = actions[:-1]

                norm_actions = (2 * ((actions-q01) / (q99-q01))) - 1

                # disc_actions = action_tokenizer.convert_actions_to_tokens(norm_actions)

                disc_actions = torch.from_numpy(norm_actions)

                image_queue = deque(maxlen=condition_frame_length)

                start_idx = 0
                disc_actions = disc_actions[start_idx+condition_frame_length-1:]

                for i in range(condition_frame_length):
                    idx = max(0, start_idx - condition_frame_length + i)
                    image_path = os.path.join(data_path, str(task_id), "images", f"{episode_id}/{idx}.png")
                    image = Image.open(image_path).convert("RGB")
                    # image = tf.convert_to_tensor(image)
                    # image = resize_image(image, image_size)  # 调整大小
                    image = np.array(image)
                    transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                        ]
                    )
                
                    processed_image = transform(image)  # 应用 Transform

                    
                    processed_image = processed_image.unsqueeze(1).unsqueeze(0).to(device).to(dtype) #  CTHW

                    x = vae.encode(processed_image)
                
                    image_queue.append(x)

                torch.manual_seed(1024)

                # original_video = [os.path.join(data_path, f"images/{episode_id:06d}/{file_name}") for file_name in sorted(os.listdir(os.path.join(data_path, f"images/{episode_id:06d}")))]
                image_files = os.listdir(os.path.join(data_path, f"{task_id}/images/{episode_id}"))
                sorted_images = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))
                image_paths = [os.path.join(data_path, f"{task_id}/images/{episode_id}", file_name) for file_name in sorted_images]
                original_video = np.stack([np.array(Image.open(image_path).convert("RGB")) for image_path in image_paths], axis=0)
                original_video = original_video[start_idx:]

                video = [original_video[0:condition_frame_length]]

                action_chunk_length = num_frames - condition_frame_length

                for i in tqdm(range(0, len(disc_actions), action_chunk_length), total=len(disc_actions)//action_chunk_length):
                    action = disc_actions[i:i+action_chunk_length]

                    if len(action) < action_chunk_length:
                        continue
                    # 处理当前的batch

                    y = action.unsqueeze(0).to(device).to(dtype)

                    y = torch.concat([torch.zeros((1, condition_frame_length-1, 7)).to(device).to(dtype), y], dim=1)
                    mask_images = torch.concat(list(image_queue), dim=2)

                    z = torch.randn(1, vae.out_channels, action_chunk_length, *latent_size[1:], device=device, dtype=dtype)
                    masks = torch.tensor([[0]*condition_frame_length+[1]*(action_chunk_length)], device=device, dtype=dtype) # torch.float32
                    z = torch.concat([mask_images, z], dim=2)

                    samples = scheduler.sample(
                        model,
                        z=z,
                        y=y,
                        device=device,
                        additional_args=model_args,
                        progress=verbose >= 2,
                        mask=masks,
                    )
                    pred_images = samples[:, :, -action_chunk_length:, :, :].to(dtype)
                    image_queue.extend(pred_images.clone().chunk(action_chunk_length, dim=2))
                    pred_images = vae.decode(pred_images)
                    pred_images = pred_images.squeeze().cpu().to(torch.float32)
                    pred_images = (
                        (pred_images.permute(1, 2, 3, 0).numpy() * 0.5 + 0.5) * 255 
                    ).clip(0, 255).astype(np.uint8) # 

                    video.append(pred_images.copy())
                    # Image.fromarray(pred_image, mode='RGB').save("./pred_image.png")
                video_np = np.concatenate(video, axis=0) # THWC
                concat_video = np.concatenate((original_video[:video_np.shape[0]], video_np), axis=2)
                os.makedirs(f'{replay_videos_dir}/{task_id}', exist_ok=True)
                imageio.mimwrite(f'{replay_videos_dir}/{task_id}/{episode_id:06d}.mp4', concat_video, fps=30)
            logger.info("Inference finished.")
            logger.info("Saved %s samples to %s", start_idx, save_dir)


if __name__ == "__main__":
    main()
