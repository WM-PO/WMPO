import os
import torch
import imageio
import numpy as np
import time
from mmengine.runner import set_random_seed
from opensora.datasets import save_sample
from opensora.datasets.datasets import resize_image
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import prepare_multi_resolution_info
from opensora.utils.misc import to_torch_dtype, create_logger


def main():
    torch.set_grad_enabled(False)

    # === Config & Setup ===
    cfg = parse_configs(training=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    set_random_seed(seed=cfg.get("seed", 1024))
    logger = create_logger()
    logger.info("Loaded config:\n%s", cfg.to_dict())

    # === Build Dataset & Model ===
    dataset = build_module(cfg.dataset, DATASETS)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    image_size = cfg.get("image_size", [128, 128])
    num_frames = cfg.get("num_frames", 16)
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    To = 2

    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
    ).to(device, dtype).eval()

    model_args = prepare_multi_resolution_info(
        cfg.get("multi_resolution", None), 1, image_size, num_frames, cfg.fps, device, dtype
    )

    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # === Load Data Sample ===
    # sample = dataset[0]
    for sample in dataset:
        obs = torch.tensor(sample["video"][:, 0:2]).unsqueeze(0).to(device, dtype)
        action = torch.tensor(sample["action"]).unsqueeze(0).to(device, dtype)


        start_time = time.time()
        # === Prepare Latents & Mask ===
        with torch.no_grad():
            x = vae.encode(obs)
            z = torch.randn(
                1, vae.out_channels, num_frames - To, *latent_size[1:], device=device, dtype=dtype
            )
            z = torch.cat([x, z], dim=2)
            masks = torch.tensor([[0] * To + [1] * (num_frames - To)], device=device, dtype=dtype)


            z = z.repeat(4, 1, 1, 1, 1)
            action = action.repeat(4, 1, 1)
            # === Sampling ===
            samples = scheduler.sample(
                model,
                z=z,
                y=action,
                device=device,
                additional_args=model_args,
                mask=masks,
            )

            # === Decode & Save ===
            pred_images = vae.decode(samples[:, :, -1:, :, :].to(dtype))
            pred_images = pred_images.cpu().to(torch.float32)
            pred_images = ((pred_images.permute(0, 2, 3, 4, 1).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

            # original_video = ((sample["video"].transpose(1,2,3,0) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            # diff_video = np.abs(original_video.astype(np.int16) - pred_images.astype(np.int16)).astype(np.uint8)
            # concat_video = np.concatenate((original_video, pred_images, diff_video), axis=2)
            # imageio.mimwrite(f"debug.mp4", concat_video, fps=1)

            # pred_images = vae.decode(samples[0:1, :, -1:, :, :].to(dtype))
            # pred_images = pred_images.squeeze().cpu().to(torch.float32)
            # pred_images = ((pred_images.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

            end_time = time.time()
            # 输出运行时间
        print(f"运行时间: {end_time - start_time:.4f} 秒")
        # imageio.imwrite('debug.png', pred_images)


if __name__ == "__main__":
    main()
