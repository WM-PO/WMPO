import os
import torch
import torch.multiprocessing as mp
from multiprocessing import Queue
import socket
import pickle
import numpy as np

from opensora.registry import build_module, MODELS, SCHEDULERS, DATASETS
from opensora.utils.config_utils import parse_configs, read_config
from opensora.utils.inference_utils import prepare_multi_resolution_info
from opensora.utils.misc import to_torch_dtype

NUM_GPUS = 8
CONFIG_PATH = "configs/realpusht/inference/sample_realpusht_A100.py"

def gpu_worker(rank, task_queue, result_queue):
    import torch.backends.cudnn as cudnn
    from mmengine.runner import set_random_seed
    torch.cuda.set_device(rank)
    # === 方法 1: 设置 deterministic 模式，避免非确定性算法导致的 CUDA 报错 ===
    cudnn.deterministic = True
    cudnn.benchmark = False

    # === 方法 3: 设置固定随机种子（每个进程不同） ===
    set_random_seed(42 + rank)

    # === 加载配置和模型 ===
    cfg = read_config(CONFIG_PATH)
    device = f"cuda:{rank}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

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

    print(f"GPU {rank} 就绪")

    while True:
        obs_list, act_list = task_queue.get()
        batch_size = len(obs_list)

        obs = torch.stack([o[:, 0:2].clone().detach() for o in obs_list]).to(device, dtype)
        action = torch.stack([a.clone().detach() for a in act_list]).to(device, dtype)

        with torch.no_grad():
            x = vae.encode(obs)  # shape: [B, C, To, H, W]
            z = torch.randn(
                batch_size, vae.out_channels, num_frames - To, *latent_size[1:],
                device=device, dtype=dtype
            )
            z = torch.cat([x, z], dim=2)
            masks = torch.tensor([[0] * To + [1] * (num_frames - To)] * batch_size, device=device, dtype=dtype)

            samples = scheduler.sample(
                model, z=z, y=action, device=device,
                additional_args=model_args, mask=masks,
            )

            pred_images = vae.decode(samples[:, :, -1:, :, :].to(dtype))
            pred_images = pred_images.cpu().to(torch.float32)
            pred_images = ((pred_images.permute(0, 2, 3, 4, 1).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

            result_queue.put(list(pred_images))


def run_server():
    mp.set_start_method('spawn', force=True)
    task_queue = Queue()
    result_queue = Queue()

    # 启动每个 GPU 的推理进程
    processes = []
    for rank in range(NUM_GPUS):
        p = mp.Process(target=worker_entry, args=(rank, task_queue, result_queue))
        p.start()
        processes.append(p)

    # 监听 socket 请求（主进程接收 obs/action）
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 23456))
    server.listen(5)

    print("主进程监听中...")

    while True:
        client, addr = server.accept()
        print(f"来自 {addr} 的连接")

        data = b""
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            data += chunk

        inputs = pickle.loads(data)
        obs_batch, action_batch = inputs['obs'], inputs['action']  # shape: [N, ...]

        # 分发任务
        num_tasks = len(obs_batch)
        tasks_per_gpu = num_tasks // NUM_GPUS
        results = []

        for i in range(NUM_GPUS):
            start = i * tasks_per_gpu
            end = (i + 1) * tasks_per_gpu if i < NUM_GPUS - 1 else num_tasks
            task_queue.put((obs_batch[start:end], action_batch[start:end]))

        # 收集结果
        for _ in range(NUM_GPUS):
            results.extend(result_queue.get())

        client.sendall(pickle.dumps(results))
        client.shutdown(socket.SHUT_RDWR)
        client.close()

def worker_entry(rank, task_queue, result_queue):
    gpu_worker(rank, task_queue, result_queue)

if __name__ == "__main__":
    run_server()
