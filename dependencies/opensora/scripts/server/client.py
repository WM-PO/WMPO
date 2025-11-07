import socket
import pickle
import torch
import time
import numpy as np
from opensora.utils.config_utils import read_config
from opensora.registry import build_module, DATASETS

CONFIG_PATH = "configs/realpusht/inference/sample_realpusht_A100.py"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 23456
BATCH_SIZE = 16

def send_to_server(obs_batch, action_batch):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((SERVER_HOST, SERVER_PORT))

    data = {'obs': obs_batch, 'action': action_batch}
    client.sendall(pickle.dumps(data))
    client.shutdown(socket.SHUT_WR)  # <<=== 添加这行！


    # 接收结果
    recv_data = b""
    while True:
        chunk = client.recv(4096)
        if not chunk:
            break
        recv_data += chunk

    client.close()
    result = pickle.loads(recv_data)
    return result

def main():
    cfg = read_config(CONFIG_PATH)
    dataset = build_module(cfg.dataset, DATASETS)
    dtype = torch.bfloat16 if cfg.get("dtype", "bf16") == "bf16" else torch.float32

    obs_batch = []
    action_batch = []

    count = 0
    for sample in dataset:
        obs = torch.tensor(sample["video"][:, 0:2], dtype=dtype)  # shape: [T, 2]
        act = torch.tensor(sample["action"], dtype=dtype)         # shape: [A]

        obs_batch.append(obs)
        action_batch.append(act)

        count += 1
        if count == BATCH_SIZE:
            # === 发送并接收结果 ===
            start_time = time.time()
            result = send_to_server(obs_batch, action_batch)

            # === 保存或打印结果 ===
            print(f"Received {len(result)} predicted frames")
            # for i, img in enumerate(result):
            #     print(f"Sample {i}: shape = {np.array(img).shape}")
            end_time = time.time()
            print(f"Time taken for {BATCH_SIZE} samples: {end_time - start_time} seconds")
            # 清空当前 batch
            obs_batch.clear()
            action_batch.clear()
            count = 0

if __name__ == "__main__":
    main()
