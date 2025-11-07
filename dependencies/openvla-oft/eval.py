#!/usr/bin/env python3
"""
Roll out OpenVLA on the LIBERO **square** task in your robomimic/mimicgen environment,
using only the agentview_image (no wrist image, no proprio), and report success rate.

Action post-processing follows your reference:
  1) clip gripper (last dim) to [0, 1]
  2) normalize gripper to [-1, +1], optionally binarize to {-1, +1}
  3) invert gripper sign
  4) env.step(inverted_action)

Defaults:
- env_config: /tmp/core_train_configs/bc_rnn_image_ds_square_D0_seed_101.json
- ckpt: /home/tiger/mg_ws/openvla-oft/Ckpts/openvla-oft/openvla-7b+square_d0_300_demos+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--debug--20000_chkpt
- runs: 10
- chunk: 8
- max_steps: 200
- task: "square"
"""
import random
import argparse
import json
import os
import re
from typing import Dict, Any
import numpy as np
import torch

from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_vla,
    get_vla_action,
)

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
import mimicgen.envs.robosuite  # noqa: F401

from PIL import Image


import os
# os['environ'] = 'PYTHONPATH=/opt/tiger/simplevla-rl:$PYTHONPATH'

# -----------------------------
# Env helpers
# -----------------------------

def _create_env(cfg):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=cfg.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=cfg.train.data,
        all_obs_keys=cfg.all_obs_keys,
        verbose=False,
    )
    if cfg.experiment.env is not None:
        env_meta["env_name"] = cfg.experiment.env
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=shape_meta["use_images"],
        use_depth_obs=shape_meta["use_depths"],
    )
    return EnvUtils.wrap_env_from_config(env, config=cfg)


def _pick_agent_image(
    obs: Dict[str, Any],
    crop_ratio: float = 0.0  # 中心裁剪比例
) -> np.ndarray:
    if "agentview_image" not in obs:
        raise KeyError("Observation does not contain 'agentview_image'")
    img = obs["agentview_image"]

    # [C,H,W] -> [H,W,C]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    # 归一化到 uint8
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)

    # 单通道扩展到三通道
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    # 中心裁剪
    if 0 < crop_ratio < 1:
        h, w, _ = img.shape
        new_h = int(h * crop_ratio)
        new_w = int(w * crop_ratio)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        cropped = img[top:top+new_h, left:left+new_w]

        # resize 回原大小
        img = np.array(Image.fromarray(cropped).resize((w, h), Image.BICUBIC))

    return img


# -----------------------------
# Gripper post-processing (from your reference)
# -----------------------------

def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """Normalize gripper action from [0,1] to [-1,+1] on the last dim, optional binarization."""
    normalized_action = action.copy()
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])
    return normalized_action


def unnormalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Unnormalize gripper action from [-1,+1] back to [0,1] on the last dim, optional binarization.
    """
    unnormalized_action = action.copy()
    orig_low, orig_high = 0.0, 1.0
    # 线性映射 [-1,1] -> [0,1]
    unnormalized_action[..., -1] = (unnormalized_action[..., -1] + 1) * (orig_high - orig_low) / 2 + orig_low
    if binarize:
        # binarize 到 {0,1}
        unnormalized_action[..., -1] = (unnormalized_action[..., -1] >= 0.5).astype(float)
    return unnormalized_action

def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Flip the sign of the gripper action (last dimension)."""
    inverted_action = action.copy()
    inverted_action[..., -1] = inverted_action[..., -1] * -1.0
    return inverted_action


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    # coffee stack_three three_piece_assembly square
    parser.add_argument("--task", type=str, default="three_piece_assembly") # square stack_three  
    # parser.add_argument("--ckpt", type=str, default="/mnt/hdfs/zhufangqi/checkpoints/openvla-oft/aug/openvla-7b+square_d0_300_demos+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--debug--150000_chkpt")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_video", default=True)
    parser.add_argument("--video_dir", type=str, default="./debug/dpo")
    args = parser.parse_args()
    args.env_config = f"/mnt/hdfs/zhufangqi/datasets/mimicgen/core_train_configs/bc_rnn_image_ds_{args.task}_D0_seed_101.json"
    if args.task == 'square':
        args.task_description = "Insert the square into the stick"
        # SFT_MODEL_PATH="/mnt/hdfs/zhufangqi/checkpoints/SimpleVLA-RL-mimicgen/square_grpo_b64/actor/global_step_1"
        # args.ckpt="/mnt/hdfs/zhufangqi/checkpoints/openvla-oft/aug/openvla-7b+square_d0_300_demos+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--debug--150000_chkpt"
        # args.ckpt="/mnt/hdfs/zhufangqi/checkpoints/SimpleVLA-RL-mimicgen/square_wm_128_grpo_b64/actor/global_step_67"
        args.ckpt="/mnt/hdfs/zhufangqi/checkpoints/SimpleVLA-RL-mimicgen/square_dpo_128/actor/global_step_10"
        
    else:
        args.task_description = args.task
        args.ckpt = "/mnt/hdfs/zhufangqi/checkpoints/SimpleVLA-RL-mimicgen/three_piece_assembly_wm_128_grpo_b64/actor/global_step_32"
        # args.ckpt = f"/mnt/hdfs/zhufangqi/checkpoints/openvla-oft/iclr2025/{args.task}_d0_300_demos/openvla-7b+{args.task}_d0_300_demos+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--{args.task}_d0_300_demos--150000_chkpt"
    
    args.task_description = args.task

    if args.task == "coffee":
        args.max_steps = 256
    elif args.task == "stack_three":
        args.max_steps = 320
    elif args.task == "three_piece_assembly":
        args.max_steps = 384
    elif args.task == "square":
        args.max_steps = 184

    # --- Load env config (robomimic) ---
    ext_cfg = json.load(open(args.env_config, "r"))
    cfg = config_factory(ext_cfg["algo_name"])
    with cfg.values_unlocked():
        cfg.update(ext_cfg)
    cfg.lock()

    ObsUtils.initialize_obs_utils_with_config(cfg)


    # --- VLA stack ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    from types import SimpleNamespace
    cfg_ns = SimpleNamespace(
        pretrained_checkpoint=args.ckpt,
        center_crop=True,
        num_images_in_input=1,
        use_proprio=False,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        load_in_8bit=False,
        load_in_4bit=False,
        num_open_loop_steps=args.chunk,
        unnorm_key=f"{args.task}_d0_300_demos",
    )

    print(f"[INFO] Using checkpoint: {args.ckpt}")
    print(f"[INFO] Building VLA from: {cfg_ns.pretrained_checkpoint}")

    vla = get_vla(cfg_ns).to(device).eval()
    processor = get_processor(cfg_ns)
    action_head = None # get_action_head(cfg_ns, llm_dim=vla.llm_dim).to(device).eval()

    # --- Rollout episodes ---
    successes = 0
    lengths = []
    os.makedirs(args.video_dir, exist_ok=True)

    states = []
    first_images = []

    import pickle
    # pos
    with open(f'/opt/tiger/simplevla-rl/verl/utils/dataset/{args.task}_d0_states_pos.pkl', 'rb') as f:
        reset_states = pickle.load(f)
    # modified_states = []
    # for state in reset_states:
    #     xml_string = state['model']
    #     m = re.search(r'<body name="peg1"[^>]*>', xml_string)
    #     block = m.group(0) if m else None
    #     y_range = random.uniform(0.06, 0.1)
    #     new_block = block.replace('0.1', str(y_range)) 
    #     new_xml_string = xml_string.replace(block, new_block)
    #     state['model'] = new_xml_string
    #     modified_states.append(state)

    # with open(f'/opt/tiger/simplevla-rl/verl/utils/dataset/{args.task}_d0_states_pos.pkl', 'wb') as f:
    #     pickle.dump(modified_states, f)

    # modified_states = []
    # for state in reset_states:
    #     old_str = '<geom name="table_visual" size="0.4 0.4 0.025" type="box" contype="0" conaffinity="0" group="1" material="table_ceramic"/>'
    #     new_str = '<geom name="table_visual" size="0.4 0.4 0.025" type="box" contype="0" conaffinity="0" group="1" material="table_mat"/>'
    #     state['model'] = state['model'].replace(old_str, new_str)
    #     modified_states.append(state)

    # with open(f'/opt/tiger/simplevla-rl/verl/utils/dataset/{args.task}_d0_states_pos.pkl', 'wb') as f:
    #     pickle.dump(modified_states, f)

    import xml.etree.ElementTree as ET

    # def add_material_look(xml_str: str) -> str:
    #     root = ET.fromstring(xml_str)

    #     # 1) 让材质显示木纹：别用纯黑，改成白色/深色调；并稍微增强高光与平铺
    #     for m in root.findall('.//asset/material'):
    #         if m.get('name') in ('piece_1_redwood_mat', 'piece_2_redwood_mat'):
    #             m.set('rgba', '0.22 0.18 0.16 1')   # 深木色调（想更亮就改成 1 1 1 1）
    #             m.set('specular', '0.35')
    #             m.set('shininess', '0.12')
    #             m.set('texuniform', 'true')
    #             m.set('texrepeat', '2 2')          # 纹理更细腻

    #     # 2) 隐藏碰撞几何（非 _vis），避免把纹理“压黑”
    #     for g in root.findall('.//worldbody//geom'):
    #         n = g.get('name', '')
    #         if (n.startswith('piece_1_') or n.startswith('piece_2_')) and not n.endswith('_vis'):
    #             g.set('rgba', '0 0 0 0')           # 透明但仍参与碰撞

    #     return ET.tostring(root, encoding='unicode')

    def add_material_look(xml_str: str) -> str:
        root = ET.fromstring(xml_str)

        # 1) 让材质显示浅绿色木纹
        # for m in root.findall('.//asset/material'):
        #     if m.get('name') in ('base_redwood_mat'):
        #         m.set('rgba', '0.3 0.8 0.3 1')   # 浅绿色
        #         m.set('specular', '0.35')
        #         m.set('shininess', '0.12')
        #         m.set('texuniform', 'true')
        #         m.set('texrepeat', '2 2')

        # 1）改成哑光材质
        for m in root.findall('.//asset/material'):
            if m.get('name') in ('base_redwood_mat',):  # 可以添加更多材质名称
                # 设置基础颜色为浅绿色
                m.set('rgba', '0.3 0.8 0.3 1')
                # m.set('rgba', '0.3 0.1 0.3 1')
        
                # 降低高光反射，创建哑光效果
                m.set('specular', '0.1')  # 降低高光强度（0-1范围，值越小越哑光）
                m.set('shininess', '0.05')  # 降低光泽度（值越小表面越粗糙）
        
                # 纹理设置
                m.set('texuniform', 'true')
                m.set('texrepeat', '2 2')
        
                # 可选：调整反射率以增强哑光效果
                m.set('reflectance', '0.1')  # 如果MuJoCo支持此属性

        # 2) 隐藏碰撞几何（非 _vis），避免压黑纹理
        for g in root.findall('.//worldbody//geom'):
            n = g.get('name', '')
            if n.startswith('base_') and not n.endswith('_vis'):
                g.set('rgba', '0 0 0 0')

        return ET.tostring(root, encoding='unicode')

    modified_states = []
    for state in reset_states:
        state['model'] = add_material_look(state['model'])
        modified_states.append(state)

    # with open(f'/opt/tiger/simplevla-rl/verl/utils/dataset/{args.task}_d0_states_pos.pkl', 'wb') as f:
    #     pickle.dump(modified_states, f)
    
    for ep in range(args.runs):
        env = _create_env(cfg)
        # reset_states[ep]['model'] = add_material_look(reset_states[ep]['model'])
        obs = env.reset_to(reset_states[ep])
        # obs = env.reset()
        # states.append(env.get_state())
        for _ in range(10):
            obs, reward, done, info = env.step(np.zeros(7))
        first_image = (obs['agentview_image']* 255.0).clip(0, 255).astype(np.uint8).transpose(1,2,0)
        # import imageio
        # imageio.imwrite(f"debug.png", first_image)

        first_images.append(first_image)
        frames = []
        ep_len = 0
        success = False

        while ep_len < args.max_steps:
            # print(ep, ep_len)
            # 1) Build observation from env
            full_img = _pick_agent_image(obs)
            policy_obs = {"full_image": full_img, "task_description": args.task_description}

            # 2) Get an action chunk from the policy
            import time
            start = time.time()
            actions = get_vla_action(cfg_ns, vla, processor, policy_obs, policy_obs["task_description"], action_head, None)
            # print(f"VLA time: {time.time() - start}")
            actions = np.asarray(actions)
            if actions.ndim == 1:
                actions = actions.reshape(1, -1)
            if actions.shape[1] != 7:
                raise RuntimeError(f"Expected action dim 7, got {actions.shape}")

            # 3) Pad/trim to exactly chunk
            if actions.shape[0] > args.chunk:
                actions = actions[: args.chunk]
            elif actions.shape[0] < args.chunk:
                pad = np.zeros((args.chunk - actions.shape[0], 7), dtype=actions.dtype)
                actions = np.concatenate([actions, pad], axis=0)

            # 4) Gripper post-processing per your reference


            # 5) Step the env
            for action in actions:
                # normalized_action = action
                # print(action[-1])
                # inverted_action = invert_gripper_action(action)
                # normalized_action = unnormalize_gripper_action(inverted_action, binarize=True)
                # action[-1] = int((action[-1])>=0)
                # print(action[-1])
                # inverted_action = normalized_action
                # 
                if args.save_video:
                    frames.append(_pick_agent_image(obs))
                import time
                start = time.time()
                obs, reward, done, info = env.step(action)
                # print(f"Step time: {time.time() - start}")
                if reward > 0.0 or done:
                    if done:
                        assert False, "Done but not success"
                    success = True
                # print(f"Done {done}, reward {reward}")
                ep_len += 1
                # if info.get("success", False):
                #     success = True
                if done or ep_len >= args.max_steps:
                    break

            if done or ep_len >= args.max_steps:
                break


        if args.save_video and len(frames) > 0:
            frames.append(_pick_agent_image(obs))
            import imageio
            out_path = os.path.join(args.video_dir, f"{args.task}_rollout_{ep+1:02d}_{success}.mp4")
            imageio.mimsave(out_path, frames, fps=20)
            print(f"Saved video -> {out_path}")

        lengths.append(ep_len)
        successes += int(success)
        print(f"[Episode {ep+1:02d}] len={ep_len}, success={success}")
        # assert False

    # import pickle
    # with open('/opt/tiger/simplevla-rl/verl/utils/dataset/square_d0_states.pkl', 'rb') as f:
    #     square_states = pickle.load(f)

    # os.makedirs(f"./debug/first_images/{args.task}", exist_ok=True)
    # for i in range(len(first_images)):
    #     imageio.imwrite(f"./debug/first_images/{args.task}/{i}.png", first_images[i])
    # import pickle
    # with open(f"/opt/tiger/simplevla-rl/verl/utils/dataset/{args.task}_d0_states.pkl", "wb") as f:
    #     pickle.dump(states, f)
    # os.makedirs(f"/mnt/hdfs/zhufangqi/datasets/mimicgen/iclr2025/first_images/{args.task}", exist_ok=True)
    # for i in range(len(first_images)):
    #     imageio.imwrite(f"/mnt/hdfs/zhufangqi/datasets/mimicgen/iclr2025/first_images/{args.task}/{i}.png", first_images[i])

    success_rate = successes / max(1, args.runs)
    print("\n=============================")
    print(f"Episodes: {args.runs}")
    print(f"Successes: {successes}")
    print(f"Success Rate: {success_rate*100:.1f}%")
    print(f"Avg Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")

    out_json = os.path.join(args.video_dir, f"openvla_{args.task}_eval.json")
    with open(out_json, "w") as f:
        json.dump({
            "episodes": args.runs,
            "successes": successes,
            "success_rate": success_rate,
            "avg_len": float(np.mean(lengths)) if lengths else 0.0,
            "std_len": float(np.std(lengths)) if lengths else 0.0,
            "chunk": args.chunk,
            "max_steps": args.max_steps,
            "ckpt": args.ckpt,
            "env_config": args.env_config,
            "task": args.task,
        }, f, indent=2)
    print(f"Saved summary -> {out_json}")
    env.env.close()

if __name__ == "__main__":
    main()

 # cp dataset_statistics.json /mnt/hdfs/zhufangqi/checkpoints/openvla-oft/lora_1000_demos/openvla-7b+square_d0_1000_demos+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--lora--150000_chkpt