import json
import pickle
from robomimic.config import config_factory
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import mimicgen.envs.robosuite  # noqa: F401
import numpy as np
import imageio
import os

def sample_env_states(env_config_path):
    # 加载环境配置
    ext_cfg = json.load(open(env_config_path, "r"))
    cfg = config_factory(ext_cfg["algo_name"])
    with cfg.values_unlocked():
        cfg.update(ext_cfg)
    cfg.lock()

    # 初始化 observation 工具
    ObsUtils.initialize_obs_utils_with_config(cfg)

    # 构建环境
    env_meta = FileUtils.get_env_metadata_from_dataset(cfg.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=cfg.train.data,
        all_obs_keys=cfg.all_obs_keys,
        verbose=True,
    )
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=shape_meta["use_images"],
        use_depth_obs=shape_meta["use_depths"],
    )
    env = EnvUtils.wrap_env_from_config(env, config=cfg)
    
    with open('/opt/tiger/simplevla-rl/verl/utils/dataset/square_d0_states.pkl', 'rb') as f:
        states = pickle.load(f)
    os.makedirs('./mimicgen_first_images', exist_ok=True)
    for idx, state in enumerate(states):
        env.reset_to(state)
        dumpy_action = np.zeros(7)
        for i in range(20):
            obs = env.step(dumpy_action)
        imageio.imwrite(f'./mimicgen_first_images/{idx}.png', (obs[0]['agentview_image'].transpose(1,2,0) * 255.0).astype(np.uint8))


if __name__ == "__main__":
    import time
    start = time.time()
    sample_env_states(
        "/mnt/hdfs/zhufangqi/datasets/mimicgen/core_train_configs/bc_rnn_image_ds_square_D0_seed_101.json"
    )
    end = time.time()
    print("time cost:", end-start)
