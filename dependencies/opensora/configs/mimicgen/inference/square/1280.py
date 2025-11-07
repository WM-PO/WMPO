# dataset = dict(
#     type="MixedSimpleVLAWebDataset",
#     shards_pattern1 = "./datasets/simplevla_rl/webdataset_shards/06/17_mp/**/*.tar",
#     shards_pattern2 = "./datasets/simplevla_rl/webdataset_shards/libero/libero_10_no_noops/**/*.tar",
#     weights = [0.7, 0.3],
#     stats_path = "./datasets/libero/dataset_statistics.json",
#     Ta = 8,
#     To = 4,
#     stride = 1,
#     action_dim = 7,
#     image_size = (256, 256),
#     episode_per_shard = 30,
#     episode_buf_size = 20, # 20
#     sample_buf_size = 1000, # 2000
# )

image_size = (256,256)
num_frames = 12
fps = 3
# data_path = "./datasets/simplevla_rl/valid_batch"
save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 4
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    action_dim=7,
    pad_action_num = 4,
)
vae = dict(
    type="SDXL",
    from_pretrained="./pretrained_models/stabilityai/stable-diffusion-xl-base-1.0/vae",
    micro_frame_size=8,
    micro_batch_size=64,
)

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

