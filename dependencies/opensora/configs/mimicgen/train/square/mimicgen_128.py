dataset = dict(
    type="SimpleVLAWebDataset",
    shards_pattern = "./datasets/simplevla_rl/mimicgen/iclr2025/square_128_demos/**/*.tar",
    stats_path = "./checkpoints/openvla-oft/aug/openvla-7b+square_d0_300_demos+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--debug--150000_chkpt/dataset_statistics.json",
    Ta = 8,
    To = 4,
    stride = 1,
    action_dim = 7,
    image_size = (256, 256),
    episode_buf_size = 20, # 20
    sample_buf_size = 500, # 1000
)

grad_checkpoint = False

# Acceleration settings
num_workers = 4 # 4
# num_bucket_build_workers =16
dtype = "bf16"
plugin = "zero2"

# Model settings
model = dict(
    type="STDiT3-XL/2",
    from_pretrained="./pretrained_models/hpcai-tech/OpenSora-STDiT-v3/model.safetensors",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    freeze_y_embedder=True,
    skip_y_embedder=True,
    cfg=False,
    training=True,
    action_dim=7,
    pad_action_num = 4
)
vae = dict(
    type="SDXL",
    from_pretrained="./pretrained_models/stabilityai/stable-diffusion-xl-base-1.0/vae",
    micro_frame_size=8,
    micro_batch_size=64,
)
# text_encoder = dict(
#     type="t5",
#     from_pretrained="DeepFloyd/t5-v1_1-xxl",
#     model_max_length=300,
#     shardformer=True,
# )
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
)

# Mask settings
mask_ratios = {
    "video_obs_4_frame": 1.0,
}

batch_size = 4
val_batch_size = 128

# Log settings
seed = 42
outputs = "./checkpoints/opensora/mimicgen/iclr2025/09/04/square_128_demos"
wandb = True
epochs = 10000
log_every = 10
ckpt_every = 10000

load = "./checkpoints/opensora/libero/08/14/openx_resume_380k/000-STDiT3-XL-2/epoch0-global_step1190000"
grad_clip = 1.0
lr = 1e-4
ema_decay = 0.99
adam_eps = 1e-15
warmup_steps = 1000

fps = 3
multi_resolution = "STDiT2"
cache_pin_memory = False
pin_memory_cache_pre_alloc_numels = [(290 + 20) * 1024**2] * (2 * 8 + 4)
