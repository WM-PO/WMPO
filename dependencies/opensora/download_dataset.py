from huggingface_hub import snapshot_download

# 数据集 repo 和目标目录
dataset_repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
target_dir = "stabilityai/stable-diffusion-xl-base-1.0"

# 下载数据集中特定分支的 .parquet 文件
snapshot_download(
    repo_id=dataset_repo_id,
    # repo_type="dataset",
    # revision="refs/convert/parquet",  # 指定分支或版本
    allow_patterns=["vae/*"],  # 仅允许下载 .parquet 文件
    local_dir=target_dir  # 将数据保存在目标目录中
)





