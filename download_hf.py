from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="fangqi/WMPO",
    repo_type="model",
    local_dir=".",
    local_dir_use_symlinks=False,
    allow_patterns=["checkpoint_files/**"]
)

snapshot_download(
    repo_id="fangqi/WMPO",
    repo_type="model",
    local_dir=".",
    local_dir_use_symlinks=False,
    allow_patterns=["data_files/**"]
)