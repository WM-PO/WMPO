from huggingface_hub import create_repo, upload_folder, upload_large_folder

repo_id = "fangqi/WMPO"
folder_path = "./checkpoint_files"
path_in_repo = "./checkpoint_files"

create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

upload_folder(
    repo_id=repo_id,
    folder_path=folder_path,
    path_in_repo=path_in_repo,
    repo_type="model",
)


repo_id = "fangqi/WMPO"
folder_path = "./data_files"
path_in_repo = "./data_files"

create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

upload_folder(
    repo_id=repo_id,
    folder_path=folder_path,
    path_in_repo=path_in_repo,
    repo_type="model",
)