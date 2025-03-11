
import os
import argparse
from huggingface_hub import upload_folder, create_repo

def push_exp_to_hub(exp_name: str, repo_id: str, repo_type: str = "model", token: str = None):
    """
    Push an experiment directory to a subdirectory in a Hugging Face Hub repository.

    Args:
        exp_name (str): Name of the experiment.
        repo_id (str): Hugging Face repo name (e.g., "username/repo-name").
        repo_type (str): Type of repo ("model", "dataset", "space"). Default is "model".
        token (str, optional): Hugging Face authentication token. If None, uses the cached login.

    Returns:
        str: URL of the uploaded repository.
    """
    # Determine paths based on exp_name
    exp_dir = os.path.join("/exps", exp_name)  # Local directory: /exps/{exp_name}
    exp_dir = exp_name
    # subdir = f"experiments/{exp_name}"  # Remote subdirectory in the repo
    subdir = 'checkpoints'

    if not os.path.exists(exp_dir):
        raise ValueError(f"Experiment directory '{exp_dir}' does not exist.")

    create_repo(repo_id, repo_type=repo_type, exist_ok=True)

    print(f"🚀 Uploading {exp_dir} to {repo_id}/{subdir} on Hugging Face Hub...")

    upload_folder(
        folder_path=exp_dir,
        repo_id=repo_id,
        # path_in_repo=subdir,  # Upload to this subdirectory
        repo_type=repo_type,
        token=token
    )

    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"✅ Upload complete: {repo_url}")
    
    return repo_url

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push an experiment directory to a subdirectory in Hugging Face Hub.")
    parser.add_argument("--exp-name", type=str, required=True, help="Name of the experiment (determines both exp_dir and subdir).")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo name (e.g., 'username/repo-name').")
    parser.add_argument("--repo-type", type=str, choices=["model", "dataset", "space"], default="model", help="Type of repo (default: model).")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (optional).")

    args = parser.parse_args()

    push_exp_to_hub(args.exp_name, args.repo_id, args.repo_type, args.token)


# import os
# import argparse
# from huggingface_hub import upload_folder, create_repo

# def push_exp_to_hub(exp_dir: str, repo_id: str, repo_type: str = "model", token: str = None):
#     """
#     Push an experiment directory to Hugging Face Hub.

#     Args:
#         exp_dir (str): Path to the experiment directory.
#         repo_id (str): Hugging Face repo name (e.g., "username/repo-name").
#         repo_type (str): Type of repo ("model", "dataset", "space"). Default is "model".
#         token (str, optional): Hugging Face authentication token. If None, uses the cached login.

#     Returns:
#         str: URL of the uploaded repository.
#     """
#     if not os.path.exists(exp_dir):
#         raise ValueError(f"Experiment directory '{exp_dir}' does not exist.")
    
#     create_repo(repo_id, repo_type=repo_type, exist_ok=True)

#     print(f"🚀 Uploading {exp_dir} to {repo_id} on Hugging Face Hub...")

#     upload_folder(
#         folder_path=exp_dir,
#         repo_id=repo_id,
#         repo_type=repo_type,
#         token=token
#     )

#     repo_url = f"https://huggingface.co/{repo_id}"
#     print(f"✅ Upload complete: {repo_url}")
    
#     return repo_url

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Push an experiment directory to Hugging Face Hub.")
#     parser.add_argument("--exp-dir", type=str, required=True, help="Path to the experiment directory.")
#     parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo name (e.g., 'username/repo-name').")
#     parser.add_argument("--repo-type", type=str, choices=["model", "dataset", "space"], default="model", help="Type of repo (default: model).")
#     parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (optional).")

#     args = parser.parse_args()

#     push_exp_to_hub(args.exp_dir, args.repo_id, args.repo_type, args.token)

# # python push_exp_to_hub.py --exp_dir "/exps/my_experiment" --repo_id "your-username/my-exp-repo" --repo_type "dataset"


# # python push_exp_to_hub.py \
# #     --exp_dir "/exps/my_experiment" \
# #     --repo_id "your-username/my-exp-repo" \
# #     --repo_type "dataset"  # Can be "model", "dataset", or "space"
