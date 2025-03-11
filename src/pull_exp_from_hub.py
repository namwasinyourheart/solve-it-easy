import os
import argparse
from huggingface_hub import snapshot_download

def has_experiment_directory(exp_name, base_dir):
    """Checks if an experiment directory exists in the specified base directory."""
    exp_path = os.path.join(base_dir, exp_name)
    return os.path.isdir(exp_path)

def pull_exp_from_hub(repo_id: str, base_dir: str, exp_name: str, repo_type: str = "dataset"):
    """Pulls an experiment directory from the Hugging Face Hub if it doesn't already exist."""
    
    local_dir = os.path.join(base_dir, exp_name)
    
    if has_experiment_directory(exp_name, base_dir):
        print(f"✅ Experiment directory '{local_dir}' already exists. Skipping download.")
        return
    
    snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir)
    print(f"✅ Successfully pulled {repo_id} to {local_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Hugging Face repository ID")
    parser.add_argument("--base-dir", required=True, help="Base directory to save the experiment", default='/exps')
    parser.add_argument("--exp-name", required=True, help="Name of the experiment directory")
    parser.add_argument("--repo-type", default="model", help="Repository type: model, dataset, or space")
    args = parser.parse_args()

    pull_exp_from_hub(args.repo_id, args.base_dir, args.exp_name, args.repo_type)



# import argparse
# from huggingface_hub import snapshot_download

# def pull_exp_from_hub(repo_id: str, local_dir: str, repo_type: str = "dataset"):
#     """Pulls an experiment directory from the Hugging Face Hub."""
#     snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir)
#     print(f"✅ Successfully pulled {repo_id} to {local_dir}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--repo-id", required=True, help="Hugging Face repository ID")
#     parser.add_argument("--local-dir", required=True, help="Local directory to save the experiment")
#     parser.add_argument("--repo-type", default="dataset", help="Repository type: model, dataset, or space")
#     args = parser.parse_args()

#     pull_exp_from_hub(args.repo_id, args.local_dir, args.repo_type)

# # python pull_exp_from_hub.py --repo_id your-username/your-exp-repo --local_dir /path/to/save --repo_type dataset
