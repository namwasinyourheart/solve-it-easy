from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

from utils.exp_utils import setup_environment

setup_environment()

def push_to_hub(repo_name: str, trained_model_path: str):
    """
    Pushes a fine-tuned model and its tokenizer to the Hugging Face Hub.

    Args:
        repo_name (str): The name of the Hugging Face Hub repository.
        trained_model_path (str): The local path to the fine-tuned model directory.
    """
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
        model = AutoModelForCausalLM.from_pretrained(trained_model_path, trust_remote_code=True)

        # Push to Hugging Face Hub
        tokenizer.push_to_hub(repo_name)
        model.push_to_hub(repo_name)
        print(f"✓ Successfully pushed model and tokenizer to {repo_name}")

    except Exception as e:
        print(f"Error pushing to hub: {e}")

def parse_args():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments containing repo_name and model_path.
    """
    parser = argparse.ArgumentParser(description="Push a fine-tuned model to the Hugging Face Hub.")
    parser.add_argument("--repo_name", type=str, help="Hugging Face repository name.")
    parser.add_argument("--trained_model_path", type=str, help="Path to the fine-tuned model directory.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    push_to_hub(args.repo_name, args.trained_model_path)
