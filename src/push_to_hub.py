from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

from utils.exp_utils import setup_environment

setup_environment()

def push_to_hub(repo_id: str, trained_model_path: str, push_model: bool = False, push_tokenizer: bool = False):
    """
    Pushes a fine-tuned model and/or its tokenizer to the Hugging Face Hub.

    Args:
        repo_id (str): The id of the Hugging Face Hub repository.
        trained_model_path (str): The local path to the fine-tuned model directory.
        push_model (bool): Whether to push the model. Default is True.
        push_tokenizer (bool): Whether to push the tokenizer. Default is True.
    """
    try:
        if push_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
            tokenizer.push_to_hub(repo_id)
            print(f"✓ Successfully pushed tokenizer to {repo_id}")

        if push_model:
            model = AutoModelForCausalLM.from_pretrained(trained_model_path, trust_remote_code=True)
            model.push_to_hub(repo_id)
            print(f"✓ Successfully pushed model to {repo_id}")

    except Exception as e:
        print(f"Error pushing to hub: {e}")


def parse_args():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments containing repo_id and model_path.
    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--trained-model-path", type=str, required=True)
    parser.add_argument("--push-model", type=bool)  
    parser.add_argument("--push-tokenizer", type=bool)  

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    push_to_hub(args.repo_id, args.trained_model_path, args.push_model, args.push_tokenizer)
