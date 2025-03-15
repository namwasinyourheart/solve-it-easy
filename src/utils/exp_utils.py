import os

def create_exp_dir(exp_name, exp_variant, exps_dir='exps', sub_dirs=['checkpoints', 'data', 'results']):
    """
    Create necessary directories for an experiment.
    
    Args:
        exp_name (str): Name of the experiment.
        sub_dirs (list): List of subdirectories to create.

    Returns:
        Tuple containing paths to experiment directories.
    """
    # os.makedirs("exps", exist_ok=True)
    os.makedirs(exps_dir, exist_ok=True)
    # exp_dir = os.path.join("exps", exp_name)
    exp_dir = os.path.join(exps_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    exp_variant_dir = os.path.join(exp_dir, exp_variant)
    os.makedirs(exp_variant_dir, exist_ok=True)
    
    sub_dirs_paths = {sub_dir: os.path.join(exp_variant_dir, sub_dir) for sub_dir in sub_dirs}
    for path in sub_dirs_paths.values():
        os.makedirs(path, exist_ok=True)


    exp_variant_data_dir = sub_dirs_paths['data']
    exp_variant_checkpoints_dir = sub_dirs_paths['checkpoints']
    exp_variant_results_dir = sub_dirs_paths['results']
    
    return (exp_dir, exp_variant_dir, exp_variant_data_dir, exp_variant_checkpoints_dir, exp_variant_results_dir)


def setup_environment() -> None:
    from dotenv import load_dotenv
    _ = load_dotenv()

    # # Disable Hugging Face cache
    # os.environ["TRANSFORMERS_CACHE"] = "/dev/null"  

    from huggingface_hub import login
    login(token=os.environ['HUGGINGFACEHUB_API_TOKEN'])