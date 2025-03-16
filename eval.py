import argparse
import os
import re
import joblib

import numpy as np
import torch
import evaluate
from tqdm.auto import tqdm

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from transformers import (
    set_seed,
    PreTrainedTokenizer,
    AutoTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    BartForConditionalGeneration,
    AutoConfig,
    AutoModel
)

from torch.utils.data import DataLoader

from src.utils.exp_utils import setup_environment, create_exp_dir
from src.utils.hieralog import hprint, pprint, fprint

from src.utils.model_utils import (
    set_torch_dtype_and_attn_implementation,
    get_quantization_config
)

from prepare_data import prepare_data, show_dataset_examples


from generate import load_model_tokenizer_for_generate, get_model_class, load_model_for_generate
from src.utils.model_utils import load_tokenizer
from src.utils.eval_utils import save_predictions, save_metrics

def load_cfg(config_path, override_args=None):

    """
    Load a configuration file using Hydra and OmegaConf.
    
    Args:
        config_path (str): Path to the configuration file.
        override_args (list, optional): List of arguments to override configuration values.

    Returns:
        cfg: Loaded configuration object.
    """

    override_args = override_args or []
    config_path = os.path.normpath(config_path)
    
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    config_dir = os.path.dirname(config_path)
    config_fn = os.path.splitext(os.path.basename(config_path))[0]
    
    try:
        with initialize(version_base=None, config_path=config_dir):
            cfg = compose(config_name=config_fn, overrides=override_args)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")
    
    # assert os.path.basename(config_path).replace('.yaml', '') == cfg.exp_manager.exp_name, \
    # assert cfg.exp_manager.phase_name + '__' + 
    # assert cfg.exp_manager.exp_name == os.path.basename(config_path).replace('.yaml', ''), \
    f"Config file name '{os.path.basename(config_path)}' does not match experiment name '{cfg.exp_manager.exp_name}' in the config."

    cfg.train.lora.task_type = cfg.train.progress_callback.model_type = cfg.model.model_type
    
    exp_args = cfg.exp_manager
    data_args = cfg.data
    tokenizer_args = cfg.tokenizer
    prompt_args = cfg.prompt
    model_args = cfg.model
    train_args = cfg.train
    eval_args = cfg.evaluate
    device_args = cfg.device
    gen_args = cfg.generate

    return cfg, exp_args, data_args, tokenizer_args, prompt_args, model_args, train_args, eval_args, gen_args, device_args


def save_cfg(cfg, config_path):
    """
    Save the configuration to a YAML file.

    Args:
        cfg (OmegaConf): The configuration object to save.
        config_path (str): The path where the configuration file will be saved.

    Returns:
        None
    """
    OmegaConf.save(cfg, config_path)


# def extract_prediction(text):
#     """
#     Extracts the text after '### Summary:' using regex.

#     Args:
#         text (str): The input string containing the summary.

#     Returns:
#         str: The extracted summary text or an empty string if not found.
#     """
#     match = re.search(r"### Summary:\s+(.*)", text, re.DOTALL)
#     prediction = match.group(1).strip() if match else ""

#     return prediction

# Function to extract the final number from generated text
def extract_prediction(response, min_value=np.iinfo(np.int32).min, max_value=np.iinfo(np.int32).max):
    try:
        # matches = re.findall(r'Solution:.*?Answer:.*([\d,]+)', response, re.DOTALL)
        matches = re.findall(r'Solution:.*?Reason:.*Answer:.*?([\d,]+)', response, re.DOTALL)

        if matches:
            answer =  matches[0]
            answer = answer.replace(',', '')

            if min_value <= int(float(answer)) <= max_value: 
                return answer
            
            else:
                return '-1'
            # answer = int(answer)    
        else:
            answer = "-1"
        return answer
    except Exception:
        return "-1"
    
import re

def extract_numeric_answer(response):
    """
    Extract the numeric answer from the response text.

    Args:
        response (str): The response text.

    Returns:
        str: The extracted numeric answer.
    """
    # Use regular expression to find the answer
    match = re.search(r'Answer:\s*(\d+)', response)
    if match:
        return match.group(1)
    else:
        return None


def eval(model, tokenizer, test_loader, eval_args, exp_args, gen_args, exp_variant_results_dir):
    """
    Evaluates a model on a test dataset, generates predictions, and saves results in multiple formats.

    Args:
        model: The model to evaluate.
        tokenizer: Tokenizer for decoding model outputs.
        test_loader: DataLoader for the test dataset.
        eval_args: Arguments related to evaluation (e.g., break_step, do_extract_prediction).
        exp_args: Experiment arguments (e.g., exp_name).
        gen_args: Generation arguments (e.g., max_new_tokens, temperature).
        exp_variant_results_dir: Directory to save results.
        result_file_types: List of file types to save results (e.g., ['txt', 'json', 'csv']).
    """

    # Ensure results directory exists
    os.makedirs(exp_variant_results_dir, exist_ok=True)

    accuracy_metric = evaluate.load("accuracy")

    predictions_list = []

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            if step == eval_args.break_step:
                break

            input_ids = batch['input_ids'].squeeze(1).to(model.device)
            attention_mask = batch['attention_mask'].squeeze(1).to(model.device)

            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # pad_token_id=tokenizer.eos_token_id,
                **gen_args.gen_args
            )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=gen_args.skip_special_tokens)

            pred_texts = [extract_prediction(output) if eval_args.do_extract_prediction else output for output in outputs]
            true_texts = [answer.replace(',', '') for answer in batch['answer']]

            try:
                accuracy_metric.add_batch(predictions=pred_texts, references=true_texts)
            except Exception as e:
                print(f"Error adding batch to accuracy metric: {e}")
                continue

            ids = batch['index']

            # Store predictions in a structured format
            for id, output, pred_text, true_text in zip(ids, outputs, pred_texts, true_texts):
                predictions_list.append({
                    "id": id,
                    "output": output,
                    "prediction": pred_text,
                    "ground_truth": true_text,
                    "correct": pred_text == true_text
                })

    # Compute final accuracy
    results = accuracy_metric.compute()
    print(f"Accuracy: {results}")

    # Save predictions
    save_predictions(predictions_list, exp_variant_results_dir, f'{exp_args.prefix_fn}_{eval_args.prediction_filename}')

    # Save accuracy metrics
    metrics = {
        "exp_name": exp_args.exp_name,
        "exp_variant": exp_args.exp_variant,
        "accuracy": results
    }
    save_metrics(metrics, exp_variant_results_dir, f'{exp_args.prefix_fn}_{eval_args.metric_filename}')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Load config.")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the YAML config file.")

    args, override_args = parser.parse_known_args()
    return args, override_args

def main():

    # Setup environment
    hprint("1: Setting up environment...")
    setup_environment()
    
    # Parse arguments
    args, override_args = parse_args()

    # Load configuration
    hprint("1: Loading configuration...")
    cfg, exp_args, data_args, tokenizer_args, prompt_args, model_args, train_args, eval_args, gen_args, device_args = load_cfg(config_path=args.config_path, override_args=override_args)
    
    if cfg.exp_manager.print_cfg:
        hprint("2: Showing configuration...")
        fprint(OmegaConf.to_yaml(cfg))
    
    # Create experiment directories
    exp_name = cfg.exp_manager.exp_name
    exps_dir = cfg.exp_manager.exps_dir
    exp_variant_dir = cfg.exp_manager.exp_variant

    (exp_dir, exp_variant_dir, exp_variant_data_dir, exp_variant_checkpoints_dir, exp_variant_results_dir) = create_exp_dir(exp_name, exp_variant_dir, exps_dir)
    # import shutil
    # shutil.copy(args.config_path, exp_dir)

    # Save configuration if have any changes from the overrides
    prefix_fn = cfg.exp_manager.prefix_fn
    if not prefix_fn:
        prefix_fn = ''
        
    config_path = os.path.join(exp_variant_dir, 'eval_' + prefix_fn + '_' + exp_name + '.yaml')
    save_cfg(cfg, config_path)
    pprint(f"2: Configuration saved to {config_path}")

    # Set seed
    set_seed(exp_args.seed)

    #Load dataset
    if data_args.is_prepared:
        hprint("1: Data is prepared, loading...")
        prepare_data_path = data_args.prepared_data_path
        dataset = joblib.load(prepare_data_path)
    else:
        hprint("1: Preparing data...")
        dataset = prepare_data(exp_args, data_args, tokenizer_args, prompt_args, model_args)

    # Get test dataset
    hprint("2: Getting test dataset...")
    test_ds = dataset['test']
    test_ds = test_ds.with_format("torch")

    hprint("2: Creating test dataloader...")   
    test_loader = DataLoader(test_ds, batch_size=eval_args.batch_size, shuffle=False)

    from accelerate import Accelerator
    accelerator = Accelerator(cpu=device_args.use_cpu)

    # Set seed
    set_seed(exp_args.seed)

    hprint("1: Loading model and tokenizer...")

    model, tokenizer = load_model_tokenizer_for_generate(model_args, device_args)

    # hprint("2: Loading model...")
    # model = load_model_for_generate(model_args, device_args)

    # hprint("2: Loading tokenizer...")
    # tokenizer = load_tokenizer(tokenizer_args, model_args, prompt_args)

    # model.resize_token_embeddings(len(tokenizer))

    
    model = model.to(accelerator.device)
    # model, test_loader = accelerator.prepare(model, test_loader)
    
    hprint("1: Evaluating model...")
    # Evaluate model
    eval(model, tokenizer, test_loader, eval_args, exp_args, gen_args, exp_variant_results_dir)




if __name__ == "__main__":
    main()
        