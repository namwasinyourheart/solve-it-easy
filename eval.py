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
from src.utils.log_utils import setup_logger

from src.utils.model_utils import (
    set_torch_dtype_and_attn_implementation,
    get_quantization_config
)

from prepare_data import prepare_data, show_dataset_examples

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
    
    assert cfg.exp_manager.exp_name == os.path.basename(config_path).replace('.yaml', ''), \
    f"Config file name '{os.path.basename(config_path)}' does not match experiment name '{cfg.exp_manager.exp_name}' in the config."
    
    exp_args = cfg.exp_manager
    data_args = cfg.prepare_data
    model_args = cfg.prepare_model
    train_args = cfg.train
    eval_args = cfg.eval
    device_args = cfg.device
    gen_args = cfg.generate
    device_args = cfg.device

    if exp_args.print_cfg:
        print(OmegaConf.to_yaml(cfg))

    return cfg, exp_args, data_args, model_args, train_args, eval_args, gen_args, device_args

from generate import get_model_class, load_model_for_generate
from src.utils.model_utils import load_tokenizer

import re
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
def extract_prediction(text, min_value=np.iinfo(np.int32).min, max_value=np.iinfo(np.int32).max):
    try:
        # matches = re.findall(r'### Solution:\s+.*?Answer:\s*([\d,]+)', text, re.DOTALL)
        matches = re.findall(r'### Solution:.*?Answer:\s*([\d,]+)', text, re.DOTALL)
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

def main():

    # Setup logging
    logger = setup_logger()

    # Setup environment
    logger.info("SETTING UP ENVIRONMENT...")
    setup_environment()


    # Parse arguments
    parser = argparse.ArgumentParser(description='Load experiment configurations.')
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='Path to the configuration file for the experiment.'
    )

    args, override_args = parser.parse_known_args()

    # Load configuration
    logger.info("LOADING CONFIGURATIONS...")
    cfg, exp_args, data_args, model_args, train_args, eval_args, gen_args, device_args = load_cfg(config_path=args.config_path, override_args=override_args)
    
    # Create experiment directories
    # logger.info("CREATING DIRECTORIES...")""
    exp_name = cfg.exp_manager.exp_name
    exps_dir = cfg.exp_manager.exps_dir

    (exp_dir, exp_data_dir, exp_checkpoints_dir, exp_results_dir) = create_exp_dir(exp_name, exps_dir)

    # import shutil
    # shutil.copy(args.config_path, exp_dir)

    OmegaConf.save(cfg, os.path.join(exp_dir, 'eval_' + exp_name) + '.yaml')

    # Set seed
    set_seed(exp_args.seed)

    #Load dataset
    if data_args.dataset.is_prepared:
        # Get the path to the processed data
        processed_data_path = os.path.normpath(data_args.dataset.prepared_data_path)
        
        # Check if the processed data exists
        if not os.path.isfile(processed_data_path):
            raise FileNotFoundError(f"Processed data not found at: {processed_data_path}")
        
        # Load the dataset
        logger.info("LOADING PROCESSED DATASET...")
        dataset = joblib.load(processed_data_path)
    else:
        # Prepare dataset
        logger.info("PREPARING DATASET...")
        dataset, processed_data_path = prepare_data(exp_args, data_args, model_args)

    test_ds = dataset['test']
    test_ds = test_ds.with_format("torch")

    test_loader = DataLoader(test_ds, batch_size=eval_args.batch_size, shuffle=False)

    from accelerate import Accelerator
    accelerator = Accelerator(cpu=device_args.use_cpu)

    # Set seed
    set_seed(exp_args.seed)
    
    tokenizer = load_tokenizer(data_args, model_args)
    # print(tokenizer)
    model = load_model_for_generate(model_args, device_args)
    
    model = model.to(accelerator.device)
    # model, test_loader = accelerator.prepare(model, test_loader)
    

    # accuracy_metric = evaluate.load("accuracy")
    accuracy_metric = evaluate.load("accuracy")
    
    logger.info("EVALUATING...")
    
    prediction_file = os.path.join(exp_results_dir, eval_args.prediction_file)
    with open(prediction_file, "w", encoding="utf-8") as f:
        f.write(f'Exp Name: {exp_args.exp_name}\n')
        f.write("-" * 96 + "\n\n")
        
        model.eval()
        with torch.no_grad():
          for step, batch in enumerate(tqdm(test_loader)):
              if step == eval_args.break_step:
                  break
              
              input_ids = batch['input_ids'].squeeze(1).to(accelerator.device)
              attention_mask = batch['attention_mask'].squeeze(1).to(accelerator.device)
              # batch.to(accelerator.device)
                
              output_ids = model.generate(input_ids=input_ids, 
                                          attention_mask=attention_mask, 
                                          pad_token_id=tokenizer.eos_token_id,
                                          max_new_tokens=gen_args['max_new_tokens'],
                                          temperature=gen_args['temperature']
                                         )
              # output_ids = model.generate(**batch ,
              #                             pad_token_id=tokenizer.eos_token_id,
              #                             max_new_tokens=gen_args['max_new_tokens'])
              
              outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

              if eval_args.do_extract_prediction:
                  pred_texts = [extract_prediction(output) for output in outputs]
                  
              else:
                  pred_texts = outputs
              
              # true_texts = batch[data_args.dataset.output_col]
              true_texts = batch['answer']
              true_texts = [answer.replace(',', '') for answer in true_texts]

              try:
                  accuracy_metric.add_batch(predictions=pred_texts, references=true_texts)
              except:
                  # print(pred_answers)
                  # print(true_answers)
                  continue

              ids = batch[data_args.dataset.id_col]
              
              # Write prediction to file
              for id, output, pred_text, true_text in zip(ids, outputs, pred_texts, true_texts):
                is_correct = True if pred_text == true_text else False # 1 = Correct, 0 = Incorrect
                
                f.write(f'Id: {id}\n')
                f.write("-" * 12 + "\n")
                
                f.write(f"Output:\n")
                f.write(f"{output}\n")
                f.write("-" * 12 + "\n")
                
                f.write(f"Prediction\n")
                f.write(f"{pred_text}\n")
                f.write("-" * 12 + "\n")
                
                f.write(f"Ground Truth: {true_text}\n")
                f.write("-" * 12 + "\n")
                # f.write("-" * 96 + "\n\n")
                
                f.write(f"Correct?: {is_correct}\n")
                f.write("-" * 96 + "\n\n")


    # Compute final accuracy
    results = accuracy_metric.compute()
    print(f"Accuracy: {results}")

    # Save to file
    result_file = eval_args.result_file
    with open(os.path.join(exp_results_dir, result_file), "w") as f:
        f.write(f'Exp Name: {exp_args.exp_name}\n')
        f.write("-" * 96 + "\n\n")
        f.write(f"Accuracy: {results}\n")


if __name__ == "__main__":
    main()
        