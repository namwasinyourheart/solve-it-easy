import os
import shutil
import joblib
import argparse
from functools import partial

import warnings

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from src.utils.model_utils import load_tokenizer

from hydra import initialize, compose
from omegaconf import OmegaConf

from transformers import set_seed
from src.utils.model_utils import load_tokenizer

from src.utils.log_utils import setup_logger
from src.utils.exp_utils import setup_environment, create_exp_dir

warnings.filterwarnings("ignore")

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
    assert cfg.exp_manager.exp_name == os.path.basename(config_path).replace('.yaml', ''), \
    f"Config file name '{os.path.basename(config_path)}' does not match experiment name '{cfg.exp_manager.exp_name}' in the config."

    if cfg.exp_manager.print_cfg:
        print(OmegaConf.to_yaml(cfg))
    
    exp_args = cfg.exp_manager
    data_args = cfg.prepare_data
    model_args = cfg.prepare_model
    train_args = cfg.train
    eval_args = cfg.eval
    device_args = cfg.device
    gen_args = cfg.generate

    return cfg, exp_args, data_args, model_args, train_args, eval_args, gen_args

def create_dataset_dict(data_path, 
                        do_split: bool=True, 
                        val_ratio: float=0.25,
                        test_ratio: float=0.2,
                        seed: int = 42):

    
    df = pd.read_csv(data_path)
    hf_dataset = Dataset.from_pandas(df)

  
    if do_split:
        # Splitting the dataset
        train_test_split = hf_dataset.train_test_split(test_size=test_ratio, seed=seed)
        train_val_split = train_test_split["train"].train_test_split(test_size=val_ratio, seed=seed)  # 0.25 x 0.8 = 0.2
    
        # Create DatasetDict
        dataset_dict = DatasetDict({
            "train": train_val_split["train"],
            "val": train_val_split["test"],
            "test": train_test_split["test"]
        })
        
    else:
        dataset_dict = DatasetDict({
            "train": hf_dataset
        })
    return dataset_dict


def show_dataset_examples(dataset_dict):
    """
    Prints the length, columns, shape of columns, and an example from each split of a DatasetDict (train, val, test).

    Parameters
    ----------
    dataset_dict : datasets.DatasetDict
        A DatasetDict containing train, val, and test splits.

    Returns
    -------
    None
    """
    for split_name, dataset in dataset_dict.items():
        # Get the length and columns of the current split
        dataset_length = len(dataset)
        dataset_columns = dataset.column_names

        print(f"\nSplit: {split_name}")
        print(f"Number of Examples: {dataset_length}")
        print(f"Columns: {dataset_columns}")

        # Calculate the shape of each column
        print("Shapes:")
        for column_name in dataset_columns:
            if column_name in dataset[0]:
                col_data = dataset[column_name]
                if isinstance(col_data[0], list):  # Multi-dimensional data (e.g., tokenized inputs)
                    print(f"  {column_name}: [{len(col_data)}, {len(col_data[0])}]")
                else:  # Single-dimensional data (e.g., strings)
                    print(f"  {column_name}: [{len(col_data)}]")

        # Get the first example from the current split
        example = dataset[0]

        print("An example:")
        for key, value in example.items():
            print(f"  {key}: \n{value}")

    print("-" * 24 + "\n")


def has_system_role_support(tokenizer):
    messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Who won the FIFA World Cup 2022?"
            },
            {
                "role": "assistant",
                "content": "Argentina won the FIFA World Cup 2022, defeating France in the final."
            }
    ]

    try: 
        tokenizer.apply_chat_template(messages, tokenize=False)
        return True
    except:
        return False



def create_prompt_formats(example,
                          tokenizer,
                          use_model_chat_template: bool=False,
                          input_col: str="question",
                          output_col: str="answer",
                          context_col: str="context",

                          use_only_input_text: str=False,
                          use_examples: str=False,
                          use_context: str=False,
                          
                          intro_text: str="You are a knowledgeable assistant for the company CMC Global.",
                          instruction_key: str="### Instruction:",
                          instruction_text: str="Your task is to providing accurate and helpful answers to the user's questions about the company.",

                          examples_key: str="### Examples:",
                          examples_template: str="",
                          examples_list: list=[],
                          
                          context_key: str="### Context:",
                          input_key: str = "### Question:",
                          response_key: str = "### Answer:",
                          end_key = None,
                          
                          do_tokenize = False, 
                          max_length = None, 
                          phase_name='eval'
):
    if not end_key:
        end_key = tokenizer.eos_token
    end = f'{end_key}'
    
    if use_only_input_text:
        input = f'{example[input_col]}'

        if phase_name == 'train':
            response = f'{example[output_col]}'
        
        elif phase_name == 'eval':
            response = None

        parts = [part for part in [input, response] if part]
        formatted_prompt = "\n\n".join(parts)

    else:
        intro = f'{intro_text}'

        instruction = f'{instruction_key}\n{instruction_text}'

        if not use_examples:
            examples = None
        else:
            example_template = examples_template
            formatted_examples = "\n".join(
                example_template.format(**example) for example in examples_list
            )
            examples = f"{examples_key}\n{formatted_examples}"

        if not use_context:
            context = None
        else:
            context = f"{context_key}\n{example['context_col']}"
    
        input = f'{input_key}\n{example[input_col]}'
    
        if phase_name == 'train':
            response = f'{response_key}\n{example[output_col]}'
        
        elif phase_name == 'eval':
            response = f'{response_key}'
        
        if not use_model_chat_template:  # Not using default model chat template
            parts = [part.strip() for part in [intro, instruction, examples, context, input, response] if part and len(part.strip())]
            formatted_prompt = "\n\n".join(parts)
        
        else:   # Using defaut model chat template
            if has_system_role_support(tokenizer):
    
                if context_col:
                    input = f'{context}\n{input}'
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": input},
                    {"role": "assistant", "content": response},   
                ]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
            else:
                if context_col:
                    input = f'{context}\n{input}'
                messages = [
                    {"role": "user", "content": instruction + '\n' + input},
                    {"role": "assistant", "content": response},
                ]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    
    if phase_name == 'train':
        if formatted_prompt.strip().endswith(end):
            example['text'] = formatted_prompt
            # print('endswith end')
        else:
            # print('does not ends with end')
            example['text'] = formatted_prompt + end
    
    elif phase_name == 'eval':
        example['text'] = formatted_prompt + '\n'
        

    if do_tokenize:
        tokenized_text = tokenizer(formatted_prompt, 
                                   truncation=True, 
                                   padding='max_length', 
                                   add_special_tokens=True, 
                                   max_length=max_length,
                                   # return_tensors='pt',
                                   padding_side='left'
                                   )

        
        example['input_ids'] = tokenized_text['input_ids']
        example['attention_mask'] = tokenized_text['attention_mask']

    return example
        
    
def save_dataset(dataset, save_path):
    joblib.dump(dataset, save_path)
    
def get_data_collator():
    pass

def prepare_data(exp_args, data_args, model_args):

    if data_args.dataset.is_dataset_dict:
        dataset_dict = load_dataset(data_args.dataset.data_path,
                                    trust_remote_code=True)
    
    else:    
        dataset_dict = create_dataset_dict(data_args.dataset.data_path, 
                                           data_args.dataset.do_split, 
                                           data_args.dataset.val_ratio, 
                                           data_args.dataset.test_ratio, 
                                           exp_args.seed)
        

    if data_args.dataset.subset_ratio and 0 < data_args.dataset.subset_ratio < 1:
        
        dataset_dict = DatasetDict({
            split: dataset_dict[split].shuffle(seed=exp_args.seed).select(range(int(len(dataset_dict[split]) * data_args.dataset.subset_ratio)))
            for split in dataset_dict.keys()
        }) 
    
    tokenizer = load_tokenizer(data_args, model_args)

    for key, value in data_args.prompt.items():
        if not key.startswith('use') and key != 'end_key' and not value:
            value = ''
            data_args.prompt[key] = value

    _create_prompt_formats = partial(
        create_prompt_formats,
          tokenizer = tokenizer,
          use_model_chat_template = data_args.prompt.use_model_chat_template,
          
          input_col = data_args.dataset.input_col,
          output_col = data_args.dataset.output_col,
          context_col = data_args.dataset.context_col,

          use_only_input_text = data_args.prompt.use_only_input_text,
          use_examples = data_args.prompt.use_examples,
          use_context = data_args.prompt.use_context,
                          
          
          intro_text = data_args.prompt.intro_text, 
          instruction_key = data_args.prompt.instruction_key, 
          instruction_text = data_args.prompt.instruction_text,           
          
          examples_key = data_args.prompt.examples_key,
          examples_template = data_args.prompt.examples_template,
          examples_list = data_args.prompt.examples_list,

          context_key =  data_args.prompt.context_key,
          input_key = data_args.prompt.input_key, 
          response_key = data_args.prompt.response_key, 
          end_key = data_args.prompt.end_key,

          do_tokenize = data_args.tokenizer.do_tokenize, 
          max_length = data_args.tokenizer.max_length,
          phase_name = exp_args.phase_name
    )
    
    columns_to_retain = data_args.dataset.columns_to_retain
    columns_to_remove = [col for col in dataset_dict['train'].column_names if col not in columns_to_retain]

    dataset = dataset_dict.map(
        _create_prompt_formats, 
         batched=False, 
         remove_columns=columns_to_remove
    )

    if data_args.dataset.do_save:
        save_path = data_args.dataset.prepared_data_path
        save_dataset(dataset, save_path)

    return dataset, save_path


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
    cfg, exp_args, data_args, model_args, train_args, eval_args, gen_args = load_cfg(config_path=args.config_path, override_args=override_args)
    
    # Create experiment directories
    logger.info("CREATING DIRECTORIES...")
    exp_name = cfg.exp_manager.exp_name
    (exp_dir, exp_data_dir, exp_checkpoints_dir, exp_results_dir) = create_exp_dir(exp_name)

    shutil.copy(args.config_path, exp_dir)

    # Set seed
    set_seed(exp_args.seed)


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

    if data_args.dataset.do_show:
        # Show dataset examples
        show_dataset_examples(dataset)


if __name__ == '__main__':
    main()
