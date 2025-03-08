import sys
sys.path.append('E:\projects\SolveItEasy')

import os
import shutil
import joblib
import argparse
from functools import partial

import warnings

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

from hydra import initialize, compose
from omegaconf import OmegaConf

from transformers import set_seed
from src.utils.model_utils1 import load_tokenizer

from src.utils.exp_utils import setup_environment, create_exp_dir

from src.utils.hieralog import hprint, fprint, pprint, progress_write
# from hieralog import hprint, fprint, pprint, progress_write

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
    # assert cfg.exp_manager.exp_name == os.path.basename(config_path).replace('.yaml', ''), \
    f"Config file name '{os.path.basename(config_path)}' does not match experiment name '{cfg.exp_manager.exp_name}' in the config."

    if cfg.exp_manager.print_cfg:
        hprint("2: Showing configuration...")
        fprint(OmegaConf.to_yaml(cfg))
    
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

# def load_cfg(file_path: str) -> dict:
#     """Loads user data from a YAML file using OmegaConf."""
#     cfg = OmegaConf.load(file_path)
#     return cfg


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
    pprint(f"Configuration saved to {config_path}")


def has_system_role_support(tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the FIFA World Cup 2022?"},
        {"role": "assistant", "content": "Argentina won the FIFA World Cup 2022, defeating France in the final."}
    ]
    try:
        tokenizer.apply_chat_template(messages, tokenize=False)
        return True
    except:
        return False

def format_examples(user_input: dict) -> str:
    """Formats the examples section if examples_list is provided, using examples_template."""
    examples_key = user_input.get("examples_key", "")
    examples_template = user_input.get("examples_template", "")
    examples_list = user_input.get("examples_list", [])

    if examples_key and examples_list:
        formatted = "\n".join(
            examples_template.format(
                index=ex.get("index", ""),
                problem=ex.get("problem", ""),
                reason=ex.get("reason", ""),
                answer=ex.get("answer", "")
            ).strip()
            for ex in examples_list if "problem" in ex and "answer" in ex
        )
        return f"{examples_key}\n{formatted}" if formatted else ""
    return ""


def generate_prompt_wrapper(tokenizer):

    def generate_prompt(sample, data_args, prompt_args, exp_args) -> str:
        """
        Generates a structured prompt based on the provided dictionary.
        Sections are included only if their corresponding key/text is present and allowed by flags.
        """

        # Ensure phase is only "train" or "eval"
        if exp_args.phase not in ["train", "eval"]:
            raise ValueError(f"Invalid phase: {exp_args.phase}. Allowed values are 'train' or 'eval'.")

        use_only_input_text = prompt_args.get("use_only_input_text", False)
        use_examples = prompt_args.get("use_examples", False)
        use_context = prompt_args.get("use_context", False)
        use_response_format_guide = prompt_args.get("use_response_format_guide", False)

        input_col = data_args.get("input_col", "problem")
        output_col = data_args.get("output_col", "solution")
        context_col = data_args.get("context_col", "context")

        if not prompt_args.get("end_key"):
            prompt_args["end_key"] = tokenizer.eos_token

        if use_only_input_text:
            prompt = sample[input_col]
            if exp_args.phase in ["train", "eval"]:
                prompt += prompt_args.get("sep", "\n\n") + sample[output_col]
        else:
            prompt_parts = {}

            def get_section(key, text_key=None):
                parts = []
                val = prompt_args.get(key)
                if val:
                    parts.append(val)
                if text_key:
                    val = prompt_args.get(text_key)
                    if val:
                        parts.append(val)
                return "\n".join(parts).strip() or None

            prompt_parts["intro"] = get_section("intro_key", "intro_text")
            prompt_parts["instruction"] = get_section("instruction_key", "instruction_text")
            prompt_parts["response_format_guide"] = get_section("response_format_guide_key", "response_format_guide_text") if use_response_format_guide else None
            prompt_parts["examples"] = format_examples(prompt_args) if use_examples else None
            prompt_parts["context"] = get_section("context_key", "context_text") if use_context else None

            # Build input section
            input_parts = []
            if prompt_args.get("input_key"):
                input_parts.append(prompt_args.get("input_key"))
            if exp_args.phase in ["train", "eval"]:
                input_parts.append(sample.get(input_col, ""))
            else:
                input_parts.append(prompt_args.get("input_text", ""))
            prompt_parts["input"] = "\n".join(filter(None, input_parts)).strip() or None

            prompt_parts["pre_response"] = prompt_args.get("pre_response_text")

            # Build response section
            response_parts = []
            if prompt_args.get("response_key"):
                response_parts.append(prompt_args.get("response_key"))
            if exp_args.phase == "train":
                response_parts.append(sample.get(output_col, ""))
            prompt_parts["response"] = "\n".join(filter(None, response_parts)).strip() or None

            if not prompt_args.get("use_model_chat_template"):
                combined = [v for v in prompt_parts.values() if v]
                prompt = prompt_args.get("sep", "\n\n").join(combined)
            else:
                # Use chat template if supported
                if has_system_role_support(tokenizer):
                    system_content = "\n\n".join(filter(None, [
                        prompt_parts.get("intro"),
                        prompt_parts.get("instruction"),
                        prompt_parts.get("response_format_guide"),
                        prompt_parts.get("examples")
                    ]))
                    user_content = "\n\n".join(filter(None, [
                        prompt_parts.get("context"),
                        prompt_parts.get("input"),
                        prompt_args.get("pre_response_text", "")
                    ]))
                    messages = [{"role": "system", "content": system_content},
                                {"role": "user", "content": user_content}]
                    if exp_args.phase == "train":
                        messages.append({"role": "assistant", "content": prompt_parts.get("response") or ""})
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    else:
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        prompt += prompt_parts.get("response") or ""
                else:
                    # Ensure context is included with input
                    user_message_content = "\n\n".join(filter(None, [
                        prompt_parts.get("intro"),
                        prompt_parts.get("instruction"),
                        prompt_parts.get("response_format_guide"),
                        prompt_parts.get("examples"),
                        prompt_parts.get("context"),
                        prompt_parts.get("input"),
                        prompt_args.get("pre_response_text", "")
                    ]))

                    messages = [{"role": "user", "content": user_message_content}]
                    
                    if exp_args.phase == "train":
                        messages.append({"role": "assistant", "content": prompt_parts.get("response") or ""})
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    else:
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if exp_args.phase == "train":
            prompt += prompt_args["end_key"]

        sample["text"] = prompt
        return sample
    
    
    return generate_prompt



# def generate_prompt_wrapper(tokenizer):

#     def generate_prompt(sample, data_args, prompt_args, exp_args) -> str:
#         """
#         Generates a structured prompt based on the provided dictionary.
#         Sections are included only if their corresponding key/text is present and allowed by flags.
#         """
#         use_only_input_text = prompt_args.get("use_only_input_text", False)
#         use_examples = prompt_args.get("use_examples", False)
#         use_context = prompt_args.get("use_context", False)
#         use_response_format_guide = prompt_args.get("use_response_format_guide", False)

#         input_col = data_args.get("input_col", "problem")
#         output_col = data_args.get("output_col", "solution")
#         context_col = data_args.get("context_col", "context")

#         if not prompt_args.get("end_key"):
#             prompt_args["end_key"] = tokenizer.eos_token

#         if use_only_input_text:
#             prompt = sample[input_col]
#             if exp_args.phase == "train":
#                 prompt += prompt_args.get("sep", "\n\n") + sample[output_col]
#         else:
#             prompt_parts = {}

#             def get_section(key, text_key=None):
#                 parts = []
#                 val = prompt_args.get(key)
#                 if val:
#                     parts.append(val)
#                 if text_key:
#                     val = prompt_args.get(text_key)
#                     if val:
#                         parts.append(val)
#                 return "\n".join(parts).strip() or None

#             prompt_parts["intro"] = get_section("intro_key", "intro_text")
#             prompt_parts["instruction"] = get_section("instruction_key", "instruction_text")
#             prompt_parts["response_format_guide"] = get_section("response_format_guide_key", "response_format_guide_text") if use_response_format_guide else None
#             prompt_parts["examples"] = format_examples(prompt_args) if use_examples else None
#             prompt_parts["context"] = get_section("context_key", "context_text") if use_context else None

#             # Build input section
#             input_parts = []
#             if prompt_args.get("input_key"):
#                 input_parts.append(prompt_args.get("input_key"))
#             if exp_args.phase == "train":
#                 input_parts.append(sample.get(input_col, ""))
#             else:
#                 input_parts.append(prompt_args.get("input_text", ""))
#             prompt_parts["input"] = "\n".join(filter(None, input_parts)).strip() or None

#             prompt_parts["pre_response"] = prompt_args.get("pre_response_text")

#             # Build response section
#             response_parts = []
#             if prompt_args.get("response_key"):
#                 response_parts.append(prompt_args.get("response_key"))
#             if exp_args.phase == "train":
#                 response_parts.append(sample.get(output_col, ""))
#             prompt_parts["response"] = "\n".join(filter(None, response_parts)).strip() or None

#             if not prompt_args.get("use_model_chat_template"):
#                 combined = [v for v in prompt_parts.values() if v]
#                 prompt = prompt_args.get("sep", "\n\n").join(combined)
#             else:
#                 # Use chat template if supported
#                 if has_system_role_support(tokenizer):
#                     system_content = "\n\n".join(filter(None, [
#                         prompt_parts.get("intro"),
#                         prompt_parts.get("instruction"),
#                         prompt_parts.get("response_format_guide"),
#                         prompt_parts.get("examples")
#                     ]))
#                     user_content = "\n\n".join(filter(None, [
#                         prompt_parts.get("context"),
#                         prompt_parts.get("input"),
#                         prompt_args.get("pre_response_text", "")
#                     ]))
#                     messages = [{"role": "system", "content": system_content},
#                                 {"role": "user", "content": user_content}]
#                     if exp_args.phase == "train":
#                         messages.append({"role": "assistant", "content": prompt_parts.get("response") or ""})
#                         prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
#                     else:
#                         prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#                         prompt += prompt_parts.get("response") or ""
#                 else:
#                     # Ensure context is included with input
#                     user_message_content = "\n\n".join(filter(None, [
#                         prompt_parts.get("intro"),
#                         prompt_parts.get("instruction"),
#                         prompt_parts.get("response_format_guide"),
#                         prompt_parts.get("examples"),
#                         prompt_parts.get("context"),
#                         prompt_parts.get("input"),
#                         prompt_args.get("pre_response_text", "")
#                     ]))

#                     messages = [{"role": "user", "content": user_message_content}]
                    
#                     if exp_args.phase == "train":
#                         messages.append({"role": "assistant", "content": prompt_parts.get("response") or ""})
#                         prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
#                     else:
#                         prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#         if exp_args.phase == "train":
#             prompt += prompt_args["end_key"]

#             sample["text"] = prompt
#             return sample
        
#         else:
#             return prompt
    
#     return generate_prompt


def do_tokenize_wrapper(tokenizer):
    def do_tokenize(sample, tokenizer_args):
        """
        Tokenizes the text in the example and updates the example dictionary with input_ids and attention_mask.
        """
        text = sample.get("text", None)  # Get text safely

        if not isinstance(text, str):  
            raise ValueError(f"Expected `text` to be a string, but got {type(text)}: {text}")

        tokenized_text = tokenizer(text, **tokenizer_args)  # Pass tokenizer_args correctly
        sample["input_ids"] = tokenized_text["input_ids"]
        sample["attention_mask"] = tokenized_text["attention_mask"]
        return sample
    
    return do_tokenize

# @capture_output
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
        hprint(f"3: Split: {split_name}")

        # Get the length and columns of the current split
        hprint("4: Showing length, columns...")
        dataset_length = len(dataset)
        dataset_columns = dataset.column_names

        pprint(f"Number of Examples: {dataset_length}")
        pprint(f"Columns: {dataset_columns}")

        # Calculate the shape of each column
        hprint("4: Calculating shapes...")
        hprint("5: Shapes:")
        for column_name in dataset_columns:
            if column_name in dataset[0]:
                col_data = dataset[column_name]
                if isinstance(col_data[0], list):  # Multi-dimensional data (e.g., tokenized inputs)
                    pprint(f"  {column_name}: [{len(col_data)}, {len(col_data[0])}]")
                else:  # Single-dimensional data (e.g., strings)
                    pprint(f"  {column_name}: [{len(col_data)}]")

        # Get the first example from the current split
        example = dataset[0]

        hprint("4: Showing an example...")
        hprint("5: An example:")
        for key, value in example.items():
            pprint(f"  {key}: \n{value}")


def prepare_data(exp_args, data_args, tokenizer_args, prompt_args, model_args):
    """
    Prepare the data for training and evaluation.

    Args:
        exp_args (dict): Experiment arguments.
        data_args (dict): Data arguments.
        tokenizer_args (dict): Tokenizer arguments.
        prompt_args (dict): Prompt arguments.
        model_args (dict): Model arguments.

    Returns:
        DatasetDict: A DatasetDict containing the prepared train, val, and test splits.
    """
    # Load the tokenizer
    hprint("2: Loading tokenizer...")
    tokenizer = load_tokenizer(tokenizer_args, model_args, prompt_args)

    # Load the dataset
    hprint("2: Loading dataset...")
    dataset = load_dataset(data_args.dataset_name_or_path)

    if data_args.subset_ratio and 0< data_args.subset_ratio < 1:
            dataset = DatasetDict({
                split: dataset[split].shuffle(seed=exp_args.seed).select(range(int(data_args.subset_ratio * len(dataset[split]))))
                for split in dataset.keys()
    })
    
    columns_to_remove = [col for col in dataset['train'].column_names if col not in data_args.columns_to_retain]

    # Generate prompts
    hprint("2: Generating prompts...")
    generate_prompt = generate_prompt_wrapper(tokenizer)
    _generate_prompt = partial(generate_prompt, data_args=data_args, prompt_args=prompt_args, exp_args=exp_args)
    dataset = dataset.map(_generate_prompt, remove_columns=columns_to_remove)

    if data_args.do_tokenize:
        # Tokenize the data
        hprint("2: Tokenizing the data...")
        do_tokenize = do_tokenize_wrapper(tokenizer)
        _do_tokenize = partial(do_tokenize, tokenizer_args=tokenizer_args.tokenizer_args)
        dataset = dataset.map(_do_tokenize)

    if data_args.save_prepared:
        # Save the prepared data
        hprint("2: Saving prepared data...")
        joblib.dump(dataset, data_args.prepared_data_path)
        pprint(f"Prepared data saved to: {data_args.prepared_data_path}")

    if data_args.do_show:
        hprint("2: Showing dataset examples...")
        # Show dataset examples
        show_dataset_examples(dataset)

    return dataset


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Load generation config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML config file for generating.")

    args, override_args = parser.parse_known_args()
    return args, override_args


def main():
    # # Setup logging
    # logger = setup_logger()

    # Setup environment
    hprint("1: Setting up environment...")
    setup_environment()
    
    # Parse arguments
    args, override_args = parse_args()

    # Load configuration
    hprint("1: Loading configuration...")
    cfg, exp_args, data_args, tokenizer_args, prompt_args, model_args, train_args, eval_args, gen_args, device_args = load_cfg(config_path=args.config_path, override_args=override_args)
    
    # Create experiment directories
    exp_name = cfg.exp_manager.exp_name
    exps_dir = cfg.exp_manager.exps_dir
    exp_dir, exp_data_dir, exp_checkpoints_dir, exp_results_dir = create_exp_dir(exp_name, exps_dir)

    # Save configuration if have any changes from the overrides
    config_path = os.path.join(exp_dir, exp_name + '.yaml')
    save_cfg(cfg, config_path)

    # Set seed
    set_seed(exp_args.seed)


    if data_args.is_prepared:
        hprint("1: Data is prepared, loading...")
        prepare_data_path = data_args.prepared_data_path
        dataset_dict = joblib.load(prepare_data_path)
    else:
        hprint("1: Preparing data...")
        dataset_dict = prepare_data(exp_args, data_args, tokenizer_args, prompt_args, model_args)

if __name__ == "__main__":
    main()
