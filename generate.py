from omegaconf import OmegaConf
from typing import List, Tuple
import importlib

from transformers import (
    # AutoModelForCausalLM,
    # AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoConfig, 
    AutoModel,
    set_seed
)


from transformers import BitsAndBytesConfig
from src.utils.model_utils import set_torch_dtype_and_attn_implementation, get_quantization_config
from src.utils.exp_utils import setup_environment



def get_model_class(pretrained_model_name_or_path):
    """
    Dynamically retrieves the model class from Hugging Face transformers 
    using the model's configuration.

    Args:
        pretrained_model_name_or_path (str): The name or local path of the model.

    Returns:
        model_class (transformers.PreTrainedModel): The corresponding model class.
    """ 
    # Load the model configuration
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

    # Extract the model class name
    model_class_name = config.architectures[0] if config.architectures else None

    if model_class_name:
        try:
            # Dynamically import the model class
            module = importlib.import_module("transformers")
            return getattr(module, model_class_name, AutoModel)
        except (ImportError, AttributeError):
            pass  # Fallback if import fails

    return AutoModel  # Default to AutoModel if class is not found

def get_quantization_config(model_args                          
) -> BitsAndBytesConfig | None:
    if model_args['load_in_4bit']:
        torch_dtype, attn_implementation = set_torch_dtype_and_attn_implementation()
        
        if model_args['bnb_4bit_compute_dtype']:
            bnb_4bit_compute_dtype = model_args['bnb_4bit_compute_dtype']
        else:
            bnb_4bit_compute_dtype = torch_dtype

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=model_args['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=model_args['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_storage=model_args['bnb_4bit_quant_storage'],
        ).to_dict()
    elif model_args['load_in_8bit']:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        ).to_dict()
    else:
        quantization_config = None

    return quantization_config



def load_model_for_generate(model_args, device_args) -> PreTrainedModel:

    # Set torch_dtype and attn_implementation
    torch_dtype, attn_implementation = set_torch_dtype_and_attn_implementation()

    if model_args['torch_dtype']:
        torch_dtype = model_args['torch_dtype']

    # QLora Config
    quantization_config = get_quantization_config(model_args)

    model_class = get_model_class(model_args['pretrained_model_name_or_path'])

    # Load model
    model = model_class.from_pretrained(
            model_args['pretrained_model_name_or_path'],
            # trust_remote_code=True,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config if not device_args['use_cpu'] else None,
            device_map=model_args['device_map'], # device_map="cpu" if device_args['use_cpu'] else "auto",
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=model_args['low_cpu_mem_usage']  # low_cpu_mem_usage=True if not device_args['use_cpu'] else False
    )
    
    return model

def load_model_vllm(model_args, device_args):
    from vllm import LLM
    
    # Set torch_dtype and attn_implementation
    torch_dtype, attn_implementation = set_torch_dtype_and_attn_implementation()

    # Load model
    llm = LLM(
        model=model_args['pretrained_model_name_or_path'],
        dtype=torch_dtype
        )

    return llm


def load_tokenizer_for_generate(
    model_args
) -> PreTrainedTokenizer:
    
    tokenizer = AutoTokenizer.from_pretrained(model_args['pretrained_tokenizer_name_or_path'])
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer

def prepare_prompt(prompt_args):
    if prompt_args['use_only_input_text']:
        return prompt_args['input_text']

    for key, value in prompt_args.items():
        if not key.startswith('use') and key != 'end_key' and not value:
            value = ''
            prompt_args[key] = value

    
    intro = f"{prompt_args['intro_text']}"
    instruction = f"{prompt_args['instruction_key']}\n{prompt_args['instruction_text']}"

    if not prompt_args['use_examples']:
        examples = None
    else:
        example_template = prompt_args['examples_template']
        formatted_examples = "\n".join(
            example_template.format(**example) for example in prompt_args['examples_list']
        )
        examples = f"{prompt_args['examples_key']}\n{formatted_examples}"

    input = f"{prompt_args['input_key']}\n{prompt_args['input_text']}"

    if not prompt_args['use_context']:
        context = None
    else:
        context = f"{prompt_args['context_key']}\n{prompt_args['context_text']}"

    response_key = f"{prompt_args['response_key']}"
            
    # Collect all non-empty parts and join them
    parts = [part.strip() for part in [intro, instruction, examples, context, input, response_key] if part and len(part.strip())]

    prompt = "\n\n".join(parts) + '\n'


    return prompt


def postprocess(prompt: str, tokenizer, output: str, response_key: str, end_key: str, return_full: bool=False):
    
    import re
    decoded = None
    fully_decoded = output
    
    # if not end_key:
    end_key = tokenizer.eos_token
    
    pattern = r".*?{}\s*(.+?){}".format(response_key, end_key)
    m = re.search(pattern, fully_decoded, flags=re.DOTALL)

    if m:
        decoded = m.group(1).strip()
    else:
        # The model might not generate the end_key sequence before reaching the max tokens. In this case,
        # return everything after response_key.
        pattern = r".*?{}\s*(.+)".format(response_key)
        m = re.search(pattern, fully_decoded, flags=re.DOTALL)
        if m:
            decoded = m.group(1).strip()
        else:
            print(f"Failed to find response in:\n{fully_decoded}")
            
    if return_full:
        decoded = f"{prompt}{decoded}"

    return decoded


def generate_response(model_args, prompt_args, gen_args, device_args):

    from accelerate import Accelerator
    accelerator = Accelerator(cpu=device_args.use_cpu)
    
    model = load_model_for_generate(model_args, device_args)
    
    tokenizer = load_tokenizer_for_generate(model_args)

    prompt = prepare_prompt(prompt_args)

    # input = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
    input = tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=512, truncation=True, padding_side='left', add_special_tokens=True)
    input_ids = input.input_ids
    attention_mask = input.attention_mask

    # device = accelerator.device

    model = model.to(accelerator.device)
    input_ids = input_ids.squeeze(1).to(accelerator.device)
    attention_mask = attention_mask.squeeze(1).to(accelerator.device)

    output_ids = model.generate(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                max_new_tokens=gen_args['max_new_tokens'], 
                                pad_token_id=tokenizer.pad_token_id)
    
    output = tokenizer.decode(output_ids[0], 
                              skip_special_tokens=gen_args['skip_special_tokens'])

    if gen_args['do_postprocess']:
        output = postprocess(prompt, tokenizer, output, 
                             prompt_args['response_key'],
                             prompt_args['end_key'], 
                             gen_args['return_full'])
    return output


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Load generation config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML config file for generating.")
    parser.add_argument("--input_text", type=str, default=None, help="The text input (question).")

    return parser.parse_args()

def main():
    args = parse_args()

    # Load the generation config file
    cfg = OmegaConf.load(args.config_path)

    # Setup environment
    setup_environment()


    # print(OmegaConf.to_yaml(cfg   ))

    model_args = cfg.model
    prompt_args = cfg.prompt
    gen_args = cfg.generate
    device_args = cfg.device

    # Set seed
    set_seed(gen_args.seed)


    if args.input_text:
        prompt_args['input_text'] = args.input_text

    response = generate_response(model_args, prompt_args, gen_args, device_args)
    print(response)

if __name__ == "__main__":
    main()
    