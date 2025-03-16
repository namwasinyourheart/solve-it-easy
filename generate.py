from omegaconf import OmegaConf
from typing import List, Tuple
import importlib

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
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

    attn_implementation = model_args['attn_implementation']

    if attn_implementation=='flash_attention_2':
        # Set torch_dtype and attn_implementation
        torch_dtype, attn_implementation = set_torch_dtype_and_attn_implementation()

    if model_args['torch_dtype']:
        torch_dtype = model_args['torch_dtype']
        
    else: 
        torch_dtype = 'float16'

    # QLora Config
    quantization_config = get_quantization_config(model_args)

    if model_args.model_type == 'CAUSAL_LM':
        model_class = AutoModelForCausalLM

    elif model_args.model_type == 'SEQ_2_SEQ_LM':
        model_class = AutoModelForSeq2SeqLM
    
    else:
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

    # # Load adapter and merge
    # if model_args.adapter_path:
    #     from peft import PeftModel
    #     merged_model = PeftModel.from_pretrained(model, model_args.adapter_path)
    #     merged_model = merged_model.merge_and_unload()

    # return merged_model


    # # print(model)
    # adapter_path = '/exps/llama-3.2-1b-instruct__gsm8k_vien/checkpoints/checkpoint-1124'
    # from peft import PeftModel
    # finetuned_model = PeftModel.from_pretrained(model, adapter_path)
    # finetuned_model = finetuned_model.merge_and_unload()

    # # print(finetuned_model)

    # from src.utils.model_utils import print_trainable_parameters
    # print_trainable_parameters(finetuned_model)


    # model.save_pretrained("/exps/tmp/toy_ft_model")
    # finetuned_model = model
    return model

def load_model_for_generate_vllm(model_args, gen_args, device_args):    
    from vllm import LLM, SamplingParams
    # # Set torch_dtype and attn_implementation
    # torch_dtype, attn_implementation = set_torch_dtype_and_attn_implementation()
    torch_dtype = model_args['torch_dtype']

    # init model torch dtype
    if torch_dtype == "" or torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif torch_dtype == "float32":
        torch_dtype = torch.float32
    elif torch_dtype == "float16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Invalid torch dtype: {torch_dtype}")
    
    # Load model
    model = LLM(
        model=model_args['pretrained_model_name_or_path'],
        dtype=torch_dtype,
        max_tokens = gen_args['max_length ']

    )

    return model


def load_tokenizer_for_generate(
    model_args
) -> PreTrainedTokenizer:
    if model_args['pretrained_tokenizer_name_or_path']:
        tokenizer = AutoTokenizer.from_pretrained(model_args['pretrained_tokenizer_name_or_path'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args['pretrained_model_name_or_path'])
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer


def load_model_tokenizer_for_generate(model_args, device_args):
    model = load_model_for_generate(model_args, device_args)
    tokenizer = load_tokenizer_for_generate(model_args)

    model.resize_token_embeddings(len(tokenizer))

    # Load adapter and merge
    if model_args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_args.adapter_path)
        model = model.merge_and_unload()

        

    return model, tokenizer


# def prepare_prompt_for_generate(prompt_args):
#     if prompt_args['use_only_input_text']:
#         return prompt_args['input_text']

#     for key, value in prompt_args.items():
#         if not key.startswith('use') and key != 'end_key' and not value:
#             value = ''
#             prompt_args[key] = value

    
#     intro = f"{prompt_args['intro_text']}"
#     instruction = f"{prompt_args['instruction_key']}\n{prompt_args['instruction_text']}"

#     if not prompt_args['use_examples']:
#         examples = None
#     else:
#         example_template = prompt_args['examples_template']
#         formatted_examples = "\n".join(
#             example_template.format(**example) for example in prompt_args['examples_list']
#         )
#         examples = f"{prompt_args['examples_key']}\n{formatted_examples}"

#     input = f"{prompt_args['input_key']}\n{prompt_args['input_text']}"

#     if not prompt_args['use_context']:
#         context = None
#     else:
#         context = f"{prompt_args['context_key']}\n{prompt_args['context_text']}"

#     response_key = f"{prompt_args['response_key']}"
            
#     # Collect all non-empty parts and join them
#     parts = [part.strip() for part in [intro, instruction, examples, context, input, response_key] if part and len(part.strip())]

#     prompt = "\n\n".join(parts) + '\n'


#     return prompt

def generate_prompt_for_inference(prompt_args):
    """
    Generates a prompt for inference using only the necessary arguments.

    Args:
        prompt_args (dict): Dictionary containing prompt formatting options, including 'input_text'.

    Returns:
        str: The formatted prompt for inference.
    """
    use_context = prompt_args.get("use_context", False)
    use_response_format_guide = prompt_args.get("use_response_format_guide", False)

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
    prompt_parts["response_format_guide"] = (
        get_section("response_format_guide_key", "response_format_guide_text")
        if use_response_format_guide
        else None
    )
    prompt_parts["context"] = get_section("context_key", "context_text") if use_context else None

    # Build input section using 'input_text' from prompt_args
    input_text = prompt_args.get("input_text", "")
    input_parts = []
    if prompt_args.get("input_key"):
        input_parts.append(prompt_args.get("input_key"))
    input_parts.append(input_text)
    prompt_parts["input"] = "\n".join(filter(None, input_parts)).strip() or None

    prompt_parts["pre_response"] = prompt_args.get("pre_response_text")

    # Construct final prompt
    combined = [v for v in prompt_parts.values() if v]
    prompt = prompt_args.get("sep", "\n\n").join(combined)

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


def generate_response_vllm(model_args, prompt_args, gen_args, device_args):
    from vllm import LLM, SamplingParams
    model = load_model_for_generate_vllm(model_args, gen_args, device_args)
    prompt = generate_prompt_for_inference(prompt_args)

    gen_params = {
        "top_p": gen_args.gen_args['top_p'],
        "top_k": gen_args.gen_args['top_k'],
        "temperature": gen_args.gen_args['temperature']
    }

    sampling_params = SamplingParams(**gen_params)

    outputs = model.generate(prompts=prompt, sampling_params=sampling_params)[0]
    generated_text = outputs.outputs[0].text
    
    return generated_text


def generate_response(model_args, tokenizer_args, prompt_args, gen_args, device_args):


    from accelerate import Accelerator
    accelerator = Accelerator(cpu=device_args.use_cpu)
    
    # model = load_model_for_generate(model_args, device_args)
    
    # tokenizer = load_tokenizer_for_generate(model_args)

    # model.resize_token_embeddings(len(tokenizer))

    model, tokenizer = load_model_tokenizer_for_generate(model_args, device_args)

    prompt = generate_prompt_for_inference(prompt_args)

    # input = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
    input = tokenizer(prompt, 
                      **tokenizer_args.tokenizer_args
    )
    
    # print(input)
    input_ids = input.input_ids
    attention_mask = input.attention_mask

    # device = accelerator.device

    model = model.to(accelerator.device)
    # input_ids = input_ids.to(accelerator.device)
    # attention_mask = attention_mask.to(accelerator.device)

    input_ids = input_ids.squeeze(1).to(accelerator.device)
    attention_mask = attention_mask.squeeze(1).to(accelerator.device)

    output_ids = model.generate(input_ids=input_ids, 
                                attention_mask=attention_mask,
                                **gen_args.gen_args
    )
    
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
    parser.add_argument("--config-path", type=str, required=True, help="Path to the YAML config file for generating.")
    parser.add_argument("--input_text", type=str, default=None, help="The text input (question).")

    return parser.parse_args()

def main():
    args = parse_args()

    # Load the generation config file

    cfg = OmegaConf.load(args.config_path)

    # Setup environment
    setup_environment()


    # print(OmegaConf.to_yaml(cfg))

    model_args = cfg.model
    tokenizer_args = cfg.tokenizer
    prompt_args = cfg.prompt
    gen_args = cfg.generate
    device_args = cfg.device

    # Set seed
    set_seed(gen_args.seed)


    if args.input_text:
        prompt_args['input_text'] = args.input_text

    if gen_args.use_vllm:
        response = generate_response_vllm(model_args, tokenizer_args, prompt_args, gen_args, device_args)

    else:
        response = generate_response(model_args, tokenizer_args, prompt_args, gen_args, device_args)
    print(response)

if __name__ == "__main__":
    main()
    