import importlib 
from typing import Tuple
import torch

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    PreTrainedTokenizer, 
    PreTrainedModel, 
    AutoModel,
    AutoConfig,
    BitsAndBytesConfig
)
     


def load_tokenizer(data_args, model_args,
                  # padding_side
) -> PreTrainedTokenizer:
    tokenizer =  AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)
    if not tokenizer.pad_token:
        
        if data_args.tokenizer.new_pad_token:
            tokenizer.padding_side = 'left'
            tokenizer.pad_token = data_args.tokenizer.new_pad_token,
            tokenizer.add_special_tokens({"pad_token": data_args.tokenizer.new_pad_token})
        else:
            tokenizer.padding_side = 'left'
            tokenizer.pad_token = tokenizer.eos_token
    
    if data_args.tokenizer.add_special_tokens:
        additional_special_tokens = [
            data_args.prompt.instruction_key, 
            data_args.prompt.input_key,  
            data_args.prompt.response_key
        ]
        
        if data_args.prompt.context_key:
            additional_special_tokens = additional_special_tokens.append(data_args.prompt.context_key)
            
        tokenizer.add_special_tokens({
            "additional_special_tokens": additional_special_tokens
        })

    tokenizer.padding_side = 'left'
                
    return tokenizer

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


def load_model(model_args, device_args) -> PreTrainedModel:

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


def get_model_tokenizer(
    data_args, model_args, device_args
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(data_args, model_args)
    model = load_model(model_args, device_args)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def set_torch_dtype_and_attn_implementation():
    # Set torch dtype and attention implementation
    try:
        if torch.cuda.get_device_capability()[0] >= 8:
            # !pip install -qqq flash-attn
            torch_dtype = torch.bfloat16
            attn_implementation = "flash_attention_2"
        else:
            torch_dtype = torch.float16
            attn_implementation = "eager"
    except:
        torch_dtype = torch.float16
        attn_implementation = "eager"

    return torch_dtype, attn_implementation



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

def get_max_length(model):
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length



def find_target_modules(model) -> list[str]:
    # Initialize a Set to Store Unique Layers
    unique_layers = set()

    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]

            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)


from peft import LoraConfig, PeftConfig
# def get_peft_config(model_args: ModelArguments) -> PeftConfig | None:
def get_peft_config(train_args) -> PeftConfig | None:
    if train_args.use_peft is False:
        return None

    peft_config = LoraConfig(
        r=train_args.lora.r,
        lora_alpha=train_args.lora.lora_alpha,
        lora_dropout=train_args.lora.lora_dropout,
        bias=train_args.lora.bias,
        task_type=train_args.lora.task_type,
        target_modules=list(train_args.lora.target_modules),
        modules_to_save=train_args.lora.modules_to_save,
    )

    return peft_config

# https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def print_parameter_datatypes(model, logger=None):
    dtypes = dict()
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    
    total = 0
    for k, v in dtypes.items(): total += v

    for k, v in dtypes.items():

        if logger is None:
            print(f'type: {k} || num: {v} || {round(v/total, 3)}')
        else:
            logger.info(f'type: {k} || num: {v} || {round(v/total, 3)}')

def param_count(model):
    params = sum([p.numel() for p in model.parameters()])/1_000_000
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])/1_000_000
    print(f"Total params: {params:.2f}M, Trainable: {trainable_params:.2f}M")
    return params, trainable_params

from pathlib import Path
import os
from transformers.trainer_utils import get_last_checkpoint

def get_checkpoint(training_args) -> Path | None:
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint