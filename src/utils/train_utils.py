from peft import PeftModel
import os

from src.utils.model_utils import get_model_tokenizer

def merge_model(data_args, model_args, device_args, adapter_path, save_path):
    """
    Loads base model and tokenizer, merges the adapter, and saves the merged model.

    Args:
        data_args: Arguments related to data processing.
        model_args: Arguments related to model configuration.
        device_args: Arguments related to device settings.
        adapter_path (str): Path to the adapter to be merged.
        save_path (str): Directory to save the merged model and tokenizer.
    """
    
    # Load base model and tokenizer
    base_model, tokenizer = get_model_tokenizer(data_args, model_args, device_args)

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Load fine-tuned model with adapter
    finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge adapter into base model
    merged_model = finetuned_model.merge_and_unload()
    
    # Save merged model and tokenizer
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


    return merged_model
