import os
import argparse
import time
import re
import json
import pandas as pd

import joblib
import wandb

from hydra import initialize, compose
from hydra.utils import instantiate

from omegaconf import OmegaConf

from peft import (
    LoraConfig, 
    prepare_model_for_kbit_training,
    get_peft_model
)

from trl import (
    DataCollatorForCompletionOnlyLM, 
    SFTTrainer
)

from transformers import DataCollatorForLanguageModeling, set_seed, Seq2SeqTrainer, Trainer, DataCollatorForSeq2Seq

from src.utils.hieralog import hprint, pprint, fprint
from prepare_data import prepare_data
from src.utils.model_utils import get_model_tokenizer, get_peft_config
from src.utils.exp_utils import setup_environment, create_exp_dir
from src.metrics import compute_metrics_wrapper

from src.utils.data_utils import get_data_subset

from src.callbacks import ProgressCallback, decode_predictions_wrapper


import warnings
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

    return (
        cfg, 
        exp_args, 
        data_args, 
        tokenizer_args, 
        prompt_args, 
        model_args, 
        train_args, 
        eval_args, 
        gen_args, 
        device_args
    )


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


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Load generation config.")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the YAML config file for generating.")

    args, override_args = parser.parse_known_args()
    return args, override_args

def finetune(
    model, tokenizer, 
    train_ds, val_ds, test_ds,
    exp_args, data_args, tokenizer_args, prompt_args, 
    model_args, train_args, eval_args, gen_args, device_args, 
    exp_variant_dir, exp_variant_data_dir, exp_variant_checkpoints_dir, exp_variant_results_dir
):
    """
    Trains and evaluates the provided model using the specified datasets and configurations.

    This function fine-tunes a preloaded model on a given training dataset, evaluates it on 
    a validation set, and optionally performs predictions on a test set. It supports various 
    configurations for model training, evaluation, and generation while handling logging, 
    checkpointing, and result saving.

    Args:
        model: The preloaded model to be fine-tuned.
        tokenizer: The tokenizer associated with the model.
        train_ds: The dataset used for training.
        val_ds: The dataset used for validation.
        test_ds: The dataset used for testing (if applicable).
        exp_args: Experiment-related arguments and configurations.
        data_args: Parameters related to dataset processing and handling.
        tokenizer_args: Arguments for tokenizer configuration.
        prompt_args: Configuration for prompt engineering or formatting.
        model_args: Model-specific settings and hyperparameters.
        train_args: Training-related parameters, including optimizer settings.
        eval_args: Evaluation-related configurations and metrics.
        gen_args: Arguments for text generation (if applicable).
        device_args: Hardware-related configurations (e.g., GPU, TPU settings).
        exp_variant_dir: Root directory for experiment outputs.
        exp_variant_data_dir: Directory for storing dataset files.
        exp_variant_checkpoints_dir: Path to save model checkpoints.
        exp_variant_results_dir: Directory for saving evaluation results and predictions.
    """

    

    if train_args.use_peft:
        # Prepare model for training
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        peft_config = get_peft_config(train_args)
        if exp_args.print_peft_config:
            pprint(peft_config)
        model = get_peft_model(model, peft_config)

    # Print trainable parameters
    if exp_args.print_trainable_parameters:
        try:
            model.print_trainable_parameters()
        except:
            from src.utils.model_utils import print_trainable_parameters
            print_trainable_parameters(model)

    # Print parameter datatypes
    if exp_args.print_parameter_datatypes:
        from src.utils.model_utils import print_parameter_datatypes
        print_parameter_datatypes(model)

    

    # Initialize W&B
    wandb.init(
        project=exp_args.wandb.project,
    )

    # Training arguments
    training_args = instantiate(
        train_args.train_args, 
        output_dir=exp_variant_checkpoints_dir, 
        report_to="wandb",
        run_name=wandb.run.name
    )

    # Log device info if required
    if exp_args.print_device:
        pprint(
            f"Process Rank: {training_args.local_rank}, Device: {training_args.device}, N_GPU: {training_args.n_gpu}, "
            f"Distributed Training: {bool(training_args.local_rank != -1)}"
        )
    if model_args.model_type == "CAUSAL_LM":

        # Setup Data Collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False, 
            return_tensors="pt"
        )

        # Initialize Trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics_wrapper(tokenizer, train_args.eval_metrics, model_args.model_type),
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    if model_args.model_type == "SEQ_2_SEQ_LM":
        
        # Setup Data Collator
        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            # pad_to_multiple_of=8
        )
            
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics_wrapper(tokenizer, train_args.eval_metrics, model_args.model_type),
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    model.config.use_cache = False  # re-enable for inference

    # Setup training progress callback
    progress_callback = ProgressCallback(
        trainer=trainer,
        tokenizer=tokenizer,
        dataset=val_ds,
        **train_args.progress_callback
    )

    all_metrics = {
        "exp_name": exp_args.exp_name,
        "exp_variant": exp_args.exp_variant,
        "run_name": wandb.run.name
    }
        

    # Training
    if training_args.do_train:
        hprint("1: Training...")
        
        trainer.add_callback(progress_callback)

        if train_args.do_resume_from_checkpoint and training_args.resume_from_checkpoint:
            checkpoint = training_args.resume_from_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        else:
            train_result = trainer.train()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        all_metrics.update(metrics)

        # Save trained adapter/models
        if train_args.use_peft:
            hprint("1: Saving adapter...")
            trainer.model.save_pretrained(os.path.join(exp_variant_results_dir, 'adapter'))
            pprint(f"2: Model saved to {os.path.join(exp_variant_results_dir, 'adapter')}")

        else:
            hprint("1: Saving finetuned model and tokenizer...")
            # Saving model
            trainer.model.save_pretrained(os.path.join(exp_variant_results_dir, 'finetuned_model'))
            
            # Save tokenizer
            tokenizer.save_pretrained(os.path.join(exp_variant_results_dir, 'finetuned_model'))
            pprint(f"2: Finetuned model and tokenizer saved to {os.path.join(exp_variant_results_dir, 'finetuned_model')}")

    # Evaluation
    if training_args.do_eval:
        hprint("1: Evaluating...")
        metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    # Prediction
    if training_args.do_predict:
        # from src.callbacks import decode_predictions_wrapper
        decode_predictions = decode_predictions_wrapper(model_args.model_type)

        # print("decode_predictions:", decode_predictions)
        
        hprint("1: Predicting...")
        predictions = trainer.predict(test_dataset=test_ds, metric_key_prefix='test')
        metrics = predictions.metrics
        all_metrics.update(metrics)
        
        predictions = decode_predictions(tokenizer, predictions)
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(os.path.join(exp_variant_results_dir, f'{exp_args.prefix_fn}_test_predictions.csv'), index=False)

    # Save metrics
    hprint("1: Saving metrics...")
    if training_args.do_train or training_args.do_eval or training_args.do_predict:
        all_metrics_fp = os.path.join(exp_variant_results_dir, f'{exp_args.prefix_fn}_{train_args.train_metric_filename}')
        with open(all_metrics_fp, "w") as fout:
            fout.write(json.dumps(all_metrics))

    # Merge model if required
    if train_args.use_peft:
        if training_args.do_train and train_args.merge_after_train:
            hprint("1: Merging model...")
            adapter_path = os.path.join(exp_variant_results_dir, 'adapter')
            base_model, tokenizer = get_model_tokenizer(model_args, tokenizer_args, prompt_args, device_args)

            from peft import PeftModel
            finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
            finetuned_model = finetuned_model.merge_and_unload()
            
            hprint("1: Saving merged model and tokenizer...")
            # Save merged model
            finetuned_model.save_pretrained(os.path.join(exp_variant_results_dir, 'merged_model'))

            # Save tokenizer
            tokenizer.save_pretrained(os.path.join(exp_variant_results_dir, 'merged_model'))
            pprint(f"2: Merged model and tokenizer saved to {os.path.join(exp_variant_results_dir, 'merged_model')}")



    # Log experiment artifact
    if exp_args.wandb.log_artifact:
        hprint("1: Logging artifact...")
        artifact = wandb.Artifact(
            name=exp_args.exp_name, 
            type="exp", 
        )

        artifact.add_dir(exp_variant_dir)
        wandb.log_artifact(artifact)

    wandb.finish()



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
        
    config_path = os.path.join(exp_variant_dir, 'sft_' + prefix_fn + '_' + exp_name + '.yaml')
    save_cfg(cfg, config_path)
    pprint(f"2: Configuration saved to {config_path}")


    # Set seed
    set_seed(exp_args.seed)

    #Load dataset
    if data_args.is_prepared:
        hprint("1: Data is prepared, loading...")
        prepared_data_path = data_args.prepared_data_path
        dataset = joblib.load(prepared_data_path)
    else:
        hprint("1: Preparing data...")
        dataset = prepare_data(exp_args, data_args, tokenizer_args, prompt_args, model_args)

        train_ds, val_ds, test_ds = dataset['train'], dataset['val'], dataset['test']

    if train_args.train_n_samples:
        train_ds = get_data_subset(train_args.train_n_samples, train_ds, exp_args.seed)

    if train_args.val_n_samples:
        val_ds = get_data_subset(train_args.val_n_samples, val_ds, exp_args.seed)

    if train_args.test_n_samples:
        test_ds = get_data_subset(train_args.test_n_samples, test_ds, exp_args.seed)

    # Loading model and tokenizer
    hprint("1: Loading model and tokenizer...")
    model, tokenizer = get_model_tokenizer(model_args, tokenizer_args, prompt_args, device_args)

    if exp_args.print_model:
        pprint(model)

    if exp_args.print_tokenizer:
        pprint(tokenizer)
   
    finetune(
        model, tokenizer, 
        train_ds, val_ds, test_ds,
        exp_args, data_args, tokenizer_args, prompt_args, 
        model_args, train_args, eval_args, gen_args, device_args, 
        exp_variant_dir, exp_variant_data_dir, exp_variant_checkpoints_dir, exp_variant_results_dir
    )
        

if __name__ == "__main__":
    main()