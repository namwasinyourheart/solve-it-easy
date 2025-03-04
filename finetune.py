import os
import argparse
import time


import joblib
import wandb

from hydra import initialize, compose
from hydra.utils import instantiate

from omegaconf import OmegaConf

from prepare_data import show_dataset_examples
from prepare_data import prepare_data

from peft import (
    LoraConfig, 
    prepare_model_for_kbit_training,
    get_peft_model
)

from trl import (
    DataCollatorForCompletionOnlyLM, 
    SFTTrainer
)

from transformers import DataCollatorForLanguageModeling, set_seed
import torch

from src.utils.model_utils import get_model_tokenizer, get_peft_config
from src.utils.log_utils import setup_logger
from src.utils.exp_utils import setup_environment, create_exp_dir

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
    
    assert cfg.exp_manager.exp_name == os.path.basename(config_path).replace('.yaml', ''), \
    f"Config file name '{os.path.basename(config_path)}' does not match experiment name '{cfg.exp_manager.exp_name}' in the config."
    
    exp_args = cfg.exp_manager
    data_args = cfg.prepare_data
    model_args = cfg.prepare_model
    train_args = cfg.train
    eval_args = cfg.eval
    gen_args = cfg.generate
    device_args = cfg.device

    if exp_args.print_cfg:
        print(OmegaConf.to_yaml(cfg))

    return cfg, exp_args, data_args, model_args, train_args, eval_args, gen_args, device_args

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Load generation config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML config file for generating.")

    args, override_args = parser.parse_known_args()
    return args, override_args

from src.metrics import * #compute_metrics_wrapper

def main():
    # Setup logging
    logger = setup_logger()

    # Setup environment
    logger.info("SETTING UP ENVIRONMENT...")
    setup_environment()
    
    # Parse arguments
    args, override_args = parse_args()

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

    OmegaConf.save(cfg, os.path.join(exp_dir, exp_name)+'.yaml')


    exp_args = cfg.exp_manager
    train_args = cfg.train
    data_args = cfg.prepare_data
    model_args = cfg.prepare_model
    device_args = cfg.device

    # Set seed
    set_seed(exp_args.seed)

    if data_args.dataset.is_prepared:
        # Get the path to the prepared data
        prepared_data_path = os.path.normpath(data_args.dataset.prepared_data_path)
        
        # Check if the processed data exists
        if not os.path.isfile(prepared_data_path):
            raise FileNotFoundError(f"Processed data not found at: {prepared_data_path}")
        
        # Load the dataset
        logger.info("LOADING PREPARED DATASET...")
        dataset = joblib.load(prepared_data_path)
    else:
        # Prepare dataset
        logger.info("PREPARING DATASET...")
        dataset, prepared_data_path = prepare_data(exp_args, data_args, model_args)

    # Show dataset examples
    if data_args.dataset.do_show:
        show_dataset_examples(dataset)

    # Set seed before initializing model.
    set_seed(exp_args.seed)

    # LOADING MODEL
    logger.info("LOADING MODEL AND TOKENIZER...")
    model, tokenizer = get_model_tokenizer(data_args, model_args, device_args)

    if exp_args.print_model:
        print(model)
    # print(tokenizer)


    # PREPARE MODEL
    # Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = get_peft_config(train_args)
    print(peft_config)

    if train_args.use_peft:
        model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    try:
        model.print_trainable_parameters()
    except:
        from src.utils.model_utils import print_trainable_parameters
        print_trainable_parameters(model)

    data_collator = DataCollatorForLanguageModeling(
            # response_template=RESPONSE_KEY,
            tokenizer=tokenizer, 
            mlm=False, 
            return_tensors="pt", 
            # pad_to_multiple_of=cfg.data.pad_to_multiple_of
        )
    
    # if exp_args.wandb.use_wandb:
    wandb.init(
        project=cfg.exp_manager.wandb.project,
        # name = cfg.exp_manager.exp_name
    )

    current_date = time.strftime("%Y%m%d_%H%M%S")
    training_args = instantiate(train_args.train_args, 
                                    output_dir=exp_checkpoints_dir, 
                                    report_to="none",
                                    # run_name=wandb.run.name
                        )

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if data_args.dataset.do_split:
        train_ds, val_ds, test_ds = dataset['train'], dataset['val'], dataset['test']
    else:
        train_ds = dataset['train']
        val_ds = test_ds = train_ds

    def get_data_subset(n_samples, dataset, seed):
        if n_samples == -1:
            subset = dataset
        else:
            subset = dataset.shuffle(seed=seed)
            subset = subset.select(range(n_samples))

        return subset

    if train_args.train_n_samples:
        # train_ds = train_ds.shuffle(seed=exp_args.seed).select(range(train_args.train_n_samples))
        train_ds = get_data_subset(train_args.train_n_samples, train_ds, exp_args.seed)

    if train_args.val_n_samples:
        val_ds = get_data_subset(train_args.val_n_samples, val_ds, exp_args.seed)

    if train_args.test_n_samples:
        test_ds = get_data_subset(train_args.test_n_samples, test_ds, exp_args.seed)

    # from src.metrics import * #compute_metrics_wrapper

    trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics_wrapper(tokenizer, train_args.eval_metrics),
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    # print(training_args)

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

    from src.callbacks import WandbPredictionProgressCallback, decode_predictions

    progress_callback = WandbPredictionProgressCallback(
        trainer=trainer,
        tokenizer=tokenizer,
        val_dataset=val_ds,
        num_samples=10,
        freq=1,
    )

    all_metrics = {"run_name": wandb.run.name}
    
    # TRAINING
    if training_args.do_train:
        logger.info("TRAINING...")
        
        # Add the callback to the trainer
        trainer.add_callback(progress_callback)
        logger.info('trainer_callback_list: %s', trainer.callback_handler.callbacks)

        if training_args.resume_from_checkpoint:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint:
        #     checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        else:
            train_result = trainer.train()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        all_metrics.update(metrics)


        # Save model
        logger.info("SAVING MODEL...")
        trainer.model.save_pretrained(os.path.join(exp_results_dir, 'adapter'))
        logger.info(f"Model saved to {os.path.join(exp_results_dir, 'adapter')}"),

        logger.info("TRAINING COMPLETED.")


    # EVALUATION
    if training_args.do_eval:
        logger.info("EVALUATING...")
        metrics = trainer.evaluate(
            eval_dataset=val_ds, 
            metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)


    import pandas as pd
    # PREDICTION
    if training_args.do_predict:
        logger.info("PREDICTING...")
        predictions = trainer.predict(
            test_dataset=test_ds,
            metric_key_prefix='test'
        )
        metrics = predictions.metrics
        all_metrics.update(metrics)
        
        # logger.info(f'predictions: {predictions}')
        # print("DECODING PREDICTION...")
        predictions = decode_predictions(tokenizer, predictions)
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(os.path.join(exp_results_dir, 'test_predictions.csv'), index=False)
    
    import json
    
    if (training_args.do_train or training_args.do_eval or training_args.do_predict):
            with open(os.path.join(exp_results_dir, "metrics.json"), "w") as fout:
                fout.write(json.dumps(all_metrics))


     # Merge model
    if training_args.do_train == True and train_args.do_merge == True:
        logger.info("MERGING MODEL...")
        # adapter_dir = os.path.join(exp_dir, 'results')
        # os.makedirs(adapter_dir, exist_ok=True)


        adapter_path = os.path.join(exp_results_dir, 'adapter')
        base_model, tokenizer = get_model_tokenizer(data_args, model_args, device_args)

        from peft import PeftModel
        finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
        finetuned_model = finetuned_model.merge_and_unload()
        
        # Save merged model
        finetuned_model.save_pretrained(os.path.join(exp_results_dir, 'merged_model'))

        # Save tokenizer
        tokenizer.save_pretrained(os.path.join(exp_results_dir, 'merged_model'))

    
    # Log exp artifact
    if exp_args.wandb.log_artifact == True:
        logger.info("LOGGING EXP ARTIFACTS...")
        # Create an artifact
        artifact = wandb.Artifact(
            name=exp_args.exp_name, 
            type="exp", 
            # description="Dummy dataset with CSV files"
        )

        # Add the directory to the artifact
        artifact.add_dir(exp_dir)

        wandb.log_artifact(artifact)

    # Finish the W&B run
    wandb.finish()
    

if __name__ == "__main__":
    main()