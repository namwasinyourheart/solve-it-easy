import numpy as np
import pandas as pd
import wandb

from transformers.integrations import WandbCallback

def decode_predictions_wrapper(model_type):

    def decode_predictions_causal(tokenizer, predictions):
        label_ids = predictions.label_ids
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        pred_ids =  predictions.predictions
        pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
        pred_ids = pred_ids.argmax(axis=-1)
        predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        return {"prediction": predictions, "label": labels}
    
    def decode_predictions_seq_2_seq(tokenizer, predictions):
        # print("predictions:", predictions)

        label_ids = predictions.label_ids
        pred_ids =  predictions.predictions

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
        pred_ids = np.argmax(pred_ids, axis=-1) 
        predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        return {"prediction": predictions, "label": labels}
        
    if model_type=="CAUSAL_LM":
        return decode_predictions_causal
    
    if model_type=="SEQ_2_SEQ_LM":
        return decode_predictions_seq_2_seq
    


def decode_predictions(tokenizer, predictions):
    label_ids = predictions.label_ids
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    pred_ids =  predictions.predictions
    pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
    pred_ids = pred_ids.argmax(axis=-1)
    predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return {"prediction": predictions, "label": labels}



class ProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    Logs model predictions at defined intervals during training, allowing 
    visualization of model progress over time.
    """

    def __init__(self, trainer, tokenizer, dataset, model_type,
                 n_samples=100, logging_strategy='step', 
                 interval_epoch=1, interval_step=100):
        """
        Initializes ProgressCallback.

        Args:
            trainer (Trainer): Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): Tokenizer for decoding predictions.
            val_dataset (Dataset): Validation dataset.
            n_samples (int, optional): Number of samples for predictions. Defaults to 100.
            logging_strategy (str, optional): 'step' or 'epoch' for logging frequency. Defaults to 'step'.
            interval_epoch (int, optional): Log predictions every N epochs. Defaults to 1.
            interval_step (int, optional): Log predictions every N steps. Defaults to 100.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.dataset = dataset.select(range(n_samples))
        self.model_type = model_type
        self.logging_strategy = logging_strategy
        self.interval_epoch = interval_epoch
        self.interval_step = interval_step

    def log_predictions(self, predictions, key):
        """Logs model predictions to wandb."""

        decode_predictions = decode_predictions_wrapper(self.model_type)

        predictions_df = pd.DataFrame(decode_predictions(self.tokenizer, predictions))
        predictions_df[key] = getattr(self.trainer.state, key, None)
        wandb.log({"progress_predictions": wandb.Table(dataframe=predictions_df)})

    def on_evaluate(self, args, state, control, **kwargs):
        """Logs predictions at specified evaluation intervals."""
        super().on_evaluate(args, state, control, **kwargs)

        if self.logging_strategy == 'epoch' and state.epoch % self.interval_epoch == 0:
            self.log_predictions(self.trainer.predict(self.dataset), key="epoch")

        elif self.logging_strategy == 'step' and state.global_step % self.interval_step == 0:
            self.log_predictions(self.trainer.predict(self.dataset), key="global_step")


