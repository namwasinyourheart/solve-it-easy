from typing import List, Dict

import evaluate
import nltk
import numpy as np
import re
import torch

from nltk.translate.bleu_score import sentence_bleu

nltk.download('punkt_tab')


def calc_bleu(predictions: List[str], references: List[str]) -> dict:
    bleu = evaluate.load('bleu')
    bleu_score = bleu.compute(predictions=predictions, references=references)

    return {'bleu': bleu_score['bleu']}

def _postprocess_text(text: str) -> str:
    # rougeLSum expects newline after each sentence
    return "\n".join(nltk.sent_tokenize(text.strip()))


def calc_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    rouge = evaluate.load("rouge")
    predictions = [_postprocess_text(text) for text in predictions]
    references = [_postprocess_text(text) for text in references]

    rouge_score =  rouge.compute(predictions=predictions, references=references, use_stemmer=True)

    return rouge_score

def calc_accuracy(predictions: List[str], references: List[str]):
    accuracy_metric = evaluate.load("accuracy")

    accuracy_score = accuracy_metric.compute(predictions=predictions, references=references)

    return accuracy_score


def extract_prediction(response, min_value=np.iinfo(np.int32).min, max_value=np.iinfo(np.int32).max, return_error: str="-123"):
    try:
        matches = re.findall(r'Solution:.*?Reason:.*Answer:.*?([\d,]+)', response, re.DOTALL)
        if matches:
            answer =  matches[0]
            answer = answer.replace(',', '')

            if min_value <= int(float(answer)) <= max_value: 
                # print('answer:', answer)
                return answer
            
            else:
                return return_error
        else:
            answer = return_error
        return answer
    except Exception:
        return return_error

from nltk.translate.bleu_score import sentence_bleu
def calc_nltk_bleu(predictions, references):
    nltk_bleu_scores = [
        sentence_bleu([reference], prediction) for prediction, reference in zip(predictions, references)
    ]
    nltk_bleu = sum(nltk_bleu_scores) / len(nltk_bleu_scores) if nltk_bleu_scores else 0.0
    return {"nltk_bleu": nltk_bleu}



def compute_metrics_wrapper(tokenizer, eval_metrics, model_type):

    def compute_metrics_seq2seq(eval_preds):
        results = {}
        if not eval_metrics:
            return results

        if not len(eval_metrics):
            return results
        
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.argmax(preds, axis=-1) 
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        predictions = decoded_preds
        labels = decoded_labels


        if 'bleu' in eval_metrics:
            # Calculate BLEU score using evaluate
            bleu_score = calc_bleu(predictions, labels)
            results.update(bleu_score)

        if 'nltk_bleu' in eval_metrics:
            # Calculate BLEU scores using NLTK
            nltk_bleu_score = calc_nltk_bleu(predictions, labels)
            results.update(nltk_bleu_score)

        if 'rouge' in eval_metrics:
            # Calculate ROUGE score
            rouge_score = calc_rouge(predictions, labels)
            results.update(rouge_score)

        if 'accuracy' in eval_metrics:
            # Calculate accuracy
            predictions = [extract_prediction(prediction, return_error='-1') for prediction in predictions]
            labels = [extract_prediction(label, return_error='-123') for label in labels]
            accuracy_score = calc_accuracy(predictions, labels)
            results.update(accuracy_score)  
    
        return results

    def compute_metrics_causal(predictions):
        results = {}
        if not eval_metrics:
            return results

        if not len(eval_metrics):
            return results
            
        label_ids = predictions.label_ids
        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        pred_ids = predictions.predictions
        pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
        pred_ids = pred_ids.argmax(axis=-1)
        predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    
        if 'bleu' in eval_metrics:
            # Calculate BLEU score using evaluate
            bleu_score = calc_bleu(predictions, labels)
            results.update(bleu_score)

        if 'nltk_bleu' in eval_metrics:
            # Calculate BLEU scores using NLTK
            nltk_bleu_score = calc_nltk_bleu(predictions, labels)
            results.update(nltk_bleu_score)

        if 'rouge' in eval_metrics:
            # Calculate ROUGE score
            rouge_score = calc_rouge(predictions, labels)
            results.update(rouge_score)

        if 'accuracy' in eval_metrics:
            # Calculate accuracy
            predictions = [extract_prediction(prediction, return_error='-1') for prediction in predictions]
            labels = [extract_prediction(label, return_error='-123') for label in labels]
            accuracy_score = calc_accuracy(predictions, labels)
            results.update(accuracy_score)  
    
        return results
    
    if model_type == "SEQ_2_SEQ_LM":
        return compute_metrics_seq2seq

    if model_type == "CAUSAL_LM":
        return compute_metrics_causal
