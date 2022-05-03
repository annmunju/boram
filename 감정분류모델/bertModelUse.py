import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from transformers import BertTokenizerFast
from bertModelLoad import model_load, NewsSubjectClassifier, NewsSubjectDataset
from tqdm import tqdm

def model_use(data, model, bert_or_albert):

    if bert_or_albert == 'bert':
        model = model.eval()
        tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
    elif bert_or_albert == 'albert':
        model = model.eval()
        tokenizer = BertTokenizerFast.from_pretrained("kykim/albert-kor-base")

    ds = NewsSubjectDataset(
        subjects=data.to_numpy(),
        targets=np.zeros(len(data)),
        tokenizer=tokenizer,
        max_len=32
        )
    dl = DataLoader(ds)

    device = torch.device("cuda:0")
    
    subject_texts = []
    predictions = []
    prediction_probs = []
    with torch.no_grad():
        for d in tqdm(dl):
            texts = d["subject_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            subject_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()

    return subject_texts, predictions, prediction_probs
