from torch import nn
import pandas as pd
import numpy as np

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast, AlbertModel

class NewsSubjectClassifier(nn.Module):
  def __init__(self, n_classes):
    super(NewsSubjectClassifier, self).__init__()
    self.bert = AlbertModel.from_pretrained("kykim/albert-kor-base")
    self.drop = nn.Dropout(p=0.5)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict=False
    )
    output = self.drop(pooled_output)
    return self.out(output)
  
class NewsSubjectDataset(Dataset):
  def __init__(self, subjects, targets, tokenizer, max_len):
    self.subjects = subjects
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  def __len__(self):
    return len(self.subjects)
  def __getitem__(self, item):
    subject = str(self.subjects[item])
    target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
      subject,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding = 'max_length',
      truncation = True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return {
      'subject_text': subject,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

def model_load(model_location, bert_or_albert):

  if bert_or_albert == 'bert':
    tokenizer_albert_kor_base = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
  elif bert_or_albert == 'albert':
    tokenizer_albert_kor_base = BertTokenizerFast.from_pretrained("kykim/albert-kor-base")
  else:
    print('실행 불가')
    return



  device = torch.device("cuda:0")


  model = torch.load(model_location)

  return model