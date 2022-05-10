import os
import re
import numpy as np
from glob import glob
import json
import requests
import tensorflow as tf
from transformers import BertModel, TFBertModel, TFRobertaModel, RobertaTokenizer, BertTokenizerFast, AlbertModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
from adabelief_pytorch import AdaBelief
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm, tqdm_notebook
import shutil
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tqdm.notebook import tqdm
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
import pymysql

L_RATE = 1e-5
MAX_LEN = 45
max_grad_norm=1
log_interval=200
NUM_CORES = os.cpu_count()
device = torch.device("cuda:0")

class TestDataset(Dataset):
    def __init__(self, df,tokenizer):
        self.df_data = df
        self.tokenizer = tokenizer
    def __getitem__(self, index):
        # get the sentence from the dataframe
        sentence = self.df_data.loc[index, 'data']
        encoded_dict = self.tokenizer(
          text = sentence,
          add_special_tokens = True, 
          max_length = MAX_LEN,
          pad_to_max_length = True,
          truncation=True,           # Pad & truncate all sentences.
          return_tensors="pt")

        padded_token_list = encoded_dict['input_ids'][0]
        token_type_id = encoded_dict['token_type_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        sample = (padded_token_list, token_type_id , att_mask)
        return sample
    def __len__(self):
        return len(self.df_data)

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


def model_test(data, model, model_type):#데이터 모델 모델타입순 모델타입으론 roberta, electra 사용
    preds = [] 
    if model_type == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large", cache_dir='bert_ckpt', do_lower_case=False)
    elif model_type == 'electra':   
        tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator", cache_dir='bert_ckpt', do_lower_case=False)
    else:
        print("error")
        return 0
    test_data = TestDataset(data,tokenizer)
    test_dataloader = torch.utils.data.DataLoader(test_data,shuffle=False,num_workers=NUM_CORES)
    model.eval()
    torch.set_grad_enabled(False)
    for batch_id, (input_id,token_type_id,attention_mask) in enumerate(tqdm(test_dataloader)):
        input_id = input_id.long().to(device)
        token_type_id = token_type_id.long().to(device)
        attention_mask = attention_mask.long().to(device)
        outputs = model(input_ids=input_id, token_type_ids=token_type_id, attention_mask=attention_mask)
        out = outputs[0]
        for inp in out:
            preds.append(inp.detach().cpu().numpy())
    Preds = np.array(preds)
    return Preds

def model_use(data, model, bert_or_albert):
    model.eval()
    if bert_or_albert == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
    elif bert_or_albert == 'albert':
        tokenizer = BertTokenizerFast.from_pretrained("kykim/albert-kor-base")
    else:
        print("error")
        return 0
    
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

    return prediction_probs

#모델생성
model_location = "./model/"
model_albert = model_load(model_location + 'model_albert_kor_base.pth', 'albert')
model_bert = model_load(model_location + 'model_bert_kor_base.pth', 'bert').cuda()
model_Roberta = torch.load(model_location+'Roberta_large_329.pt')
model_KoELECTRA_113 = torch.load(model_location+'KoELECTRA_113.pt')
model_Roberta_328 = torch.load(model_location+'model_roberta_328.pth')

# 데이터 입력
conn=pymysql.connect(host='34.64.181.43', user='root', password='1234', db='novelmusic')

sql_music="""select * from music"""
song_row = pd.read_sql_query(sql_music, conn)
song = song_row[['lyrics']]
song.rename({'lyrics':'data'}, axis=1, inplace=True)

sql_novel="""select * from novel fields terminated by '\t'"""
novel = pd.read_sql_query(sql_novel, conn)
story = novel[['story', 'review', 'piece']]
story.fillna('', inplace=True)
story['data'] = story['story'] + story['review'] + story['piece']
story.drop(['story', 'review', 'piece'], axis=1, inplace=True)

sql_color="""select * from color"""
color_raw = pd.read_sql_query(sql_color, conn)
color_raw['data'] = color_raw['cbaAlikeWord'] + [',' for _ in range(color_raw.shape[0])] + color_raw['cbaKeyword'] 
color = color_raw[['data']]

albert_lyrics = model_use(song, model_albert, bert_or_albert='albert')
bert_lyrics = model_use(song, model_bert, bert_or_albert = 'bert')
roberta_329_lyrics = model_test(song, model_Roberta,'roberta')
electra_lyrics = model_test(song, model_KoELECTRA_113,'electra')
roberta_328_lyrics = model_test(song, model_Roberta_328,'roberta')

total_lyrics = 0.3 * albert_lyrics + 0.075 * bert_lyrics + 0.425 * roberta_329_lyrics + 0.05 * electra_lyrics + 0.15 * roberta_328_lyrics

albert_story = model_use(story, model_albert, bert_or_albert='albert')
bert_story = model_use(story, model_bert, bert_or_albert = 'bert')
roberta_329_story = model_test(story, model_Roberta,'roberta')
electra_story = model_test(story, model_KoELECTRA_113,'electra')
roberta_328_story = model_test(story, model_Roberta_328,'roberta')

total_story = 0.3 * albert_story + 0.075 * bert_story + 0.425 * roberta_329_story + 0.05 * electra_story + 0.15 * roberta_328_story

albert_color = model_use(color, model_albert, bert_or_albert='albert')
bert_color = model_use(color, model_bert, bert_or_albert = 'bert')
roberta_329_color = model_test(color, model_Roberta,'roberta')
electra_color = model_test(color, model_KoELECTRA_113,'electra')
roberta_328_color = model_test(color, model_Roberta_328,'roberta')

total_color = 0.3 * albert_color + 0.075 * bert_color + 0.425 * roberta_329_color + 0.05 * electra_color + 0.15 * roberta_328_color

lyrics_ls = []
for lyrics in total_lyrics:
  lyrics_ls.append(torch.Tensor(lyrics))
song_row['vector'] = lyrics_ls

story_ls = []
for s in total_story:
  story_ls.append(torch.Tensor(s))
novel['vector'] = story_ls

color_ls = []
for c in total_color:
  color_ls.append(torch.Tensor(c))
color_raw['vector'] = color_ls

cursor = conn.cursor()
for i,v in zip(song_row['id'], song_row['vector']):
    sql_update = f'update music set vector = {v} where id = {i}'
    cursor.execute(sql_update)
    
conn.commit()

for i,v in zip(novel['id'], novel['vector']):
    sql_update = f'update novel set vector = {v} where id = {i}'
    cursor.execute(sql_update)
    
conn.commit()

for i,v in zip(color_raw['id'], color_raw['vector']):
    sql_update = f'update color set vector = {v} where id = {i}'
    cursor.execute(sql_update)
    
conn.commit()

conn.close()