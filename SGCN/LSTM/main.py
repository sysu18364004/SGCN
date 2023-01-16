import torch
from transformers import BertConfig,BertModel,BertTokenizer
import os
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from tqdm import tqdm 
import pandas as pd
from data import *
from train import *
from model import *
import json
train_file = '/home/zengdl/project/multi_label/rmsc/rmsc.train.tsv'
val_file = '/home/zengdl/project/multi_label/rmsc/rmsc.val.tsv'
save_path = 'best_embeding_rmsc'
dict_file = "dict_rmsc.json"
n_class = 54
if __name__ == '__main__':
    devices = torch.device('cuda:2')
    sentences,labels = load_tsv(train_file)
    wordid_map = build_dict(sentences)
    tokenizier = Tokenizier(wordid_map)
    with open(dict_file,'w',encoding='utf-8') as f:
        json.dump(wordid_map,f,indent=4,ensure_ascii=False)
    # print(len(wordid_map)
    train_dataset = LSTMDataset(train_file,tokenizier,rate=1.0)
    train_dataloader = DataLoader(train_dataset,collate_fn=train_dataset.collate_fn,batch_size=64,shuffle=True)
    test_dataset = LSTMDataset(val_file,tokenizier,rate=1.0)
    test_dataloader = DataLoader(test_dataset,collate_fn=test_dataset.collate_fn,batch_size=64)
    embeding_dim = 300
    hidden_dim = 64
    dropout = 0.5
    model = LSTM(len(wordid_map)+1,embeding_dim // 2,n_class,dropout).to(devices)
    

    optim = torch.optim.AdamW(model.parameters(),lr = 5e-4,weight_decay=0.1)
    criterian = nn.BCELoss()
    best = 0

    for i in range(1000):
        
        train_epoch(model,train_dataloader,criterian,optim,n_class,embeding_dim,devices)
        train_f1 = evaluate(model,train_dataloader,n_class,embeding_dim,devices)
        print("train",train_f1)
        test_f1 = evaluate(model,test_dataloader,n_class,embeding_dim,devices)
        print("test",test_f1,best)
        if test_f1 > best:
            torch.save({"embeding":model.embeding,"lstm":model.lstm},save_path)
            model.to(devices)
        best = test_f1 if test_f1 > best else best
        # stop training to avoid overfitting
        if train_f1 > 0.95:
            break
        

        