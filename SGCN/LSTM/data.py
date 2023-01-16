import nltk
from nltk.corpus import stopwords
import re
import torch
import pandas as pd
import random
from torch.utils.data import Dataset,DataLoader
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def build_dict(sentences):
    random.shuffle(sentences)
    word_dict = {}
    for sentence in sentences:
        # print(sentence)
        # words = clean_str(sentence).split()
        words = sentence.split()
        # print(words)
        for word in words:
            if word in stop_words:
                continue

            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
    word_id_map = {}
    i = 0
    for word in word_dict:
        if word_dict[word] > 15:
            word_id_map[word] = i
            i += 1
    return word_id_map
def clean_str(string):
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_tsv(filename,rate=1):
    labels = []
    sentences = []
    

    with open(filename,'r',encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        length = int(rate * len(lines))
        for line in lines[:length]:
            # print(line.split())
            label,*word = line.split()

            labels.append(list(map(int,label)))
            sentences.append(" ".join(word))

    return sentences,labels
class Tokenizier():
    def __init__(self,dic):
        self.dic = dic
    def single_sentence(self,sentence):
        sentence = clean_str(sentence)
        input_ids = []
        for word in sentence.split():
            if word in self.dic:
                # print(word)
                input_ids.append(self.dic[word]+1)
        return input_ids
    def encode(self,sentences):
        ans = {}
        if isinstance(sentences,str):
            ans['input_ids'] = self.single_sentence(sentences)
            attention_mask = [1]*len(input_ids)
            ans['attention_mask'] = attention_mask
            return ans
        
        input_ids = []
        max_l = 0
        for sentence in sentences:
            input_ids_item = self.single_sentence(sentence)
            input_ids.append(input_ids_item)
            max_l = max_l if max_l > len(input_ids_item) else len(input_ids_item)
        
        padding_input_ids = []
        attention_mask = []
        for input_id in input_ids:
            padding_input_ids.append(input_id+[0]*(max_l - len(input_id)))
            attention_mask.append([1]*len(input_id)+[0]*(max_l - len(input_id)))
        ans['input_ids'] = padding_input_ids
        ans['attention_mask'] = attention_mask
        return ans
class LSTMDataset(Dataset):
    def __init__(self,path,tokenizier,rate=0.1):
        sentences,labels = load_tsv(path,rate)
        self.sentences = sentences
        self.labels = labels
        self.tokenizier = tokenizier
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self,index):
        return self.sentences[index],self.labels[index]
    
    def collate_fn(self,batch):
        sentences = []
        labels = []
        for b in batch:
            sentences.append(b[0])
            labels.append(b[1])
        # features = self.tokenizier.encode(sentences,max_length=512,truncation=True,padding=True)
        features = self.tokenizier.encode(sentences)
        inputs = features['input_ids']
        mask = features['attention_mask']
        return torch.LongTensor(inputs),torch.LongTensor(mask),torch.FloatTensor(labels)