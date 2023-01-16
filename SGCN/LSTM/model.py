import torch.nn as nn
from utils import *
import numpy as np
import torch

class LSTM(nn.Module):
    def __init__(self,input_size,embeding_size,out_size,dropout):
        super(LSTM,self).__init__()
        self.embeding = nn.Embedding(input_size,embeding_size)
        self.end = MLP(embeding_size*2,128, out_size,2,0.1)
        self.sigmoid = nn.Sigmoid()
        self.lstm = nn.LSTM(input_size=embeding_size,hidden_size=embeding_size,num_layers=2,
                        bias=True,batch_first=False,dropout=0.2,bidirectional=True)

        self.dropout = nn.Dropout(dropout)
    def attention_net(self, x, query, mask=None): 
        
        d_k = query.size(-1)     
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  
        
        alpha_n = F.softmax(scores, dim=-1) 
        context = torch.matmul(alpha_n, x).sum(1)
        return context, alpha_n

    def forward(self,x):
        y_ = self.infence(x)
        y_ = torch.sigmoid(y_)
        return y_




