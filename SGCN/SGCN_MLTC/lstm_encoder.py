import torch
import re

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

import math
import torch.nn.functional as F
def attention_net(x, query, mask=None): 
        
        d_k = query.size(-1)     
       
        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
#         print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  
#         print("score: ", scores.shape)  # torch.Size([128, 38, 38])
        
        alpha_n = F.softmax(scores, dim=-1) 
#         print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])

        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        context = torch.matmul(alpha_n, x).sum(1)
        
        return context, alpha_n

@torch.no_grad()
def lstm_encode(text,info,tokenizier):
    embeding = info['embeding'].to('cuda:2')
    lstm = info['lstm'].to('cuda:2')
    inputs = tokenizier.encode([text])['input_ids']
    inputs = torch.LongTensor(inputs).to('cuda:2')
    out = lstm(embeding(inputs.permute(1,0)))[0].permute(1,0,2)
    attn_output, alpha_n = attention_net(out, out)
    return attn_output[0]


