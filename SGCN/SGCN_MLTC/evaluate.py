
import numpy as np
from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from lstm_encoder import *
def fix(t):
    t[t < 0.] = 0.
    t[t > 1.] = 1.
def get_weights_hidden(model,features,adj,train_size,test_size,vocab_length):
    model.eval()
    hidden_output, label_embedding = model.encode(features, adj)
    weights1 = model.gc1.weight[:vocab_length]
    bias1 = model.gc1.bias
    weights1.require_grad = False
    bias1.require_grad = False
    weights2 = model.gc2.weight[:vocab_length]
    bias2 = model.gc2.bias
    weights2.require_grad = False
    bias2.require_grad = False
    fc = model.fc
    fc.require_grad = False
    return [hidden_output[train_size+test_size:train_size+test_size+vocab_length],label_embedding, weights1, bias1, weights2, bias2,fc]
## normalize the test edge
def normalize_edge(tokenized_test_edge,norm_item,train_size,test_size,vocab_length):
    word_doc, one = tokenized_test_edge[:,:-1],tokenized_test_edge[:,-1]
    rowsum = tokenized_test_edge.sum(1)
    word_doc_sqrt = np.power(rowsum, -0.5).flatten()
    word_doc_sqrt[np.isinf(word_doc_sqrt)] = 0.
    word_doc_sqrt = word_doc_sqrt.reshape(1,-1)
    word_diag = norm_item[train_size+test_size:train_size+test_size+vocab_length].reshape(1,-1)
    normal_p = word_doc_sqrt.T.dot(word_diag)
    test_edges = sp.coo_matrix(word_doc)
    test_edges_emb = test_edges.multiply(normal_p)
    another_cal_ = test_edges_emb.toarray()
    tokenized_test_edge1 = np.concatenate([another_cal_,(word_doc_sqrt**2).reshape(-1,1)],axis=1)
    return tokenized_test_edge1
def get_test_emb(tokenize_test_sentences, word_id_map, vocab_length, word_doc_freq, word_list, train_size,norm_item,args):
    test_size = len(tokenize_test_sentences)
    test_emb = [[0]*vocab_length for _ in range(test_size)]
    test_emb = []
    info = torch.load(args.preModel_path)

    tokenized_test_edge = [[0]*vocab_length+[1] for _ in range(test_size)]
    for i in range(test_size):
        tokenized_test_sample = tokenize_test_sentences[i]
        test_emb.append(lstm_encode(" ".join(tokenized_test_sample),info,args.tokenizier).cpu().numpy())
        word_freq_list = [0]*vocab_length
        for word in tokenized_test_sample:
            if word in word_id_map:
                word_freq_list[word_id_map[word]]+=1
            
        for word in tokenized_test_sample:
            if word in word_id_map:
                j = word_id_map[word]
                freq = word_freq_list[j]   
                idf = log(1.0 * train_size / (word_doc_freq[word_list[j]]+1))
                w = idf
                tokenized_test_edge[i][j] = w
         

    tokenized_test_edge = np.array(tokenized_test_edge)
    tokenized_test_edge = normalize_edge(tokenized_test_edge,norm_item,train_size,test_size,vocab_length)
    return test_emb, tokenized_test_edge



@torch.no_grad()
def test_model(model, test_emb, tokenized_test_edge,model_weights_list,device):
    hidden_output,label_embedding, weights1, bias1, weights2, bias2,fc = model_weights_list
    test_result = []
    test_size = len(tokenized_test_edge[0])

    for ind in range(len(test_emb)):
        tokenized_test_edge_temp = torch.FloatTensor([tokenized_test_edge[ind]]).to(device)

        hidden_temp = torch.FloatTensor([test_emb[ind].tolist()]).to(device)
        hidden_temp = F.relu(torch.mm(tokenized_test_edge_temp, torch.vstack((weights1, hidden_temp))) + bias1)
        test_hidden_temp = torch.cat((hidden_output,hidden_temp))
        test_output_temp = torch.mm(tokenized_test_edge_temp, torch.mm(test_hidden_temp, weights2)) + bias2

        test_output_temp = F.relu(test_output_temp)
        predict_temp = (fc(test_output_temp)).cpu() 

        test_result.append(torch.sigmoid(predict_temp).cpu())

    return test_result

