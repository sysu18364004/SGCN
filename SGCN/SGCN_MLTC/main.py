from build_dataset import get_dataset
from preprocess import encode_labels,preprocess_data
from build_graph import get_adj
from train import train_model
from utils import *
from model import GCN
from evaluate import get_weights_hidden, get_test_emb, test_model
import argparse
import torch
import torch.optim as optim
import scipy.sparse as sp
import torch.nn as nn
from sklearn.metrics import classification_report
import json
import pickle as pkl
from utils import cal_accuracy,getall
import torch
from lstm_encoder import clean_str
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
import json


parser = argparse.ArgumentParser()
parser.add_argument('--train_size',  type=float, default=1, help='If it is larger than 1, it means the number of training samples. If it is from 0 to 1, it means the proportion of the original training set.')
parser.add_argument('--test_size',  type=float, default=1, help='If it is larger than 1, it means the number of training samples. If it is from 0 to 1, it means the proportion of the original training set.')
parser.add_argument('--remove_limit', type=int, default=10, help='Remove the words showing fewer than 2 times')
parser.add_argument('--use_gpu', type=int, default=1, help='Whether to use GPU, 1 means True and 0 means False. If True and no GPU available, will use CPU instead.')
parser.add_argument('--shuffle_seed',type = int, default = None, help="If not specified, train/val is shuffled differently in each experiment")
parser.add_argument('--hidden_dim',type = int, default = 300, help="The hidden dimension of GCN model")
parser.add_argument('--dropout',type = float, default = 0.5, help="The dropout rate of GCN model")
parser.add_argument('--learning_rate',type = float, default = 5e-3, help="Learning rate")
parser.add_argument('--weight_decay',type = float, default = 0, help="Weight decay, normally it is 0")
parser.add_argument('--early_stopping',type = int, default = 10, help="Number of epochs of early stopping.")
parser.add_argument('--epochs',type = int, default = 1500, help="Number of maximum epochs")
parser.add_argument('--multiple_times',type = int, default = 2, help="Running multiple experiments, each time the train/val split is different")
parser.add_argument('--easy_copy',type = int, default = 1, help="For easy copy of the experiment results. 1 means True and 0 means False.")
parser.add_argument('--lambda1',type = float, default = 0.5, help="The parameter that balance the 1 label and 0 label")
parser.add_argument('--dict_path',type = str, default = '/home/zengdl/project/multi_label/LSTM/dict_rmsc.json', help="the word dict of the dataset calculated by LSTM")
parser.add_argument('--preModel_path',type = str, default = '/home/zengdl/project/multi_label/LSTM/best_embeding_rmsc', help="LSTM model path, which is trained before.")
args = parser.parse_args()
with open(args.dict_path,'r',encoding='utf-8') as f:
        wordid_map = json.load(f)
tokenizier = Tokenizier(wordid_map)
args.tokenizier = tokenizier
device = decide_device(args)

# Get dataset
sentences, labels, train_size, val_size,test_size = get_dataset(args)
train_size += val_size
train_sentences = sentences[:train_size]
test_sentences = sentences[train_size:]
train_labels = labels[:train_size]
test_labels = labels[train_size:]


num_class = len(labels[0])
tokenize_sentences, word_list = preprocess_data(train_sentences, test_sentences, args)




vocab_length = len(word_list)
word_id_map = {}
for i in range(vocab_length):
    word_id_map[word_list[i]] = i
if not args.easy_copy:
    print("There are", vocab_length, "unique words in total.")   

# Generate Graph
adj, doc_emb, word_doc_freq = get_adj(tokenize_sentences,labels,train_size,test_size,word_id_map,word_list,args)
doc_emb = doc_emb.reshape(-1,300)

adj, norm_item = normalize_adj(adj + sp.eye(adj.shape[0]))
adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
features = torch.FloatTensor(doc_emb).to(device)


# Generate Test input

test_emb, tokenized_test_edge = get_test_emb(tokenize_sentences[train_size:], word_id_map, vocab_length, word_doc_freq, word_list, train_size, norm_item)

labels = torch.FloatTensor(labels).to(device)    
### one-hot code
label_truth = torch.eye(num_class,num_class).long().to(device)  


criterion = nn.MSELoss()


if args.multiple_times:
    test_acc_list = []
    for t in range(args.multiple_times):
        if not args.easy_copy:
            print("Round",t+1)

        model = GCN(nfeat=vocab_length+num_class, nhid=args.hidden_dim, nclass=num_class, dropout=args.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        idx_train, idx_val,idx_test = generate_train_val(args, train_size,val_size,test_size)

        train_model(args, model, optimizer,criterion, features, adj, labels,label_truth, idx_train, idx_val,idx_test,show_result=True)
        model = torch.load('model/best_model.pth')
        model_weights_list = get_weights_hidden(model,features,adj,train_size,test_size,vocab_length)

        test_result = test_model(model, test_emb, tokenized_test_edge,model_weights_list,device)
        test_acc_list.append(getall(test_result,labels[train_size:].cpu()))
        print(test_acc_list[-1])
    if args.easy_copy:

        print("%.4f"%np.mean(test_acc_list), end = ' ± ')
        print("%.4f"%np.std(test_acc_list))

    else: 
        for t in test_acc_list:
            print("%.4f"%t)       
        print("Test Accuracy:",np.round(test_acc_list,4).tolist())
        print("Mean:%.4f"%np.mean(test_acc_list))
        print("Std:%.4f"%np.std(test_acc_list))
