from build_dataset import get_dataset
from preprocess import encode_labels,preprocess_data
from build_graph import get_adj
from train import train_model
from utils import *

from evaluate import get_weights_hidden, get_test_emb, test_model
import argparse
import torch
import torch.optim as optim
import scipy.sparse as sp
import torch.nn as nn
from sklearn.metrics import classification_report
import json
import os
import numpy as np
import pickle as pkl
from utils import cal_accuracy,getall
import torch
from model_util import GCN
import random
import torch
import time
def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
###保存json
def save_json(filename,obj):
    with open(filename,'w',encoding='utf-8') as f:
        return json.dump(obj,f,indent=4,ensure_ascii=False)

def load_vocab(filename):
    vocab_list = []
    with open(filename,'r',encoding='utf-8') as f:
        for line in f.readlines():
            vocab_list.append(line.strip())
    return vocab_list
def load_json(filename):
    with open(filename,'r',encoding='utf-8') as f:
        return json.load(f)
def save_vocab(vocab_list,filename):
    with open(filename,'w',encoding='utf-8') as f:
        print("\n".join(vocab_list),file=f)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='R8', help='Dataset string: R8, R52, OH, 20NGnew, MR')
parser.add_argument('--train_size',  type=float, default=1, help='If it is larger than 1, it means the number of training samples. If it is from 0 to 1, it means the proportion of the original training set.')
parser.add_argument('--test_size',  type=float, default=1, help='If it is larger than 1, it means the number of training samples. If it is from 0 to 1, it means the proportion of the original training set.')
parser.add_argument('--remove_limit', type=int, default=10, help='Remove the words showing fewer than 2 times')
parser.add_argument('--use_gpu', type=int, default=1, help='Whether to use GPU, 1 means True and 0 means False. If True and no GPU available, will use CPU instead.')
parser.add_argument('--shuffle_seed',type = int, default = None, help="If not specified, train/val is shuffled differently in each experiment")
parser.add_argument('--hidden_dim',type = int, default = 768, help="The hidden dimension of GCN model")
parser.add_argument('--dropout',type = float, default = 0.5, help="The dropout rate of GCN model")
parser.add_argument('--learning_rate',type = float, default = 3e-4, help="Learning rate")
parser.add_argument('--weight_decay',type = float, default = 0, help="Weight decay, normally it is 0")
parser.add_argument('--early_stopping',type = int, default = 300, help="Number of epochs of early stopping.")
parser.add_argument('--epochs',type = int, default = 10, help="Number of maximum epochs")
parser.add_argument('--multiple_times',type = int, default = 2, help="Running multiple experiments, each time the train/val split is different")
parser.add_argument('--easy_copy',type = int, default = 1, help="For easy copy of the experiment results. 1 means True and 0 means False.")
parser.add_argument('--use_emb',type = int, default = 0, help="if use the pre training embdeing")
parser.add_argument('--norm_edge',type = int, default = 0, help="if use the pre training embdeing")
parser.add_argument('--batch_size',type = int, default = 32, help="if use the pre training embdeing")
parser.add_argument('--lambda1',type = float, default = 0.5, help="negetivete rate")
parser.add_argument('--lambda2',type = float, default = 0.0001, help="label loss rate,current best is 0.00007 for reusters")
parser.add_argument('--resrate',type = float, default = 0.5, help="recall the input rate")
parser.add_argument('--activate_type',type = str, default = 'relu', help="recall the input rate")
parser.add_argument('--dataset_type',type = str, default ='aapd', help="decision which dataset to train and evaluate")
parser.add_argument('--use_loaded',type = int, default =0, help="decision whether use the load file")
parser.add_argument('--loc_path',type = str, default ="/home/zengdl/project/init_project/InductTGCN", help="decision whether use the load file")
parser.add_argument('--gpu_order',type = str, default ="cuda:4", help="decision whether use the load file")
parser.add_argument('--output_dir',type = str, default ="aapd", help="decision whether use the load file")
parser.add_argument('--emb_path',type = str, default ="/home/zengdl/project/init_project/SGCN/embeddings/aapd/lstm", help="decision whether use the load file")

args = parser.parse_args()
start_time = time.time()
print(args.lambda1)
if not os.path.exists(f"output_{args.dataset_type}/{args.output_dir}"):
    os.mkdir(f"output_{args.dataset_type}/{args.output_dir}")
device = decide_device(args)

# Get dataset
sentences, labels, train_size, val_size,test_size = get_dataset(args)
train_size += val_size
train_sentences = sentences[:train_size]
test_sentences = sentences[train_size:]
train_labels = labels[:train_size]
test_labels = labels[train_size:]

# Preprocess text and labels
# labels, num_class = encode_labels(train_labels, test_labels, args)
if args.use_loaded == 1:
    use_load = True
else:
    use_load = False

emb_root = args.emb_path
train_emb = pkl.load(open(f"{emb_root}/train", "rb"))
val_emb = pkl.load(open(f"{emb_root}/val", "rb"))
test_emb = pkl.load(open(f"{emb_root}/test", "rb"))
# print(train_emb.shape,val_emb.shape,test_emb.shape)
doc_emb = np.concatenate([train_emb, val_emb, test_emb])
print(doc_emb.shape)
# num_class = labels.shape[1]
args.hidden_dim = doc_emb.shape[1]
print(args.hidden_dim)
num_class = len(labels[0])
tokenize_sentences, word_list = preprocess_data(train_sentences, test_sentences, args)
ori_sentences = train_sentences + test_sentences
if args.norm_edge == 1:
    load_dataset = f"load_dataset_{args.dataset_type}_norm_edge"
else:
    load_dataset = f"load_dataset_{args.dataset_type}"
loc_path = args.loc_path

### when set use load, we could save the graph so that not to build graph next times
if not use_load:
    if not os.path.exists(f'{loc_path}/{load_dataset}'):
        os.mkdir(f'{loc_path}/{load_dataset}')
        print("mk dir: ",f'{loc_path}/{load_dataset}')
    save_vocab(word_list,f'{loc_path}/{load_dataset}/word_list.txt')
    vocab_length = len(word_list)
    word_id_map = {}
    for i in range(vocab_length):
        word_id_map[word_list[i]] = i
    if not args.easy_copy:
        print("There are", vocab_length, "unique words in total.")   

    # Generate Graph
    adj, word_doc_freq = get_adj(ori_sentences,tokenize_sentences,labels,train_size,test_size,word_id_map,word_list,args)
    save_json(f"{loc_path}/{load_dataset}/word_doc_freq.json",word_doc_freq)
    with open(f"{loc_path}/{load_dataset}/ind.graph", 'wb') as f:
        pkl.dump(adj, f)
    adj, norm_item = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    features = torch.FloatTensor(doc_emb).to(device)
    # Generate Test input
    
    # test_emb, tokenized_test_edge = get_test_emb(tokenize_sentences[train_size-val_size:train_size], word_id_map, vocab_length, word_doc_freq, word_list, train_size,norm_item)
    tokenized_test_edge = get_test_emb(args, test_sentences,tokenize_sentences[train_size:], word_id_map, vocab_length, word_doc_freq, word_list, train_size, norm_item)

    with open(f"{loc_path}/{load_dataset}/ind.tokenized_test_edge", 'wb') as f:
        pkl.dump(tokenized_test_edge, f)
else:
    word_list = load_vocab(f'{loc_path}/{load_dataset}/word_list.txt')
    vocab_length = len(word_list)
    word_id_map = {}
    for i in range(vocab_length):
        word_id_map[word_list[i]] = i
    if not args.easy_copy:
        print("There are", vocab_length, "unique words in total.")   
    adj = pkl.load(open(f"{loc_path}/{load_dataset}/ind.graph", "rb"))
    
    # doc_emb = pkl.load(open(f"{loc_path}/{load_dataset}/ind.doc_emb", "rb"))
    # doc_emb = doc_emb.reshape(-1,args.hidden_dim)
    word_doc_freq = load_json(f"{loc_path}/{load_dataset}/word_doc_freq.json")
    adj, norm_item = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    features = torch.FloatTensor(doc_emb).to(device)
    tokenized_test_edge = get_test_emb(args, test_sentences,tokenize_sentences[train_size:], word_id_map, vocab_length, word_doc_freq, word_list, train_size, norm_item)
    tokenized_test_edge = pkl.load(open(f"{loc_path}/{load_dataset}/ind.tokenized_test_edge", "rb"))


labels = torch.FloatTensor(labels).to(device)    
label_truth = torch.eye(num_class,num_class).long().to(device)  
# print(labels[:train_size].sum(axis=0).cpu().numpy()/train_size)
print(adj.shape)
class_freq = (labels[:train_size].sum(axis=0).cpu().numpy()/train_size).tolist()
# labels.te
criterion = nn.MSELoss()
seed_num = 11
file_name = args.activate_type + "lambda1_"+str(args.lambda1)+"_lambda2_"+str(args.lambda2)+"_resrate_"+str(args.resrate)+"_lr_"+str(args.learning_rate)+ "_norm_edge_" + str(args.norm_edge)
if args.multiple_times:
    seed_everything(seed_num)
    seed_num += 1
    test_acc_list = []
    for t in range(args.multiple_times):
        start_time = time.time()
        if not args.easy_copy:
            print("Round",t+1)
        # model = GCN(nfeat=vocab_length, nhid=args.hidden_dim, nclass=num_class, dropout=args.dropout).to(device)
        model = GCN(nfeat=vocab_length+num_class, nhid=args.hidden_dim, nclass=num_class, dropout=args.dropout,activate_type= args.activate_type,resrate= args.resrate).to(device)
    
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        idx_train, idx_val,idx_test = generate_train_val(args, train_size,val_size,test_size)
        
        # train_model(args, model, optimizer, scheduler,criterion, features, adj, labels, idx_train, idx_val,idx_test,show_result=True)
        train_model(args, model, optimizer,criterion, features, adj, labels,label_truth, idx_train, idx_val,idx_test,show_result=True)



        model = torch.load(f'model/best_model_{file_name}.pth').to(device)
        # model_weights_list = get_weights_hidden(model,features,adj,train_size,test_size,vocab_length)
        # model_weights_list = get_weights_hidden(model,features,adj,train_size)
        model.eval()
        with torch.no_grad():
            _, output = model(features, adj)
        test_result = output[idx_test]
        # test_result = test_model(model, test_emb, tokenized_test_edge,model_weights_list,device,resrate= args.resrate)
        # test_acc_list.append(cal_accuracy(test_result,labels[train_size-val_size:train_size].cpu()))
        test_acc_list.append(result_show(test_result,labels[train_size:].cpu()))

        with open(f"output_{args.dataset_type}/{args.output_dir}/"+file_name,'a',encoding='utf-8') as f:
            # print(test_acc_list[-1],file=f)
            for n,v in zip(["accuracy", "micro_f1", "macro_f1", "ndcg1", "ndcg3", "ndcg5", "p1", "p3", "p5"],test_acc_list[-1]):
                print(n,v,file=f)
            
    test_acc_list = np.array(test_acc_list)
    
    if args.easy_copy:
        file_name = args.activate_type +"lambda1_"+str(args.lambda1)+"_lambda2_"+str(args.lambda2)+"_resrate_"+str(args.resrate)+"_lr_"+str(args.learning_rate) + "_norm_edge_" + str(args.norm_edge)
        with open(f"output_{args.dataset_type}/{args.output_dir}/"+file_name,'a',encoding='utf-8') as f:
            for k,n in enumerate(["accuracy", "micro_f1", "macro_f1", "ndcg1", "ndcg3", "ndcg5", "p1", "p3", "p5"]):
                print(n,file=f)
                print("%.4f"%np.mean(test_acc_list[:, k]), end = ' ± ',file=f)
                print("%.4f"%np.std(test_acc_list[:, k]),file=f)

    else: 
        for t in test_acc_list:
            print("%.4f"%t)       
        print("Test Accuracy:",np.round(test_acc_list,4).tolist())
        print("Mean:%.4f"%np.mean(test_acc_list))
        print("Std:%.4f"%np.std(test_acc_list))
