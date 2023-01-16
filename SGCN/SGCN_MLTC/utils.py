import scipy.sparse as sp
import numpy as np
import torch
from sklearn.metrics import accuracy_score

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo(), d_inv_sqrt
    
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def decide_device(args):
    if args.use_gpu:
        if torch.cuda.is_available():
            if not args.easy_copy:
                print("Use CUDA")
            device = torch.device("cuda:4")
        else:
            if not args.easy_copy:
                print("CUDA not avaliable, use CPU instead")
            device = torch.device("cpu")
    else:
        if not args.easy_copy:
            print("Use CPU")
        device = torch.device("cpu")
    return device


def generate_train_val(args, train_size, val_size,test_size, train_pro=0.9):
    idx_train = list([_ for _ in range(train_size-val_size)])
    idx_val  = list([_ for _ in range(train_size-val_size,train_size)])
    idx_test  = list([_ for _ in range(train_size,train_size+test_size)])
    return idx_train, idx_val,idx_test


def cal_accuracy(predictions,labels):
    if isinstance(predictions,list):
        predictions = torch.cat(predictions,0)
    pred = (predictions > 0.5).cpu()
    
    lab = labels.cpu()
    lab = lab > 0.5
    mask = lab
    tp = (mask*(lab==pred)).sum()
    pred_1 = pred.sum()
    truth_1 = lab.sum()
    if pred_1 + truth_1 == 0:
        return 0
    else:
        return (2*tp/(pred_1 + truth_1)).item()
        
def getall(predictions,labels):
    if isinstance(predictions,list):
        predictions = torch.cat(predictions,0)
    pred = (predictions > 0.5).cpu()
    
    lab = labels.cpu()
    lab = lab > 0.5
    mask = lab
    tp = (mask*(lab==pred)).sum()
    # print(tp)
    pred_1 = pred.sum()
    truth_1 = lab.sum()
    if pred_1 + truth_1 == 0:
        return (tp/pred_1).item(),(tp/truth_1).item(), 0
    else:
        return (tp/pred_1).item(),(tp/truth_1).item(),(2*tp/(pred_1 + truth_1)).item()