import scipy.sparse as sp
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn import metrics
from torchmetrics import Precision
def patk(actual, pred, k):
    #we return 0 if k is 0 because 
    #   we can't divide the no of common values by 0 
    if k == 0:
        return 0

    #taking only the top k predictions in a class 
    k_pred = pred[:k]

    # taking the set of the actual values 
    actual_set = set(actual)
    # print(list(actual_set))
    # taking the set of the predicted values 
    pred_set = set(k_pred)
    # print(list(pred_set))
    
    # 求预测值与真实值得交集
    common_values = actual_set.intersection(pred_set)
    # print(common_values)
    if len(pred[:k]) == 0:
        return 0
    else:
        return len(common_values) / len(pred[:k])
    
def mapk(acutal, pred, k):

    #creating a list for storing the Average Precision Values
    average_precision = []
    #interating through the whole data and calculating the apk for each 
    for i in range(len(acutal)):
        ap = patk(np.where(acutal[i] > 0.5)[0], np.argsort(pred[i])[::-1], k)
        # print(f"AP@k: {ap}")
        average_precision.append(ap)

    #returning the mean of all the data
    return np.mean(average_precision)
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
            device = torch.device(args.gpu_order)
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

def evaluate_f1_ml(predict, truth):
    """
    F1-score for multi-label classification
    :param predict: (batch, labels)
    :param truth: (batch, labels)
    :return:
    """
    label_same = []
    label_predict = []
    label_truth = []
    label_f1 = []

    division = lambda x, y: (x * 1.0 / y) if y else 0
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0

    batch, label_size = predict.size()
    for i in range(label_size):
        cur_predict = predict[:, i]
        cur_truth = truth[:, i]

        predict_max = cur_predict.gt(0.5).long()
        cur_eq_num = (predict_max * cur_truth).sum().item()

        cur_predict_num = predict_max.sum().item()
        cur_truth_num = cur_truth.sum().item()

        cur_precision = division(cur_eq_num, cur_predict_num)
        cur_recall = division(cur_eq_num, cur_truth_num)
        cur_f1 = f1(cur_precision, cur_recall)

        label_same.append(cur_eq_num)
        label_predict.append(cur_predict_num)
        label_truth.append(cur_truth_num)
        label_f1.append(cur_f1)

    macro_f1 = sum(label_f1) / len(label_f1)
    micro_precision = division(sum(label_same), sum(label_predict))
    micro_recall = division(sum(label_same), sum(label_truth))
    micro_f1 = f1(micro_precision, micro_recall)

    return macro_f1, micro_f1, micro_precision, micro_recall, label_f1
    
def cal_accuracy(predictions,labels):
    if isinstance(predictions,list):
        predictions = torch.cat(predictions,0)
    predicted_labels = predictions.cpu().numpy()
    target_labels = labels.cpu().numpy()
    micro_f1 = metrics.f1_score(target_labels, predicted_labels.round(), average='micro')
    return micro_f1
    # pred = torch.argmax(predictions,-1).cpu().tolist()
    # pred = (predictions > 0.5).cpu()
    
    # lab = labels.cpu()
    # lab = lab > 0.5
    # mask = lab
    # tp = (mask*(lab==pred)).sum()
    # # print(tp)
    # pred_1 = pred.sum()
    # truth_1 = lab.sum()
    # if pred_1 + truth_1 == 0:
    #     return 0
    # else:
    #     return (2*tp/(pred_1 + truth_1)).item()
    # return accuracy_score(lab,pred)
    # return 
def result_show(predictions,labels):
    if isinstance(predictions,list):
        predictions = torch.cat(predictions,0)
    # pred = torch.argmax(predictions,-1).cpu().tolist()
    predicted_labels = predictions.cpu().numpy()
    target_labels = labels.cpu().numpy()
    n_classes = labels.shape[1]
    # print(n_classes)
    # predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
    accuracy = metrics.accuracy_score(target_labels, predicted_labels.round())
    micro_f1 = metrics.f1_score(target_labels, predicted_labels.round(), average='micro')
    macro_f1 = metrics.f1_score(target_labels, predicted_labels.round(), average='macro')

    ndcg1 = metrics.ndcg_score(target_labels, predicted_labels, k=1)
    ndcg3 = metrics.ndcg_score(target_labels, predicted_labels, k=3)
    ndcg5 = metrics.ndcg_score(target_labels, predicted_labels, k=5)

    # p1 = Precision(num_classes=n_classes, top_k=1, task = "multilabel")(torch.tensor(predicted_labels), torch.tensor(target_labels) , num_labels=1)
    # p3 = Precision(num_classes=n_classes, top_k=3, task = "multilabel")(torch.tensor(predicted_labels), torch.tensor(target_labels), num_labels=3)
    # p5 = Precision(num_classes=n_classes, top_k=5, task = "multilabel")(torch.tensor(predicted_labels), torch.tensor(target_labels), num_labels=5)
    # p1 , p3, p5 = 0, 0, 0
    p1 = mapk(target_labels, predicted_labels, 1)
    p3 = mapk(target_labels, predicted_labels, 3)
    p5 = mapk(target_labels, predicted_labels, 5)
    return [accuracy, micro_f1, macro_f1, ndcg1, ndcg3, ndcg5, p1, p3, p5]
def getall(predictions,labels):
    if isinstance(predictions,list):
        predictions = torch.cat(predictions,0)
    # pred = torch.argmax(predictions,-1).cpu().tolist()
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