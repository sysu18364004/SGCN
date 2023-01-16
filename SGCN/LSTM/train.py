from tqdm import tqdm
import torch
import torch.nn as nn
def train_epoch(model,train_loader,criterion,optim,n_class,embeding_dim,device):
    model.train()
    pbar = tqdm(train_loader)
    acc_loss = []
    # A_label = torch.eye(n_class,n_class)
    for inputs,mask,labels in pbar:
        
        inputs,mask,labels = inputs.to(device),mask.to(device),labels.to(device)
        batch_size,token_l = inputs.shape[0],inputs.shape[1]
       
        # A_token = torch.eye(token_l,token_l)
        # row = torch.LongTensor(range(token_l))
        # col = row
        # A_token[abs(col.unsqueeze(1)-row)==1]=1

        # A_tokens = A_token.repeat(batch_size,1,1).to(device)
        # A_labels = A_label.repeat(batch_size,1,1).to(device)
        # A_tls = torch.zeros(token_l,n_class).repeat(batch_size,1,1).to(device)
        # labels_init = torch.eye(n_class,embeding_dim).repeat(batch_size,1,1).to(device)

        # predict = model(inputs,A_tokens,A_labels,A_tls,labels_init)
        predict = model(inputs)
        lambdas = 0.5
        mask = labels > 0.5
        loss = criterion(predict*mask, labels*mask) + lambdas*criterion(predict, labels)
        # loss = criterion(predict, labels)
        acc_loss.append(loss.item())
        pbar.set_description("loss %s" % loss.item()+" (%s)" % (sum(acc_loss)/len(acc_loss)))

        optim.zero_grad()
        loss.backward()
        optim.step()
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

@torch.no_grad()
def evaluate(model,test_loader,n_class,embeding_dim,device):
    model.eval()
    results = []
    alabel = []
    pbar = tqdm(test_loader)
    # A_label = torch.eye(n_class,n_class)
    for inputs,mask,labels in pbar:
        inputs,mask,labels = inputs.to(device),mask.to(device),labels.to(device)
        predict = model(inputs)

    
        results.append(predict)
        alabel.append(labels)
    results = torch.cat(results,0)
    lab =  torch.cat(alabel,0)
    macro_f1, micro_f1, micro_precision, micro_recall, label_f1 = evaluate_f1_ml(results, lab)
    return micro_f1
# @torch.no_grad()
# def evaluate(model,test_loader,n_class,embeding_dim,device):
#     model.eval()
#     results = []
#     alabel = []
#     pbar = tqdm(test_loader)
#     # A_label = torch.eye(n_class,n_class)
#     for inputs,mask,labels in pbar:
#         inputs,mask,labels = inputs.to(device),mask.to(device),labels.to(device)


#         # batch_size,token_l = inputs.shape[0],inputs.shape[1]
       
#         # A_token = torch.eye(token_l,token_l)
#         # row = torch.LongTensor(range(token_l))
#         # col = row
#         # A_token[abs(col.unsqueeze(1)-row)==1]=1

#         # A_tokens = A_token.repeat(batch_size,1,1).to(device)
#         # A_labels = A_label.repeat(batch_size,1,1).to(device)
#         # A_tls = torch.zeros(token_l,n_class).repeat(batch_size,1,1).to(device)
#         # labels_init = torch.eye(n_class,embeding_dim).repeat(batch_size,1,1).to(device)


#         # predict = model(inputs,A_tokens,A_labels,A_tls,labels_init)
#         predict = model(inputs,mask)

    
#         results.append(predict)
#         alabel.append(labels)
#     results = torch.cat(results,0)
#     # pred = torch.argmax(predictions,-1).cpu().tolist()
#     pred = (results > 0.5)
    
#     lab =  torch.cat(alabel,0)
#     lab = lab > 0.5
#     mask = lab
#     tp = (mask*(lab==pred)).sum()
    
#     pred_1 = pred.sum()
#     truth_1 = lab.sum()
#     tp = ((mask*(lab==pred)).sum())
#     ans = 2*tp/(pred_1+truth_1)
#     # ans[torch.isnan(ans)] = 0.0
#     if torch.isnan(ans):
#         return 0
#     else:
#         # return ans.mean().item()
#         return ans.item()
#     # print(tp,pred_1,truth_1)
#     # if pred_1 + truth_1 == 0:
#     #     return 0
#     # else:
#     #     return (2*tp/(pred_1 + truth_1)).item()