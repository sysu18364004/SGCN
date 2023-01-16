import time
import numpy as np
from utils import cal_accuracy
from tqdm import tqdm
import torch
import random 
# def adjust_learning_rate(optimizer, epoch, args):
#             """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#             n = 1000
#             lrt = (0.1**((1 % n) / n))
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] *= lrt
def adjust_learning_rate(optimizer, epoch, args):
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            # if epoch == 100:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = 1e-2
            # if epoch == 1000:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = 2e-3
            if epoch == 1700 and epoch == 2000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
def train_model(args, model, optimizer,criterion, features, adj, labels,label_truth, idx_train, idx_val, idx_test,show_result = True):
    val_loss = []

    save_path = 'model/best_model.pth'
    best_val_f1 = 0
    idx_train_out = idx_train.copy()
    idx_label_out = idx_train.copy()
    idx_orders = [i for i in range(len(idx_train))]
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        _, output= model(features, adj)
        # print(output.shape)
        lambdas = args.lambda1
        # sub_size = len(idx_train) // 4
        # idx_train_out_order = idx_orders[(epoch % sub_size)*4:(epoch % sub_size)*4+4]
        # idx_train_out_sub = [idx_train_out[j] for j in idx_train_out_order]
        # idx_label_out_sub = [idx_label_out[j] for j in idx_train_out_order]
        train_output = output[idx_train_out]
        train_label = labels[idx_label_out]
        # train_label = torch.cat((labels[idx_label_out],label_truth),axis=0)
        # print(train_label.shape,torch.cat((train_label,label_truth),axis=0).shape)
        # return
        # mask = labels[idx_train] > 0.5
        # loss_train = criterion(output[idx_train]*mask, labels[idx_train]*mask) + lambdas*criterion(output[idx_train], labels[idx_train])
        mask = train_label > 0.5
        loss_train = criterion(train_output*mask, train_label*mask) + lambdas*criterion(train_output*(~mask), train_label*(~mask))
        # loss = criterion(train_output, train_label)
        acc_train = cal_accuracy(output[idx_train], labels[idx_train])

        loss =  loss_train + 0.0001*criterion(output[-label_truth.shape[0]:], label_truth.float())
        loss.backward()
        optimizer.step()
        
        
        model.eval()
        _, output = model(features, adj)
        mask = labels[idx_val] > 0.5
        loss_val = criterion(output[idx_val]*mask, labels[idx_val]*mask)+lambdas*criterion(output[idx_val]*(~mask), labels[idx_val]*(~mask))
        loss = criterion(output[idx_val], labels[idx_val])
        val_loss.append(loss_val.item())

        # scheduler.step(loss_val)

        acc_val = cal_accuracy(output[idx_val], labels[idx_val])
        if acc_val > best_val_f1:
            best_val_f1 = acc_val
            torch.save(model,save_path)
        acc_test = cal_accuracy(output[idx_test], labels[idx_test])
        if show_result:
            print(  'Epoch: {:04d}'.format(epoch+1),
                    'loss_train: {:.4f}'.format(loss_train.item()),
                    'f1_train: {:.4f}'.format(acc_train),
                    'loss_val: {:.4f}'.format(loss_val.item()),
                    'f1_val: {:.4f}'.format(acc_val),
                    'f1_test: {:.4f}'.format(acc_test)
            )

        # adjust_learning_rate(optimizer,epoch,1)
        # if epoch > args.early_stopping and np.min(val_loss[-args.early_stopping:]) > np.min(val_loss[:-args.early_stopping]) :
        #     if show_result:
        #         print("Early Stopping...")
        #     break