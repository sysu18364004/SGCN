import time
import numpy as np
from utils import cal_accuracy
from tqdm import tqdm
import torch
import random 
import os 
def adjust_learning_rate(optimizer, epoch, args):
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            if epoch == 1500:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
def train_model(args, model, optimizer,criterion, features, adj, labels,label_truth, idx_train, idx_val, idx_test,show_result = True):
    val_loss = []
    file_name = args.activate_type + "lambda1_"+str(args.lambda1)+"_lambda2_"+str(args.lambda2)+"_resrate_"+str(args.resrate)+"_lr_"+str(args.learning_rate)+ "_norm_edge_" + str(args.norm_edge)
    save_path = f'model/best_model_{file_name}.pth'
    best_val_f1 = 0
    idx_train_out = idx_train.copy()
    idx_label_out = idx_train.copy()
    idx_orders = [i for i in range(len(idx_train))]
    if not os.path.exists(f"output_{args.dataset_type}"):
        os.mkdir(f"output_{args.dataset_type}")
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        # print("显存占用 (字节):", torch.cuda.memory_allocated())
        _, output= model(features, adj)
        lambdas = args.lambda1
        train_output = output[idx_train_out]
        train_label = labels[idx_label_out]
        mask = train_label > 0.5
        loss_train = criterion(train_output*mask, train_label*mask) + lambdas*criterion(train_output*(~mask), train_label*(~mask))
        loss =  loss_train + args.lambda2*criterion(output[-label_truth.shape[0]:], label_truth.float())
        loss.backward()
        optimizer.step()
        
        
        model.eval()
        with torch.no_grad():
            _, output = model(features, adj)
        mask = labels[idx_val] > 0.5
        loss_val = criterion(output[idx_val]*mask, labels[idx_val]*mask)+lambdas*criterion(output[idx_val]*(~mask), labels[idx_val]*(~mask))
        loss = criterion(output[idx_val], labels[idx_val])
        val_loss.append(loss_val.item())

        # scheduler.step(loss_val)
        acc_train = cal_accuracy(output[idx_train], labels[idx_train])
        acc_val = cal_accuracy(output[idx_val], labels[idx_val])
        if acc_val > best_val_f1:
            best_val_f1 = acc_val
            torch.save(model,save_path)
        acc_test = cal_accuracy(output[idx_test], labels[idx_test])
        if show_result:
            file_name = args.activate_type +"lambda1_"+str(args.lambda1)+"_lambda2_"+str(args.lambda2)+"_resrate_"+str(args.resrate)+"_lr_"+str(args.learning_rate)+ "_norm_edge_" + str(args.norm_edge)
            with open(f"output_{args.dataset_type}/{args.output_dir}/"+file_name,'a',encoding='utf-8') as f:
                print(  'Epoch: {:04d}'.format(epoch+1),
                        'loss_train: {:.4f}'.format(loss_train.item()),
                        'f1_train: {:.4f}'.format(acc_train),
                        'loss_val: {:.4f}'.format(loss_val.item()),
                        'f1_val: {:.4f}'.format(acc_val),
                        'f1_test: {:.4f}'.format(acc_test),
                        'time: {:.4f} s'.format(time.time()-t),
                        file=f
                )

        # adjust_learning_rate(optimizer,epoch,1)
        if epoch > 1200 and np.min(val_loss[-args.early_stopping:]) > np.min(val_loss[:-args.early_stopping]) :
            if show_result:
                print("Early Stopping...")
            break