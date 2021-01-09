# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from elsefile import savefile,savefile1
# from pytorch_pretrained_bert.optimization import BertAdam
from first import *


# class beipoyuanli():
#     # 权重初始化，默认xavier
#     np.set_printoptions(threshold=np.inf)
#     def __init__(self):
#         pass
#     def init_network(model, method='xavier', exclude='embedding', seed=123):
#         for name, w in model.named_parameters():
#             if exclude not in name:
#                 if len(w.size()) < 2:
#                     continue
#                 if 'weight' in name:
#                     if method == 'xavier':
#                         nn.init.xavier_normal_(w)
#                     elif method == 'kaiming':
#                         nn.init.kaiming_normal_(w)
#                     else:
#                         nn.init.normal_(w)
#                 elif 'bias' in name:
#                     nn.init.constant_(w, 0)
#                 else:
#                     pass
#
#     #guang ce shi zhu xiao for epoch in range(config.num_epochs):        if flag: break
#     def train(config, model,train_iter,dev_iter, test_iter):
#     # def train(config, model, test_iter):
#         start_time = time.time()
#         # print("test_iter",test_iter)
#         # model.train()
#         # param_optimizer = list(model.named_parameters())
#         # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#         # optimizer_grouped_parameters = [
#         #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#         #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
#         # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
#         # # print("***********optimizer******************", optimizer)
#         # # optimizer = BertAdam(optimizer_grouped_parameters,
#         # #                      lr=config.learning_rate,
#         # #                      warmup=0.05,
#         # #                      t_total=len(train_iter) * config.num_epochs)
#         # total_batch = 0  # 记录进行到多少batch
#         # dev_best_loss = float('inf')
#         # last_improve = 0  # 记录上次验证集loss下降的batch数
#         # flag = False  # 记录是否很久没有效果提升
#         # model.train()
#
#         #从下一行到 test(config, model, test_iter) 为训练
#         # for epoch in range(config.num_epochs):
#         #     print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
#         #     for i, (trains, labels) in enumerate(train_iter):
#         #         outputs = model(trains)
#         #         model.zero_grad()
#         #         loss = F.cross_entropy(outputs, labels)
#         #         loss.backward()
#         #         optimizer.step()
#         #         if total_batch % 1 == 0:
#         #             # 每多少轮输出在训练集和验证集上的效果
#         #             true = labels.data.cpu()
#         #             predic = torch.max(outputs.data, 1)[1].cpu()
#         #             train_acc = metrics.accuracy_score(true, predic)
#         #             dev_acc, dev_loss = evaluate(config, model, dev_iter)
#         #             print ("*********************")
#         #             if dev_loss < dev_best_loss:
#         #                 dev_best_loss = dev_loss
#         #                 torch.save(model.state_dict(), config.save_path)
#         #                 improve = '*'
#         #                 last_improve = total_batch
#         #             else:
#         #                 improve = ''
#         #             time_dif = get_time_dif(start_time)
#         #             msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
#         #             print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
#         #             savefile(epoch,loss.item(), train_acc, dev_loss, dev_acc)
#         #             model.train()
#         #         total_batch += 1
#         #         if total_batch - last_improve > config.require_improvement:
#         #             # 验证集loss超过1000batch没下降，结束训练
#         #             print("No optimization for a long time, auto-stopping...")
#         #             flag = True
#         #             break
#         #     if flag:
#         #         break
#         beipoyuanli.test(config, model, test_iter)
#
#
#     def test(config, model, test_iter):
#         # test
#
#         state_dict = torch.load(config.save_path,map_location=lambda storage,loc:storage)
#         # from collections import  OrderedDict
#         # new_state_dict= OrderedDict()
#         # for k,v in state_dict.items():
#         #     name=k[7:]
#         #     new_state_dict[name]=v
#         model.load_state_dict(state_dict)
#
#         start_time = time.time()
#         with torch.no_grad():
#             # test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
#             beipoyuanli.evaluate(config, model, test_iter, test=True)
#             # msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
#             # print(msg.format(test_loss, test_acc))
#             # print("Precision, Recall and F1-Score...")
#             # print(test_report)
#             # print("Confusion Matrix...")
#             # print(test_confusion)
#             # time_dif = get_time_dif(start_time)
#             # print("Time usage:", time_dif)
#
#
#     def evaluate(config, model, data_iter, test=False):
#         model.eval()
#         loss_total = 0
#         i=0
#         predictchange = np.array([], dtype=int)
#         labelschange = np.array([], dtype=int)
#         predict_all = np.array([], dtype=int)
#         labels_all = np.array([], dtype=int)
#         with torch.no_grad():
#             for texts, labels in data_iter:
#                 i=i+1
#                 outputs = model(texts)
#                 loss = F.cross_entropy(outputs, labels)
#                 # print ("*************************",float(loss))
#                 loss_total += float(loss)
#                 labels = labels.data.cpu().numpy()
#                 # print ("*******************",labels)
#                 labelschange= np.append(labelschange,labels)
#                 predic = torch.max(outputs.data, 1)[1].cpu().numpy()#每一次预测的值
#                 # print ("********predic***********", predic)
#                 #xq
#                 login.woyouhuilaile(predic)
#                 #xqgai
#                 predictchange = np.append(predictchange, predic)
#                 acc=metrics.accuracy_score(labelschange, predictchange)#每一次预测的准确值
#
#                 # savefile1(i,float(loss),acc)
#                 labels_all = np.append(labels_all, labels)
#                 # print(labels_all)
#                 predict_all = np.append(predict_all, predic)#预测结果总和
#                 # print(predict_all)
#                 # print(config.class_list)
#         # print ("********predict_all***********", predict_all)
#
#         acc_total = metrics.accuracy_score(labels_all, predict_all)
#         if test:
#             # report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
#             # confusion = metrics.confusion_matrix(labels_all, predict_all)
#             # return acc_total, loss_total / len(data_iter), report, confusion
#             pass
#         return acc_total, loss_total / len(data_iter)

#
# if __name__ == '__main__':
#     dataset = 'THUCNews'  # 数据集

np.set_printoptions(threshold=np.inf)
def __init__(self):
    pass
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

#guang ce shi zhu xiao for epoch in range(config.num_epochs):        if flag: break
def train(config, model, test_iter):
    predic=test(config, model, test_iter)
    return predic

def test(config, model, test_iter):
    # test

    state_dict = torch.load(config.save_path,map_location=lambda storage,loc:storage)
    # from collections import  OrderedDict
    # new_state_dict= OrderedDict()
    # for k,v in state_dict.items():
    #     name=k[7:]
    #     new_state_dict[name]=v
    model.load_state_dict(state_dict)

    start_time = time.time()
    with torch.no_grad():
        # test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
        predic=evaluate(config, model, test_iter, test=True)

        # msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        # print(msg.format(test_loss, test_acc))
        # print("Precision, Recall and F1-Score...")
        # print(test_report)
        # print("Confusion Matrix...")
        # print(test_confusion)
        # time_dif = get_time_dif(start_time)
        # print("Time usage:", time_dif)
    return predic
def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    i=0
    predictchange = np.array([], dtype=int)
    labelschange = np.array([], dtype=int)
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            i=i+1
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            # print ("*************************",float(loss))
            loss_total += float(loss)
            labels = labels.data.cpu().numpy()
            # print ("*******************",labels)
            labelschange= np.append(labelschange,labels)
            # global predic
            predic= torch.max(outputs.data, 1)[1].cpu().numpy()#每一次预测的值
    #         # login.woyouhuilaile(login,predic)
    #         #xqgai
    #         predictchange = np.append(predictchange, predic)
    #         acc=metrics.accuracy_score(labelschange, predictchange)#每一次预测的准确值
    #
    #         # savefile1(i,float(loss),acc)
    #         labels_all = np.append(labels_all, labels)
    #         # print(labels_all)
    #         predict_all = np.append(predict_all, predic)#预测结果总和
    #         # print(predict_all)
    #         # print(config.class_list)
    # # print ("********predict_all***********", predict_all)
    return predic
    acc_total = metrics.accuracy_score(labels_all, predict_all)
    if test:
        # report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        # confusion = metrics.confusion_matrix(labels_all, predict_all)
        # return acc_total, loss_total / len(data_iter), report, confusion
        pass
    # return acc_total, loss_total / len(data_iter)
# def zaineibudehanshu(meiyong):
#     meiyong+='1'
#     return predic