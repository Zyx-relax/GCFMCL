import numpy as np
import random
import torch
import pandas as pd
np.random.seed(2)
random.seed(2)


def load_data():
    embedding1 = np.loadtxt('./mlp/data/embedding1.txt', dtype=float, delimiter=" ")
    embedding2 = np.loadtxt('./mlp/data/embedding2.txt', dtype=float, delimiter=" ")
    adj = np.loadtxt('./mlp/data/interaction.txt', dtype=float, delimiter=" ")
    pos_data = []
    neg_data = []

    for i in range(len(adj)):
        for j in range(len(adj[0])):
            if adj[i][j] == 1:
                pos_data.append([embedding1[i], embedding2[j]])
            else:
                neg_data.append([embedding1[i], embedding2[j]])
    return pos_data, neg_data


def split_data(pos_data, neg_data):
    n_pos = len(pos_data)

    pos_list = pos_data
    neg_list = []
    times = n_pos
    while times:
        index = np.random.randint(len(neg_data))
        neg_list.append(neg_data[index])
        del neg_data[index]
        times = times - 1
    return pos_list, neg_list


def deal_embedding(list):
    number = len(list)

    result = []
    for i in range(number):
        result.append(np.concatenate([list[i][0], list[i][1]], axis=0))
    return result


def split_train_test(pos_list, neg_list, i):
    train_data,train_label,test_data,test_label=[],[],[],[]
    for index in range(len(pos_list)):
        if index%5==i:
            test_data.append(pos_list[index])
            test_label.append(1)
            test_data.append(neg_list[index])
            test_label.append(0)
        else:
            train_data.append(pos_list[index])
            train_label.append(1)
            train_data.append(neg_list[index])
            train_label.append(0)

    train_data, train_label, test_data, test_label = torch.tensor(train_data), torch.tensor(train_label), torch.tensor(test_data), torch.tensor(test_label)
    train_data = train_data.to(torch.float32)
    train_label = train_label.to(torch.float32)
    test_data = test_data.to(torch.float32)
    test_label = test_label.to(torch.float32)
    return train_data, train_label, test_data, test_label


def calculate_metrics(y_true, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    precision = TP / (TP + FP)
    specificity = TN / (TN + FP)
    return accuracy, sensitivity, precision, specificity




