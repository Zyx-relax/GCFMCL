from sklearn.ensemble import RandomForestClassifier
from util import *
from model import mlp
import numpy as np
import torch
import warnings
import random
import tqdm
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,roc_curve, auc,precision_recall_curve,average_precision_score,f1_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
colors = list(mcolors.TABLEAU_COLORS.keys())
def test():
    model.eval()
    with torch.no_grad():
        out = model(test_data)
    aucc = roc_auc_score(test_label.unsqueeze(-1), out.cpu())
    temp = torch.tensor(out)
    temp[temp >= 0.5] = 1
    temp[temp < 0.5] = 0
    acc, sen, pre, spe = calculate_metrics(test_label, temp)
    F1=f1_score(test_label.cpu(), temp.cpu())
    aupr=average_precision_score(test_label.cpu().numpy(), out.cpu().numpy())
    print("auc:{},acc:{},pre:{},sen:{},f1:{},aupr:{}".format(aucc*100, acc*100,pre*100, sen*100, F1*100,aupr*100))
    
    fpr, tpr, thresholds = roc_curve(test_label, out.cpu())
    roc_auc = auc(fpr, tpr)

    x = np.linspace(0, 1, 100)
    f = interp1d(fpr, tpr)
    tpr = f(x)
    fpr = x

    plt.plot(fpr, tpr, color=mcolors.TABLEAU_COLORS[colors[fold]], label=f'Fold {fold} ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color=mcolors.TABLEAU_COLORS[colors[fold]], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

pos_data, neg_data = load_data()
pos_list, neg_list = split_data(pos_data, neg_data)

pos_data = deal_embedding(pos_list)
neg_data = deal_embedding(neg_list)
for fold in range(5):
    train_data, train_label, test_data, test_label = split_train_test(pos_data, neg_data, fold)

    model = mlp(len(train_data[0]), 256, 64, 1)
    opt = torch.optim.Adam(params=model.parameters(), lr=0.005, weight_decay=5e-4)
    loss_fn = torch.nn.BCELoss()
    best_loss = 1
    best_model = model
    for epoch in tqdm.tqdm(range(2000),desc=f'FOLD {fold} Training~'):
        model.train()
        out = model(train_data)
        loss = loss_fn(out, train_label.unsqueeze(-1))
        if loss < best_loss:
            beat_model = model
        opt.zero_grad()
        loss.backward()
        opt.step()
    test()

plt.show()



