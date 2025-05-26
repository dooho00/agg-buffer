import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score

def kl_div(x, y):
    x = F.log_softmax(x, dim=1)
    y = F.softmax(y, dim=1)
    return F.kl_div(x, y, reduction='batchmean')

def calculate_metrics(logits_, labels, nodes):
    pred = logits_[nodes]
    true = labels[nodes]
    acc = np.mean(pred == true)
    f1 = f1_score(true, pred, average='macro')
    return acc, f1