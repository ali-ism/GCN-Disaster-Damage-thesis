import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing_extensions import Tuple

"""
def parse_ordinal_output(out: torch.Tensor) -> torch.Tensor:
    idx = torch.where(out < 0.5, 1, 0)
    idx = torch.argmax(idx, dim=1)
    idx = torch.where(idx == 0, 4, idx) - 1
    return idx


def to_onehot(y_ord: torch.Tensor, num_classes=4) -> torch.Tensor:
    y_ord = parse_ordinal_output(y_ord)
    return F.one_hot(y_ord.long(), num_classes=num_classes)
"""

def score(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float]:
    """
    Claculates the accuracy, macro F1 score, weighted F1 score, xview2 F1 score and the ROC AUC score. The xview2 F1 score is calculated according to https://github.com/DIUx-xView/xView2_scoring.
    
    Args:
        y_true (torch.Tensor) of shape (n_samples) containing true labels.
        y_pred (torch.Tensor) of shape (n_samples,n_classes) containing the log softmax probabilities.
    """
    accuracy = accuracy_score(y_true, y_pred.argmax(dim=1, keepdim=True))
    f1_macro = f1_score(y_true, y_pred.argmax(dim=1, keepdim=True), average='macro')
    f1_weighted = f1_score(y_true, y_pred.argmax(dim=1, keepdim=True), average='weighted')
    f1_classes = f1_score(y_true, y_pred.argmax(dim=1, keepdim=True), average=None)
    epsilon = 1e-6
    xview2_f1 = len(f1_classes) / sum((f1+epsilon)**-1 for f1 in f1_classes)
    #auc = roc_auc_score(y_true, torch.exp(y_pred), average='macro', multi_class='ovr')
    auc = 0
    return accuracy, f1_macro, f1_weighted, xview2_f1, auc