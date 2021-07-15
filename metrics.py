import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def parse_ordinal_output(out: torch.Tensor) -> np.ndarray:
    idx = torch.where(out < 0.5, 1, 0)
    idx = torch.argmax(idx, dim=1)
    idx = torch.where(idx == 0, 4, idx)
    return idx.detach().numpy()


def to_onehot(y_ord: torch.Tensor, num_classes=4):
    y_ord = parse_ordinal_output(y_ord) - 1
    return F.one_hot(torch.LongTensor(y_ord), num_classes=num_classes)


def xview2_f1_score(y_true: torch.Tensor, out: torch.Tensor) -> float:
    """
    According to https://github.com/DIUx-xView/xView2_scoring:
        the df1 is calculated by taking the harmonic mean of the 4 damage f1 scores (no damage, minor damage, major damage, and destroyed) df1 = 4 / sum((f1+epsilon)**-1 for f1 in [no_damage_f1, minor_damage_f1, major_damage_f1, destroyed_f1]), where epsilon = 1e-6
    
    Args:
        y_true (torch.Tensor)
        out (torch.Tensor)
    """
    y_pred = parse_ordinal_output(out)
    y_true = parse_ordinal_output(y_true)
    f1_classes = f1_score(y_true, y_pred, average=None)
    epsilon = 1e-6
    return len(f1_classes) / sum((f1+epsilon)**-1 for f1 in f1_classes)