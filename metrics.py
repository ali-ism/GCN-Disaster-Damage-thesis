import torch
from sklearn.metrics import f1_score

def parse_ordinal_output(out: torch.Tensor) -> torch.Tensor:
    idx = torch.where(out < 0.5, 1, 0)
    idx = torch.argmax(idx, dim=1)
    idx = torch.where(idx == 0, 4, idx)
    return idx

def xview2_f1_score(y_true: torch.Tensor, out: torch.Tensor) -> float:
    """
    According to https://github.com/DIUx-xView/xView2_scoring:
        the df1 is calculated by taking the harmonic mean of the 4 damage f1 scores (no damage, minor damage, major damage, and destroyed) df1 = 4 / sum((f1+epsilon)**-1 for f1 in [no_damage_f1, minor_damage_f1, major_damage_f1, destroyed_f1]), where epsilon = 1e-6
    
    Args:
        y_true (torch.Tensor)
        out (torch.Tensor)
    """
    y_pred = parse_ordinal_output(out).detach().numpy()
    y_true = parse_ordinal_output(y_true).detach().numpy()
    f1_classes = f1_score(y_true, y_pred, average=None)
    epsilon = 1e-6
    return 4 / sum((f1+epsilon)**-1 for f1 in f1_classes)