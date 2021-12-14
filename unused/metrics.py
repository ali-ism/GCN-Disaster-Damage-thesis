import torch
import torch.nn.functional as F


def parse_ordinal_output(out: torch.Tensor) -> torch.Tensor:
    idx = torch.where(out < 0.5, 1, 0)
    idx = torch.argmax(idx, dim=1)
    idx = torch.where(idx == 0, 4, idx) - 1
    return idx


def to_onehot(y_ord: torch.Tensor, num_classes=4) -> torch.Tensor:
    y_ord = parse_ordinal_output(y_ord)
    return F.one_hot(y_ord.long(), num_classes=num_classes)