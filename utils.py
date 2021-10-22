import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data, Dataset


def get_class_weights(disasters: List[str], dataset: Dataset, num_classes: int, leaked: bool=False) -> torch.Tensor:
    """
        Computes the class weights yo be used in the loss function for mitigating the effect of class imbalance.

        Args:
            disasters (List[str]): names of the included datasets.
            dataset (torch_geometric.data.Dataset): PyG dataset instance.
            num_classes (int): number of classes in the dataset.
        
        Returns:
            class_weights (Tensor): class weights tensor of shape (n_classes).
    """
    name = '_'.join(text.replace('-', '_') for text in disasters)
    if leaked:
        name = name + '_leaked'
    if os.path.isfile(f'weights/class_weights_{name}_{num_classes}.pt'):
        return torch.load(f'weights/class_weights_{name}_{num_classes}.pt')
    else:
        y_all = [data.y for data in dataset]
        y_all = torch.cat(y_all).numpy()
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_all), y=y_all)
        class_weights = torch.Tensor(class_weights)
        torch.save(class_weights, f'weights/class_weights_{name}_{num_classes}.pt')
        return class_weights


def merge_classes(data: Data):
    data.y[data.y==3] = 2
    return data


def score(y_true: torch.Tensor, y_pred: torch.Tensor) -> Tuple[float]:
    """
    Calculates the accuracy, macro F1 score, weighted F1 score and the ROC AUC score.
    
    Args:
        y_true (torch.Tensor) of shape (n_samples) containing true labels.
        y_pred (torch.Tensor) of shape (n_samples,n_classes) containing the log softmax probabilities.
    """
    accuracy = accuracy_score(y_true, y_pred.argmax(dim=1, keepdim=True))
    f1_macro = f1_score(y_true, y_pred.argmax(dim=1, keepdim=True), average='macro')
    f1_weighted = f1_score(y_true, y_pred.argmax(dim=1, keepdim=True), average='weighted')
    auc = roc_auc_score(y_true, torch.exp(y_pred), average='macro', multi_class='ovr')
    return accuracy, f1_macro, f1_weighted, auc


def score_cm(cm: np.ndarray) -> Tuple[float]:
    accuracy = np.trace(cm) / np.sum(cm)
    fp = cm.sum(axis=0) - np.diag(cm) 
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = (2*tp) / (2*tp + fp + fn)
    return accuracy, precision.mean(), recall.mean(), specificity.mean(), f1.mean()


def make_plot(train: np.ndarray, test: np.ndarray, plot_type: str, model_name: str) -> None:
    plt.figure()
    plt.plot(train)
    plt.plot(test)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel(plot_type)
    plt.savefig('results/'+model_name+'_'+plot_type+'.pdf')
    plt.close()


def stratified_graph_leak(dataset: Dataset, split: float=0.1):
    num_negative = 0
    for data in dataset:
        if not data.y.sum():
            num_negative += 1
    
    size_split = round(split * len(dataset))
    num_negative_split = round(num_negative/len(dataset) * size_split)
    num_positive_split = size_split - num_negative_split
    idx = torch.empty(len(dataset), dtype=bool)

    for i, data in enumerate(dataset):
        if data.y.sum() and num_positive_split > 0:
            idx[i] = True
            num_positive_split -= 1
        elif not data.y.sum() and num_negative_split > 0:
            idx[i] = True
            num_negative_split -= 1
        else:
            idx[i] = False
    
    return dataset.index_select(idx), dataset.index_select(~idx)