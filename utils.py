import os
from math import sqrt
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import sort_edge_index
from torch_sparse import SparseTensor


def build_edge_idx(num_nodes: int) -> torch.Tensor:
    """
        Build a complete undirected graph for the edge_index parameter.

        Args:
            num_nodes (int): number of nodes in the graph.
        
        Returns:
            E (LongTensor): graph connectivity with shape [2, num_edges].

        Adapted from:
            https://github.com/rusty1s/pytorch_geometric/issues/964
    """
    # Initialize edge index matrix
    E = torch.zeros((2, (num_nodes * (num_nodes - 1))//2), dtype=torch.long)
    # Populate 1st row
    i = 0
    for node in range(num_nodes):
        for _ in range(num_nodes - node - 1):
            E[0, i] = node
            i+=1
    # Populate 2nd row
    neighbors = []
    for node in range(num_nodes):
        neighbors.extend(np.arange(node+1, num_nodes))
    E[1, :] = torch.Tensor(neighbors)
    return E


def euclidean_similarity(coords1: Tuple[float], coords2: Tuple[float]) -> float:
    """
    Normalized euclidean distance similarity.

    Args:
        coords1 (Tuple[float]): xy coordinates of the first node.
        coords2 (Tuple[float]): xy coordinates of the second node.
    """
    x1 = coords1[0]
    y1 = coords1[1]
    x2 = coords2[0]
    y2 = coords2[1]
    euc_dist = sqrt((x1-x2)**2 + (y1-y2)**2)
    euc_sim = 1 / (1 + euc_dist)
    return euc_sim


def get_edge_features(node1: torch.Tensor, node2: torch.Tensor, coords1: Tuple[float], coords2: Tuple[float]) -> Tuple[float]:
    """
        Computes the edge weights between two given nodes.

        Args:
            node1 (Tensor): feature vector of the first node.
            node2 (Tensor): feature vector of the second node.
            coords1 (Tuple[float]): xy pixel coordinates of node1.
            coords2 (Tuple[float]): xy pixel coordinates of node2.
        
        Returns:
            node_sim (float): normalized node feature similarity.
            euc_sim (float): normalized euclidean distance similarity between the node image objects.
        
        Node feature similarity is based on the adjacency matrix built by:
            S. Saha, L. Mou, X. X. Zhu, F. Bovolo and L. Bruzzone, "Semisupervised Change Detection Using Graph Convolutional Network," in IEEE Geoscience and Remote Sensing Letters, vol. 18, no. 4, pp. 607-611, April 2021, doi: 10.1109/LGRS.2020.2985340.
    """
    D = node1.shape[0]
    s = (torch.abs(node1 - node2)) / (torch.abs(node1) + torch.abs(node2))
    s[s.isnan()] = 1
    node_sim = 1 - torch.sum(s)/D
    euc_sim = euclidean_similarity(coords1, coords2)
    return node_sim.item(), euc_sim


def get_class_weights(disasters: List[str], dataset: torch_geometric.data.Dataset, num_classes: int, leaked: bool=False) -> torch.Tensor:
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


def merge_classes(data: torch_geometric.data.Data):
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


def make_plot(train: np.ndarray, test: np.ndarray, plot_type: str, model_name: str) -> None:
    plt.figure()
    plt.plot(train)
    plt.plot(test)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel(plot_type)
    plt.savefig('results/'+model_name+'_'+plot_type+'.pdf')
    plt.close()


def stratified_graph_leak(dataset: torch_geometric.data.Dataset, split: float=0.1):
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


class ToSparseTensor(BaseTransform):
    r"""Converts the :obj:`edge_index` attributes of a homogeneous or
    heterogeneous data object into a (transposed)
    :class:`torch_sparse.SparseTensor` type with key :obj:`adj_.t`.

    .. note::

        In case of composing multiple transforms, it is best to convert the
        :obj:`data` object to a :obj:`SparseTensor` as late as possible, since
        there exist some transforms that are only able to operate on
        :obj:`data.edge_index` for now.

    Args:
        attr: (str, optional): The name of the attribute to add as a value to
            the :class:`~torch_sparse.SparseTensor` object (if present).
            (default: :obj:`edge_weight`)
        remove_edge_index (bool, optional): If set to :obj:`False`, the
            :obj:`edge_index` tensor will not be removed.
            (default: :obj:`True`)
        fill_cache (bool, optional): If set to :obj:`False`, will not fill the
            underlying :obj:`SparseTensor` cache. (default: :obj:`True`)
    """
    def __init__(self, attr: Optional[str] = 'edge_weight',
                 remove_edge_index: bool = True, fill_cache: bool = True):
        self.attr = attr
        self.remove_edge_index = remove_edge_index
        self.fill_cache = fill_cache

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            nnz = store.edge_index.size(1)

            keys, values = [], []
            for key, value in store.items():
                if isinstance(value, Tensor) and value.size(0) == nnz:
                    keys.append(key)
                    values.append(value)

            store.edge_index, values = sort_edge_index(store.edge_index,
                                                       values,
                                                       sort_by_row=False)

            for key, value in zip(keys, values):
                store[key] = value

            store.adj_t = SparseTensor(
                row=store.edge_index[1], col=store.edge_index[0],
                value=None if self.attr is None or self.attr not in store else
                store[self.attr], sparse_sizes=store.size()[::-1],
                is_sorted=True)

            if self.remove_edge_index:
                del store['edge_index']
                if self.attr is not None and self.attr in store:
                    del store[self.attr]

            if self.fill_cache:  # Pre-process some important attributes.
                store.adj_t.storage.rowptr()
                store.adj_t.storage.csr2csc()

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
