import os
from math import sqrt
import numpy as np
import torch
from typing import List, Tuple
from metrics import parse_ordinal_output
from sklearn.utils.class_weight import compute_class_weight


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
    s[s.isnan()] = 0
    node_sim = 1 - torch.sum(s)/D
    euc_sim = euclidean_similarity(coords1, coords2)
    return node_sim.item(), euc_sim


def get_class_weights(set_id: int, train_data_list: List) -> torch.Tensor:
    if os.path.isfile(f'weights/class_weights_{set_id}.pt'):
        return torch.load(f'weights/class_weights_{set_id}.pt')
    else:
        y_all = []
        for dataset in train_data_list:
            y_all.extend([data.y for data in dataset])
        y_all = torch.cat(y_all)
        y_all = parse_ordinal_output(y_all)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_all), y=y_all)
        class_weights = torch.Tensor(class_weights)
        torch.save(class_weights, f'weights/class_weights_{set_id}.pt')
        return class_weights