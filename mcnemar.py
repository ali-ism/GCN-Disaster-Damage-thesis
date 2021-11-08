import json
import os.path as osp
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import RandomNodeSampler

from dataset import xBDBatch, xBDImages
from model import CNNSage, SiameseNet
from utils import merge_classes

from statsmodels.stats.contingency_tables import mcnemar

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

batch_size = settings_dict['data_sup']['batch_size']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@torch.no_grad()
def infer_cnn() -> Tuple[np.ndarray]:
    y_true = []
    y_pred = []
    for data in hold_loader:
        x = data['x'].to(device)
        y = data['y']
        out = model(x).cpu()
        y_pred.append(out)
        y_true.append(y)
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    return y_true.numpy(), y_pred.numpy()


@torch.no_grad()
def infer_sage() -> Tuple[np.ndarray]:
    y_true = []
    y_pred = []
    for data in hold_dataset:
        if data.num_nodes > batch_size:
            sampler = RandomNodeSampler(data, num_parts=data.num_nodes//batch_size, num_workers=2)
            for subdata in sampler:
                subdata = subdata.to(device)
                out = model(subdata.x, subdata.edge_index).cpu()
                y_pred.append(out)
                y_true.append(subdata.y.cpu())
        else:
            data = data.to(device)
            out = model(data.x, data.edge_index).cpu()
            y_pred.append(out)
            y_true.append(data.y.cpu())
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    return y_true.numpy(), y_pred.numpy()


def contingency_table(y_target: np.ndarray, y_model1: np.ndarray, y_model2: np.ndarray) -> np.ndarray:
    """
    Compute a 2x2 contigency table for McNemar's test.
    Source: Raschka, Sebastian (2018) MLxtend: Providing machine learning and data science utilities and extensions to Python's scientific computing stack. J Open Source Softw 3(24).
    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True class labels as 1D NumPy array.
    y_model1 : array-like, shape=[n_samples]
        Predicted class labels from model as 1D NumPy array.
    y_model2 : array-like, shape=[n_samples]
        Predicted class labels from model 2 as 1D NumPy array.
    Returns
    ----------
    tb : array-like, shape=[2, 2]
       2x2 contingency table with the following contents:
       a: tb[0, 0]: # of samples that both models predicted correctly
       b: tb[0, 1]: # of samples that model 1 got right and model 2 got wrong
       c: tb[1, 0]: # of samples that model 2 got right and model 1 got wrong
       d: tb[1, 1]: # of samples that both models predicted incorrectly
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar_table/
    """
    for ary in (y_target, y_model1, y_model2):
        if len(ary.shape) != 1:
            raise ValueError('One or more input arrays are not 1-dimensional.')

    if y_target.shape[0] != y_model1.shape[0]:
        raise ValueError('y_target and y_model1 contain a different number'
                         ' of elements.')

    if y_target.shape[0] != y_model2.shape[0]:
        raise ValueError('y_target and y_model2 contain a different number'
                         ' of elements.')

    m1_vs_true = (y_target == y_model1).astype(int)
    m2_vs_true = (y_target == y_model2).astype(int)

    plus_true = m1_vs_true + m2_vs_true
    minus_true = m1_vs_true - m2_vs_true

    tb = np.zeros((2, 2), dtype=int)

    tb[0, 0] = np.sum(plus_true == 2)
    tb[0, 1] = np.sum(minus_true == 1)
    tb[1, 0] = np.sum(minus_true == -1)
    tb[1, 1] = np.sum(plus_true == 0)

    return tb


if __name__ == "__main__":

    if settings_dict['merge_classes']:
        transform = merge_classes
    else:
        transform = None
    
    if osp.isfile('results/y_true_sage.npy') and osp.isfile('results/y_pred_sage.npy'):
        y_true_sage = np.load('results/y_true_sage.npy')
        y_pred_sage = np.load('results/y_pred_sage.npy')
    else:
        hold_dataset = xBDBatch(
            '/home/ami31/scratch/datasets/xbd_graph/socal_hold',
            '/home/ami31/scratch/datasets/xbd/hold_bldgs/',
            'socal-fire',
            transform=transform
        )
        num_classes = 3 if settings_dict['merge_classes'] else hold_dataset.num_classes
        model = CNNSage(
            settings_dict['model']['hidden_units'],
            num_classes,
            settings_dict['model']['num_layers'],
            settings_dict['model']['dropout_rate']
        )
        name = settings_dict['model']['name'] + '_sage'
        model_path = 'weights/' + name
        model.load_state_dict(torch.load(model_path+'_best.pt'))
        model.eval()
        model = model.to(device)
        y_true_sage, y_pred_sage = infer_sage()
        np.save('results/y_true_sage.npy', y_true_sage)
        np.save('results/y_pred_sage.npy', y_pred_sage)
    
    if osp.isfile('results/y_true_cnn.npy') and osp.isfile('results/y_pred_cnn.npy'):
        y_true_cnn = np.load('results/y_true_cnn.npy')
        y_pred_cnn = np.load('results/y_pred_cnn.npy')
    else:
        hold_dataset = xBDImages(
            ['/home/ami31/scratch/datasets/xbd/hold_bldgs/'],
            ['socal-fire'],
            merge_classes
        )
        hold_loader = DataLoader(hold_dataset, batch_size)
        model = SiameseNet(
            settings_dict['model']['hidden_units'],
            num_classes,
            settings_dict['model']['dropout_rate']
        )
        name = settings_dict['model']['name'] + '_siamese'
        model_path = 'weights/' + name
        model.load_state_dict(torch.load(model_path+'_best.pt'))
        model.eval()
        model = model.to(device)
        y_true_cnn, y_pred_cnn = infer_cnn()
        np.save('results/y_true_cnn.npy', y_true_cnn)
        np.save('results/y_pred_cnn.npy', y_pred_cnn)

    assert np.array_equal(y_true_sage, y_true_cnn)

    table = contingency_table(y_true_cnn, y_pred_sage, y_pred_cnn)

    if table[0,1] + table[1,0] < 25:
        stat, p_value = mcnemar(table, exact=True)
    else:
        stat, p_value = mcnemar(table, exact=False, correction=True)
    
    print(f'p value: {p_value}')
    if p_value <= 0.05:
        print('Null hypothesis rejected')
    else:
        print('Failed to reject null hypothesis')