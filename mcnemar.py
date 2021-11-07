import json
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import RandomNodeSampler

from dataset import xBDBatch, xBDImages
from model import CNNSage, SiameseNet
from utils import merge_classes

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


if __name__ == "__main__":

    if settings_dict['merge_classes']:
        transform = merge_classes
    else:
        transform = None

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

    print('\nComparing the two y_preds:')
    print(np.array_equal(y_true_sage, y_true_cnn))