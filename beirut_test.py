import json
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor
from torch_geometric.transforms import Compose, GCNNorm, ToSparseTensor

from dataset import BeirutFullGraph
from model import CNNGCN
from utils import score_cm

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

n_epochs = settings_dict['epochs']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train() -> Tuple[float]:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(input=out, target=data.y[train_idx], weight=class_weights.to(device))
    loss.backward()
    optimizer.step()
    cm = confusion_matrix(data.y[train_idx].cpu(), out.detach().cpu().argmax(dim=1, keepdims=True))
    accuracy, precision, recall, specificity, f1 = score_cm(cm)
    return loss.detach().cpu().item(), accuracy, precision, recall, specificity, f1


@torch.no_grad()
def test(mask: Tensor) -> Tuple[float]:
    model.eval()
    out = model(data.x, data.adj_t)[mask].cpu()
    loss = F.nll_loss(input=out, target=data.y[mask].cpu(), weight=class_weights)
    cm = confusion_matrix(data.y[mask].cpu(), out.argmax(dim=1, keepdims=True))
    accuracy, precision, recall, specificity, f1 = score_cm(cm)
    return loss.item(), accuracy, precision, recall, specificity, f1


def merge_classes(data):
    """
    Merges the first two classes into a single class.
    
    Args:
        data: torch_geometric.data.Data object.

    Returns:
        data: transformed torch_geometric.data.Data object.
    """
    data.y[data.y>0] -= 1
    return data


if __name__ == "__main__":

    if settings_dict['merge_classes']:
        transform = Compose([merge_classes, GCNNorm(), ToSparseTensor()])
    else:
        transform = Compose([GCNNorm(), ToSparseTensor()])
    
    train_loss = np.empty(n_epochs)
    train_acc = np.empty(n_epochs)
    train_precision = np.empty(n_epochs)
    train_recall = np.empty(n_epochs)
    train_specificity = np.empty(n_epochs)
    train_f1 = np.empty(n_epochs)
    test_loss = np.empty(n_epochs)
    test_acc = np.empty(n_epochs)
    test_precision = np.empty(n_epochs)
    test_recall = np.empty(n_epochs)
    test_specificity = np.empty(n_epochs)
    test_f1 = np.empty(n_epochs)
    hold_loss = np.empty(n_epochs)
    hold_acc = np.empty(n_epochs)
    hold_precision = np.empty(n_epochs)
    hold_recall = np.empty(n_epochs)
    hold_specificity = np.empty(n_epochs)
    hold_f1 = np.empty(n_epochs)

    train_loss_meta = np.empty(n_epochs)
    train_acc_meta = np.empty(n_epochs)
    train_precision_meta = np.empty(n_epochs)
    train_recall_meta = np.empty(n_epochs)
    train_specificity_meta = np.empty(n_epochs)
    train_f1_meta = np.empty(n_epochs)
    test_loss_meta = np.empty(n_epochs)
    test_acc_meta = np.empty(n_epochs)
    test_precision_meta = np.empty(n_epochs)
    test_recall_meta = np.empty(n_epochs)
    test_specificity_meta = np.empty(n_epochs)
    test_f1_meta = np.empty(n_epochs)
    hold_loss_meta = np.empty(n_epochs)
    hold_acc_meta = np.empty(n_epochs)
    hold_precision_meta = np.empty(n_epochs)
    hold_recall_meta = np.empty(n_epochs)
    hold_specificity_meta = np.empty(n_epochs)
    hold_f1_meta = np.empty(n_epochs)

    print('########################### No Meta Features ###########################')
    dataset = BeirutFullGraph(
        '/home/ami31/scratch/datasets/beirut_bldgs/beirut_graph',
        '/home/ami31/scratch/datasets/beirut_bldgs',
        settings_dict['data_ss']['reduced_size'],
        meta_features=False,
        transform=transform
    )

    num_classes = 3 if settings_dict['merge_classes'] else dataset.num_classes

    data = dataset[0]

    #extract hold set
    idx, hold_idx = train_test_split(
        np.arange(data.y.shape[0]), test_size=0.5,
        stratify=data.y, random_state=42
    )
    n_labeled_samples = round(settings_dict['data_ss']['labeled_size'] * data.y.shape[0])
    #select labeled samples
    train_idx, test_idx = train_test_split(
        np.arange(idx.shape[0]), train_size=n_labeled_samples,
        stratify=data.y[idx], random_state=42
    )

    print(f'Number of labeled samples: {train_idx.shape[0]}')
    print(f'Number of test samples: {test_idx.shape[0]}')
    print(f'Number of hold samples: {hold_idx.shape[0]}')

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(data.y.numpy()),
        y=data.y.numpy()[train_idx]
    )
    class_weights = torch.Tensor(class_weights)

    data = data.to(device)

    model = CNNGCN(
        settings_dict['model']['hidden_units'],
        num_classes,
        settings_dict['model']['num_layers'],
        settings_dict['model']['dropout_rate'],
        num_meta_features=0
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings_dict['model']['lr'])

    best_test_f1 = best_epoch = 0
    
    for epoch in range(1, n_epochs+1):
        
        results = train()
        train_loss[epoch-1] = results[0]
        train_acc[epoch-1] = results[1]
        train_precision[epoch-1] = results[2]
        train_recall[epoch-1] = results[3]
        train_specificity[epoch-1] = results[4]
        train_f1[epoch-1] = results[5]

        results = test(test_idx)
        test_loss[epoch-1] = results[0]
        test_acc[epoch-1] = results[1]
        test_precision[epoch-1] = results[2]
        test_recall[epoch-1] = results[3]
        test_specificity[epoch-1] = results[4]
        test_f1[epoch-1] = results[5]

        results = test(hold_idx)
        hold_loss[epoch-1] = results[0]
        hold_acc[epoch-1] = results[1]
        hold_precision[epoch-1] = results[2]
        hold_recall[epoch-1] = results[3]
        hold_specificity[epoch-1] = results[4]
        hold_f1[epoch-1] = results[5]

        if test_f1[epoch-1] > best_test_f1:
            best_test_f1 = test_f1[epoch-1]
            best_epoch = epoch

    print(f'\nBest test F1 {best_test_f1} at epoch {best_epoch}.\n')

    print('########################### With Meta Features ###########################')
    dataset = BeirutFullGraph(
        '/home/ami31/scratch/datasets/beirut_bldgs/beirut_graph_meta',
        '/home/ami31/scratch/datasets/beirut_bldgs',
        settings_dict['data_ss']['reduced_size'],
        meta_features=True,
        transform=transform
    )

    data = dataset[0]

    #extract hold set
    idx, hold_idx = train_test_split(
        np.arange(data.y.shape[0]), test_size=0.5,
        stratify=data.y, random_state=42
    )
    n_labeled_samples = round(settings_dict['data_ss']['labeled_size'] * data.y.shape[0])
    #select labeled samples
    train_idx, test_idx = train_test_split(
        np.arange(idx.shape[0]), train_size=n_labeled_samples,
        stratify=data.y[idx], random_state=42
    )

    data = data.to(device)

    model = CNNGCN(
        settings_dict['model']['hidden_units'],
        num_classes,
        settings_dict['model']['num_layers'],
        settings_dict['model']['dropout_rate'],
        num_meta_features=2
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings_dict['model']['lr'])

    best_test_f1 = best_epoch = 0
    
    for epoch in range(1, n_epochs+1):
        
        results = train()
        train_loss_meta[epoch-1] = results[0]
        train_acc_meta[epoch-1] = results[1]
        train_precision_meta[epoch-1] = results[2]
        train_recall_meta[epoch-1] = results[3]
        train_specificity_meta[epoch-1] = results[4]
        train_f1_meta[epoch-1] = results[5]

        results = test(test_idx)
        test_loss_meta[epoch-1] = results[0]
        test_acc_meta[epoch-1] = results[1]
        test_precision_meta[epoch-1] = results[2]
        test_recall_meta[epoch-1] = results[3]
        test_specificity_meta[epoch-1] = results[4]
        test_f1_meta[epoch-1] = results[5]

        results = test(hold_idx)
        hold_loss_meta[epoch-1] = results[0]
        hold_acc_meta[epoch-1] = results[1]
        hold_precision_meta[epoch-1] = results[2]
        hold_recall_meta[epoch-1] = results[3]
        hold_specificity_meta[epoch-1] = results[4]
        hold_f1_meta[epoch-1] = results[5]

        if test_f1[epoch-1] > best_test_f1:
            best_test_f1 = test_f1[epoch-1]
            best_epoch = epoch

    print(f'\nBest test F1 {best_test_f1} at epoch {best_epoch}.\n')

    fig = make_subplots(rows=2, cols=3)

    fig.add_trace(go.Scatter(y=train_loss, name='train', legendgroup='train', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=test_loss, name='test', legendgroup='test', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(y=hold_loss, name='hold', legendgroup='hold', line=dict(color='purple')), row=1, col=1)
    fig.add_trace(go.Scatter(y=train_loss_meta, name='train_meta', legendgroup='train_meta', line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(y=test_loss_meta, name='test_meta', legendgroup='test_meta', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(y=hold_loss_meta, name='hold_meta', legendgroup='hold_meta', line=dict(color='brown')), row=1, col=1)

    fig.add_trace(go.Scatter(y=train_acc, name='train', legendgroup='train', line=dict(color='blue'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(y=test_acc, name='test', legendgroup='test', line=dict(color='orange'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(y=hold_acc, name='hold', legendgroup='hold', line=dict(color='purple'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(y=train_acc_meta, name='train_meta', legendgroup='train_meta', line=dict(color='green'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(y=test_acc_meta, name='test_meta', legendgroup='test_meta', line=dict(color='red'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(y=hold_acc_meta, name='hold_meta', legendgroup='hold_meta', line=dict(color='brown'), showlegend=False), row=1, col=2)    

    fig.add_trace(go.Scatter(y=train_precision, name='train', legendgroup='train', line=dict(color='blue'), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(y=test_precision, name='test', legendgroup='test', line=dict(color='orange'), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(y=hold_precision, name='hold', legendgroup='hold', line=dict(color='purple'), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(y=train_precision_meta, name='train_meta', legendgroup='train_meta', line=dict(color='green'), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(y=test_precision_meta, name='test_meta', legendgroup='test_meta', line=dict(color='red'), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(y=hold_precision_meta, name='hold_meta', legendgroup='hold_meta', line=dict(color='brown'), showlegend=False), row=1, col=3)

    fig.add_trace(go.Scatter(y=train_recall, name='train', legendgroup='train', line=dict(color='blue'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(y=test_recall, name='test', legendgroup='test', line=dict(color='orange'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(y=hold_recall, name='hold', legendgroup='hold', line=dict(color='purple'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(y=train_recall_meta, name='train_meta', legendgroup='train_meta', line=dict(color='green'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(y=test_recall_meta, name='test_meta', legendgroup='test_meta', line=dict(color='red'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(y=hold_recall_meta, name='hold_meta', legendgroup='hold_meta', line=dict(color='brown'), showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(y=train_specificity, name='train', legendgroup='train', line=dict(color='blue'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(y=test_specificity, name='test', legendgroup='test', line=dict(color='orange'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(y=hold_specificity, name='hold', legendgroup='hold', line=dict(color='purple'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(y=train_specificity_meta, name='train_meta', legendgroup='train_meta', line=dict(color='green'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(y=test_specificity_meta, name='test_meta', legendgroup='test_meta', line=dict(color='red'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(y=hold_specificity_meta, name='hold_meta', legendgroup='hold_meta', line=dict(color='brown'), showlegend=False), row=2, col=2)

    fig.add_trace(go.Scatter(y=train_f1, name='train', legendgroup='train', line=dict(color='blue'), showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(y=test_f1, name='test', legendgroup='test', line=dict(color='orange'), showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(y=hold_f1, name='hold', legendgroup='hold', line=dict(color='purple'), showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(y=train_f1_meta, name='train_meta', legendgroup='train_meta', line=dict(color='green'), showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(y=test_f1_meta, name='test_meta', legendgroup='test_meta', line=dict(color='red'), showlegend=False), row=2, col=3)
    fig.add_trace(go.Scatter(y=hold_f1_meta, name='hold_meta', legendgroup='hold_meta', line=dict(color='brown'), showlegend=False), row=2, col=3)

    fig['layout']['xaxis']['title']='epochs'
    fig['layout']['xaxis2']['title']='epochs'
    fig['layout']['xaxis3']['title']='epochs'
    fig['layout']['xaxis4']['title']='epochs'
    fig['layout']['xaxis5']['title']='epochs'
    fig['layout']['xaxis6']['title']='epochs'
    fig['layout']['yaxis']['title']='loss'
    fig['layout']['yaxis2']['title']='accuracy'
    fig['layout']['yaxis3']['title']='precision'
    fig['layout']['yaxis4']['title']='recall'
    fig['layout']['yaxis5']['title']='specificity'
    fig['layout']['yaxis6']['title']='f1'

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.write_html('results/meta_comparsion_metrics.html')

    print(f"\nLabeled size: {settings_dict['data_ss']['labeled_size']}")
    print(f"Reduced dataset size: {settings_dict['data_ss']['reduced_size']}")