import json
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.tensor import Tensor
from torch_geometric.transforms import Compose, GCNNorm, ToSparseTensor

from dataset import xBDFull
from model import CNNGCN
from utils import merge_classes, score_cm

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

name = settings_dict['model']['name'] + '_gcn'
model_path = 'weights/' + name
disaster = 'pinery-bushfire'
path = '/home/ami31/scratch/datasets/xbd/tier3_bldgs/'
root = '/home/ami31/scratch/datasets/xbd_graph/pinery_full_reduced'
n_epochs = settings_dict['epochs']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train() -> None:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_mask]
    loss = F.nll_loss(input=out, target=data.y[train_mask], weight=class_weights.to(device))
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(mask: Tensor) -> Tuple[float]:
    model.eval()
    out = model(data.x, data.adj_t)[mask].cpu()
    cm = confusion_matrix(data.y[mask].cpu(), out.argmax(dim=1, keepdims=True))
    accuracy, precision, recall, specificity, f1 = score_cm(cm)
    return accuracy, precision, recall, specificity, f1


if __name__ == "__main__":

    if settings_dict['data']['merge_classes']:
        transform = Compose([merge_classes, GCNNorm(), ToSparseTensor()])
    else:
        transform = Compose([GCNNorm(), ToSparseTensor()])

    dataset = xBDFull(root, path, disaster, settings_dict['data']['reduced_size'], transform=transform)
    
    num_classes = 3 if settings_dict['data']['merge_classes'] else dataset.num_classes
    
    data = dataset[0]

    accuracy = np.empty(100)
    precision = np.empty(100)
    recall = np.empty(100)
    specificity = np.empty(100)
    f1 = np.empty(100)

    for seed in range(100):

        #create masks for labeled samples
        train_idx, test_idx = train_test_split(
            np.arange(data.y.shape[0]), train_size=settings_dict['data']['labeled_size'],
            stratify=data.y, random_state=seed)
        train_mask = torch.zeros(data.y.shape[0]).bool()
        train_mask[train_idx] = True
        print('\nLabeled sample distribution:')
        print(torch.unique(data.y[train_mask], return_counts=True))
        #split remaining unlabeled samples into test and hold
        test_idx, _ = train_test_split(
            np.arange(test_idx.shape[0]), test_size=0.2,
            stratify=data.y[test_idx], random_state=42)
        test_mask = torch.zeros(data.y.shape[0]).bool()
        test_mask[test_idx] = True
        
        y_all = data.y.numpy()
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_all), y=y_all[train_mask])
        class_weights = torch.Tensor(class_weights)
        
        data = data.to(device)

        model = CNNGCN(
            settings_dict['model']['hidden_units'],
            num_classes,
            settings_dict['model']['num_layers'],
            settings_dict['model']['dropout_rate']
        )
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=settings_dict['model']['lr'])

        best_test_f1 = 0
        for epoch in range(1, n_epochs+1):
            train()
            results = test(test_mask)
            test_f1 = results[4]
            if test_f1 > best_test_f1:
                accuracy[seed], precision[seed], recall[seed],\
                    specificity[seed], f1[seed] = test(torch.ones(data.y.shape[0]).bool())
                
    np.save('results/gcn_acc_ttest.npy', accuracy)
    np.save('results/gcn_prec_ttest.npy', precision)
    np.save('results/gcn_rec_ttest.npy', recall)
    np.save('results/gcn_spec_ttest.npy', specificity)
    np.save('results/gcn_f1_ttest.npy', f1)