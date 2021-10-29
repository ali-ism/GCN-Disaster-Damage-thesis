import json
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor
from torch_geometric.transforms import Compose, GCNNorm, ToSparseTensor

from dataset import xBDFull
from model import CNNGCN
from utils import merge_classes, score_cm

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

name = settings_dict['model']['name'] + '_gcn'
model_path = 'weights/' + name
disaster = settings_dict['data_ss']['disaster']
path = '/home/ami31/scratch/datasets/xbd/tier3_bldgs/'
root = settings_dict['data_ss']['root']
n_epochs = settings_dict['epochs']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train() -> None:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(input=out, target=data.y[train_idx], weight=class_weights.to(device))
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

    if settings_dict['merge_classes']:
        transform = Compose([merge_classes, GCNNorm(), ToSparseTensor()])
    else:
        transform = Compose([GCNNorm(), ToSparseTensor()])

    dataset = xBDFull(root, path, disaster, settings_dict['data']['reduced_size'], transform=transform)
    
    num_classes = 3 if settings_dict['data']['merge_classes'] else dataset.num_classes
    
    data = dataset[0]

    #extract hold set
    idx, hold_idx = train_test_split(
        np.arange(data.y.shape[0]), test_size=0.5,
        stratify=data.y, random_state=42
    )

    labeled_sizes = [0.1, 0.2, 0.3, 0.4]

    accuracy = np.empty(len(labeled_sizes))
    precision = np.empty(len(labeled_sizes))
    recall = np.empty(len(labeled_sizes))
    specificity = np.empty(len(labeled_sizes))
    f1 = np.empty(len(labeled_sizes))

    for i, labeled_size in enumerate(labeled_sizes):

        data = data.cpu()

        n_labeled_samples = round(labeled_size * data.y.shape[0])

        #select labeled samples
        train_idx, test_idx = train_test_split(
            np.arange(idx.shape[0]), train_size=n_labeled_samples,
            stratify=data.y[idx], random_state=42
        )

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
            settings_dict['model']['dropout_rate']
        )
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=settings_dict['model']['lr'])

        best_test_f1 = 0
        for epoch in range(1, n_epochs+1):
            train()
            test_f1 = test(test_idx)[4]
            if test_f1 > best_test_f1:
                accuracy[i], precision[i], recall[i], specificity[i], f1[i] = test(hold_idx)
                
    np.save('results/gcn_acc_sens.npy', accuracy)
    np.save('results/gcn_prec_sens.npy', precision)
    np.save('results/gcn_rec_sens.npy', recall)
    np.save('results/gcn_spec_sens.npy', specificity)
    np.save('results/gcn_f1_sens.npy', f1)