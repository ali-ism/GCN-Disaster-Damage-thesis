import json
import os
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
from utils import make_plot, merge_classes, score_cm

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


def train() -> Tuple[float]:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_mask]
    loss = F.nll_loss(input=out, target=data.y[train_mask], weight=class_weights.to(device))
    loss.backward()
    optimizer.step()
    cm = confusion_matrix(data.y[train_mask].cpu(), out.detach().cpu().argmax(dim=1, keepdims=True))
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


def save_results() -> None:
    make_plot(train_loss, test_loss, 'loss', name)
    make_plot(train_acc, test_acc, 'accuracy', name)
    make_plot(train_precision, test_precision, 'precision', name)
    make_plot(train_recall, test_recall, 'recall', name)
    make_plot(train_specificity, test_specificity, 'specificity', name)
    make_plot(train_f1, test_f1, 'f1', name)
    print('\n\nTrain results for last model.')
    print(f'Train accuracy: {train_acc[-1]:.4f}')
    print(f'Train precision: {train_precision[-1]:.4f}')
    print(f'Train recall: {train_recall[-1]:.4f}')
    print(f'Train specificity: {train_specificity[-1]:.4f}')
    print(f'Train f1: {train_f1[-1]:.4f}')
    print('\nTest results for last model.')
    print(f'Test accuracy: {test_acc[-1]:.4f}')
    print(f'Test precision: {test_precision[-1]:.4f}')
    print(f'Test recall: {test_recall[-1]:.4f}')
    print(f'Test specificity: {test_specificity[-1]:.4f}')
    print(f'Test f1: {test_f1[-1]:.4f}')
    hold_scores = test(hold_mask)
    print('\nHold results for last model.')
    print(f'Hold accuracy: {hold_scores[1]:.4f}')
    print(f'Hold precision: {hold_scores[2]:.4f}')
    print(f'Hold recall: {hold_scores[3]:.4f}')
    print(f'Hold specificity: {hold_scores[4]:.4f}')
    print(f'Hold f1: {hold_scores[5]:.4f}')
    all_scores = test(torch.ones(data.y.shape[0]).bool())
    print('\nFull results for last model.')
    print(f'Full accuracy: {all_scores[1]:.4f}')
    print(f'Full precision: {all_scores[2]:.4f}')
    print(f'Full recall: {all_scores[3]:.4f}')
    print(f'Full specificity: {all_scores[4]:.4f}')
    print(f'Full f1: {all_scores[5]:.4f}')
    print('\n\nTrain results for best model.')
    print(f'Train accuracy: {train_acc[best_epoch-1]:.4f}')
    print(f'Train precision: {train_precision[best_epoch-1]:.4f}')
    print(f'Train recall: {train_recall[best_epoch-1]:.4f}')
    print(f'Train specificity: {train_specificity[best_epoch-1]:.4f}')
    print(f'Train f1: {train_f1[best_epoch-1]:.4f}')
    print('\nTest results for best model.')
    print(f'Test accuracy: {test_acc[best_epoch-1]:.4f}')
    print(f'Test precision: {test_precision[best_epoch-1]:.4f}')
    print(f'Test recall: {test_recall[best_epoch-1]:.4f}')
    print(f'Test specificity: {test_specificity[best_epoch-1]:.4f}')
    print(f'Test f1: {test_f1[best_epoch-1]:.4f}')
    model.load_state_dict(torch.load(model_path+'_best.pt'))
    hold_scores = test(hold_mask)
    print('\nHold results for best model.')
    print(f'Hold accuracy: {hold_scores[1]:.4f}')
    print(f'Hold precision: {hold_scores[2]:.4f}')
    print(f'Hold recall: {hold_scores[3]:.4f}')
    print(f'Hold specificity: {hold_scores[4]:.4f}')
    print(f'Hold f1: {hold_scores[5]:.4f}')
    all_scores = test(torch.ones(data.y.shape[0]).bool())
    print('\nFull results for best model.')
    print(f'Full accuracy: {all_scores[1]:.4f}')
    print(f'Full precision: {all_scores[2]:.4f}')
    print(f'Full recall: {all_scores[3]:.4f}')
    print(f'Full specificity: {all_scores[4]:.4f}')
    print(f'Full f1: {all_scores[5]:.4f}')


if __name__ == "__main__":

    if settings_dict['data']['merge_classes']:
        transform = Compose([merge_classes, GCNNorm(), ToSparseTensor()])
    else:
        transform = Compose([GCNNorm(), ToSparseTensor()])

    dataset = xBDFull(root, path, disaster, settings_dict['data']['reduced_size'], transform=transform)
    
    num_classes = 3 if settings_dict['data']['merge_classes'] else dataset.num_classes
    
    data = dataset[0]

    #create masks for labeled samples
    train_idx, test_idx = train_test_split(
		np.arange(data.y.shape[0]), train_size=settings_dict['data']['labeled_size'],
		stratify=data.y, random_state=42)
    train_mask = torch.zeros(data.y.shape[0]).bool()
    train_mask[train_idx] = True
    print('\nLabeled sample distribution:')
    print(torch.unique(data.y[train_mask], return_counts=True))
    #split remaining unlabeled samples into test and hold
    test_idx, hold_idx = train_test_split(
        np.arange(test_idx.shape[0]), test_size=0.2,
        stratify=data.y[test_idx], random_state=42)
    test_mask = torch.zeros(data.y.shape[0]).bool()
    test_mask[test_idx] = True
    hold_mask = torch.zeros(data.y.shape[0]).bool()
    hold_mask[hold_idx] = True
    assert train_idx.shape[0] + test_idx.shape[0] + hold_idx.shape[0] == data.y.shape[0]

    
    if os.path.isfile(f'weights/class_weights_{disaster}_gcn_{num_classes}.pt'):
        class_weights = torch.load(f'weights/class_weights_{disaster}_gcn_{num_classes}.pt')
    else:
        y_all = data.y.numpy()
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_all), y=y_all[train_mask])
        class_weights = torch.Tensor(class_weights)
        torch.save(class_weights, f'weights/class_weights_{disaster}_gcn_{num_classes}.pt')
    
    data = data.to(device)

    model = CNNGCN(
        settings_dict['model']['hidden_units'],
        num_classes,
        settings_dict['model']['num_layers'],
        settings_dict['model']['dropout_rate']
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings_dict['model']['lr'])

    best_test_f1 = best_epoch = 0

    train_loss = np.empty(n_epochs)
    test_loss = np.empty(n_epochs)
    train_acc = np.empty(n_epochs)
    test_acc = np.empty(n_epochs)
    train_precision = np.empty(n_epochs)
    test_precision = np.empty(n_epochs)
    train_recall = np.empty(n_epochs)
    test_recall = np.empty(n_epochs)
    train_specificity = np.empty(n_epochs)
    test_specificity = np.empty(n_epochs)
    train_f1 = np.empty(n_epochs)
    test_f1 = np.empty(n_epochs)

    for epoch in range(1, n_epochs+1):
        
        results = train()
        train_loss[epoch-1] = results[0]
        train_acc[epoch-1] = results[1]
        train_precision[epoch-1] = results[2]
        train_recall[epoch-1] = results[3]
        train_specificity[epoch-1] = results[4]
        train_f1[epoch-1] = results[5]
        print('**********************************************')
        print(f'Epoch {epoch}, Train Loss: {train_loss[epoch-1]:.4f}')

        results = test(test_mask)
        test_loss[epoch-1] = results[0]
        test_acc[epoch-1] = results[1]
        test_precision[epoch-1] = results[2]
        test_recall[epoch-1] = results[3]
        test_specificity[epoch-1] = results[4]
        test_f1[epoch-1] = results[5]

        if test_f1[epoch-1] > best_test_f1:
            best_test_f1 = test_f1[epoch-1]
            best_epoch = epoch
            print(f'New best model saved with test F1 {best_test_f1} at epoch {best_epoch}.')
            torch.save(model.state_dict(), model_path+'_best.pt')
    
    print(f'\nBest test F1 {best_test_f1} at epoch {best_epoch}.\n')
    save_results()
    print(f"\nLabeled size: {settings_dict['data']['labeled_size']}")
    print(f"Reduced dataset size: {settings_dict['data']['reduced_size']}")