import json
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from dataset import xBDFull
from model import CNNGCN
from utils import make_plot, merge_classes, score

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

name = settings_dict['model']['name'] + '_gcn'
model_path = 'weights/' + name
disaster = 'pinery-bushfire'
path = '/home/ami31/scratch/datasets/xbd/tier3_bldgs/'
root = '/home/ami31/scratch/datasets/xbd_graph/pinery_full_reduced'
n_epochs = settings_dict['epochs']
starting_epoch = 1
assert starting_epoch > 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train() -> Tuple[float]:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)[train_mask]
    loss = F.nll_loss(input=out, target=data.y[train_mask], weight=class_weights.to(device))
    loss.backward()
    optimizer.step()
    accuracy, f1_macro, f1_weighted, auc = score(data.y[train_mask].cpu(), out.detach().cpu())
    return loss.detach().cpu().item(), accuracy, f1_macro, f1_weighted, auc


@torch.no_grad()
def test(mask) -> Tuple[float]:
    model.eval()
    out = model(data.x, data.edge_index, data.edge_attr)[mask].cpu()
    loss = F.nll_loss(input=out, target=data.y[mask].cpu(), weight=class_weights)
    accuracy, f1_macro, f1_weighted, auc = score(data.y[mask].cpu(), out)
    return loss.item(), accuracy, f1_macro, f1_weighted, auc


def save_results(hold: bool=False) -> None:
    make_plot(train_loss, test_loss, 'loss', name)
    make_plot(train_acc, test_acc, 'accuracy', name)
    make_plot(train_f1_macro, test_f1_macro, 'macro_f1', name)
    make_plot(train_f1_weighted, test_f1_weighted, 'weighted_f1', name)
    make_plot(train_auc, test_auc, 'auc', name)
    np.save('results/'+name+'_loss_train.npy', train_loss)
    np.save('results/'+name+'_loss_test.npy', test_loss)
    np.save('results/'+name+'_acc_train.npy', train_acc)
    np.save('results/'+name+'_acc_test.npy', test_acc)
    np.save('results/'+name+'_macro_f1_train.npy', train_f1_macro)
    np.save('results/'+name+'_macro_f1_test.npy', test_f1_macro)
    np.save('results/'+name+'_weighted_f1_train.npy', train_f1_weighted)
    np.save('results/'+name+'_weighted_f1_test.npy', test_f1_weighted)
    np.save('results/'+name+'_auc_train.npy', train_auc)
    np.save('results/'+name+'_auc_test.npy', test_auc)
    if hold:
        print('\n\nTrain results for last model.')
        print(f'Train accuracy: {train_acc[-1]:.4f}')
        print(f'Train macro F1: {train_f1_macro[-1]:.4f}')
        print(f'Train weighted F1: {train_f1_weighted[-1]:.4f}')
        print(f'Train auc: {train_auc[-1]:.4f}')
        print('\nTest results for last model.')
        print(f'Test accuracy: {test_acc[-1]:.4f}')
        print(f'Test macro F1: {test_f1_macro[-1]:.4f}')
        print(f'Test weighted F1: {test_f1_weighted[-1]:.4f}')
        print(f'Test auc: {test_auc[-1]:.4f}')
        hold_scores = test(hold_mask)
        print('\nHold results for last model.')
        print(f'Hold accuracy: {hold_scores[1]:.4f}')
        print(f'Hold macro F1: {hold_scores[2]:.4f}')
        print(f'Hold weighted F1: {hold_scores[3]:.4f}')
        print(f'Hold auc: {hold_scores[4]:.4f}')
        print('\n\nTrain results for best model.')
        print(f'Train accuracy: {train_acc[best_epoch-1]:.4f}')
        print(f'Train macro F1: {train_f1_macro[best_epoch-1]:.4f}')
        print(f'Train weighted F1: {train_f1_weighted[best_epoch-1]:.4f}')
        print(f'Train auc: {train_auc[best_epoch-1]:.4f}')
        print('\nTest results for best model.')
        print(f'Test accuracy: {test_acc[best_epoch-1]:.4f}')
        print(f'Test macro F1: {test_f1_macro[best_epoch-1]:.4f}')
        print(f'Test weighted F1: {test_f1_weighted[best_epoch-1]:.4f}')
        print(f'Test auc: {test_auc[best_epoch-1]:.4f}')
        model.load_state_dict(torch.load(model_path+'_best.pt'))
        hold_scores = test(hold_mask)
        print('\nHold results for best model.')
        print(f'Hold accuracy: {hold_scores[1]:.4f}')
        print(f'Hold macro F1: {hold_scores[2]:.4f}')
        print(f'Hold weighted F1: {hold_scores[3]:.4f}')
        print(f'Hold auc: {hold_scores[4]:.4f}')


if __name__ == "__main__":

    if settings_dict['data']['merge_classes']:
        transform = merge_classes
    else:
        transform = None

    dataset = xBDFull(root, path, disaster, transform=transform)

    data = dataset[0]

    train_idx, test_idx = train_test_split(np.arange(data.y.shape[0]), test_size=0.8, stratify=data.y, random_state=42)
    train_mask = torch.zeros(data.y.shape[0])
    train_mask[train_idx] = 1
    train_mask = train_mask.bool()

    test_idx, hold_idx = train_test_split(np.arange(len(test_idx)), test_size=0.2, stratify=data.y[test_idx], random_state=42)
    test_mask = torch.zeros(data.y.shape[0])
    test_mask[test_idx] = 1
    test_mask = test_mask.bool()
    hold_mask = torch.zeros(data.y.shape[0])
    hold_mask[hold_idx] = 1
    hold_mask = hold_mask.bool()

    num_classes = 3 if settings_dict['data']['merge_classes'] else dataset.num_classes
    
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
    if starting_epoch > 1:
        model.load_state_dict(torch.load(model_path+'_last.pt'))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings_dict['model']['lr'])

    best_test_auc = best_epoch = 0

    if starting_epoch == 1:
        train_loss = np.empty(n_epochs)
        test_loss = np.empty(n_epochs)
        train_acc = np.empty(n_epochs)
        test_acc = np.empty(n_epochs)
        train_f1_macro = np.empty(n_epochs)
        test_f1_macro = np.empty(n_epochs)
        train_f1_weighted = np.empty(n_epochs)
        test_f1_weighted = np.empty(n_epochs)
        train_auc = np.empty(n_epochs)
        test_auc = np.empty(n_epochs)
    else:
        train_loss = np.load('results/'+name+'_loss_train.npy')
        test_loss = np.load('results/'+name+'_loss_test.npy')
        train_acc = np.load('results/'+name+'_acc_train.npy')
        test_acc = np.load('results/'+name+'_acc_test.npy')
        train_f1_macro = np.load('results/'+name+'_macro_f1_train.npy')
        test_f1_macro = np.load('results/'+name+'_macro_f1_test.npy')
        train_f1_weighted = np.load('results/'+name+'_weighted_f1_train.npy')
        test_f1_weighted = np.load('results/'+name+'_weighted_f1_test.npy')
        train_auc = np.load('results/'+name+'_auc_train.npy')
        test_auc = np.load('results/'+name+'_auc_test.npy')

    for epoch in range(starting_epoch, n_epochs+1):
        
        train_loss[epoch-1], train_acc[epoch-1], train_f1_macro[epoch-1],\
            train_f1_weighted[epoch-1], train_auc[epoch-1] = train()
        print('**********************************************')
        print(f'Epoch {epoch:02d}, Train Loss: {train_loss[epoch-1]:.4f}')
    
        torch.save(model.state_dict(), model_path+'_last.pt')

        test_loss[epoch-1], test_acc[epoch-1], test_f1_macro[epoch-1],\
            test_f1_weighted[epoch-1], test_auc[epoch-1] = test(test_mask)

        if test_auc[epoch-1] > best_test_auc:
            best_test_auc = test_auc[epoch-1]
            best_epoch = epoch
            print(f'New best model saved with AUC {best_test_auc} at epoch {best_epoch}.')
            torch.save(model.state_dict(), model_path+'_best.pt')
        
        save_results()
    
    print(f'\nBest test AUC {best_test_auc} at epoch {best_epoch}.\n')
    save_results(True)
