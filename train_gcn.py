import json
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from torch.tensor import Tensor
from torch_geometric.transforms import Compose, GCNNorm, ToSparseTensor

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
    out = model(data.x, data.adj_t)[train_mask]
    loss = F.nll_loss(input=out, target=data.y[train_mask], weight=class_weights.to(device))
    loss.backward()
    optimizer.step()
    accuracy, f1_macro, f1_weighted, auc = score(data.y[train_mask].cpu(), out.detach().cpu())
    return loss.detach().cpu().item(), accuracy, f1_macro, f1_weighted, auc


@torch.no_grad()
def test(mask: Tensor) -> Tuple[float]:
    model.eval()
    out = model(data.x, data.adj_t)[mask].cpu()
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
        all_scores = test(torch.ones(data.y.shape[0]).bool())
        print('\nFull results for last model.')
        print(f'Full accuracy: {all_scores[1]:.4f}')
        print(f'Full macro F1: {all_scores[2]:.4f}')
        print(f'Full weighted F1: {all_scores[3]:.4f}')
        print(f'Full auc: {all_scores[4]:.4f}')
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
        all_scores = test(torch.ones(data.y.shape[0]).bool())
        print('\nFull results for best model.')
        print(f'Full accuracy: {all_scores[1]:.4f}')
        print(f'Full macro F1: {all_scores[2]:.4f}')
        print(f'Full weighted F1: {all_scores[3]:.4f}')
        print(f'Full auc: {all_scores[4]:.4f}')


if __name__ == "__main__":

    if settings_dict['data']['merge_classes']:
        transform = Compose([merge_classes, GCNNorm(), ToSparseTensor()])
    else:
        transform = Compose([GCNNorm(), ToSparseTensor()])

    dataset = xBDFull(root, path, disaster, settings_dict['data']['reduced_size'], transform=transform)
    
    num_classes = 3 if settings_dict['data']['merge_classes'] else dataset.num_classes
    
    data = dataset[0]

    nb_per_class = int(settings_dict['data']['labeled_size']*data.y.shape[0] // num_classes)
    sampling_strat = {}
    _, count = torch.unique(data.y, return_counts=True)
    for i in range(num_classes):
        sampling_strat[i] = min(nb_per_class, count[i].item())
    rus = RandomUnderSampler(sampling_strategy=sampling_strat, random_state=42)
    train_idx, y_train = rus.fit_resample(np.expand_dims(np.arange(data.y.shape[0]),axis=1), data.y)
    train_idx = np.squeeze(train_idx)
    print('\nLabeled sample distribution:')
    print(np.unique(y_train, return_counts=True))
    train_mask = torch.zeros(data.y.shape[0]).bool()
    train_mask[train_idx] = True

    test_idx, hold_idx = train_test_split(
        np.delete(np.arange(data.y.shape[0]),train_idx), test_size=0.2,
        stratify=np.delete(data.y.numpy(),train_idx), random_state=42)
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
    print(f"\nLabeled size: {settings_dict['data']['labeled_size']}")
    print(f"Reduced dataset size: {settings_dict['data']['reduced_size']}")
