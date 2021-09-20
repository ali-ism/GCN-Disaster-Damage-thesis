import json
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset
from torch_geometric.data import DataLoader, RandomNodeSampler
from tqdm import tqdm
from dataset import xBD
from model import CNNSage
from utils import get_class_weights, merge_classes, score

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

batch_size = settings_dict['data']['batch_size']
name = settings_dict['model']['name']
train_disasters = settings_dict['data']['train_disasters']
train_paths = settings_dict['data']['train_paths']
train_roots = settings_dict['data']['train_roots']
test_root = "/home/ami31/scratch/datasets/delaunay/socal_test"
hold_root = "/home/ami31/scratch/datasets/delaunay/socal_hold"
edge_features = settings_dict['model']['edge_features']
n_epochs = settings_dict['epochs']
starting_epoch = settings_dict['starting_epoch']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(epoch: int) -> float:
    model.train()
    pbar = tqdm(total=len(train_dataset))
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = total_examples = 0
    for data in train_loader:
        if data.num_nodes > batch_size:
            sampler = RandomNodeSampler(data, num_parts=data.num_nodes//batch_size, num_workers=2)
            for subdata in sampler:
                subdata = subdata.to(device)
                optimizer.zero_grad()
                if edge_features:
                    out = model(subdata.x, subdata.edge_index, subdata.edge_attr[:,0])
                else:
                    out = model(subdata.x, subdata.edge_index)
                loss = F.nll_loss(input=out, target=subdata.y, weight=class_weights.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * subdata.num_nodes
                total_examples += subdata.num_nodes
        else:
            data = data.to(device)
            optimizer.zero_grad()
            if edge_features:
                out = model(data.x, data.edge_index, data.edge_attr[:,0])
            else:
                out = model(data.x, data.edge_index)
            loss = F.nll_loss(input=out, target=data.y, weight=class_weights.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_nodes
            total_examples += data.num_nodes
        pbar.update()
    pbar.close()
    return total_loss / total_examples


@torch.no_grad()
def test(dataset) -> Tuple[float]:
    model.eval()
    y_true = []
    y_pred = []
    if dataset is train_loader:
        total_loss = 0
        total_examples = 0
    for data in dataset:
        if data.num_nodes > batch_size:
            sampler = RandomNodeSampler(data, num_parts=data.num_nodes//batch_size, num_workers=2)
            for subdata in sampler:
                subdata = subdata.to(device)
                if edge_features:
                    out = model(subdata.x, subdata.edge_index, subdata.edge_attr[:,0]).cpu()
                    y_pred.append(out)
                else:
                    out = model(subdata.x, subdata.edge_index).cpu()
                    y_pred.append(out)
                y_true.append(subdata.y.cpu())
                if dataset is train_loader:
                    loss = F.nll_loss(input=out, target=subdata.y, weight=class_weights)
                    total_loss += loss.item() * subdata.num_nodes
                    total_examples += subdata.num_nodes
        else:
            data = data.to(device)
            if edge_features:
                out = model(data.x, data.edge_index, data.edge_attr[:,0]).cpu()
                y_pred.append(out)
            else:
                out = model(data.x, data.edge_index).cpu()
                y_pred.append(out)
            y_true.append(data.y.cpu())
            if dataset is train_loader:
                loss = F.nll_loss(input=out, target=data.y, weight=class_weights)
                total_loss += loss.item() * data.num_nodes
                total_examples += data.num_nodes
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    accuracy, f1_macro, f1_weighted, auc = score(y_true, y_pred)
    if dataset is train_loader:
        total_loss = total_loss / total_examples
    else:
        total_loss = None
    return accuracy, f1_macro, f1_weighted, auc, total_loss


def make_plot(train: np.ndarray, test: np.ndarray, type: str) -> None:
    plt.figure()
    plt.plot(train)
    plt.plot(test)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel(type)
    plt.savefig('results/'+name+'_'+type+'.pdf')
    plt.close()

def save_results(hold: bool=False) -> None:
    make_plot(train_loss, test_loss, 'loss')
    make_plot(train_acc, test_acc, 'accuracy')
    make_plot(train_f1_macro, test_f1_macro, 'macro_f1')
    make_plot(train_f1_weighted, test_f1_weighted, 'weighted_f1')
    make_plot(train_auc, test_auc, 'auc')
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
        hold_dataset = xBD(
            hold_root,
            '/home/ami31/scratch/datasets/xbd/hold_bldgs/',
            'socal-fire',
            transform=transform
        )
        hold_scores = test(hold_dataset)
        print('\nHold results for last model.')
        print(f'Hold accuracy: {hold_scores[0]:.4f}')
        print(f'Hold macro F1: {hold_scores[1]:.4f}')
        print(f'Hold weighted F1: {hold_scores[2]:.4f}')
        print(f'Hold auc: {hold_scores[3]:.4f}')
        print('\n\nTrain results for best model.')
        print(f'Train accuracy: {train_acc[best_epoch]:.4f}')
        print(f'Train macro F1: {train_f1_macro[best_epoch]:.4f}')
        print(f'Train weighted F1: {train_f1_weighted[best_epoch]:.4f}')
        print(f'Train auc: {train_auc[best_epoch]:.4f}')
        print('\nTest results for best model.')
        print(f'Test accuracy: {test_acc[best_epoch]:.4f}')
        print(f'Test macro F1: {test_f1_macro[best_epoch]:.4f}')
        print(f'Test weighted F1: {test_f1_weighted[best_epoch]:.4f}')
        print(f'Test auc: {test_auc[best_epoch]:.4f}')
        model_path = 'weights/' + name + '_best.pt'
        model.load_state_dict(torch.load(model_path))
        hold_scores = test(hold_dataset)
        print('\nHold results for best model.')
        print(f'Hold accuracy: {hold_scores[0]:.4f}')
        print(f'Hold macro F1: {hold_scores[1]:.4f}')
        print(f'Hold weighted F1: {hold_scores[2]:.4f}')
        print(f'Hold auc: {hold_scores[3]:.4f}')


if __name__ == "__main__":

    if settings_dict['data']['merge_classes']:
        transform = merge_classes
    else:
        transform = None

    train_dataset = []
    for root, path, disaster in zip(train_roots, train_paths, train_disasters):
        train_dataset.append(xBD(root, path, disaster, transform=transform))
    
    if len(train_dataset) > 1:
        train_dataset = ConcatDataset(train_dataset)
    else:
        train_dataset = train_dataset[0]
    train_loader = DataLoader(train_dataset, shuffle=True)

    test_dataset = xBD(
        test_root,
        '/home/ami31/scratch/datasets/xbd/test_bldgs/',
        'socal-fire',
        transform=transform
    )

    num_classes = 3 if settings_dict['data']['merge_classes'] else train_dataset.num_classes
    class_weights = get_class_weights(train_disasters, train_dataset, num_classes)

    model = CNNSage(
        settings_dict['model']['hidden_units'],
        num_classes,
        settings_dict['model']['num_layers'],
        settings_dict['model']['dropout_rate'],
        settings_dict['model']['enc_diff']
    )
    if starting_epoch != 1:
        model_path = 'weights/' + name + '_best.pt'
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings_dict['model']['lr'])
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=0.00001, verbose=True)

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
        
        train_loss[epoch-1] = train(epoch)
        print('**********************************************')
        print(f'Epoch {epoch:02d}, Train Loss: {train_loss[epoch-1]:.4f}')
    
        if not settings_dict['save_best_only']:
            model_path = 'weights/' + name + '.pt'
            torch.save(model.state_dict(), model_path)

        train_acc[epoch-1], train_f1_macro[epoch-1],\
            train_f1_weighted[epoch-1], train_auc[epoch-1], _ = test(train_loader)
        test_acc[epoch-1], test_f1_macro[epoch-1], test_f1_weighted[epoch-1],\
            test_auc[epoch-1], test_loss[epoch-1] = test(test_dataset)
        #scheduler.step(test_loss[epoch-1])

        if test_auc[epoch-1] > best_test_auc:
            best_test_auc = test_auc[epoch-1]
            best_epoch = epoch
            model_path = 'weights/' + name + '_best.pt'
            print(f'New best model saved with AUC {best_test_auc} at epoch {best_epoch}.')
            torch.save(model.state_dict(), model_path)
        
        if not (epoch % 5):
            save_results()
    
    print(f'\nBest test AUC {best_test_auc} at epoch {best_epoch}.\n')
    save_results(hold=True)