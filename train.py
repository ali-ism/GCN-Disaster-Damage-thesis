# ------------------------------------------------------------------------------
# This code is (modified) from
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_proteins_deepgcn.py
# and
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/graph_saint.py
# Licensed under the MIT License.
# Written by Matthias Fey (http://rusty1s.github.io)
# ------------------------------------------------------------------------------
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import GraphSAINTNodeSampler
from tqdm import tqdm
from dataset import xBD
from model import DeeperGCN
from metrics import score
from utils import get_class_weights

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

seed = 42
batch_size = settings_dict['data']['batch_size']
num_steps = settings_dict['data']['saint_num_steps']
name = settings_dict['model']['name']
train_set = settings_dict['train_set']
if len(train_set) == 1:
    if train_set[0] == 'mexico-earthquake':
        train_root = "/home/ami31/scratch/datasets/pixel/mexico_train"
        test_root = "/home/ami31/scratch/datasets/pixel/mexico_test"
    else:
        train_root = "/home/ami31/scratch/datasets/pixel/palu_train"
        test_root = "/home/ami31/scratch/datasets/pixel/palu_test"
else:
    train_root = "/home/ami31/scratch/datasets/pixel/palu_matthew_rosa_train"
    test_root = "/home/ami31/scratch/datasets/pixel/palu_matthew_rosa_test"
hold_root = "/home/ami31/scratch/datasets/pixel/mexico_hold"
hidden_units = settings_dict['model']['hidden_units']
num_layers = settings_dict['model']['num_layers']
dropout_rate = settings_dict['model']['dropout_rate']
lr = settings_dict['model']['lr']
n_epochs = settings_dict['epochs']
starting_epoch = settings_dict['starting_epoch']
path = settings_dict['model']['path']
save_best_only = settings_dict['save_best_only']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(epoch):
    model.train()
    pbar = tqdm(total=len(train_dataset))
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = 0
    for data in train_dataset:
        sampler = GraphSAINTNodeSampler(data, batch_size=batch_size, num_steps=num_steps, num_workers=2)
        data_loss = 0
        total_examples = 0
        for subdata in sampler:
            subdata = subdata.to(device)
            optimizer.zero_grad()
            out = model(subdata.x, subdata.edge_index, subdata.edge_attr)
            loss = F.nll_loss(input=out, target=subdata.y, weight=class_weights.to(device))
            loss.backward()
            optimizer.step()
            data_loss += loss.item() * subdata.num_nodes
            total_examples += subdata.num_nodes
        total_loss += data_loss / total_examples
        pbar.update()
    pbar.close()
    return total_loss / len(train_dataset)


@torch.no_grad()
def test(dataset):
    model.eval()
    y_true = []
    y_pred = []
    for data in dataset:
        sampler = GraphSAINTNodeSampler(data, batch_size=batch_size, num_steps=num_steps, num_workers=2)
        for subdata in sampler:
            subdata = subdata.to(device)
            y_pred.append(model(subdata.x, subdata.edge_index, subdata.edge_attr).cpu())
            y_true.append(subdata.y.cpu())
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    accuracy, f1_macro, f1_weighted, xview2_f1, auc = score(y_true, y_pred)
    if dataset is not train_dataset:
        loss = F.nll_loss(input=y_pred, target=y_true, weight=class_weights)
    else:
        loss = None
    return accuracy, f1_macro, f1_weighted, xview2_f1, auc, loss


def save_results(hold=False) -> None:
    plt.figure()
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('results/'+name+'_loss.eps')
    plt.figure()
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig('results/'+name+'_acc.eps')
    plt.figure()
    plt.plot(train_f1_macro)
    plt.plot(test_f1_macro)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('macro f1')
    plt.savefig('results/'+name+'_macro_f1.eps')
    plt.figure()
    plt.plot(train_f1_weighted)
    plt.plot(test_f1_weighted)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('weighted f1')
    plt.savefig('results/'+name+'_weighted_f1.eps')
    plt.figure()
    plt.plot(train_xview2)
    plt.plot(test_xview2)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('xview2 f1')
    plt.savefig('results/'+name+'_xview2.eps')
    plt.figure()
    plt.plot(train_auc)
    plt.plot(test_auc)
    plt.legend(['train', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('auc')
    plt.savefig('results/'+name+'_auc.eps')
    np.save('results/'+name+'_loss_train.npy', train_loss)
    np.save('results/'+name+'_loss_test.npy', test_loss)
    np.save('results/'+name+'_acc_train.npy', train_acc)
    np.save('results/'+name+'_acc_test.npy', test_acc)
    np.save('results/'+name+'_macro_f1_train.npy', train_f1_macro)
    np.save('results/'+name+'_macro_f1_test.npy', test_f1_macro)
    np.save('results/'+name+'_weighted_f1_train.npy', train_f1_weighted)
    np.save('results/'+name+'_weighted_f1_test.npy', test_f1_weighted)
    np.save('results/'+name+'_xview2_train.npy', train_xview2)
    np.save('results/'+name+'_xview2_test.npy', test_xview2)
    np.save('results/'+name+'_auc_train.npy', train_auc)
    np.save('results/'+name+'_auc_test.npy', test_auc)
    best_test = {'best_test_f1': best_test_auc, 'best_epoch': best_epoch}
    with open('results/'+name+'_best_test.json', 'w') as JSON:
        json.dump(best_test, JSON)
    with open('results/'+name+'_exp_progress.txt', 'a') as file:
        file.write(f'Best epoch: {best_epoch}')
    if hold:
        hold_dataset = xBD(hold_root, 'hold', ['mexico-earthquake'])
        np.save('results/'+name+'_hold_scores.npy', np.asarray(test(hold_dataset)))


if __name__ == "__main__":
    
    train_dataset = xBD(train_root, 'train', train_set).shuffle()
    test_dataset = xBD(train_root, 'test', train_set)

    class_weights = get_class_weights(train_set, train_dataset)

    model = DeeperGCN(train_dataset.num_node_features,
                      train_dataset.num_edge_features,
                      hidden_units,
                      train_dataset.num_classes,
                      num_layers,
                      dropout_rate)
    if starting_epoch != 1:
        model_path = path + '/' + name + '_best.pt'
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001)

    if starting_epoch == 1:
        best_test_auc = best_epoch = 0
        train_loss = np.empty(n_epochs)
        test_loss = np.empty(n_epochs-5)
        train_acc = np.empty(n_epochs-5)
        test_acc = np.empty(n_epochs-5)
        train_f1_macro = np.empty(n_epochs-5)
        test_f1_macro = np.empty(n_epochs-5)
        train_f1_weighted = np.empty(n_epochs-5)
        test_f1_weighted = np.empty(n_epochs-5)
        train_xview2 = np.empty(n_epochs-5)
        test_xview2 = np.empty(n_epochs-5)
        train_auc = np.empty(n_epochs-5)
        test_auc = np.empty(n_epochs-5)
    else:
        with open('results/'+name+'_best_test.json', 'r') as JSON:
            best_test = json.load(JSON)
        best_test_auc = best_test['best_test_auc']
        best_epoch = best_test['best_epoch']
        train_loss = np.load('results/'+name+'_loss_train.npy')
        test_loss = np.load('results/'+name+'_loss_test.npy')
        train_acc = np.load('results/'+name+'_acc_train.npy')
        test_acc = np.load('results/'+name+'_acc_test.npy')
        train_f1_macro = np.load('results/'+name+'_macro_f1_train.npy')
        test_f1_macro = np.load('results/'+name+'_macro_f1_test.npy')
        train_f1_weighted = np.load('results/'+name+'_weighted_f1_train.npy')
        test_f1_weighted = np.load('results/'+name+'_weighted_f1_test.npy')
        train_xview2 = np.load('results/'+name+'_xview2_train.npy')
        test_xview2 = np.load('results/'+name+'_xview2_test.npy')
        train_auc = np.load('results/'+name+'_auc_train.npy')
        test_auc = np.load('results/'+name+'_auc_test.npy')

    for epoch in range(starting_epoch, n_epochs+1):

        with open('results/'+name+'_exp_progress.txt', 'w') as file:
            file.write(f'Last epoch: {epoch}\n')
        
        loss = train(epoch)
        train_loss[epoch-1] = loss
        print('**********************************************')
        print(f'Epoch {epoch:02d}, Train Loss: {loss:.4f}')
    
        if not save_best_only:
            model_path = path + '/' + name + '.pt'
            torch.save(model.state_dict(), model_path)

        if epoch > 5:
            train_acc[epoch-6], train_f1_macro[epoch-6], train_f1_weighted[epoch-6],\
            train_xview2[epoch-6], train_auc[epoch-6], _ = test(train_dataset)
            test_acc[epoch-6], test_f1_macro[epoch-6], test_f1_weighted[epoch-6],\
            test_xview2[epoch-6], test_auc[epoch-6], test_loss[epoch-6] = test(test_dataset)
            scheduler.step(test_loss[epoch-6])

            if test_auc[epoch-6] > best_test_auc:
                best_test_auc = test_auc[epoch-6]
                best_epoch = epoch
                model_path = path + '/' + name + '_best.pt'
                print(f'New best model saved to: {model_path}')
                torch.save(model.state_dict(), model_path)
        
        if not (epoch % 10):
            save_results()
    
    with open('results/'+name+'_exp_settings.json', 'w') as JSON:
        json.dump(settings_dict, JSON)
    
    save_results(hold=True)