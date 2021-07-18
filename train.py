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
from model import DeeperGCN, SplineNet
from metrics import to_onehot, xview2_f1_score
from utils import get_class_weights

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

seed = settings_dict['seed']
batch_size = settings_dict['data']['batch_size']
num_steps = settings_dict['data']['saint_num_steps']
name = settings_dict['model']['name']
train_set = settings_dict['train_set']
if len(train_set) == 1:
    if train_set[0] == 'mexico-earthquake':
        train_root = settings_dict['data']['mexico_train_root']
        test_root = settings_dict['data']['mexico_test_root']
    else:
        train_root = settings_dict['data']['palu_train_root']
        test_root = settings_dict['data']['palu_test_root']
else:
    train_root = settings_dict['data']['palu_matthew_rosa_train_root']
    test_root = settings_dict['data']['palu_matthew_rosa_test_root']
hold_root = settings_dict['data']['mexico_hold_root']
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
            y_true = to_onehot(subdata.y)
            loss = F.nll_loss(input=out, target=y_true.float(), weight=class_weights.to(device))
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
    ys = []
    outs = []
    for data in dataset:
        sampler = GraphSAINTNodeSampler(data, batch_size=batch_size, num_steps=num_steps, num_workers=2)
        for subdata in sampler:
            subdata = subdata.to(device)
            outs.append(model(subdata.x, subdata.edge_index, subdata.edge_attr).cpu())
            ys.append(subdata.y.cpu())
    outs = torch.cat(outs)
    ys = torch.cat(ys)
    f1 = xview2_f1_score(ys, outs)
    if dataset is not train_dataset:
        y = to_onehot(ys)
        loss = F.nll_loss(input=outs, target=y.float(), weight=class_weights)
    else:
        loss = None
    return f1, loss


def save_results() -> None:
    plt.figure()
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.plot(hold_losses)
    plt.legend(['train', 'test', 'hold'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('results/'+name+'_loss.eps')
    plt.figure()
    plt.plot(train_f1s)
    plt.plot(test_f1s)
    plt.plot(hold_f1s)
    plt.legend(['train', 'test', 'hold'])
    plt.xlabel('epochs')
    plt.ylabel('xview2 f1')
    plt.savefig('results/'+name+'_f1.eps')
    np.save('results/'+name+'_loss_train.npy', train_losses)
    np.save('results/'+name+'_loss_test.npy', test_losses)
    np.save('results/'+name+'_loss_hold.npy', hold_losses)
    np.save('results/'+name+'_f1_train.npy', train_f1s)
    np.save('results/'+name+'_f1_test.npy', test_f1s)
    np.save('results/'+name+'_f1_hold.npy', hold_f1s)
    best_test_epoch = {'best_test_f1': best_test_f1, 'best_epoch': best_epoch}
    with open('results/'+name+'_best_test_epoch.json', 'w') as JSON:
        json.dump(best_test_epoch, JSON)
    with open('results/'+name+'_exp_progress.txt', 'a') as file:
        file.write(f'Best epoch: {best_epoch}')


if __name__ == "__main__":
    
    train_dataset = xBD(train_root, 'train', train_set)
    test_dataset = xBD(train_root, 'test', train_set)
    hold_dataset = xBD(hold_root, 'hold', ['mexico-earthquake'])

    class_weights = get_class_weights(train_set, train_dataset)

    model = DeeperGCN(train_dataset.num_node_features,
                      train_dataset.num_edge_features,
                      hidden_units,
                      train_dataset.num_classes,
                      num_layers,
                      dropout_rate)
    """ model = SplineNet(train_dataset.num_node_features,
                      hidden_units,
                      train_dataset.num_classes,
                      num_layers,
                      dropout_rate) """
    if starting_epoch != 1:
        model_path = path + '/' + name + '_best.pt'
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001)

    if starting_epoch == 1:
        best_test_f1 = best_epoch = 0
        train_losses = np.empty(n_epochs)
        test_losses = np.empty(n_epochs-5)
        hold_losses = np.empty(n_epochs-5)
        train_f1s = np.empty(n_epochs-5)
        test_f1s = np.empty(n_epochs-5)
        hold_f1s = np.empty(n_epochs-5)
    else:
        with open('results/'+name+'_best_test_epoch.json', 'r') as JSON:
            best_test_epoch = json.load(JSON)
        best_test_f1 = best_test_epoch['best_test_f1']
        best_epoch = best_test_epoch['best_epoch']
        train_losses = np.load('results/'+name+'_loss_train.npy')
        test_losses = np.load('results/'+name+'_loss_test.npy')
        hold_losses = np.load('results/'+name+'_loss_hold.npy')
        train_f1s = np.load('results/'+name+'_f1_train.npy')
        test_f1s = np.load('results/'+name+'_f1_test.npy')
        hold_f1s = np.load('results/'+name+'_f1_hold.npy')

    for epoch in range(starting_epoch, n_epochs+1):

        with open('results/'+name+'_exp_progress.txt', 'w') as file:
            file.write(f'Last epoch: {epoch}\n')
        
        loss = train(epoch)
        train_losses[epoch-1] = loss
        print('**********************************************')
        print(f'Epoch {epoch:02d}, Train Loss: {loss:.4f}')
    
        if not save_best_only:
            model_path = path + '/' + name + '.pt'
            torch.save(model.state_dict(), model_path)

        if epoch > 5:
            train_f1, _ = test(train_dataset)
            test_f1, test_loss = test(test_dataset)
            hold_f1, hold_loss = test(hold_dataset)
            scheduler.step(test_loss)
            train_f1s[epoch-6] = train_f1
            test_f1s[epoch-6] = test_f1
            hold_f1s[epoch-6] = hold_f1
            test_losses[epoch-6] = test_loss
            hold_losses[epoch-6] = hold_loss
            print(f'Test Loss: {test_loss:.4f}, Hold Loss: {hold_loss:.4f}')
            print(f'Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Hold F1: {hold_f1:.4f}')

            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_epoch = epoch
                model_path = path + '/' + name + '_best.pt'
                print(f'New best model saved to: {model_path}')
                torch.save(model.state_dict(), model_path)
        
        if not (epoch % 10):
            save_results()
    
    with open('results/'+name+'_exp_settings.json', 'w') as JSON:
        json.dump(settings_dict, JSON)
    
    save_results()