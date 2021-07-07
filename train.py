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
from metrics import xview2_f1_score
from utils import get_class_weights

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

seed = settings_dict['seed']
batch_size = settings_dict['data']['batch_size']
num_steps = settings_dict['data']['saint_num_steps']
name = settings_dict['model']['name']
train_set = settings_dict['train_set']
set_id = settings_dict['train_set_id']
train_roots = []
test_roots = []
for set in train_set:
    if set == 'mexico-earthquake':
        train_roots.append(settings_dict['data']['mexico_train_root'])
        test_roots.append(settings_dict['data']['mexico_test_root'])
    elif set == 'palu-tsunami':
        train_roots.append(settings_dict['data']['palu_train_root'])
        test_roots.append(settings_dict['data']['palu_test_root'])
    elif set == 'hurricane-matthew':
        train_roots.append(settings_dict['data']['matthew_train_root'])
        test_roots.append(settings_dict['data']['matthew_test_root'])
    elif set == 'santa-rosa-wildfire':
        train_roots.append(settings_dict['data']['rosa_train_root'])
        test_roots.append(settings_dict['data']['rosa_test_root'])
hold_roots = settings_dict['data']['mexico_hold_root']
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
    pbar = tqdm(total=sum([len(dataset) for dataset in train_data_list]))
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = 0
    for dataset in train_data_list:
        for data in dataset:
            sampler = GraphSAINTNodeSampler(data, batch_size=batch_size, num_steps=num_steps, num_workers=2)
            batch_loss = 0
            total_examples = 0
            for subdata in sampler:
                subdata = subdata.to(device)
                optimizer.zero_grad()
                out = model(subdata.x, subdata.edge_index, subdata.edge_attr)
                loss = F.binary_cross_entropy(input=out, target=subdata.y.float(), weight=class_weights)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item() * subdata.num_nodes
                total_examples += subdata.num_nodes
            pbar.update()
            total_loss += batch_loss / total_examples
    pbar.close()
    return total_loss / len(train_data_list)


@torch.no_grad()
def test(dataset_list):
    model.eval()
    ys = []
    outs = []
    for dataset in dataset_list:
        for data in dataset:
            sampler = GraphSAINTNodeSampler(data, batch_size=batch_size, num_steps=num_steps, num_workers=2)
            for subdata in sampler:
                subdata = subdata.to(device)
                outs.append(model(subdata.x, subdata.edge_index, subdata.edge_attr).cpu())
                ys.append(subdata.y.cpu())
    outs = torch.cat(outs)
    ys = torch.cat(ys)
    f1 = xview2_f1_score(ys, outs)
    if dataset_list is not train_data_list:
        loss = F.binary_cross_entropy(input=outs, target=ys.float(), weight=class_weights)
    else:
        loss = None
    return f1, loss


def save_results() -> None:
    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.plot(test_losses)
    plt.legend(['train', 'val', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('results/'+name+'_loss.eps')
    plt.savefig('results/'+name+'_loss.png')
    plt.figure()
    plt.plot(train_f1s)
    plt.plot(val_f1s)
    plt.plot(test_f1s)
    plt.legend(['train', 'val', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('xview2 f1')
    plt.savefig('results/'+name+'_f1.eps')
    plt.savefig('results/'+name+'_loss.png')
    np.save('results/'+name+'_loss_train.npy', train_losses)
    np.save('results/'+name+'_loss_val.npy', val_losses)
    np.save('results/'+name+'_loss_test.npy', test_losses)
    np.save('results/'+name+'_f1_train.npy', train_f1s)
    np.save('results/'+name+'_f1_val.npy', val_f1s)
    np.save('results/'+name+'_f1_test.npy', test_f1s)
    best_val_epoch = {'best_val_f1': best_val_f1, 'best_epoch': best_epoch}
    with open('results/'+name+'_best_val_epoch.json', 'w') as JSON:
        json.dump(best_val_epoch, JSON)
    with open('results/'+name+'_exp_progress.txt', 'a') as file:
        file.write(f'Best epoch: {best_epoch}')


if __name__ == "__main__":

    train_data_list = []
    test_data_list = []
    for disaster, train_root, test_root in zip(train_set, train_roots, test_roots):
        dataset = xBD(train_root, disaster, 'train')
        train_data_list.append(dataset)
        dataset = xBD(test_root, disaster, 'test')
        test_data_list.append(dataset)

    hold_data_list = [xBD(hold_roots, 'mexico-earthquake', 'hold')]

    class_weights = get_class_weights(set_id, train_data_list)

    model = DeeperGCN(dataset.num_node_features,
                      dataset.num_edge_features,
                      hidden_units,
                      dataset.num_classes,
                      num_layers,
                      dropout_rate)
    if starting_epoch != 1:
        model_path = path + '/' + name + '_best.pth'
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001)

    if starting_epoch == 1:
        best_val_f1 = best_epoch = 0
        train_losses = np.empty(n_epochs)
        train_f1s = val_f1s = test_f1s = val_losses = test_losses = np.empty(n_epochs-5)
    else:
        with open('results/'+name+'_best_val_epoch.json', 'r') as JSON:
            best_val_epoch = json.load(JSON)
        best_val_f1 = best_val_epoch['best_val_f1']
        best_epoch = best_val_epoch['best_epoch']
        train_losses = np.load('results/'+name+'_loss_train.npy')
        val_losses = np.load('results/'+name+'_loss_val.npy')
        test_losses = np.load('results/'+name+'_loss_test.npy')
        train_f1s = np.load('results/'+name+'_f1_train.npy')
        val_f1s = np.load('results/'+name+'_f1_val.npy')
        test_f1s = np.load('results/'+name+'_f1_test.npy')

    for epoch in range(starting_epoch, n_epochs+1):

        with open('results/'+name+'_exp_progress.txt', 'w') as file:
            file.write(f'Last epoch: {epoch}\n')
        
        loss = train(epoch)
        train_losses[epoch-1] = loss
        print('**********************************************')
        print(f'Epoch {epoch:02d}, Train Loss: {loss:.4f}')
    
        if not save_best_only:
            model_path = path + '/' + name + '.pth'
            torch.save(model.state_dict(), model_path)

        if epoch > 5:
            train_f1, _ = test(train_data_list)
            val_f1, val_loss = test(test_data_list)
            test_f1, test_loss = test(hold_data_list)
            scheduler.step(val_loss)
            train_f1s[epoch-6] = train_f1
            val_f1s[epoch-6] = val_f1
            test_f1s[epoch-6] = test_f1
            val_losses[epoch-6] = val_loss
            test_losses[epoch-6] = test_loss
            print(f'Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')
            print(f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, 'f'Test F1: {test_f1:.4f}')

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                model_path = path + '/' + name + '_best.pth'
                print(f'New best model saved to: {model_path}')
                torch.save(model.state_dict(), model_path)
        
        if not (epoch % 10):
            save_results()
    
    with open('results/'+name+'_exp_settings.json', 'w') as JSON:
        json.dump(settings_dict, JSON)
    
    save_results()