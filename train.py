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
from dataset import IIDxBD
from model import DeeperGCN
from metrics import xview2_f1_score

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(settings_dict['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_dataset))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = 0
    for data in train_dataset:
        sampler = GraphSAINTNodeSampler(data, batch_size=settings_dict['data']['batch_size'],
                                        sample_coverage=0, num_steps=5, num_workers=2)
        batch_loss = 0
        total_examples = 0
        for subdata in sampler:
            subdata = subdata.to(device)
            optimizer.zero_grad()
            out = model(subdata.x, subdata.edge_index, subdata.edge_attr)
            loss = F.binary_cross_entropy(out, subdata.y)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item() * subdata.num_nodes
            total_examples += subdata.num_nodes
            pbar.update()
        
        pbar.close()
        total_loss += batch_loss / total_examples
    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    ys = []
    outs = []

    for data in loader:
        sampler = GraphSAINTNodeSampler(data, batch_size=settings_dict['data']['batch_size'],
                                        sample_coverage=0, num_steps=5, num_workers=2)
        for subdata in sampler:
            subdata = subdata.to(device)
            outs.append(model(subdata.x, subdata.edge_index, subdata.edge_attr).cpu())
            ys.append(subdata.y.cpu())

    outs = torch.stack(outs)
    ys = torch.stack(ys)
    
    f1 = xview2_f1_score(ys, outs)
    if loader is not train_dataset:
        loss = F.binary_cross_entropy(outs, ys)
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
    plt.savefig('results/'+settings_dict['model']['name']+'_loss.eps')
    plt.savefig('results/'+settings_dict['model']['name']+'_loss.png')
    plt.figure()
    plt.plot(train_f1s)
    plt.plot(val_f1s)
    plt.plot(test_f1s)
    plt.legend(['train', 'val', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('xview2 f1')
    plt.savefig('results/'+settings_dict['model']['name']+'_f1.eps')
    plt.savefig('results/'+settings_dict['model']['name']+'_loss.png')

    np.save('results/'+settings_dict['model']['name']+'_loss_train.npy', train_losses)
    np.save('results/'+settings_dict['model']['name']+'_loss_val.npy', val_losses)
    np.save('results/'+settings_dict['model']['name']+'_loss_test.npy', test_losses)
    np.save('results/'+settings_dict['model']['name']+'_f1_train.npy', train_f1s)
    np.save('results/'+settings_dict['model']['name']+'_f1_val.npy', val_f1s)
    np.save('results/'+settings_dict['model']['name']+'_f1_test.npy', test_f1s)

    best_val_epoch = {'best_val_f1': best_val_f1, 'best_epoch': best_epoch}
    with open('results/'+settings_dict['model']['name']+'_best_val_epoch.json', 'w') as JSON:
        json.dump(best_val_epoch, JSON)

    with open('results/'+settings_dict['model']['name']+'_exp_progress.txt', 'a') as file:
        file.write(f'Best epoch: {best_epoch}')


if __name__ == "__main__":

    train_dataset = IIDxBD(settings_dict['data']['iid_xbd_train_root'], 'train')
    test_dataset = IIDxBD(settings_dict['data']['iid_xbd_test_root'], 'test')
    hold_dataset = IIDxBD(settings_dict['data']['iid_xbd_hold_root'], 'hold')

    #train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=1)
    #hold_loader = DataLoader(hold_dataset, batch_size=1)

    model = DeeperGCN(train_dataset.num_node_features,
                      train_dataset.num_edge_features,
                      settings_dict['model']['hidden_units'],
                      train_dataset.num_classes,
                      settings_dict['model']['num_layers'],
                      settings_dict['model']['dropout_rate'])
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings_dict['model']['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001)

    n_epochs = settings_dict['epochs']

    if settings_dict['starting_epoch'] == 1:
        best_val_f1 = best_epoch = 0
        train_losses = np.empty(n_epochs)
        train_f1s = val_f1s = test_f1s = val_losses = test_losses = np.empty(n_epochs-5)
    else:
        with open('results/'+settings_dict['model']['name']+'_best_val_epoch.json', 'r') as JSON:
            best_val_epoch = json.load(JSON)
        best_val_f1 = best_val_epoch['best_val_f1']
        best_epoch = best_val_epoch['best_epoch']
        train_losses = np.load('results/'+settings_dict['model']['name']+'_loss_train.npy')
        val_losses = np.load('results/'+settings_dict['model']['name']+'_loss_val.npy')
        test_losses = np.load('results/'+settings_dict['model']['name']+'_loss_test.npy')
        train_f1s = np.load('results/'+settings_dict['model']['name']+'_f1_train.npy')
        val_f1s = np.load('results/'+settings_dict['model']['name']+'_f1_val.npy')
        test_f1s = np.load('results/'+settings_dict['model']['name']+'_f1_test.npy')

    for epoch in range(settings_dict['starting_epoch'], n_epochs+1):

        with open('results/'+settings_dict['model']['name']+'_exp_progress.txt', 'w') as file:
            file.write(f'Last epoch: {epoch}\n')
        
        loss = train(epoch)
        train_losses[epoch-1] = loss
        print('**********************************************')
        print(f'Epoch {epoch:02d}, Train Loss: {loss:.4f}')
    
        if not settings_dict['save_best_only']:
            model_path = settings_dict['model']['path'] + '/' + settings_dict['model']['name'] + '.pth'
            torch.save(model.state_dict(), model_path)

        if epoch > 5:
            train_f1, _ = test(train_dataset)
            val_f1, val_loss = test(test_dataset)
            test_f1, test_loss = test(hold_dataset)
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
                model_path = settings_dict['model']['path'] + '/' + settings_dict['model']['name'] + '_best.pth'
                print(f'New best model saved to: {model_path}')
                torch.save(model.state_dict(), model_path)
        
        if not (epoch % 10):
            save_results()
    
    with open('results/'+settings_dict['model']['name']+'_exp_settings.json', 'w') as JSON:
        json.dump(settings_dict, JSON)
    
    save_results()