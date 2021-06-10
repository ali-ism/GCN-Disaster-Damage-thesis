# ------------------------------------------------------------------------------
# This code is (modified) from
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/reddit.py
# Licensed under the MIT License.
# Written by Matthias Fey (http://rusty1s.github.io)
# ------------------------------------------------------------------------------
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from tqdm import tqdm
from dataset import IIDxBD
from model import PNANet
from metrics import xview2_f1_score

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(settings_dict['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(epoch):
    model.train()

    pbar = tqdm(total=int(batch.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.binary_cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pbar.update(batch_size)

    pbar.close()

    return total_loss


@torch.no_grad()
def test(loader):
    model.eval()

    ys = []
    outs = []

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        outs.append(out.cpu())
        ys.append(data.y.cpu())
    
    outs = torch.stack(outs)
    ys = torch.stack(ys)
    
    f1 = xview2_f1_score(ys, outs)

    loss = F.binary_cross_entropy(outs, ys)

    return f1, loss


if __name__ == "__main__":
    
    root = settings_dict['data']['root']
    if not os.path.isdir(root):
        os.mkdir(root)

    train_dataset = IIDxBD(root, 'train',
                           resnet_pretrained=settings_dict['resnet']['pretrained'],
                           resnet_diff=settings_dict['resnet']['diff'],
                           resnet_shared=settings_dict['resnet']['shared'])
    
    test_dataset = IIDxBD(root, 'test',
                          resnet_pretrained=settings_dict['resnet']['pretrained'],
                          resnet_diff=settings_dict['resnet']['diff'],
                          resnet_shared=settings_dict['resnet']['shared'])
    
    hold_dataset = IIDxBD(root, 'hold',
                          resnet_pretrained=settings_dict['resnet']['pretrained'],
                          resnet_diff=settings_dict['resnet']['diff'],
                          resnet_shared=settings_dict['resnet']['shared'])

    train_loader = DataLoader(train_dataset, batch_size=settings_dict['data']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=settings_dict['data']['batch_size'])
    hold_loader = DataLoader(hold_dataset, batch_size=settings_dict['data']['batch_size'])

    n_epochs = settings_dict['epochs']

    # Compute in-degree histogram over training data.
    deg = torch.zeros(5, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    model = PNANet(train_dataset.num_node_features, train_dataset.num_edge_features,
                   settings_dict['model']['hidden_units'], train_dataset.num_classes, deg) #TODO num_classes
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings_dict['model']['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001)

    best_val_f1 = final_test_f1 = best_epoch = 0
    train_losses = np.empty(n_epochs)
    train_f1s = val_f1s = test_f1s = val_losses = test_losses = np.empty(n_epochs-5)

    for epoch in range(1, n_epochs+1):
        
        loss = train(epoch)
        train_losses[epoch-1] = loss
        print('**********************************************')
        print(f'Epoch {epoch:02d}, Train Loss: {loss:.4f}')
    
        if not settings_dict['save_best_only']:
            model_path = settings_dict['model']['path'] + '/' + settings_dict['model']['name'] + '.pth'
            torch.save(model.state_dict(), model_path)

        if epoch > 5:
            train_f1, _ = test(train_loader)
            val_f1, val_loss = test(test_loader)
            test_f1, test_loss = test(hold_loader)
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
                final_test_f1 = test_f1
                model_path = settings_dict['model']['path'] + '/' + settings_dict['model']['name'] + '_best.pth'
                print(f'New best model saved to: {model_path}')
                torch.save(model.state_dict(), model_path)
    
    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.plot(test_losses)
    plt.legend(['train', 'val', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('results/'+settings_dict['model']['name']+'_loss.eps')
    plt.figure()
    plt.plot(train_f1s)
    plt.plot(val_f1s)
    plt.plot(test_f1s)
    plt.legend(['train', 'val', 'test'])
    plt.xlabel('epochs')
    plt.ylabel('xview2 f1')
    plt.savefig('results/'+settings_dict['model']['name']+'_f1.eps')

    np.save('results/'+settings_dict['model']['name']+'_loss_train.npy', train_losses)
    np.save('results/'+settings_dict['model']['name']+'_loss_val.npy', val_losses)
    np.save('results/'+settings_dict['model']['name']+'_loss_test.npy', test_losses)
    np.save('results/'+settings_dict['model']['name']+'_f1_train.npy', train_f1s)
    np.save('results/'+settings_dict['model']['name']+'_f1_val.npy', val_f1s)
    np.save('results/'+settings_dict['model']['name']+'_f1_test.npy', test_f1s)

    with open('results/'+settings_dict['model']['name']+'_exp_settings.json', 'w') as JSON:
        json.dump(settings_dict, JSON)