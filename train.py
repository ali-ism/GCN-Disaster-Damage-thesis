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
from torch_geometric.data import DataLoader, NeighborSampler
from tqdm import tqdm
from dataset import IIDxBD
from model import SAGENet
from metrics import xview2_f1_score

with open('exp_setting.json', 'r') as JSON:
    settings_dict = json.load(JSON)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(settings_dict['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(epoch):
    model.train()

    pbar = tqdm(total=int(batch.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.binary_cross_entropy(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        #total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    #approx_acc = total_correct / int(batch.train_mask.sum())

    return loss


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x, subgraph_loader).cpu()
    y_true = y.cpu().unsqueeze(-1)
    
    train_f1 = xview2_f1_score(y_true[batch.train_mask], out[batch.train_mask])
    val_f1 = xview2_f1_score(y_true[batch.val_mask], out[batch.val_mask])
    test_f1 = xview2_f1_score(y_true[batch.test_mask], out[batch.test_mask])

    val_loss = F.binary_cross_entropy(out[batch.val_mask], y_true[batch.val_mask])

    return train_f1, val_f1, test_f1, val_loss


if __name__ == "__main__":
    
    root = settings_dict['data']['root']
    if not os.path.isdir(root):
        os.mkdir(root)

    dataset = IIDxBD(root, resnet_pretrained=settings_dict['resnet']['pretrained'],
                           resnet_diff=settings_dict['resnet']['diff'],
                           resnet_shared=settings_dict['resnet']['shared'])

    loader = DataLoader(dataset, batch_size=len(dataset))
    batch = list(loader)[0]

    nbr_sizes = settings_dict['model']['neighbor_sizes']
    train_loader = NeighborSampler(batch.edge_index, node_idx=batch.train_mask,
                                   sizes=nbr_sizes, batch_size=settings_dict['data']['batch_size'],
                                   shuffle=True, num_workers=12)
    subgraph_loader = NeighborSampler(batch.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=4096, shuffle=False,
                                      num_workers=12) #TODO what's batch size here
    x = batch.x.to(device)
    y = batch.y.squeeze().to(device)

    hidden_units = settings_dict['model']['hidden_units']
    n_epochs = settings_dict['epochs']

    model = SAGENet(dataset.num_features, dataset.num_edge_features, hidden_units,
                    dataset.num_classes, num_layers=len(nbr_sizes))
    model = model.to(device)
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=settings_dict['model']['lr'])

    best_val_f1 = final_test_f1 = best_epoch = 0
    train_losses = np.empty(n_epochs)
    train_f1s = val_f1s = test_f1s = val_losses = np.empty(n_epochs-5)

    for epoch in range(1, n_epochs+1):
        
        loss = train(epoch)
        train_losses[epoch-1] = loss
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    
        if not settings_dict['save_best_only']:
            model_path = settings_dict['model']['path'] + '/' + settings_dict['model']['name'] + '.pth'
            torch.save(model.state_dict(), model_path)

        if epoch > 5:
            train_f1, val_f1, test_f1, val_loss = test()
            train_f1s[epoch-6] = train_f1
            val_f1s[epoch-6] = val_f1
            test_f1s[epoch-6] = test_f1
            val_losses[epoch-6] = val_loss
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
    plt.legend(['train', 'val'])
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
    np.save('results/'+settings_dict['model']['name']+'_f1_train.npy', train_f1s)
    np.save('results/'+settings_dict['model']['name']+'_f1_val.npy', val_f1s)
    np.save('results/'+settings_dict['model']['name']+'_f1_test.npy', test_f1s)

    with open('results/'+settings_dict['model']['name']+'_exp_settings.json', 'w') as JSON:
        json.dump(settings_dict, JSON)