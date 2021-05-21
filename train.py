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
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum()) #TODO modify this
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(batch.train_mask.sum())

    return loss, approx_acc


@torch.no_grad()
def test(): #TODO
    model.eval()

    out = model.inference(x, subgraph_loader)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    #train_acc = evaluator.eval({
    #    'y_true': y_true[split_idx['train']],
    #    'y_pred': y_pred[split_idx['train']],
    #})['acc']
    #val_acc = evaluator.eval({
    #    'y_true': y_true[split_idx['valid']],
    #    'y_pred': y_pred[split_idx['valid']],
    #})['acc']
    #test_acc = evaluator.eval({
    #    'y_true': y_true[split_idx['test']],
    #    'y_pred': y_pred[split_idx['test']],
    #})['acc']

    #return train_acc, val_acc, test_acc


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

    best_val_acc = final_test_acc = best_epoch = 0
    train_losses = np.empty(n_epochs)

    for epoch in range(1, n_epochs+1):
        
        loss, acc = train(epoch)
        train_losses[epoch-1] = loss
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    
        if not settings_dict['save_best_only']:
            model_path = settings_dict['model']['path'] + '/' + settings_dict['model']['name'] + '.pth'
            torch.save(model.state_dict(), model_path)

        if epoch > 5:
            train_acc, val_acc, test_acc = test()
            print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, 'f'Test: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                final_test_acc = test_acc
                model_path = settings_dict['model']['path'] + '/' + settings_dict['model']['name'] + '_best.pth'
                print(f'New best model saved to: {model_path}')
                torch.save(model.state_dict(), model_path)
    
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('results/'+settings_dict['model']['name']+'_loss.eps')

    np.save('results/'+settings_dict['model']['name']+'_loss_train.npy', train_losses)