# ------------------------------------------------------------------------------
# This code is (modified) from
# https://github.com/rusty1s/pytorch_geometric/blob/master/examples/reddit.py
# Licensed under the MIT License.
# Written by Matthias Fey (http://rusty1s.github.io)
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from tqdm import tqdm
from dataset import IIDxBD
from model import SAGENet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(epoch):
    model.train()

    pbar = tqdm(total=int(data.train_mask.sum()))
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
    approx_acc = total_correct / int(data.train_mask.sum())

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

    dataset = IIDxBD()
    #split_idx = dataset.get_idx_split()
    nb_sizes = [15, 10, 5]
    hidden_units = 256
    n_epochs = 20

    model = SAGENet(dataset.num_features, dataset.num_edge_features, hidden_units,
                    dataset.num_classes, num_layers=len(nb_sizes))
    model = model.to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    best_val_acc = final_test_acc = best_epoch = 0

    train_losses = np.empty(n_epochs)

    for epoch in range(1, n_epochs+1):

        for data in dataset:

            train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                           sizes=nb_sizes, batch_size=1024,
                                           shuffle=True, num_workers=12)

            subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                              batch_size=4096, shuffle=False,
                                              num_workers=12)

            x = data.x.to(device)
            y = data.y.squeeze().to(device)
            loss, acc = train(epoch)
            train_losses[epoch-1] = loss
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

        if epoch > 5:
            train_acc, val_acc, test_acc = test()
            print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, 'f'Test: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                final_test_acc = test_acc
    
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('exp.eps')