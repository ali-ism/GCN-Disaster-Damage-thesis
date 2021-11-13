import json
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from celluloid import Camera
from matplotlib.animation import PillowWriter
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor
from torch_geometric.transforms import Compose, GCNNorm, ToSparseTensor

from dataset import xBDFull
from model import CNNGCN
from utils import make_plot, merge_classes, score_cm

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

name = settings_dict['model']['name'] + '_gcn'
model_path = 'weights/' + name
disaster = settings_dict['data_ss']['disaster']
path = '/home/ami31/scratch/datasets/xbd/tier3_bldgs/'
root = settings_dict['data_ss']['root']
n_epochs = settings_dict['epochs']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train() -> Tuple[float]:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
    out = out[train_idx]
    loss = F.nll_loss(input=out, target=data.y[train_idx], weight=class_weights.to(device))
    loss.backward()
    optimizer.step()
    cm = confusion_matrix(data.y[train_idx].cpu(), out.detach().cpu().argmax(dim=1, keepdims=True))
    accuracy, precision, recall, specificity, f1 = score_cm(cm)
    return loss.detach().cpu().item(), accuracy, precision, recall, specificity, f1, z


@torch.no_grad()
def test(mask: Tensor) -> Tuple[float]:
    model.eval()
    out = model(data.x, data.adj_t)[mask].cpu()
    loss = F.nll_loss(input=out, target=data.y[mask].cpu(), weight=class_weights)
    cm = confusion_matrix(data.y[mask].cpu(), out.argmax(dim=1, keepdims=True))
    accuracy, precision, recall, specificity, f1 = score_cm(cm)
    return loss.item(), accuracy, precision, recall, specificity, f1


def save_results() -> None:
    make_plot(train_loss, test_loss, 'loss', name)
    make_plot(train_acc, test_acc, 'accuracy', name)
    make_plot(train_precision, test_precision, 'precision', name)
    make_plot(train_recall, test_recall, 'recall', name)
    make_plot(train_specificity, test_specificity, 'specificity', name)
    make_plot(train_f1, test_f1, 'f1', name)
    print('\n\nTrain results for last model.')
    print(f'Train accuracy: {train_acc[-1]:.4f}')
    print(f'Train precision: {train_precision[-1]:.4f}')
    print(f'Train recall: {train_recall[-1]:.4f}')
    print(f'Train specificity: {train_specificity[-1]:.4f}')
    print(f'Train f1: {train_f1[-1]:.4f}')
    print('\nTest results for last model.')
    print(f'Test accuracy: {test_acc[-1]:.4f}')
    print(f'Test precision: {test_precision[-1]:.4f}')
    print(f'Test recall: {test_recall[-1]:.4f}')
    print(f'Test specificity: {test_specificity[-1]:.4f}')
    print(f'Test f1: {test_f1[-1]:.4f}')
    hold_scores = test(hold_idx)
    print('\nHold results for last model.')
    print(f'Hold accuracy: {hold_scores[1]:.4f}')
    print(f'Hold precision: {hold_scores[2]:.4f}')
    print(f'Hold recall: {hold_scores[3]:.4f}')
    print(f'Hold specificity: {hold_scores[4]:.4f}')
    print(f'Hold f1: {hold_scores[5]:.4f}')
    all_scores = test(torch.ones(data.y.shape[0]).bool())
    print('\nFull results for last model.')
    print(f'Full accuracy: {all_scores[1]:.4f}')
    print(f'Full precision: {all_scores[2]:.4f}')
    print(f'Full recall: {all_scores[3]:.4f}')
    print(f'Full specificity: {all_scores[4]:.4f}')
    print(f'Full f1: {all_scores[5]:.4f}')
    print('\n\nTrain results for best model.')
    print(f'Train accuracy: {train_acc[best_epoch-1]:.4f}')
    print(f'Train precision: {train_precision[best_epoch-1]:.4f}')
    print(f'Train recall: {train_recall[best_epoch-1]:.4f}')
    print(f'Train specificity: {train_specificity[best_epoch-1]:.4f}')
    print(f'Train f1: {train_f1[best_epoch-1]:.4f}')
    print('\nTest results for best model.')
    print(f'Test accuracy: {test_acc[best_epoch-1]:.4f}')
    print(f'Test precision: {test_precision[best_epoch-1]:.4f}')
    print(f'Test recall: {test_recall[best_epoch-1]:.4f}')
    print(f'Test specificity: {test_specificity[best_epoch-1]:.4f}')
    print(f'Test f1: {test_f1[best_epoch-1]:.4f}')
    print('\nHold results for best model.')
    print(f'Hold accuracy: {hold_scores_best[1]:.4f}')
    print(f'Hold precision: {hold_scores_best[2]:.4f}')
    print(f'Hold recall: {hold_scores_best[3]:.4f}')
    print(f'Hold specificity: {hold_scores_best[4]:.4f}')
    print(f'Hold f1: {hold_scores_best[5]:.4f}')
    print('\nFull results for best model.')
    print(f'Full accuracy: {all_scores_best[1]:.4f}')
    print(f'Full precision: {all_scores_best[2]:.4f}')
    print(f'Full recall: {all_scores_best[3]:.4f}')
    print(f'Full specificity: {all_scores_best[4]:.4f}')
    print(f'Full f1: {all_scores_best[5]:.4f}')


if __name__ == "__main__":

    if settings_dict['merge_classes']:
        transform = Compose([merge_classes, GCNNorm(), ToSparseTensor()])
    else:
        transform = Compose([GCNNorm(), ToSparseTensor()])

    dataset = xBDFull(root, path, disaster, settings_dict['data_ss']['reduced_size'], transform=transform)

    num_classes = 3 if settings_dict['merge_classes'] else dataset.num_classes

    data = dataset[0]

    #extract hold set
    idx, hold_idx = train_test_split(
        np.arange(data.y.shape[0]), test_size=0.5,
        stratify=data.y, random_state=42
    )
    n_labeled_samples = round(settings_dict['data_ss']['labeled_size'] * data.y.shape[0])
    #select labeled samples
    train_idx, test_idx = train_test_split(
        np.arange(idx.shape[0]), train_size=n_labeled_samples,
        stratify=data.y[idx], random_state=42
    )

    print(f'Number of labeled samples: {train_idx.shape[0]}')
    print(f'Number of test samples: {test_idx.shape[0]}')
    print(f'Number of hold samples: {hold_idx.shape[0]}')

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(data.y.numpy()),
        y=data.y.numpy()[train_idx]
    )
    class_weights = torch.Tensor(class_weights)

    data = data.to(device)

    model = CNNGCN(
        settings_dict['model']['hidden_units'],
        num_classes,
        settings_dict['model']['num_layers'],
        settings_dict['model']['dropout_rate']
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=settings_dict['model']['lr'])

    best_test_f1 = best_epoch = 0

    train_loss = np.empty(n_epochs)
    train_acc = np.empty(n_epochs)
    train_precision = np.empty(n_epochs)
    train_recall = np.empty(n_epochs)
    train_specificity = np.empty(n_epochs)
    train_f1 = np.empty(n_epochs)
    test_loss = np.empty(n_epochs)
    test_acc = np.empty(n_epochs)
    test_precision = np.empty(n_epochs)
    test_recall = np.empty(n_epochs)
    test_specificity = np.empty(n_epochs)
    test_f1 = np.empty(n_epochs)

    fig_tsne = plt.figure(figsize=(10,10))
    camera = Camera(fig_tsne)
    plt.xticks([])
    plt.yticks([])

    for epoch in range(1, n_epochs+1):
        
        results = train()
        train_loss[epoch-1] = results[0]
        train_acc[epoch-1] = results[1]
        train_precision[epoch-1] = results[2]
        train_recall[epoch-1] = results[3]
        train_specificity[epoch-1] = results[4]
        train_f1[epoch-1] = results[5]
        print('**********************************************')
        print(f'Epoch {epoch}, Train Loss: {train_loss[epoch-1]:.4f}')

        z = results[6]
        plt.scatter(z[:, 0], z[:, 1], s=70, c=data.y.cpu().numpy(), cmap="Set1")
        camera.snap()

        results = test(test_idx)
        test_loss[epoch-1] = results[0]
        test_acc[epoch-1] = results[1]
        test_precision[epoch-1] = results[2]
        test_recall[epoch-1] = results[3]
        test_specificity[epoch-1] = results[4]
        test_f1[epoch-1] = results[5]

        if test_f1[epoch-1] > best_test_f1:
            best_test_f1 = test_f1[epoch-1]
            best_epoch = epoch
            hold_scores_best = test(hold_idx)
            all_scores_best = test(torch.ones(data.y.shape[0]).bool())
    
    animation = camera.animate()
    animation.save('results/'+name+'_tsne.gif', writer=PillowWriter(fps=2))

    print(f'\nBest test F1 {best_test_f1} at epoch {best_epoch}.\n')
    save_results()
    print(f"\nLabeled size: {settings_dict['data_ss']['labeled_size']}")
    print(f"Reduced dataset size: {settings_dict['data_ss']['reduced_size']}")