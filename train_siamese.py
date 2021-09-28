import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from typing import Callable, List, Tuple
from model import SiameseNet
from utils import score
from tqdm import tqdm
from train import make_plot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

batch_size = settings_dict['data']['batch_size']
name = settings_dict['model']['name'] + '_SiameseClf'
train_disasters = settings_dict['data']['train_disasters']
train_paths = settings_dict['data']['train_paths']
assert len(train_disasters) == len(train_paths)
merge_classes = settings_dict['data']['merge_classes']
n_epochs = settings_dict['epochs']
starting_epoch = 1


to_tensor = ToTensor()

class xBDImages(Dataset):
    """
    xBD building image dataset.

    Args:
        paths (List[str]): paths to the desired data split (train, test, hold or tier3).
        disasters (List[str]): names of the included disasters.
    """
    def __init__(
        self,
        paths: List[str],
        disasters: List[str],
        merge_classes: bool=False,
        transform: Callable=None) -> None:

        list_labels = []
        for disaster, path in zip(disasters, paths):
            labels = pd.read_csv(list(Path(path + disaster).glob('*.csv*'))[0], index_col=0)
            labels.drop(columns=['long','lat', 'xcoord', 'ycoord'], inplace=True)
            labels.drop(index=labels[labels['class'] == 'un-classified'].index, inplace = True)
            labels['image_path'] = path + disaster + '/'
            list_labels.append(labels)
        
        self.labels = pd.concat(list_labels)
        self.label_dict = {'no-damage':0,'minor-damage':1,'major-damage':2,'destroyed':3}
        self.num_classes = 3 if merge_classes else 4
        self.merge_classes = merge_classes
        self.transform = transform
    
    def __len__(self) -> int:
        return self.labels.shape[0]
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor]:

        if torch.is_tensor(idx):
            idx = idx.tolist()

        post_image_file = self.labels['image_path'][idx] + self.labels.index[idx]
        pre_image_file = post_image_file.replace('post', 'pre')
        pre_image = Image.open(pre_image_file)
        post_image = Image.open(post_image_file)
        pre_image = pre_image.resize((128, 128))
        post_image = post_image.resize((128, 128))
        pre_image = to_tensor(pre_image)
        post_image = to_tensor(post_image)
        images = torch.cat((pre_image, post_image),0).flatten()

        if self.transform is not None:
            images = self.transform(images)

        y = torch.tensor(self.label_dict[self.labels['class'][idx]])

        if self.merge_classes:
            y[y==3] = 2
        
        sample = {'x': images, 'y': y}

        return sample


def train(epoch: int) -> float:
    model.train()
    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = 0
    y_true = []
    y_pred = []
    for data in train_loader:
        x = data['x'].to(device)
        y = data['y'].to(device)
        optimizer.zero_grad()
        out = model(x)
        y_pred.append(out.cpu())
        y_true.append(y.cpu())
        loss = F.nll_loss(input=out, target=y, weight=class_weights.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.update(x.shape[0])
    pbar.close()
    y_pred = torch.cat(y_pred).detach()
    y_true = torch.cat(y_true).detach()
    accuracy, f1_macro, f1_weighted, auc = score(y_true, y_pred)
    return total_loss / len(train_loader), accuracy, f1_macro, f1_weighted, auc


@torch.no_grad()
def test(dataloader) -> Tuple[float]:
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0
    for data in dataloader:
        x = data['x'].to(device)
        y = data['y']
        out = model(x).cpu()
        y_pred.append(out)
        y_true.append(y)
        loss = F.nll_loss(input=out, target=y, weight=class_weights)
        total_loss += loss.item()
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    accuracy, f1_macro, f1_weighted, auc = score(y_true, y_pred)
    total_loss = total_loss / len(dataloader)
    return total_loss, accuracy, f1_macro, f1_weighted, auc


@torch.no_grad()
def save_results(hold: bool=False) -> None:
    make_plot(train_loss, test_loss, 'loss', name)
    make_plot(train_acc, test_acc, 'accuracy', name)
    make_plot(train_f1_macro, test_f1_macro, 'macro_f1', name)
    make_plot(train_f1_weighted, test_f1_weighted, 'weighted_f1', name)
    make_plot(train_auc, test_auc, 'auc', name)
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
        hold_dataset = xBDImages(
            ['/home/ami31/scratch/datasets/xbd/hold_bldgs/'],
            ['socal-fire'],
            merge_classes
        )
        hold_loader = DataLoader(hold_dataset, batch_size)
        hold_scores = test(hold_loader)
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
        hold_scores = test(hold_loader)
        print('\nHold results for best model.')
        print(f'Hold accuracy: {hold_scores[0]:.4f}')
        print(f'Hold macro F1: {hold_scores[1]:.4f}')
        print(f'Hold weighted F1: {hold_scores[2]:.4f}')
        print(f'Hold auc: {hold_scores[3]:.4f}')


if __name__ == "__main__":

    train_dataset = xBDImages(train_paths, train_disasters, merge_classes)
    test_dataset = xBDImages(['/home/ami31/scratch/datasets/xbd/test_bldgs/'], ['socal-fire'], merge_classes)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size)

    cw_name = '_'.join(text.replace('-', '_') for text in train_disasters) + '_siameseclf'
    if os.path.isfile(f'weights/class_weights_{cw_name}_{train_dataset.num_classes}.pt'):
        class_weights = torch.load(f'weights/class_weights_{cw_name}_{train_dataset.num_classes}.pt')
    else:
        y_all = np.fromiter((data['y'].item() for data in train_dataset), dtype=int)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_all), y=y_all)
        class_weights = torch.Tensor(class_weights)
        torch.save(class_weights, f'weights/class_weights_{cw_name}_{train_dataset.num_classes}.pt')

    model = SiameseNet(
        settings_dict['model']['hidden_units'],
        train_dataset.num_classes,
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
        
        train_loss[epoch-1], train_acc[epoch-1], train_f1_macro[epoch-1],\
            train_f1_weighted[epoch-1], train_auc[epoch-1] = train(epoch)
        print('**********************************************')
        print(f'Epoch {epoch:02d}, Train Loss: {train_loss[epoch-1]:.4f}')
    
        if not settings_dict['save_best_only']:
            model_path = 'weights/' + name + '.pt'
            torch.save(model.state_dict(), model_path)

        test_loss[epoch-1], test_acc[epoch-1], test_f1_macro[epoch-1],\
            test_f1_weighted[epoch-1], test_auc[epoch-1] = test(test_loader)
        #scheduler.step(test_loss[epoch-1])

        if test_auc[epoch-1] > best_test_auc:
            best_test_auc = test_auc[epoch-1]
            best_epoch = epoch
            model_path = 'weights/' + name + '_best.pt'
            print(f'New best model saved with AUC {best_test_auc} at epoch {best_epoch}.')
            torch.save(model.state_dict(), model_path)
        
        if not (epoch % 2):
            save_results()
    
    print(f'\nBest test AUC {best_test_auc} at epoch {best_epoch}.\n')
    save_results(hold=True)