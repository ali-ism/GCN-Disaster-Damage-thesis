import json
import os.path as osp
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch_geometric
from PIL import Image
from statsmodels.stats.contingency_tables import mcnemar
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, RandomNodeSampler
from torch_geometric.transforms import Compose, Delaunay, FaceToEdge
from torchvision.transforms import ToTensor

from model import CNNSage, SiameseNet
from utils import merge_classes

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

batch_size = settings_dict['data_sup']['batch_size']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

to_tensor = ToTensor()
delaunay = Compose([Delaunay(), FaceToEdge()])

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
            #labels.drop(index=labels[labels['class'] == 'un-classified'].index, inplace = True)
            labels['image_path'] = path + disaster + '/'
            zone_func = lambda row: '_'.join(row.name.split('_', 2)[:2])
            labels['zone'] = labels.apply(zone_func, axis=1)
            zones = labels['zone'].value_counts()[labels['zone'].value_counts()<=1].index.tolist()
            for zone in zones:
                labels.drop(index=labels.loc[labels['zone']==zone].index, inplace=True)
            for zone in labels['zone'].unique():
                if (labels[labels['zone'] == zone]['class'] == 'un-classified').all() or \
                   (labels[labels['zone'] == zone]['class'] != 'un-classified').sum() == 1:
                    labels.drop(index=labels.loc[labels['zone']==zone].index, inplace=True)
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
        
        sample = {'key': self.labels.index[idx], 'x': images, 'y': y}

        return sample


class xBDBatch(torch_geometric.data.Dataset):
    """
    xBD graph dataset.
    Every image chip is a graph.
    Every building (pre and post) is a node.
    Edges are created accoring to the Delaunay triangulation.
    Edge features are calculated as a similarity measure between the nodes.

    Args:
        root (str): path where the processed dataset is saved.
        data_path (str): path to the desired data split (train, test, hold or tier3).
        disaster_name (str): name of the included disaster.
    """
    def __init__(
        self,
        root: str,
        data_path: str,
        disaster_name: str,
        transform: Callable=None,
        pre_transform: Callable=None) -> None:
        
        self.path = data_path
        self.disaster = disaster_name
        self.labels = pd.read_csv(list(Path(self.path + self.disaster).glob('*.csv*'))[0], index_col=0)
        self.labels.drop(columns=['long','lat'], inplace=True)
        zone_func = lambda row: '_'.join(row.name.split('_', 2)[:2])
        self.labels['zone'] = self.labels.apply(zone_func, axis=1)
        self.zones = self.labels['zone'].value_counts()[self.labels['zone'].value_counts()>1].index.tolist()
        self.num_classes = 4

        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> List:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        processed_files = []
        for zone in self.zones:
            if not ((self.labels[self.labels['zone'] == zone]['class'] == 'un-classified').all() or \
                    (self.labels[self.labels['zone'] == zone]['class'] != 'un-classified').sum() == 1):
                processed_files.append(osp.join(self.processed_dir, f'{zone}.pt'))
        return processed_files

    def process(self) -> None:
        label_dict = {'no-damage':0,'minor-damage':1,'major-damage':2,'destroyed':3}
        for zone in self.zones:
            if osp.isfile(osp.join(self.processed_dir, f'{zone}.pt')) or \
            (self.labels[self.labels['zone'] == zone]['class'] == 'un-classified').all() or \
            (self.labels[self.labels['zone'] == zone]['class'] != 'un-classified').sum() == 1:
                continue
            print(f'Building {zone}...')
            list_pre_images = list(map(str, Path(self.path + self.disaster).glob(f'{zone}_pre_disaster*')))
            list_post_images = list(map(str, Path(self.path + self.disaster).glob(f'{zone}_post_disaster*')))
            x = []
            y = []
            key = []
            coords = []

            for pre_image_file, post_image_file in zip(list_pre_images, list_post_images):
                
                annot = self.labels.loc[osp.split(post_image_file)[1],'class']
                if annot == 'un-classified':
                    continue
                key.append(osp.split(post_image_file)[1])
                y.append(label_dict[annot])
                coords.append((self.labels.loc[osp.split(post_image_file)[1],'xcoord'],
                                self.labels.loc[osp.split(post_image_file)[1],'ycoord']))

                pre_image = Image.open(pre_image_file)
                post_image = Image.open(post_image_file)
                pre_image = pre_image.resize((128, 128))
                post_image = post_image.resize((128, 128))
                pre_image = to_tensor(pre_image)
                post_image = to_tensor(post_image)
                images = torch.cat((pre_image, post_image),0)
                x.append(images.flatten())

            x = torch.stack(x)
            y = torch.tensor(y)
            coords = torch.tensor(coords)

            data = Data(x=x, y=y, pos=coords)
            data = delaunay(data)

            edge_index = data.edge_index

            edge_attr = torch.empty((edge_index.shape[1],1))
            for i in range(edge_index.shape[1]):
                node1 = x[edge_index[0,i]]
                node2 = x[edge_index[1,i]]
                s = (torch.abs(node1 - node2)) / (torch.abs(node1) + torch.abs(node2))
                s[s.isnan()] = 1
                s = 1 - torch.sum(s)/node1.shape[0]
                edge_attr[i,0] = s.item()
            data.edge_attr = edge_attr

            data.key = key

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            torch.save(data, osp.join(self.processed_dir, f'{zone}.pt'))
    
    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int):
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        return data

@torch.no_grad()
def infer_cnn() -> Tuple[np.ndarray]:
    y_true = []
    y_pred = []
    keys = []
    for data in hold_loader:
        x = data['x'].to(device)
        y = data['y']
        keys.extend(data['key'])
        out = model(x).cpu()
        y_pred.append(out)
        y_true.append(y)
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    return keys, y_true.numpy(), y_pred.numpy()


@torch.no_grad()
def infer_sage() -> Tuple[np.ndarray]:
    y_true = []
    y_pred = []
    keys = []
    for data in hold_dataset:
        if data.num_nodes > batch_size:
            sampler = RandomNodeSampler(data, num_parts=data.num_nodes//batch_size, num_workers=2)
            for subdata in sampler:
                subdata = subdata.to(device)
                out = model(subdata.x, subdata.edge_index).cpu()
                y_pred.append(out)
                y_true.append(subdata.y.cpu())
                keys.extend(subdata.key)
        else:
            data = data.to(device)
            out = model(data.x, data.edge_index).cpu()
            y_pred.append(out)
            y_true.append(data.y.cpu())
            keys.extend(data.key)
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    return keys, y_true.numpy(), y_pred.numpy()


def contingency_table(y_target: np.ndarray, y_model1: np.ndarray, y_model2: np.ndarray) -> np.ndarray:
    """
    Compute a 2x2 contigency table for McNemar's test.
    Source: Raschka, Sebastian (2018) MLxtend: Providing machine learning and data science utilities and extensions to Python's scientific computing stack. J Open Source Softw 3(24).
    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True class labels as 1D NumPy array.
    y_model1 : array-like, shape=[n_samples]
        Predicted class labels from model as 1D NumPy array.
    y_model2 : array-like, shape=[n_samples]
        Predicted class labels from model 2 as 1D NumPy array.
    Returns
    ----------
    tb : array-like, shape=[2, 2]
       2x2 contingency table with the following contents:
       a: tb[0, 0]: # of samples that both models predicted correctly
       b: tb[0, 1]: # of samples that model 1 got right and model 2 got wrong
       c: tb[1, 0]: # of samples that model 2 got right and model 1 got wrong
       d: tb[1, 1]: # of samples that both models predicted incorrectly
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar_table/
    """
    for ary in (y_target, y_model1, y_model2):
        if len(ary.shape) != 1:
            raise ValueError('One or more input arrays are not 1-dimensional.')

    if y_target.shape[0] != y_model1.shape[0]:
        raise ValueError('y_target and y_model1 contain a different number'
                         ' of elements.')

    if y_target.shape[0] != y_model2.shape[0]:
        raise ValueError('y_target and y_model2 contain a different number'
                         ' of elements.')

    m1_vs_true = (y_target == y_model1).astype(int)
    m2_vs_true = (y_target == y_model2).astype(int)

    plus_true = m1_vs_true + m2_vs_true
    minus_true = m1_vs_true - m2_vs_true

    tb = np.zeros((2, 2), dtype=int)

    tb[0, 0] = np.sum(plus_true == 2)
    tb[0, 1] = np.sum(minus_true == 1)
    tb[1, 0] = np.sum(minus_true == -1)
    tb[1, 1] = np.sum(plus_true == 0)

    return tb


if __name__ == "__main__":

    if settings_dict['merge_classes']:
        transform = merge_classes
    else:
        transform = None
    
    hold_dataset = xBDBatch(
        '/home/ami31/scratch/datasets/xbd_graph/socal_hold_mcnemar',
        '/home/ami31/scratch/datasets/xbd/hold_bldgs/',
        'socal-fire',
        transform=transform
    )
    num_classes = 3 if settings_dict['merge_classes'] else hold_dataset.num_classes
    model = CNNSage(
        settings_dict['model']['hidden_units'],
        num_classes,
        settings_dict['model']['num_layers'],
        settings_dict['model']['dropout_rate']
    )
    name = settings_dict['model']['name'] + '_sage'
    model_path = 'weights/' + name
    model.load_state_dict(torch.load(model_path+'_best.pt'))
    model.eval()
    model = model.to(device)
    keys_sage, y_true_sage, y_pred_sage = infer_sage()
    np.save('results/keys_sage.npy', keys_sage)
    np.save('results/y_true_sage.npy', y_true_sage)
    np.save('results/y_pred_sage.npy', y_pred_sage)
    df_sage = pd.DataFrame({'y_true': y_true_sage, 'y_pred': y_pred_sage}, index=keys_sage).sort_index()

    hold_dataset = xBDImages(
        ['/home/ami31/scratch/datasets/xbd/hold_bldgs/'],
        ['socal-fire'],
        merge_classes
    )
    hold_loader = DataLoader(hold_dataset, batch_size)
    model = SiameseNet(
        settings_dict['model']['hidden_units'],
        num_classes,
        settings_dict['model']['dropout_rate']
    )
    name = settings_dict['model']['name'] + '_siamese'
    model_path = 'weights/' + name
    model.load_state_dict(torch.load(model_path+'_best.pt'))
    model.eval()
    model = model.to(device)
    keys_cnn, y_true_cnn, y_pred_cnn = infer_cnn()
    np.save('results/keys_cnn.npy', keys_cnn)
    np.save('results/y_true_cnn.npy', y_true_cnn)
    np.save('results/y_pred_cnn.npy', y_pred_cnn)
    df_cnn = pd.DataFrame({'y_true': y_true_cnn, 'y_pred': y_pred_cnn}, index=keys_cnn).sort_index()

    assert np.array_equal(df_cnn['y_true'].values, df_sage['y_true'].values)

    table = contingency_table(df_cnn['y_true'].values, df_sage['y_pred'].values, df_cnn['y_pred'].values)

    if table[0,1] + table[1,0] < 25:
        stat, p_value = mcnemar(table, exact=True)
    else:
        stat, p_value = mcnemar(table, exact=False, correction=True)
    
    print(f'p value: {p_value}')
    if p_value <= 0.05:
        print('Null hypothesis rejected')
    else:
        print('Failed to reject null hypothesis')