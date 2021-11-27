import os
import os.path as osp
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch_geometric
import utm
from PIL import Image
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import Compose, Delaunay, FaceToEdge
from torchvision.transforms import ToTensor

torch.manual_seed(42)

to_tensor = ToTensor()

class xBDImages(torch.utils.data.Dataset):
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
        assert pre_image.shape == post_image.shape == (3,128,128)
        images = torch.cat((pre_image, post_image),0).flatten()

        if self.transform is not None:
            images = self.transform(images)

        y = torch.tensor(self.label_dict[self.labels['class'][idx]])

        if self.merge_classes:
            y[y==3] = 2
        
        sample = {'x': images, 'y': y}

        return sample


delaunay = Compose([Delaunay(), FaceToEdge()])

class xBDMiniGraphs(torch_geometric.data.Dataset):
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
                processed_files.append(os.path.join(self.processed_dir, f'{zone}.pt'))
        return processed_files

    def process(self) -> None:
        label_dict = {'no-damage':0,'minor-damage':1,'major-damage':2,'destroyed':3}
        for zone in self.zones:
            if os.path.isfile(os.path.join(self.processed_dir, f'{zone}.pt')) or \
            (self.labels[self.labels['zone'] == zone]['class'] == 'un-classified').all() or \
            (self.labels[self.labels['zone'] == zone]['class'] != 'un-classified').sum() == 1:
                continue
            print(f'Building {zone}...')
            list_pre_images = list(map(str, Path(self.path + self.disaster).glob(f'{zone}_pre_disaster*')))
            list_post_images = list(map(str, Path(self.path + self.disaster).glob(f'{zone}_post_disaster*')))
            x = []
            y = []
            coords = []

            for pre_image_file, post_image_file in zip(list_pre_images, list_post_images):
                
                annot = self.labels.loc[os.path.split(post_image_file)[1],'class']
                if annot == 'un-classified':
                    continue
                y.append(label_dict[annot])
                coords.append((self.labels.loc[os.path.split(post_image_file)[1],'xcoord'],
                                self.labels.loc[os.path.split(post_image_file)[1],'ycoord']))

                pre_image = Image.open(pre_image_file)
                post_image = Image.open(post_image_file)
                pre_image = pre_image.resize((128, 128))
                post_image = post_image.resize((128, 128))
                pre_image = to_tensor(pre_image)
                post_image = to_tensor(post_image)
                assert pre_image.shape == post_image.shape == (3,128,128)
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

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, f'{zone}.pt'))
    
    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data


class xBDFullGraph(InMemoryDataset):
    def __init__(
        self,
        root: str, 
        data_path: str,
        disaster_name: str,
        reduced_dataset_size: int,
        transform: Callable=None,
        pre_transform: Callable=None) -> None:

        self.path = data_path
        self.disaster = disaster_name
        self.labels = pd.read_csv(list(Path(self.path + self.disaster).glob('*.csv*'))[0], index_col=0)
        self.labels.drop(columns=['xcoord','ycoord'], inplace=True)
        self.labels.drop(index=self.labels.loc[self.labels['class']=='un-classified'].index, inplace=True)
        zone_func = lambda row: '_'.join(row.name.split('_', 2)[:2])
        self.labels['zone'] = self.labels.apply(zone_func, axis=1)

        for zone in self.labels['zone'].unique():
            if (self.labels[self.labels['zone'] == zone]['class'] == 'no-damage').all():
                self.labels.drop(index=self.labels.loc[self.labels['zone']==zone].index, inplace=True)
        
        label_dict = {'no-damage':0,'minor-damage':1,'major-damage':2,'destroyed':3}
        self.labels['class_num'] = self.labels['class'].apply(lambda x: label_dict[x])
        
        if self.labels.shape[0] > reduced_dataset_size:
            idx, _ = train_test_split(
                np.arange(self.labels.shape[0]), train_size=reduced_dataset_size,
                stratify=self.labels['class_num'].values, random_state=42)
            self.labels = self.labels.iloc[idx,:]

        self.labels['easting'], self.labels['northing'], *_ = utm.from_latlon(
            self.labels['lat'].values, self.labels['long'].values
        )

        super().__init__(root, transform, pre_transform)
        self.labels.to_csv(os.path.join(self.processed_dir, self.disaster+'_metadata.csv'))
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return [os.path.join(self.processed_dir, self.disaster+'_data.pt')]
    
    def download(self) -> None:
        pass

    def process(self) -> None:
        x = []
        y = []
        coords = []

        for post_image_file in self.labels.index.tolist():
            
            y.append(self.labels.loc[post_image_file,'class_num'])
            coords.append((self.labels.loc[post_image_file,'easting'],
                            self.labels.loc[post_image_file,'northing']))

            pre_image = Image.open(os.path.join(self.path, self.disaster, post_image_file.replace('post', 'pre')))
            post_image = Image.open(os.path.join(self.path, self.disaster, post_image_file))
            pre_image = pre_image.resize((128, 128))
            post_image = post_image.resize((128, 128))
            pre_image = to_tensor(pre_image)
            post_image = to_tensor(post_image)
            assert pre_image.shape == post_image.shape == (3,128,128)
            images = torch.cat((pre_image, post_image),0)
            x.append(images.flatten())

        x = torch.stack(x)
        print(f'Size of x matrix: {x.element_size()*x.nelement()*1e-9:.4f} GB')
        y = torch.tensor(y)
        coords = torch.tensor(coords)

        data_ = Data(x=x, y=y, pos=coords)
        data_ = delaunay(data_)

        edge_index = data_.edge_index

        edge_attr = torch.empty(edge_index.shape[1])
        for i in range(edge_index.shape[1]):
            node1 = x[edge_index[0,i]]
            node2 = x[edge_index[1,i]]
            s = (torch.abs(node1 - node2)) / (torch.abs(node1) + torch.abs(node2))
            s[s.isnan()] = 1
            s = 1 - torch.sum(s)/node1.shape[0]
            edge_attr[i] = s.item()
        
        data_.edge_attr = edge_attr

        data_list = [data_]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class BeirutFullGraph(InMemoryDataset):
    def __init__(
        self,
        root: str,
        data_path: str,
        reduced_dataset_size: int,
        meta_features: bool=False,
        transform: Callable=None,
        pre_transform: Callable=None) -> None:

        self.meta_features = meta_features
        self.path = data_path

        cols = ['OID_', 'damage', 'Type', 'MIN_BLDG_N', 'MAX_BLDG_N', 'SUM_N_KWH', 'Shape_Leng', 'ORIG_FID', 'BUFF_DIST', 'ORIG_FID_1', 'Shape_Length', 'Shape_Area', 'Floors_fin', 'NbreEtages', 'Era_all', 'era_usj', 'Era_fin', 'era_usj_1']
        self.labels = pd.read_csv(osp.join(self.path, 'buffered_masks.csv')).drop(columns=cols)
        self.labels['built_year_final'] = self.labels.apply(lambda row: row['built_year'] if row['built_year'] else row['Annee'] , axis = 1)
        self.labels['Floors_final'] = self.labels.apply(lambda row: row['Floors'] if row['Floors'] else row['Estim_Etag'] , axis = 1)
        self.labels.drop(columns=['built_year', 'Annee', 'Estim_Etag', 'Floors'], inplace=True)
        self.labels.replace(r'\s+', np.nan, regex=True, inplace=True)
        self.labels['Const_Year'].fillna(0, inplace=True)
        self.labels['Fonction'].fillna('Autre', inplace=True)
        self.labels = pd.get_dummies(self.labels, drop_first=True)
        
        if self.labels.shape[0] > reduced_dataset_size:
            idx, _ = train_test_split(
                np.arange(self.labels.shape[0]), train_size=reduced_dataset_size,
                stratify=self.labels['damage_num'].values, random_state=42)
            self.labels = self.labels.iloc[idx,:]

        self.labels['Easting'], self.labels['Northing'], *_ = utm.from_latlon(
            self.labels['Latitude'].values, self.labels['Longitude'].values
        )

        self.labels.drop(columns=['Latitude', 'Longitude'], inplace=True)
        num_cols = ['NbreAppts', 'MEAN_DSM_O', 'MEAN_Blg_H', 'Area', 'perimeter', 'era_final', 'built_year_final']
        self.labels[num_cols] = self.labels[num_cols]/self.labels[num_cols].max()
        if self.meta_features:
            self.num_meta_features = self.labels.drop(columns=['Easting', 'Northing', 'damage_num']).shape[1]
        else:
            self.num_meta_features = 0

        super().__init__(root, transform, pre_transform)
        self.labels.to_csv(osp.join(self.processed_dir, 'beirut_metadata.csv'))
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ['beirut_data.pt']
    
    def download(self) -> None:
        pass

    def process(self) -> None:
        x = []
        y = []
        coords = []

        pre_image_files = sorted(os.listdir(osp.join(self.path, 'pre_bldgs')))
        post_image_files = sorted(os.listdir(osp.join(self.path, 'post_bldgs')))

        for i in self.labels.index.tolist():
            y.append(self.labels['damage_num'][i])
            coords.append((self.labels['Easting'][i], self.labels['Northing'][i]))
            pre_image = Image.open(osp.join(self.path, 'pre_bldgs', pre_image_files[i]))
            post_image = Image.open(osp.join(self.path, 'post_bldgs', post_image_files[i]))
            pre_image = pre_image.resize((128, 128))
            post_image = post_image.resize((128, 128))
            pre_image = to_tensor(pre_image)
            pre_image = pre_image[:3,:,:]
            post_image = to_tensor(post_image)
            post_image = post_image[:3,:,:]
            assert pre_image.shape == post_image.shape == (3,128,128)
            images = torch.cat((pre_image, post_image),0)
            x.append(images.flatten())

        x = torch.stack(x)
        if self.meta_features:
            meta = torch.from_numpy(self.labels.drop(columns=['Easting', 'Northing', 'damage_num']).values)
            x = torch.cat([x, meta], dim=1).float()
        print(f'Size of x matrix: {x.element_size()*x.nelement()*1e-9:.4f} GB')
        y = torch.tensor(y)
        coords = torch.tensor(coords)

        data_ = Data(x=x, y=y, pos=coords)
        data_ = delaunay(data_)

        edge_index = data_.edge_index

        edge_attr = torch.empty(edge_index.shape[1])
        for i in range(edge_index.shape[1]):
            node1 = x[edge_index[0,i]]
            node2 = x[edge_index[1,i]]
            s = (torch.abs(node1 - node2)) / (torch.abs(node1) + torch.abs(node2))
            s[s.isnan()] = 1
            s = 1 - torch.sum(s)/node1.shape[0]
            edge_attr[i] = s.item()
        
        data_.edge_attr = edge_attr

        data_list = [data_]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])