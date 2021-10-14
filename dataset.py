import os
from pathlib import Path
import numpy as np
import pandas as pd
import utm
from typing import List, Callable, Tuple
import torch
from torchvision.transforms import ToTensor
from torch_geometric.transforms import Compose, Delaunay, FaceToEdge
from PIL import Image
from torch_geometric.data import Data, Dataset, InMemoryDataset
from sklearn.model_selection import train_test_split


torch.manual_seed(42)

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


delaunay = Compose([Delaunay(), FaceToEdge()])

class xBDBatch(Dataset):
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
        transform=None,
        pre_transform=None) -> None:
        
        self.path = data_path
        self.disaster = disaster_name
        self.labels = pd.read_csv(list(Path(self.path + self.disaster).glob('*.csv*'))[0], index_col=0)
        self.labels.drop(columns=['long','lat'], inplace=True)
        zone = lambda row: '_'.join(row.name.split('_', 2)[:2])
        self.labels['zone'] = self.labels.apply(zone, axis=1)
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


class xBDFull(InMemoryDataset):
    def __init__(
        self,
        root: str, 
        data_path: str,
        disaster_name: str,
        transform=None,
        pre_transform=None) -> None:

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
        self.labels['class_num'] = self.labels['class'].apply(lambda x: label_dict)
        
        idx, _ = train_test_split(
            np.arange(self.labels.shape[0]), test_size=0.3,
            stratify=self.labels['class_num'].values, random_state=42)
        
        self.labels = self.labels.iloc[[idx]]
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

        for post_image_file in self.labels.index.values.tolist():
            
            y.append(self.labels.loc[post_image_file,'class_num'])
            coords.append((self.labels.loc[post_image_file,'easting'],
                            self.labels.loc[post_image_file,'northing']))

            pre_image = Image.open(os.path.join(self.path, self.disaster, post_image_file.replace('post', 'pre')))
            post_image = Image.open(os.path.join(self.path, self.disaster, post_image_file))
            pre_image = pre_image.resize((128, 128))
            post_image = post_image.resize((128, 128))
            pre_image = to_tensor(pre_image)
            post_image = to_tensor(post_image)
            images = torch.cat((pre_image, post_image),0)
            x.append(images.flatten())

        x = torch.stack(x)
        print(f'Size of x matrix: {x.element_size()*x.nelement()*1e-9:.4f} GB')
        y = torch.tensor(y)
        coords = torch.tensor(coords)

        data_ = Data(x=x, y=y, pos=coords)
        data_ = delaunay(data_)

        edge_index = data_.edge_index

        edge_attr = torch.empty((edge_index.shape[1],1))
        for i in range(edge_index.shape[1]):
            node1 = x[edge_index[0,i]]
            node2 = x[edge_index[1,i]]
            s = (torch.abs(node1 - node2)) / (torch.abs(node1) + torch.abs(node2))
            s[s.isnan()] = 1
            s = 1 - torch.sum(s)/node1.shape[0]
            edge_attr[i,0] = s.item()
        
        data_.edge_attr = edge_attr

        data_list = [data_]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":

    train_path = "/home/ami31/scratch/datasets/xbd/train_bldgs/"
    test_path = "/home/ami31/scratch/datasets/xbd/test_bldgs/"
    hold_path = "/home/ami31/scratch/datasets/xbd/hold_bldgs/"
    tier3_path = "/home/ami31/scratch/datasets/xbd/tier3_bldgs/"

    root = "/home/ami31/scratch/datasets/xbd_graph/pinery_full_reduced"
    if not os.path.isdir(root):
        os.mkdir(root)
    xBDFull(root, tier3_path, 'pinery-bushfire')