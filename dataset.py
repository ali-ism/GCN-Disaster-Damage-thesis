import os
from pathlib import Path
import pandas as pd
from typing import List
import torch
from torchvision.transforms import ToTensor
from torch_geometric.transforms import Compose, Delaunay, FaceToEdge
from PIL import Image
from torch_geometric.data import Data, Dataset


train_path = "/home/ami31/scratch/datasets/xbd/train_bldgs/"
test_path = "/home/ami31/scratch/datasets/xbd/test_bldgs/"
hold_path = "/home/ami31/scratch/datasets/xbd/hold_bldgs/"
tier3_path = "/home/ami31/scratch/datasets/xbd/tier3_bldgs/"

torch.manual_seed(42)

transform = ToTensor()
delaunay = Compose([Delaunay(), FaceToEdge()])

class xBD(Dataset):
    """
    xBD graph dataset.
    Every building (pre and post) is a node.
    Edge are created accoring to the Delaunay triangulation.
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
                pre_image = transform(pre_image)
                post_image = transform(post_image)
                images = torch.cat((pre_image, post_image),0)
                x.append(images.flatten())

            x = torch.stack(x)
            y = torch.tensor(y)
            coords = torch.tensor(coords)

            data = Data(x=x, y=y, pos=coords)
            data = delaunay(data)

            edge_index = data.edge_index

            #edge features
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


if __name__ == "__main__":

    root = "/home/ami31/scratch/datasets/xbd_graph/midwest"
    if not os.path.isdir(root):
        os.mkdir(root)
    xBD(root, train_path, 'midwest-flooding')