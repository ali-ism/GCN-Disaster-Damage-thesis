import os
from pathlib import Path
from shutil import copy2
import json
import pandas as pd
from typing import List
import torch
import torchvision.transforms as tr
from PIL import Image
from torch_geometric.data import Data, Dataset
from feature_extractor import load_feature_extractor
from utils import build_edge_idx, get_edge_features

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

seed = settings_dict['seed']
train_path = settings_dict['data']['train_bldgs']
test_path = settings_dict['data']['test_bldgs']
hold_path = settings_dict['data']['hold_bldgs']
resnet_pretrained = settings_dict['resnet']['pretrained']
resnet_shared = settings_dict['resnet']['shared']
resnet_diff = settings_dict['resnet']['diff']
train_root = settings_dict['data']['iid_xbd_train_root']
test_root = settings_dict['data']['iid_xbd_test_root']
hold_root = settings_dict['data']['iid_xbd_hold_root']
mexico_train = settings_dict['data']['mexico_train_root']
mexico_test = settings_dict['data']['mexico_test_root']
mexico_hold = settings_dict['data']['mexico_hold_root']
palu_train = settings_dict['data']['palu_train_root']
palu_test = settings_dict['data']['palu_test_root']
palu_hold = settings_dict['data']['palu_hold_root']
matthew_train = settings_dict['data']['matthew_train_root']
matthew_test = settings_dict['data']['matthew_test_root']
matthew_hold = settings_dict['data']['matthew_hold_root']
rosa_train = settings_dict['data']['rosa_train_root']
rosa_test = settings_dict['data']['rosa_test_root']
rosa_hold = settings_dict['data']['rosa_hold_root']

del settings_dict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#normalizer = tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = tr.ToTensor()

class xBD(Dataset):
    def __init__(self,
                 root: str,
                 disaster: str,
                 subset: str,
                 transform=None,
                 pre_transform=None) -> None:

        if subset == 'train':
            self.path = train_path
        elif subset == 'test':
            self.path = test_path
        elif subset == 'hold':
            self.path = hold_path
        else:
            raise ValueError("Subset can be either 'train', 'test' or 'hold'.")
        
        self.disaster = disaster

        self.labels = pd.read_csv(list(Path(self.path + disaster).glob('*.csv*'))[0], index_col=0)
        self.labels.drop(columns=['long','lat'], inplace=True)
        zone = lambda row: '_'.join(row.name.split('_', 2)[:2])
        self.labels['zone'] = self.labels.apply(zone, axis=1)
        
        self.num_classes = 4

        super(xBD, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> List:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        processed_files = []
        zones = self.labels['zone'].value_counts()[self.labels['zone'].value_counts()>1].index.tolist()
        for zone in zones:
            if not ((self.labels[self.labels['zone'] == zone]['class'] == 'un-classified').all() or \
                   (self.labels[self.labels['zone'] == zone]['class'] != 'un-classified').sum() == 1):
                processed_files.append(os.path.join(self.processed_dir, f'{zone}.pt'))
        return processed_files

    def process(self):
        resnet50 = load_feature_extractor(resnet_pretrained, resnet_shared, resnet_diff)
        resnet50 = resnet50.to(device)
        
        zones = self.labels['zone'].value_counts()[self.labels['zone'].value_counts()>1].index.tolist()
        for zone in zones:
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
                
                #ordinal encoding of labels
                label = self.labels.loc[os.path.split(post_image_file)[1],'class']
                if label == 'un-classified':
                    continue
                elif label == 'no-damage':
                    y.append([1,0,0,0])
                elif label == 'minor-damage':
                    y.append([1,1,0,0])
                elif label == 'major-damage':
                    y.append([1,1,1,0])
                elif label == 'destroyed':
                    y.append([1,1,1,1])
                else:
                    raise ValueError(f'Label class {label} undefined.')

                coords.append((self.labels.loc[os.path.split(post_image_file)[1],'xcoord'],
                               self.labels.loc[os.path.split(post_image_file)[1],'ycoord']))

                pre_image = Image.open(pre_image_file)
                post_image = Image.open(post_image_file)
                pre_image = pre_image.resize((256, 256))
                post_image = post_image.resize((256, 256))
                pre_image = transform(pre_image)
                post_image = transform(post_image)
                images = torch.cat((pre_image, post_image),0)
                images = images.to(device)
                with torch.no_grad():
                    x.append(resnet50(images.unsqueeze(0)).flatten().cpu())

            x = torch.stack(x)
            y = torch.tensor(y)

            #edge index matrix
            edge_index = build_edge_idx(x.shape[0])

            #edge features
            edge_attr = torch.empty((edge_index.shape[1],2))
            for i in range(edge_index.shape[1]):
                node1 = x[edge_index[0,i]]
                node2 = x[edge_index[1,i]]
                coords1 = coords[edge_index[0,i]]
                coords2 = coords[edge_index[1,i]]
                attr1, attr2 = get_edge_features(node1, node2, coords1, coords2)
                edge_attr[i,0] = attr1
                edge_attr[i,1] = attr2
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, f'{zone}.pt'))
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data


def copy_from_full(disaster: str, subset: str, root: str) -> None:
    if subset == 'train':
        source = train_root + '/processed/'
    elif subset == 'test':
        source = test_root + '/processed/'
    else:
        source = hold_root + '/processed/'
    for source_file in Path(source).glob(f'{disaster}*'):
        copy2(source_file, root + '/processed/')


if __name__ == "__main__":
    disasters = ['mexico-earthquake', 'palu-tsunami', 'hurricane-matthew', 'santa-rosa-wildfire']
    subsets = ['train', 'test', 'hold']
    roots = [[mexico_train, mexico_test, mexico_hold],
             [palu_train, palu_test, palu_hold],
             [matthew_train, matthew_test, matthew_hold],
             [rosa_train, rosa_test, rosa_hold]]
    for disaster, root in zip(disasters, roots):
        for subset, root_dir in zip(subsets, root):
            print(f'Building dataset for {disaster} {subset}...')
            if not os.path.isdir(root_dir):
                os.mkdir(root_dir)
                #os.mkdir(root_dir + '/processed')
            #copy_from_full(disaster, subset, root_dir)
            xBD(root_dir, disaster, subset)
            print(f'****{disaster} {subset} done****')