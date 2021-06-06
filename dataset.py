import os
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
from typing import List
import torch
import torchvision.transforms as tr
from PIL import Image
from torch_geometric.data import Data, Dataset
from feature_extractor import load_feature_extractor
from utils import build_edge_idx, get_edge_weight

with open('exp_settings.json', 'r') as JSON:
    settings_dict = json.load(JSON)

device = torch.device('cpu')#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(settings_dict['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#normalizer = tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = tr.ToTensor()

class IIDxBD(Dataset):
    def __init__(self,
                 root: str,
                 subset: str,
                 resnet_pretrained=False,
                 resnet_shared=False,
                 resnet_diff=True,
                 transform=None,
                 pre_transform=None) -> None:
        
        self.resnet_pretrained = resnet_pretrained
        self.resnet_shared = resnet_shared
        self.resnet_diff = resnet_diff

        if subset == 'train':
            self.path = 'datasets/xbd/train_bldgs/'
        elif subset == 'test':
            self.path = 'datasets/xbd/test_bldgs/'
        elif subset == 'hold':
            self.path = 'datasets/xbd/hold_bldgs/'
        else:
            raise ValueError("Subset can be either 'train', 'test' or 'hold'.")
        
        self.disaster_folders = os.listdir(self.path)

        self.list_labels = []
        for disaster in self.disaster_folders:
            labels = pd.read_csv(list(Path(self.path + disaster).glob('*.csv*'))[0], index_col=0)
            labels.drop(columns=['long','lat'], inplace=True)
            zone = lambda row: '_'.join(row.name.split('_', 2)[:2])
            labels['zone'] = labels.apply(zone, axis=1)
            self.list_labels.append(labels)

        super(IIDxBD, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> List:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        processed_files = []
        for labels in self.list_labels:
            zones = labels['zone'].value_counts()[labels['zone'].value_counts()>1].index.tolist()
            for zone in zones:
                processed_files.append(os.path.join(self.processed_dir, f'{zone}.pt'))
        return processed_files

    def process(self):
        resnet50 = load_feature_extractor(self.resnet_pretrained, self.resnet_shared, self.resnet_diff)
        resnet50 = resnet50.to(device)

        for disaster, labels in zip(self.disaster_folders, self.list_labels):
            zones = labels['zone'].value_counts()[labels['zone'].value_counts()>1].index.tolist()
            for zone in zones:
                if os.path.isfile(os.path.join(self.processed_dir, f'{zone}.pt')):
                    print('File exists already!')
                    continue
                list_pre_images = list(map(str, Path(self.path + disaster).glob(f'{zone}_pre_disaster*')))
                list_post_images = list(map(str, Path(self.path + disaster).glob(f'{zone}_post_disaster*')))
                x = []
                y = []
                coords = []

                pbar = tqdm(total=len(list_post_images))
                pbar.set_description(f'Building {zone} node features')

                for pre_image_file, post_image_file in zip(list_pre_images, list_post_images):
                    
                    #ordinal encoding of labels
                    label = labels.loc[os.path.split(post_image_file)[1],'class']
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

                    coords.append((labels.loc[os.path.split(post_image_file)[1],'xcoord'],
                                   labels.loc[os.path.split(post_image_file)[1],'ycoord']))

                    pre_image = Image.open(pre_image_file)
                    post_image = Image.open(post_image_file)
                    pre_image = pre_image.resize((256, 256))
                    post_image = post_image.resize((256, 256))
                    pre_image = transform(pre_image)
                    post_image = transform(post_image)
                    images = torch.cat((pre_image, post_image),0)
                    x.append(images)
                    pbar.update()
                
                pbar.close()
                x = torch.stack(x).to(device)
                with torch.no_grad():
                    x = resnet50(x).cpu()
                y = torch.tensor(y)

                #edge index matrix
                edge_index = build_edge_idx(x.shape[0])

                #edge features
                pbar = tqdm(total=edge_index.shape[1])
                pbar.set_description(f'Building {zone} edge features')
                
                edge_attr = torch.empty((edge_index.shape[1],2))
                for i in range(edge_index.shape[1]):
                    node1 = x[edge_index[0,i]]
                    node2 = x[edge_index[1,i]]
                    coords1 = coords[edge_index[0,i]]
                    coords2 = coords[edge_index[1,i]]
                    attr = get_edge_weight(node1, node2, coords1, coords2)
                    edge_attr[i,0] = attr[0]
                    edge_attr[i,1] = attr[1]
                    pbar.update()
                
                pbar.close()
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


if __name__ == "__main__":
    roots = [settings_dict['data']['iid_xbd_train_root'],
             settings_dict['data']['iid_xbd_test_root'],
             settings_dict['data']['iid_xbd_hold_root']]
    subsets = ['train', 'test', 'hold']
    for subset, root in zip(subsets, roots):
        print(f'Building dataset for subset {subset}.')
        if not os.path.isdir(root):
            os.mkdir(root)
        IIDxBD(root, subset,
               resnet_pretrained=settings_dict['resnet']['pretrained'],
               resnet_diff=settings_dict['resnet']['diff'],
               resnet_shared=settings_dict['resnet']['shared'])