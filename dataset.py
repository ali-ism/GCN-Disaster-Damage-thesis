import os
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(settings_dict['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open('disaster_dirs.json', 'r') as JSON:
    disasters_dict = json.load(JSON)

#normalizer = tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class IIDxBDTrain(Dataset):
    def __init__(self,
                 root,
                 resnet_pretrained=False,
                 resnet_shared=False,
                 resnet_diff=True,
                 transform=None,
                 pre_transform=None) -> None:
        
        self.resnet_pretrained = resnet_pretrained
        self.resnet_shared = resnet_shared
        self.resnet_diff = resnet_diff
        self.xbd_path = 'datasets/xbd'

        super(IIDxBDTrain, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> List:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return

    def process(self):
        resnet50 = load_feature_extractor(self.resnet_pretrained, self.resnet_shared, self.resnet_diff)
        resnet50 = resnet50.to(device)
        disaster_folders = os.listdir(self.xbd_path + '/train_bldgs/')

        self.annot_list = []

        for disaster in disaster_folders:
            if os.path.isfile(os.path.join(self.processed_dir, 'iid_data_{}.pt'.format(disaster))):
                continue
            x = []
            y = []
            coords = []
            split = []

            list_pre_images = disasters_dict[disaster + '_pre']
            list_post_images = disasters_dict[disaster + '_post']

            annotation_train = pd.read_csv(disasters_dict[disaster + '_labels'][0], index_col=0)
            annotation_train['split'] = ['train'] * annotation_train.shape[0]
            annotation_hold = pd.read_csv(disasters_dict[disaster + '_labels'][1], index_col=0)
            annotation_hold['split'] = ['hold'] * annotation_hold.shape[0]
            annotation_test = pd.read_csv(disasters_dict[disaster + '_labels'][2], index_col=0)
            annotation_test['split'] = ['test'] * annotation_test.shape[0]
            annotation = pd.concat((annotation_train, annotation_hold, annotation_test))
            self.annot_list.append(annotation.drop(annotation[annotation['class']=='un-classified'].index))

            pbar = tqdm(total=len(list_post_images))
            pbar.set_description(f'Building {disaster}, node features')

            for pre_image_file, post_image_file in zip(list_pre_images, list_post_images):
                if annotation.loc[os.path.split(post_image_file)[1],'class'] == 'un-classified':
                    continue

                #ordinal encoding of labels
                label = annotation.loc[os.path.split(post_image_file)[1],'class']
                if label == 'no-damage':
                    y.append([1,0,0,0])
                elif label == 'minor-damage':
                    y.append([1,1,0,0])
                elif label == 'major-damage':
                    y.append([1,1,1,0])
                elif label == 'destroyed':
                    y.append([1,1,1,1])
                else:
                    raise ValueError(f'Label class {label} undefined.')

                coords.append((annotation.loc[os.path.split(post_image_file)[1],'coords1'],
                               annotation.loc[os.path.split(post_image_file)[1],'coords2']))
                split.append(annotation.loc[os.path.split(post_image_file)[1],'split'])

                pre_image = Image.open(pre_image_file)
                post_image = Image.open(post_image_file)
                pre_image = pre_image.resize((256, 256))
                post_image = post_image.resize((256, 256))
                pre_image = tr.ToTensor(pre_image)
                post_image = tr.ToTensor(post_image)
                images = torch.cat((pre_image, post_image),0)
                images = images.to(device)
                with torch.no_grad():
                    node_features = resnet50(images.unsqueeze(0)).flatten()
                x.append(node_features.cpu())
                pbar.update()
            
            pbar.close()
            x = torch.stack(x)
            y = torch.tensor(y)

            #mask as train/val/test according to
            #https://stackoverflow.com/questions/65670777/loading-a-single-graph-into-a-pytorch-geometric-data-object-for-node-classificat
            train_mask = torch.zeros(x.shape[0])
            hold_mask = torch.zeros(x.shape[0])
            test_mask = torch.zeros(x.shape[0])

            split = pd.Series(split)
            train_mask[split[split=='train'].index] = 1
            hold_mask[split[split=='hold'].index] = 1
            test_mask[split[split=='test'].index] = 1

            train_mask = train_mask.type(torch.bool)
            hold_mask = hold_mask.type(torch.bool)
            test_mask = test_mask.type(torch.bool)

            #edge index matrix
            edge_index = build_edge_idx(x.shape[0])

            #edge features
            pbar = tqdm(total=edge_index.shape[1])
            pbar.set_description(f'Building {disaster}, edge features')
            
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
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        train_mask=train_mask, val_mask=test_mask, test_mask=hold_mask)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            torch.save(data, os.path.join(self.processed_dir, 'iid_data_{}.pt'.format(disaster)))
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data


if __name__ == "__main__":
    root = settings_dict['data']['root']
    if not os.path.isdir(root):
        os.mkdir(root)
    IIDxBDTrain(root, resnet_pretrained=settings_dict['resnet']['pretrained'],
                 resnet_diff=settings_dict['resnet']['diff'],
                 resnet_shared=settings_dict['resnet']['shared'])