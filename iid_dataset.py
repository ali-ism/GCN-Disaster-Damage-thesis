import os
import json
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tr
from PIL import Image
from torch_geometric.data import Data, InMemoryDataset
from feature_extractor import load_feature_extractor
from generate_disaster_dict import generate_disaster_dict


if not os.path.isdir('disaster_dirs.json'):
    generate_disaster_dict()

with open('disaster_dirs.json', 'r') as JSON:
    disasters_dict = json.load(JSON)

normalizer = tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = tr.ToTensor()


def build_edge_idx(num_nodes: int) -> torch.Tensor:
    """
    Build a complete graph for the edge_index parameter.

    Adapted from: https://github.com/rusty1s/pytorch_geometric/issues/964
    """
    # Initialize edge index matrix
    E = torch.zeros((2, (num_nodes * (num_nodes - 1))//2), dtype=torch.long)
    
    # Populate 1st row
    i = 0
    for node in range(num_nodes):
        for neighbor in range(num_nodes - node - 1):
            E[0, i] = node
            i+=1

    # Populate 2nd row
    neighbors = []
    for node in range(num_nodes):
        neighbors.extend(np.arange(node+1, num_nodes))
    E[1, :] = torch.Tensor(neighbors)
    
    return E


class IIDxBD(InMemoryDataset):
    def __init__(self,
                 xbd_path: str,
                 root,
                 resnet_pretrained=False,
                 resnet_shared=False,
                 resnet_diff=True,
                 transform=None,
                 pre_transform=None) -> None:

        super(IIDxBD, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

        self.resnet50 = load_feature_extractor(resnet_pretrained, resnet_shared, resnet_diff)
        self.disaster_folders = os.listdir(xbd_path + '/train_bldgs/')

    @property
    def raw_file_names(self) -> dict:
        return disasters_dict

    @property
    def processed_file_names(self) -> list(str):
        return ['data.pt']

    def download(self) -> None:
        pass

    def process(self):
        data_list = []

        for disaster in self.disaster_folders:
            x = []
            y = []
            coords = []

            list_pre_images = disasters_dict[disaster + '_pre']
            list_post_images = disasters_dict[disaster + '_post']

            annotation_train = pd.read_csv(disasters_dict[disaster + '_labels'][0], index_col=0)
            annotation_hold = pd.read_csv(disasters_dict[disaster + '_labels'][1], index_col=0)
            annotation_test = pd.read_csv(disasters_dict[disaster + '_labels'][2], index_col=0)
            annotation = pd.concat((annotation_train, annotation_hold, annotation_test))

            for pre_image_file, post_image_file in zip(list_pre_images, list_post_images):
                if annotation[os.path.split(post_image_file)[1]] == 'un-classified':
                    continue

                y.append(annotation.loc[os.path.split(post_image_file)[1],'class'])
                coords.append(annotation.loc[os.path.split(post_image_file)[1],'coords'])

                pre_image = Image.open(pre_image_file)
                post_image = Image.open(post_image_file)
                pre_image = pre_image.resize((256, 256))
                post_image = post_image.resize((256, 256))
                pre_image = transform(pre_image)
                post_image = transform(post_image)
                images = torch.cat((pre_image, post_image),1)

                with torch.no_grad():
                    node_features = self.resnet50(images)
                
                x.append(node_features)
            
            x = torch.stack(x)

            label_en = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3}
            y = torch.from_numpy(pd.Series(y).replace(label_en).values)

            #mask as train/hold according to https://stackoverflow.com/questions/65670777/loading-a-single-graph-into-a-pytorch-geometric-data-object-for-node-classificat
            train_mask = torch.zeros(x.shape[0])
            hold_mask = torch.zeros(x.shape[0])
            test_mask = torch.zeros(x.shape[0])
            train_mask[:annotation_train.shape[0]] = 1
            hold_mask[annotation_train.shape[0]:annotation_hold.shape[0]] = 1
            test_mask[annotation_hold.shape[0]:annotation_test.shape[0]] = 1
            train_mask = train_mask.type(torch.bool)
            hold_mask = hold_mask.type(torch.bool)
            test_mask = test_mask.type(torch.bool)

            #build edge info and weight
            edge_index = build_edge_idx(x.shape[0])
            #create Data object for the current disaster


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])