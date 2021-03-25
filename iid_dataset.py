import json
from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from feature_extractor import load_feature_extractor

with open('exp_setting.json', r) as JSON:
    settings_dict = json.load(JSON)

class IIDxBD(InMemoryDataset):
    def __init__(self, xbd_path: str, root, transform=None, pre_transform=None) -> None:
        super(IIDxBD, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.xbd_path = xbd_path
        self.resnet50 = load_feature_extractor(settings_dict)
        self.subsets = ('/train_bldgs/', '/hold_bldgs/', '/test_bldgs/')

    @property
    def raw_file_names(self) -> None:
        pass

    @property
    def processed_file_names(self) -> list(str):
        return ['data.pt']

    def download(self) -> None:
        pass

    def process(self):
        data_list = []

        for subset in self.subsets:
            disaster_folders = list(Path(self.xbd_path + subset).glob('*'))
            for disaster in disaster_folders:
                list_pre_images = list(disaster.glob('*pre_disaster*'))
                list_post_images = list(disaster.glob('*post_disaster*'))
                annotation = pd.read_csv(list(disaster.glob('*.csv'))[0], index_col=0)
                #TODO
                #read pre img
                #read post img
                #preprocess images according to benson and ecker
                #feed into resnet
                #append to x matrix according to Data class
                #build edge info and weight
                #mask as train/hold according to https://stackoverflow.com/questions/65670777/loading-a-single-graph-into-a-pytorch-geometric-data-object-for-node-classificat


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])