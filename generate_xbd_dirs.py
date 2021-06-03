import os
import json
from pathlib import Path
from collections import defaultdict
from typing import List, DefaultDict
import pandas as pd

xbd_path = 'datasets/xbd'


def write_to_dict(dict_: DefaultDict, subset: str, disaster_folders: List) -> None:
    subset_marker = '_'+subset[subset.find('/')+len('/'):subset.rfind('_')]
    for disaster in disaster_folders:
        labels = pd.read_csv(list(Path(xbd_path + subset + disaster).glob('*.csv*'))[0])
        labels.columns = ['name', 'xcoords', 'ycoords', 'long', 'lat', 'class']
        labels.drop(columns=['class','xcoords','ycoords','long','lat'], inplace=True)
        zone = lambda row: '_'.join(row['name'].split('_', 2)[:2])
        labels['zone'] = labels.apply(zone, axis=1)
        zones = labels['zone'].value_counts()[labels['zone'].value_counts()>1].index.tolist()
        for zone in zones:
            dict_[disaster+subset_marker+'_pre_'+zone] = list(map(str, Path(xbd_path + subset + disaster).glob(f'{zone}_pre_disaster*')))
            dict_[disaster+subset_marker+'_post_'+zone] = list(map(str, Path(xbd_path + subset + disaster).glob(f'{zone}_post_disaster*')))


def generate_disaster_dict() -> None:
    disaster_dict = defaultdict(defaultdict(defaultdict(defaultdict(list))))

    subsets = ('/train_bldgs/', '/hold_bldgs/', '/test_bldgs/')
    disaster_folders = os.listdir(xbd_path + subsets[0])

    disaster_folders_tier3 = os.listdir(xbd_path + '/tier3_bldgs/')

    for subset in subsets:
        write_to_dict(disaster_dict, subset, disaster_folders)
    write_to_dict(disaster_dict, '/tier3_bldgs/', disaster_folders_tier3)

    with open('xbd_dirs.json', "w") as outfile: 
        json.dump(disaster_dict, outfile)

    print('Successfully created "xbd_dirs.json".')


if __name__== "__main__":
    generate_disaster_dict()