import os
import json
from pathlib import Path
from collections import defaultdict
from sys import argv

def generate_disaster_dict(xbd_path) -> None:
    disaster_dict = defaultdict(list)
    subsets = ('/train_bldgs/', '/hold_bldgs/', '/test_bldgs/', '/tier3_bldgs/')

    disaster_folders = os.listdir(xbd_path + subsets[0]) + os.listdir(xbd_path + subsets[-1])

    for subset in subsets:
        for disaster in disaster_folders:
            disaster_dict[disaster + '_pre'].extend(map(str, Path(xbd_path + subset + disaster).glob('*pre_disaster*')))
            disaster_dict[disaster + '_post'].extend(map(str, Path(xbd_path + subset + disaster).glob('*post_disaster*')))
            disaster_dict[disaster + '_labels'].extend(map(str, Path(xbd_path + subset + disaster).glob('*.csv')))

    with open('disaster_dirs.json', "w") as outfile: 
        json.dump(disaster_dict, outfile)

    print('Successfully created "disaster_dirs.json".')

if __name__== "__main__":
    xbd_path = argv[1]
    generate_disaster_dict(xbd_path)