import os
import json
from pathlib import Path
from collections import defaultdict
from math import sqrt
import pandas as pd
from sklearn.metrics import pairwise_distances
from utils import euclidean_similarity


def define_neigborhood(labels: pd.DataFrame):
    pairwise_euc = pairwise_distances(labels.values, metric=euclidean_similarity, njobs=-1)
    results = {}
    for idx, row in labels.iterrows():
        similar_indices = pairwise_euc[idx].argsort()
        similar_indices = similar_indices[:-len(similar_indices)-1:-1]
        similar_items = [(pairwise_euc[idx][i], labels['name'][i]) for i in similar_indices]
        results[row['id']] = similar_items[1:]

    norm = lambda row: sqrt(row[1]**2 + row[2]**2)
    labels['norm'] = labels.apply(norm, axis=1)
    labels.sort_values(by='norm', inplace=True)
    i = 0
    neighbors = defaultdict(list)
    while i < labels.shape[0]:
        neighbors[labels['name'][0]] = results[0][:7600]
    return

def generate_disaster_dict() -> None:
    xbd_path = 'datasets/xbd'
    disaster_dict = defaultdict(list)
    subsets = ('/train_bldgs/', '/hold_bldgs/', '/test_bldgs/')

    disaster_folders = os.listdir(xbd_path + subsets[0])
    disaster_folders_tier3 = os.listdir(xbd_path + '/tier3_bldgs/')

    for subset in subsets:
        subset_marker = subset[subset.find('/')+len('/'):subset.rfind('_')]
        for disaster in disaster_folders:
            labels = pd.read_csv(list(Path(xbd_path + subset + disaster).glob('*.csv*'))[0])
            labels.columns = ['name', 'xcoords', 'ycoords', 'class']
            labels.drop(columns='class', inplace=True)
            if labels.shape[0] > 7600:
                define_neigborhood(labels)
            disaster_dict[disaster][subset_marker]['pre'].extend(map(str, Path(xbd_path + subset + disaster).glob('*pre_disaster*')))
            disaster_dict[disaster][subset_marker]['post'].extend(map(str, Path(xbd_path + subset + disaster).glob('*post_disaster*')))

    with open('disaster_dirs.json', "w") as outfile: 
        json.dump(disaster_dict, outfile)

    print('Successfully created "disaster_dirs.json".')

if __name__== "__main__":
    generate_disaster_dict()