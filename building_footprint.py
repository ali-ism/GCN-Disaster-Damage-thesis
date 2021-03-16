from pathlib import Path
import json
import os
from collections import defaultdict
from math import ceil
from shapely import wkt
import pandas as pd
import cv2
from tqdm import tqdm

subsets = ('./train/', './tier3/', './hold/', './test/')

for subset in subsets:
    subset_path = subset[:-1] + '_bldgs/'
    if not os.path.isdir(subset_path):
        os.mkdir(subset_path)

    post_labels = list(Path(subset + 'labels/').glob('*_post_disaster.json'))
    disaster_dict_post = defaultdict(list)
    for label in post_labels:
        disaster_type = label.name.split('_')[0]
        disaster_dict_post[disaster_type].append(label)
    
    for disaster in disaster_dict_post:
        disaster_path = subset_path + disaster + '/'
        if not os.path.isdir(disaster_path):
            os.mkdir(disaster_path)
            print(f'Started disaster {disaster} in subset {subset}.')
        elif os.path.isfile(disaster_path + disaster + '_' + subset[2:-1] + '_labels.csv'):
            print(f'Disaster {disaster} already completed, skipping to next disaster.')
            continue
        else:
            print(f'Resuming disaster {disaster} in subset {subset}.')
        disaster_labels = disaster_dict_post[disaster]
        class_dict = defaultdict(list)

        for label in tqdm(disaster_labels):
            annotation = json.load(open(label))
            image_name = label.name.split('.')[0] + '.png'
            post_image = cv2.imread(subset + 'images/' + image_name)
            pre_image = cv2.imread(subset + 'images/' + image_name.replace('_post_', '_pre_'))
            for index, bldg_annotation in enumerate(annotation['features']['xy']):
                bldg_image_name_post = label.name.split('.')[0] + f'_{index}.png'
                if os.path.isfile(disaster_path + bldg_image_name_post):
                    continue
                bldg = wkt.loads(bldg_annotation['wkt'])
                minx, miny, maxx, maxy = bldg.bounds
                minx = ceil(minx)
                miny = ceil(miny)
                maxx = ceil(maxx)
                maxy = ceil(maxy)
                pre_im_bldg = pre_image[miny:maxy,minx:maxx]
                post_im_bldg = post_image[miny:maxy,minx:maxx]
                cv2.imwrite(disaster_path + bldg_image_name_post, post_im_bldg)
                cv2.imwrite(disaster_path + bldg_image_name_post.replace('_post_','_pre_'), pre_im_bldg)
                class_dict[bldg_image_name_post] = [list(bldg.centroid.coords)[0], bldg_annotation['properties']['subtype']]
        df = pd.DataFrame.from_dict(class_dict, orient='index', columns=['coords', 'class'])
        df.to_csv(disaster_path + disaster + '_' + subset[2:-1] + '_labels.csv')