'''
Reads the images and labels from the xBD dataset and extracts the buildings.
'''
import json
import os
from collections import defaultdict
from math import ceil
from pathlib import Path

import cv2
import pandas as pd
from shapely import wkt
from tqdm import tqdm

subsets = ('datasets/xbd/train/', 'datasets/xbd/tier3/', 'datasets/xbd/hold/', 'datasets/xbd/test/')

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
            for index,(bldg_annotationxy,bldg_annotationlnglat) in enumerate(zip(annotation['features']['xy'],annotation['features']['lng_lat'])):
                bldg_image_name_post = label.name.split('.')[0] + f'_{index}.png'
                bldg = wkt.loads(bldg_annotationxy['wkt'])
                if not os.path.isfile(disaster_path + bldg_image_name_post):
                    minx, miny, maxx, maxy = bldg.bounds
                    minx = ceil(minx)
                    miny = ceil(miny)
                    maxx = ceil(maxx)
                    maxy = ceil(maxy)
                    pre_im_bldg = pre_image[miny:maxy,minx:maxx]
                    post_im_bldg = post_image[miny:maxy,minx:maxx]
                    cv2.imwrite(disaster_path + bldg_image_name_post, post_im_bldg)
                    cv2.imwrite(disaster_path + bldg_image_name_post.replace('_post_','_pre_'), pre_im_bldg)
                coords = list(bldg.centroid.coords)[0]
                bldg_lnglat = wkt.loads(bldg_annotationlnglat['wkt'])
                lng_lat = list(bldg_lnglat.centroid.coords)[0]
                class_dict[bldg_image_name_post] = [coords[0], coords[1], lng_lat[0], lng_lat[1], bldg_annotationxy['properties']['subtype']]
        df = pd.DataFrame.from_dict(class_dict, orient='index', columns=['xcoord', 'ycoord', 'long', 'lat', 'class'])
        df.to_csv(disaster_path + disaster + '_' + subset[13:-1] + '_labels.csv')