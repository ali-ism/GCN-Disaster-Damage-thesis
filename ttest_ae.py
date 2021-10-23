import json
import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array

from autoencoder import learn_representationSS, cluster_embeddings

tensorflow.random.set_seed(42)


if __name__ == "__main__":

    with open('exp_settings.json', 'r') as JSON:
        settings_dict = json.load(JSON)

    name = settings_dict['model']['name'] + '_ssae'
    model_path = 'weights/' + name
    disaster = 'pinery-bushfire'
    path = '/home/ami31/scratch/datasets/xbd/tier3_bldgs/'

    labels = pd.read_csv(list(Path(path + disaster).glob('*.csv*'))[0], index_col=0)
    labels.drop(columns=['xcoord','ycoord', 'long', 'lat'], inplace=True)
    labels.drop(index=labels.loc[labels['class']=='un-classified'].index, inplace=True)

    zone_func = lambda row: '_'.join(row.name.split('_', 2)[:2])
    labels['zone'] = labels.apply(zone_func, axis=1)
    for zone in labels['zone'].unique():
        if (labels[labels['zone'] == zone]['class'] == 'no-damage').all():
            labels.drop(index=labels.loc[labels['zone']==zone].index, inplace=True)

    if settings_dict['merge_classes']:
        label_dict = {'no-damage':0,'minor-damage':1,'major-damage':2,'destroyed':2}
    else:
        label_dict = {'no-damage':0,'minor-damage':1,'major-damage':2,'destroyed':3}

    labels['class'] = labels['class'].apply(lambda x: label_dict[x])

    #sample dataset so it fits in memory
    if labels.shape[0] > settings_dict['data_ss']['reduced_size']:
        idx, _ = train_test_split(
            np.arange(labels.shape[0]), train_size=settings_dict['data_ss']['reduced_size'],
            stratify=labels['class'].values, random_state=42
        )
        labels = labels.iloc[idx,:]

    x = []
    y = []

    for post_image_file in labels.index.values.tolist():  
        y.append(labels.loc[post_image_file,'class'])
        pre_image = Image.open(osp.join(path, disaster, post_image_file.replace('post', 'pre')))
        post_image = Image.open(osp.join(path, disaster, post_image_file))
        pre_image = pre_image.resize((128, 128))
        post_image = post_image.resize((128, 128))
        pre_image = img_to_array(pre_image)
        post_image = img_to_array(post_image)
        images = np.concatenate((pre_image, post_image))
        x.append(images.flatten())

    x = np.stack(x)
    x = MinMaxScaler().fit_transform(x)

    y = np.array(y)

    accuracy = np.empty(100)
    precision = np.empty(100)
    recall = np.empty(100)
    specificity = np.empty(100)
    f1 = np.empty(100)

    for seed in range(100):
        #split into train and test
        train_idx, test_idx = train_test_split(
            np.arange(y.shape[0]), test_size=0.4,
            stratify=y, random_state=seed
        )
        #split test into test and hold
        test_idx, hold_idx = train_test_split(
            np.arange(test_idx.shape[0]), test_size=0.5,
            stratify=y[test_idx], random_state=seed
        )
        #select labeled samples
        labeled_idx, _ = train_test_split(
            np.arange(train_idx.shape[0]), train_size=settings_dict['data_ss']['labeled_size'],
            stratify=y[train_idx], random_state=seed
        )
        y_labeled = y[labeled_idx]

        embeddings = learn_representationSS(x, train_idx, labeled_idx, y_labeled, 30, verbose=False)
        accuracy[seed], precision[seed], recall[seed], specificity[seed], f1[seed] = cluster_embeddings(embeddings)
    
    np.save('results/ae_acc_ttest.npy', accuracy)
    np.save('results/ae_prec_ttest.npy', precision)
    np.save('results/ae_rec_ttest.npy', recall)
    np.save('results/ae_spec_ttest.npy', specificity)
    np.save('results/ae_f1_ttest.npy', f1)