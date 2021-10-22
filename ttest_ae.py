import json
import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import img_to_array

from utils import score_cm
from autoencoder import learn_representationSS, _make_cost_m


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

    if settings_dict['data']['merge_classes']:
        label_dict = {'no-damage':0,'minor-damage':1,'major-damage':2,'destroyed':2}
    else:
        label_dict = {'no-damage':0,'minor-damage':1,'major-damage':2,'destroyed':3}

    labels['class'] = labels['class'].apply(lambda x: label_dict[x])

    #sample dataset so it fits in memory
    if labels.shape[0] > settings_dict['data']['reduced_size']:
        idx, _ = train_test_split(
            np.arange(labels.shape[0]), train_size=settings_dict['data']['reduced_size'],
            stratify=labels['class'].values, random_state=42)
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
        #select labeled samples
        train_idx, _ = train_test_split(
            np.arange(y.shape[0]), train_size=settings_dict['data']['labeled_size'],
            stratify=y, random_state=seed)
        y_train = y[train_idx]

        new_feat_ssae = learn_representationSS(x, train_idx, y_train, 30)
        clusters = KMeans(n_clusters=len(np.unique(y)), random_state=42).fit_predict(new_feat_ssae)

        cm = confusion_matrix(y, clusters)
        indexes = linear_sum_assignment(_make_cost_m(cm))
        cm2 = cm[:, indexes[1]]

        accuracy[seed], precision[seed], recall[seed], specificity[seed], f1[seed] = score_cm(cm2)
    
    np.save('results/ae_acc_ttest.npy', accuracy)
    np.save('results/ae_prec_ttest.npy', precision)
    np.save('results/ae_rec_ttest.npy', recall)
    np.save('results/ae_spec_ttest.npy', specificity)
    np.save('results/ae_f1_ttest.npy', f1)