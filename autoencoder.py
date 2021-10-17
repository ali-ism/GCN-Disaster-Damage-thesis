import json
import math
import os.path as osp
from pathlib import Path
from random import randint

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, img_to_array
from utils import score


def step_decay(epoch):
	if epoch < 100:
		#print "epoch %d learning rate %f" % (epoch, 0.005)
		return 0.0005
	else:
		#print "epoch %d learning rate %f" % (epoch, 0.0001)
		return 0.0001


def deepSSAEMulti(n_dim, n_hidden1, n_hidden2, n_classes):
	input_layer = Input(shape=(n_dim,))
	encoded = Dense(n_hidden1, activation='relu')(input_layer)
	encoded = Dense(n_hidden2, activation='relu', name="low_dim_features")(encoded)
	decoded = Dense(n_hidden1, activation='relu')(encoded)
	decoded = Dense(n_dim, activation='sigmoid')(decoded)
	
	classifier = Dense(n_classes, activation='softmax')(encoded)
	
	adamDino = optimizers.RMSprop(lr=0.0005)
	adamDino1 = optimizers.RMSprop(lr=0.0005)
	autoencoder = Model(inputs=[input_layer], outputs=[decoded])
	autoencoder.compile(optimizer=adamDino, loss=['mse'])
	
	ssautoencoder = Model(inputs=[input_layer], outputs=[decoded, classifier])
	ssautoencoder.compile(optimizer=adamDino1, loss=['mse','categorical_crossentropy'], loss_weights=[1., 1.])
	return [autoencoder, ssautoencoder]
	

def feature_extraction(model, data, layer_name):
	feat_extr = Model(inputs= model.input, outputs= model.get_layer(layer_name).output)
	return feat_extr.predict(data)


def learn_SingleReprSS(X_tot, idx_train, y_train):
	n_classes = len(np.unique(y_train))
	idx_train = idx_train.astype("int")
	X_train = X_tot[idx_train]
	encoded_Y_train = to_categorical(y_train, n_classes)
	_, n_col = X_tot.shape
		
	n_feat = math.ceil( n_col -1)
	n_feat_2 = math.ceil( n_col * 0.5)
	n_feat_4 = math.ceil( n_col * 0.25)
	
	n_hidden1 = randint(n_feat_2, n_feat)
	n_hidden2 = randint(n_feat_4, n_feat_2-1)
		
	ae, ssae = deepSSAEMulti(n_col, n_hidden1, n_hidden2, n_classes)
	for _ in range(200):	
		ae.fit(X_tot, X_tot, epochs=1, batch_size=16, shuffle=True, verbose=1)
		ssae.fit(X_train, [X_train, encoded_Y_train], epochs=1, batch_size=8, shuffle=True, verbose=1)			
	new_train_feat = feature_extraction(ae, X_tot, "low_dim_features")
	return new_train_feat


def learn_representationSS(X_tot, idx_train, Y_train, ens_size):
	intermediate_reprs = np.array([])
	for l in range(ens_size): 
		print(f'learn representation {l}')
		embeddings = learn_SingleReprSS(X_tot, idx_train, Y_train)
		if intermediate_reprs.size == 0:
			intermediate_reprs = embeddings
		else:
			intermediate_reprs = np.column_stack((intermediate_reprs, embeddings))
	return intermediate_reprs


def clusters_to_classes(labels, clusters):
	class_map = {}
	for cluster in np.unique(clusters):
		class_map[cluster] = np.bincount(labels[clusters == cluster]).argmax()
	mapper = np.vectorize(lambda x: class_map[x])
	return mapper(clusters)


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

	idx, _ = train_test_split(
		np.arange(labels.shape[0]), train_size=settings_dict['data']['reduced_size'],
		stratify=labels['class_num'].values, random_state=42)
	labels = labels.iloc[idx,:]

	x = []
	y = []

	for post_image_file in labels.index.values.tolist():  
		y.append(labels.loc[post_image_file,'class_num'])
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
	train_idx, test_idx = train_test_split(
		np.arange(y.shape[0]), train_size=settings_dict['data']['labeled_size'],
		stratify=y, random_state=42)
	y_train = y[train_idx]

	new_feat_ssae = learn_representationSS(x, train_idx, y_train, 30)
	clusters = KMeans(n_clusters=len(np.unique(y)), random_state=42).fit_predict(new_feat_ssae)
	y_pred = clusters_to_classes(y, clusters)

	accuracy, f1_macro, f1_weighted, auc = score(y, y_pred)
	print(f'\nAccuracy: {accuracy:.4f}')
	print(f'Macro F1: {f1_macro:.4f}')
	print(f'Weighted F1: {f1_weighted:.4f}')
	print(f'Auc: {auc:.4f}')