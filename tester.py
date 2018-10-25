#author:vamshiteja
import matplotlib
matplotlib.use('Agg')

import sys
import dateutil.parser
import pickle
import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from sys import argv
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from utils.loader import load_batched_data
from model.model_w import InfoGAN
from tqdm import tqdm

import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score

import pandas as pd


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

#define hyperparameters
batch_size = 1
learning_rate = 5e-5
num_epochs = 100
num_iters = 10000
disc_iters = 2
is_Train = 1
grad_clip = 1
z_dim = 50
cat_dim = 20
cont_dim = 30
imsize = 256
smooth_labels = 1
std = 0.0
restore = 1
mode = "dc-gan"
dataset = "mammo"

# 'kaggle' or 'mammo'
dataset = "mammo"

if(dataset=="mammo"):
	from model.model_w_mammo import InfoGAN
	datapath = "./datasets/mammo_cancer/dataset_mammo_train_"+str(imsize)+".pickle"
	checkpoint_dir = "./checkpoints/gan/mammo/"
	num_channels = 1
elif(dataset=="kaggle"):
	from model.model_w_kaggle import InfoGAN
	datapath = "./datasets/kaggle_cancer/test/stage1_"+str(imsize)+".pickle"
	checkpoint_dir = "./checkpoints/gan/kaggle/"
	num_channels = 3

if(mode=="dc-gan"):
	disc_iters = 1
	learning_rate = 2e-4
elif(mode=="dc-wgan"):
	disc_iters = 5
	learning_rate = 5e-5

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Tester():
	def _default_configs(self):
	  return {'batch_size': batch_size,
			  'is_Train' : is_Train,
			  'learning_rate': learning_rate,
			  'num_epochs': num_epochs,
			  'grad_clip': grad_clip,
			  'z_dim': z_dim,
			  'cat_dim': cat_dim,
			  'cont_dim': cont_dim,
			  'imsize': imsize,
			  'num_channels': num_channels,
			  'mode': mode,
			  'smooth_labels': smooth_labels,
			  'std': std,
			  'restore': restore,
			}

	def load_mammo_data(self,datapath):
		'''
			Function to load data
		'''
		with open(datapath,"rb") as f:
			data = pickle.load(f)

		normal = np.array(data['normal'])
		labels_n = np.zeros(shape=(normal.shape[0],))
		cancer = np.array(data['cancer'])
		labels_c = np.ones(shape=(cancer.shape[0],))
		data = np.concatenate((normal,cancer),axis=0)
		labels = np.concatenate((labels_n, labels_c), axis=0)
		return data, labels

	def load_kaggle_data(self,datapath):
		'''
			Function to load data
		'''
		with open(datapath,"rb") as f:
			data = pickle.load(f)

		imgs = data['data']
		labels = data['labels']
		return imgs,labels


	def run(self):
		'''
			Function to get features of samples
		'''
		args_dict = self._default_configs()
		args = dotdict(args_dict)

		if(dataset=="mammo"):
			data, labels = self.load_mammo_data(datapath)
		elif(dataset=="kaggle"):
			data, labels = self.load_kaggle_data(datapath)

		model = InfoGAN(args)
		print("[INFO]: Building graph")
		model.build_graph(args)
		num_samples = int(len(data)/args.batch_size)
		print("Numer of test samples are %d"%num_samples)

		with tf.Session(graph=model.graph) as sess:
			writer = tf.summary.FileWriter("logging",graph=model.graph)
			if(1==1):
				print("[INFO]: Restoring Weights")
			#model.saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
			sess.run(model.initial_op)
			features_cat =[]
			features_cont = []
			features_gap = []
			for sample in tqdm(range(num_samples)):
				if(dataset=="mammo"):
					X_mb = np.expand_dims(data[sample],axis=0)
					X_mb  = np.expand_dims(X_mb, axis=-1)
				else:
					X_mb  = np.expand_dims(X_mb, axis=2)
				Z_noise = model.sample_z(args.batch_size, args.z_dim)
				cat_noise = model.sample_cat(args.batch_size, args.cat_dim)
				cont_noise = model.sample_cont(args.batch_size, args.cont_dim)

				Q_cat_given_x, Q_cont_given_x, gap_features = sess.run([model.Q_cat_given_x, model.Q_cont_given_x, model.gap_shared_out], 
					feed_dict={model.X: X_mb, model.z: Z_noise, model.cat: cat_noise, model.cont: cont_noise})

				features_cont.append(Q_cont_given_x[0])
				features_cat.append(Q_cat_given_x[0])
				features_gap.append(gap_features[0])
		
		return features_cat, features_cont, features_gap




if __name__ == '__main__':
	tester = Tester()
	features_cat, features_cont, features_gap = tester.run()
	features_cont = np.vstack(features_cont)
	features_gap = np.vstack(features_gap)

	if(dataset=="kaggle"):
		TEST_LABELS = "./datasets/kaggle_cancer/test/solution_stg1_release.csv"

		labels = pd.read_csv(TEST_LABELS,names=['image_name','Type_1', 'Type_2', 'Type_3'],header=0)
		labels = labels.drop(columns=['image_name'])
		labels = labels.as_matrix()
		labels = np.argmax(labels, axis=1)
	elif(dataset=="mammo"):
		_, labels = tester.load_mammo_data(datapath)

	X_tsne = TSNE(n_components=2).fit_transform(features_gap)
	plt.scatter(X_tsne[:,0], X_tsne[:,1], marker="s", c= labels, s=20, cmap="rainbow")
	plt.savefig("./cluster_res/actual_"+dataset+".png")
	plt.close()

	kmeans = KMeans(n_clusters=3, random_state=0).fit(features_gap)
	est_labels = kmeans.labels_

	rand_score = adjusted_rand_score(labels, est_labels)
	nmi_score = normalized_mutual_info_score(labels, est_labels)
	s_score = silhouette_score(features_gap, est_labels)
	print("\nFor Kmeans")
	print("rand score is: {}".format(rand_score))
	print("nmi score is: {}".format(nmi_score))
	print("silhouette_score is: {}".format(s_score))

	dbscan = DBSCAN()
	est_labels = dbscan.fit_predict(features_gap)

	rand_score = adjusted_rand_score(labels, est_labels)
	nmi_score = normalized_mutual_info_score(labels, est_labels)
	#s_score = silhouette_score(features_gap, est_labels)
	print("\nFor DBSCAN")
	print("rand score is: {}".format(rand_score))
	print("nmi score is: {}".format(nmi_score))
	#print("silhouette_score is: {}".format(s_score))

	agglo = AgglomerativeClustering(n_clusters=3)
	est_labels = agglo.fit_predict(features_gap)

	rand_score = adjusted_rand_score(labels, est_labels)
	nmi_score = normalized_mutual_info_score(labels, est_labels)
	s_score = silhouette_score(features_gap, est_labels)
	print("\nFor AgglomerativeClustering")
	print("rand score is: {}".format(rand_score))
	print("nmi score is: {}".format(nmi_score))
	print("silhouette_score is: {}".format(s_score))
	
	plt.scatter(X_tsne[:,0], X_tsne[:,1], marker="s", c= est_labels, s=20, cmap="rainbow")
	plt.savefig("./pred.png")
	plt.close()




