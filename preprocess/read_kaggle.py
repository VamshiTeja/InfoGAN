import os
import sys
import numpy as np 
import cv2
import re
import glob
from tqdm import tqdm
import pickle 
from natsort import natsorted, ns
import pandas as pd

mode = "test"

TRAIN_DIR = "../data_cancer_kaggle/train"
TEST_DIR  = "../data_cancer_kaggle/test/stage1"
TEST_LABELS = "../data_cancer_kaggle/test/solution_stg1_release.csv"
imsize = 256


if(mode=="train"):
	label_data = []
	for c in os.listdir(TRAIN_DIR):
		data = []
		for file in tqdm(os.listdir(os.path.join(TRAIN_DIR,c))):
			img_file = os.path.join(TRAIN_DIR+"/"+c, file)
			try:
				img = cv2.imread(img_file,1)
				img = cv2.resize(img,(imsize,imsize),interpolation=cv2.INTER_CUBIC)

			except cv2.error as e:
				print ("error while reading")
				continue

			#print(img.shape)
			#cv2.imshow('img',img)
			#cv2.waitKey(0)
			img = img.astype(int)
			img = (img.astype(np.float32)/255.0)*2.0 - 1.0

			data.append(img)
		print(len(data))
		label_data.append(data)

	save_file = "../data_cancer_kaggle/dataset_kaggle_train_"+str(imsize)+".pickle"
	with open(save_file,"wb") as f:
		data  = {'1':label_data[0],
			  '2':label_data[1],
			  '3':label_data[2]}
		pickle.dump(data,f)

elif(mode=="test"):
	data = []
	for file in tqdm(natsorted(os.listdir(TEST_DIR), key=lambda y:y.lower())):
		img_file = os.path.join(TEST_DIR,file)
		img = cv2.imread(img_file,1)
		img = cv2.resize(img,(imsize,imsize),interpolation=cv2.INTER_CUBIC)

		img = img.astype(int)
		img = (img.astype(np.float32)/255.0)*2.0 - 1.0

		data.append(img)

	labels = pd.read_csv(TEST_LABELS,names=['image_name','Type_1', 'Type_2', 'Type_3'],header=0)
	labels = labels.drop(columns=['image_name'])
	labels = labels.as_matrix()
	labels = np.argmax(labels, axis=0)

	save_file = "../data_cancer_kaggle/test/stage1_"+str(imsize)+".pickle"
	with open(save_file,"wb") as f:
		data = {
		'data': data,
		'labels': labels
		}
		pickle.dump(data,f)

