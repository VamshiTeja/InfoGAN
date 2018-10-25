#author:vamshi

import os
import sys
import numpy as np 
import cv2
import re
import glob
from tqdm import tqdm
import pickle 

#file to read 

TRAIN_DIR = "../datasets/mammo_cancer/train/"
VAL_DIR  = "../data_mammo/val/"

imsize = 256


def read_imgs(data_dir,label="cancer",d="LEFT", mode="MLO"):
	'''
		returns all images in the folder with dir, mode of photo

	'''
	data = []
	if(mode==None and d!=None):
		file = "*"+d+"_"+"*"+".png"
	elif(mode==None and d==None):
		file = "*.png"
	elif(d==None and mode!=None):
		file = "*"+"_"+mode+".png"
	else:
		file = "*"+d+"_"+mode+".png"

	for file in tqdm(glob.glob(TRAIN_DIR+label+"/"+file)):
		img = cv2.imread(file,0)
		img = cv2.resize(img,(imsize,imsize),interpolation=cv2.INTER_CUBIC)

		#print(img.shape)
		#cv2.imshow('img',img)
		#cv2.waitKey(0)
		img = img.astype(int)
		img = (img.astype(np.float32)/255.0)*2.0 - 1.0
		data.append(img)
	return data

def remove_text(img):
	'''
		removes text present in the images
	'''
	#edge = cv2.Canny(img,10,200)
	#blur = cv2.GaussianBlur(img,(5,5),0)
	#ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	#cv2.imshow('img',edge)
	#cv2.waitKey(0)
	#print(img)
	return img


cancer = read_imgs(TRAIN_DIR,label="cancer",d=None,mode=None)
print(len(cancer))
normal = read_imgs(TRAIN_DIR,label="normal",d=None,mode=None)
print(len(normal))

save_file = "../datasets/mammo_cancer/dataset_mammo_train_"+str(imsize)+".pickle"
with open(save_file,"wb") as f:
	data  = {'cancer':cancer,
		  'normal':normal}
	pickle.dump(data,f)

print(len(data))
#read_imgs(TRAIN_DIR,d="LEFT",mode="MLO")
#read_imgs(TRAIN_DIR,d="RIGHT",mode="CC")
#read_imgs(TRAIN_DIR,d="LEFT",mode="CC")
