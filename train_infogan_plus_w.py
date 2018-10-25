#author:vamshiteja

import sys
sys.path.append("../model")
sys.path.append("../")

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
from utils.loader import load_batched_data
from model.model_w_kaggle import InfoGAN
 

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

#define hyperparameters
batch_size = 32
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
std = 0.02
restore = 1
restore_vae = 0
mode = "dc-gan"
dataset = "kaggle"

if(dataset=="mammo"):
	from model.model_w_mammo import InfoGAN
	datapath           = "./datasets/mammo_cancer/dataset_mammo_train_"+str(imsize)+".pickle"
	checkpoint_dir     = "./checkpoints/gan/mammo/"
	vae_checkpoint_dir = "./checkpoints/vae/mammo/"
	num_channels = 1
elif(dataset=="kaggle"):
	from model.model_w_kaggle import InfoGAN
	datapath           = "./datasets/kaggle_cancer/dataset_kaggle_train_"+str(imsize)+".pickle"
	checkpoint_dir     = "./checkpoints/gan/kaggle/"
	vae_checkpoint_dir = "./checkpoints/vae/kaggle/"
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

def make_str(config):
	'''
		returns str representation of a dict
	'''
	config_str = ""
	for k,v in config.iteritems():
		config_str += str(k)+":"+str(v)+"_"
	return config_str

class Trainer():
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
			  'restore_vae': restore_vae,
			  'dataset': dataset
			}

	def load_mammo_data(self,datapath):
		'''
			Function to load data
		'''
		with open(datapath,"rb") as f:
			data = pickle.load(f)

		normal = np.array(data['normal'])
		cancer = np.array(data['cancer'])
		data = np.concatenate((normal,cancer),axis=0)
		return data

	def load_kaggle_data(self,datapath):
		'''
			Function to load data
		'''
		with open(datapath,"rb") as f:
			data = pickle.load(f)

		a = np.array(data['1'])
		b = np.array(data['2'])
		c = np.array(data['3'])
		data = np.concatenate((a,b,c),axis=0)
		return data


	def run(self):
		'''
			Function to train and test
		'''
		args_dict = self._default_configs()
		args = dotdict(args_dict)

		if(dataset=="mammo"):
			data = self.load_mammo_data(datapath)
		elif(dataset=="kaggle"):
			data = self.load_kaggle_data(datapath)

		print("[INFO]: Data Loaded")
		model = InfoGAN(args)
		model.build_graph(args)
		num_batches = int(len(data)/args.batch_size)
		print("[Info]: Number of batches are %d " %num_batches)

		with tf.Session(graph=model.graph) as sess:
			writer = tf.summary.FileWriter("logging",graph=model.graph)
			if(args.restore_vae==1):
				vae_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator'))
				vae_saver.restore(sess, tf.train.latest_checkpoint(vae_checkpoint_dir))
				print("[Info]: Generator weights restored	 from VAE decoder")
			if(args.restore==1):
				model.saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
			else:
				sess.run(model.initial_op)

			for iteration in range(num_iters):

				disc_loss = 0
				for i in range(disc_iters):
					X_mb = next(load_batched_data(data,args.batch_size,args.imsize,args.num_channels))
					Z_noise = model.sample_z(args.batch_size, args.z_dim)
					cat_noise = model.sample_cat(args.batch_size, args.cat_dim)
					cont_noise = model.sample_cont(args.batch_size, args.cont_dim)

					if(args.mode=="dc-gan"):
						_, D_loss_curr, summary= sess.run([model.D_solver, model.D_loss, model.summary_op],
							feed_dict={model.X: X_mb, model.z: Z_noise, model.cat: cat_noise, model.cont: cont_noise})
						disc_loss += D_loss_curr
						_, Q_loss_cur = sess.run([model.Q_solver, model.Q_loss], 
							feed_dict={model.z: Z_noise, model.cat: cat_noise, model.cont: cont_noise})

					elif(args.mode=="dc-wgan"):
						_, D_loss_curr, summary, _= sess.run([model.D_solver, model.D_loss, model.summary_op, model.clip_D],
							feed_dict={model.X: X_mb, model.z: Z_noise, model.cat: cat_noise, model.cont: cont_noise})
						
					disc_loss += D_loss_curr

						#_, Q_loss_cur = sess.run([model.Q_solver, model.Q_loss], 
							#feed_dict={model.z: Z_noise, model.cat: cat_noise, model.cont: cont_noise})

				if(iteration>0):
					X_mb = next(load_batched_data(data,args.batch_size,args.imsize,args.num_channels))
					Z_noise = model.sample_z(args.batch_size, args.z_dim)
					cat_noise = model.sample_cat(args.batch_size, args.cat_dim)
					cont_noise = model.sample_cont(args.batch_size, args.cont_dim)

					_, G_loss_curr = sess.run([model.G_solver, model.G_loss],
			                              feed_dict={model.z: Z_noise, model.cat: cat_noise, model.cont: cont_noise})
				
				disc_loss = disc_loss/disc_iters
				writer.add_summary(summary, iteration)
				if(iteration!=0 & iteration%50==0):
					print("[Info]: Iter:%d/%d , G_loss: %f, D_loss: %f"%(iteration+1,num_iters,G_loss_curr, disc_loss))
					#print("Q_loss: %f\n"%Q_loss_cur)

				if(iteration%100==0):
					model.saver.save(sess, checkpoint_dir+args.dataset+"_infogan.ckpt", global_step = model.global_step)

				#test after every 2 epochs
				i=0
				if(iteration%100 == 0):
					Z_noise = model.sample_z(args.batch_size, args.z_dim)

					idx = np.random.randint(0, args.cat_dim)
					cat_noise = np.zeros([args.batch_size, args.cat_dim])
					cat_noise[range(args.cat_dim), idx] = 1

					cont_noise = model.sample_cont(args.batch_size, args.cont_dim)

					samples = sess.run(model.G_sample,
					                   feed_dict={model.z: Z_noise, model.cat: cat_noise, model.cont: cont_noise})
					for i in range(args.batch_size):
						img_d = samples[i]*255.0/2.0 + 255.0/2.0
						img_d = img_d.astype(int)
						img_d = img_d.astype(np.uint8)
						if not os.path.exists("out/gan_"+args.dataset+"_"+str(args.std)):
    							os.makedirs("out/gan_"+args.dataset+"_"+str(args.std))
						cv2.imwrite('out/gan_{}_{}/iter_{}_{}.png'.format(args.dataset, args.std,str(iteration),str(i)),img_d)



if __name__ == '__main__':
	runner = Trainer()
	runner.run()
