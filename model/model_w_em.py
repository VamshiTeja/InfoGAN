#author:vamshiteja

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf 
import tensorflow.contrib.layers as layers

import os,sys

LAMBDA = 10

def lrelu(x, alpha):
	return tf.nn.relu(x) - alpha*tf.nn.relu(-x)

def conv2d(inp, num_kernels=32, kernel_size=3, stride=2,activation_fn='relu', stddev=0.02, is_train=True, keep_prob=0.9,name="conv2d"):
	'''
		Wrapper on tf.layers.conv2d
	'''
	with tf.variable_scope(name):
		conv = tf.layers.conv2d(inp,filters=num_kernels,kernel_size=kernel_size,strides=[stride,stride], padding='same',use_bias=True,
				kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1),trainable=is_train)
		conv = layers.dropout(conv,keep_prob=keep_prob)
		if activation_fn == 'relu':
			conv = layers.batch_norm(conv, center=True, scale=True, decay=0.9, is_training=is_train, updates_collections=None)
			conv = lrelu(conv,0.1)
		elif activation_fn =='tanh':
			conv = tf.nn.tanh(conv)
		elif activation_fn == None:
			conv = conv
	return conv

def deconv2d(inp, num_filters, kernel_size, stride, keep_prob=0.9, is_train=True, name="deconv2d", stddev=0.02,activation_fn='relu'):
	'''
		Wrapper on tf.layers.conv2d_transpose
	'''
	with tf.variable_scope(name):
	    deconv = layers.conv2d_transpose(inp,num_outputs=num_filters,kernel_size=[kernel_size, kernel_size], stride=[stride,stride], padding='same',
	    	weights_initializer=tf.contrib.layers.xavier_initializer(seed=1))
	    deconv = layers.dropout(deconv,keep_prob=keep_prob)
	    if activation_fn == 'relu':
	        #deconv = tf.nn.local_response_normalization(deconv, depth_radius=5, bias=1e-4, alpha=1, beta=0.5)
	        deconv = tf.contrib.layers.batch_norm(deconv, center=True, scale=True, decay=0.9, is_training=is_train, updates_collections=None)
	        deconv = tf.nn.relu(deconv)
	    elif activation_fn == 'tanh':
	        deconv = tf.nn.tanh(deconv)
	    elif activation_fn == None:
	    	deconv = deconv
	    return deconv

def max_pool(inp,pool_size,strides,padding="SAME",name="pool2d"):
	'''
		Max pooling operation 
	'''
	with tf.variable_scope(name):
		return tf.layers.max_pooling2d(inp,pool_size=pool_size,strides=strides,padding=padding,name=name)


class InfoGAN:
	'''
		Info GAN class with graph of the network
	'''

	def __init__(self,args):
		self.args = args
		self.g_reuse = False
		self.d_reuse = False
		self.q_cat_reuse = False
		self.q_cont_reuse = False	
		self.s_reuse = False


	def sample_z(self, m, n):
		'''
			sample z from normal dist
		'''
		return np.random.normal(loc=0, scale=1.0, size=[m, n])

	def sample_cat(self, m, n):
		'''
			sample categorical variables from multinomial distribution.
		'''
		return np.random.multinomial(1, n*[1.0/float(n)], size=m)
	
	def sample_cont(self,m,n):
		'''
			sample continuos variables to be learned from normal distribution.
		'''
		return np.random.normal(loc=0, scale=1.0, size=[m, n])

	def Generator(self,z,cat,cont):
		'''
			Generator: generates images of size 512x512x3
			inputs : z-> z_dim dimensional, cat-> cat_dim dimensional, cont-> cont_dim dimensional
		'''
		inp = tf.concat(values=[z,cat,cont],axis=1)
		inp = tf.cast(inp,dtype=tf.float32)
		with tf.variable_scope("generator", reuse=self.g_reuse) as scope:
			dense = tf.layers.dense(inp, units=8*8*128,activation=tf.nn.relu,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())
			#reshape to batch_sx8x8x128 
			inp = tf.reshape(dense, shape=(self.args.batch_size,4,4,512))

			deconv1_1 = deconv2d(inp, num_filters=256, kernel_size=5, stride=2, name="deconv2d_11", stddev=0.02,activation_fn='relu')
			deconv1_2 = deconv2d(deconv1_1, num_filters=256, kernel_size=5, stride=1, name="deconv2d_12", stddev=0.02,activation_fn='relu')
			deconv2_1 = deconv2d(deconv1_2, num_filters=128, kernel_size=5, stride=2, name="deconv2d_21", stddev=0.02,activation_fn='relu')
			deconv2_2 = deconv2d(deconv2_1, num_filters=128, kernel_size=5, stride=1, name="deconv2d_22", stddev=0.02,activation_fn='relu')
			deconv3_1 = deconv2d(deconv2_2, num_filters=64, kernel_size=3, stride=2, name="deconv2d_31", stddev=0.02,activation_fn='relu')
			deconv3_2 = deconv2d(deconv3_1, num_filters=64, kernel_size=3, stride=1, name="deconv2d_32", stddev=0.02,activation_fn='relu')
			deconv4_1 = deconv2d(deconv3_2, num_filters=32, kernel_size=3, stride=2, name="deconv2d_41", stddev=0.02,activation_fn='relu')
			deconv4_2 = deconv2d(deconv4_1, num_filters=32, kernel_size=3, stride=1, name="deconv2d_42", stddev=0.02,activation_fn='relu')
			deconv5_1 = deconv2d(deconv4_2, num_filters=16, kernel_size=3, stride=2, name="deconv2d_51", stddev=0.02,activation_fn='relu')
			deconv5_2 = deconv2d(deconv5_1, num_filters=16, kernel_size=3, stride=1, name="deconv2d_52", stddev=0.02,activation_fn='relu')
			deconv6_1 = deconv2d(deconv5_2, num_filters=8, kernel_size=3, stride=2, name="deconv2d_61", stddev=0.02,activation_fn='relu')
			deconv6_2 = deconv2d(deconv6_1, num_filters=self.args.num_channels, kernel_size=3, stride=1, name="deconv2d_62", stddev=0.02,activation_fn='tanh')
		self.g_reuse = True
		return deconv6_2

	def shared_layers(self,X):
		'''
			shared layers of discriminator
		'''

		with tf.variable_scope("shared",reuse=self.s_reuse):
			conv1_1 = conv2d(X, num_kernels=8, kernel_size=5, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_1_1")
			conv1_2 = conv2d(conv1_1, num_kernels=8, kernel_size=5, stride=2,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_1_2")

			conv2_1 = conv2d(conv1_2, num_kernels=16, kernel_size=5, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_2_1")
			conv2_2 = conv2d(conv2_1, num_kernels=16, kernel_size=5, stride=2,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_2_2")

			conv3_1 = conv2d(conv2_2, num_kernels=32, kernel_size=3, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_3_1")
			conv3_2 = conv2d(conv3_1, num_kernels=32, kernel_size=3, stride=2,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_3_2")

			conv4_1 = conv2d(conv3_2, num_kernels=64, kernel_size=3, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_4_1")
			conv4_2 = conv2d(conv4_1, num_kernels=64, kernel_size=3, stride=2,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_4_2")

			conv5_1 = conv2d(conv4_2, num_kernels=128, kernel_size=3, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_5_1")
			conv5_2 = conv2d(conv5_1, num_kernels=128, kernel_size=3, stride=2,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_5_2")

			conv6_1 = conv2d(conv5_2, num_kernels=256, kernel_size=3, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_6_1")
			conv6_2 = conv2d(conv6_1, num_kernels=256, kernel_size=3, stride=2,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_6_2")

		self.s_reuse = True
		return conv6_2


	def Discriminator(self,X):
		'''
			Discriminator as in vanilla GAN 
			x: [None,256,256,3] image
			returns 0 or 1 (real or fake) 
		'''

		conv6_2 = self.shared_layers(X)
		reshape = tf.contrib.layers.flatten(conv6_2,scope="flatten")

		with tf.variable_scope("discriminator",reuse=self.d_reuse):

			#add fully connected layers
			n_h1 = 2048
			n_h2 = 128
			n_h3 = 1

			fc1 = tf.layers.dense(reshape, units=n_h2,activation=None,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())
			fc1 = lrelu(fc1, 0.1)

			fc2 = tf.layers.dense(fc1, units=n_h2,activation=None,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())
			fc2 = lrelu(fc2, 0.1)

			fc3 = tf.layers.dense(fc2, units=n_h3,activation=None,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())
			#fc3 = lrelu(fc3, 0.1)

		self.d_reuse = True
		return fc3


	def Q(self,X):
		'''
			Network which learns p(c/X) distribution
			returns c
			x: [None,512,512,3] image
			returns categorical variables, continuos variables
		'''

		conv6_2 = self.shared_layers(X)
		reshape = tf.contrib.layers.flatten(conv6_2,scope="flatten")

		with tf.variable_scope("cat",reuse=self.q_cat_reuse):

			#add fully connected layers
			n_h1 = 2048
			n_h2 = 256
			n_h3 = self.args.cat_dim

			cat_fc1 = tf.layers.dense(reshape, units=n_h2,activation=None,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())
			cat_fc1 = lrelu(cat_fc1, 0.1)

			cat_fc2 = tf.layers.dense(cat_fc1, units=n_h2,activation=tf.nn.relu,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())
			cat_fc2 = lrelu(cat_fc2, 0.1)

			cat_fc3 = tf.layers.dense(cat_fc2, units=n_h3,activation=None,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())
			#cat_fc3 = lrelu(cat_fc3, 0.1)
		
		self.q_cat_reuse = True

		with tf.variable_scope("cont",reuse=self.q_cont_reuse):

			#add fully connected layers
			n_h1 = 2048
			n_h2 = 256
			n_h3 = self.args.cont_dim

			cont_fc1 = tf.layers.dense(reshape, units=n_h2,activation=tf.nn.relu,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())
			cont_fc1 = lrelu(cont_fc1,0.1)

			cont_fc2 = tf.layers.dense(cont_fc1, units=n_h2,activation=tf.nn.relu,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())
			cont_fc2 = lrelu(cont_fc2,0.1)

			cont_fc3 = tf.layers.dense(cont_fc2, units=n_h3,activation=tf.nn.tanh,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())
			cont_fc3 = lrelu(cont_fc3,0.1)

		self.q_cont_reuse = True

		return cat_fc3, cont_fc3

	def build_graph(self,args):
		'''
			builds graph of the architecture
		'''
		self.graph = tf.Graph()
		with self.graph.as_default():
			#define placeholders
			self.X = tf.placeholder(dtype=tf.float32, shape=(None,args.imsize,args.imsize,args.num_channels))
			self.z = tf.placeholder(dtype=tf.float32, shape=(None,args.z_dim))
			self.cat = tf.placeholder(dtype=tf.float32, shape=(None,args.cat_dim))
			self.cont = tf.placeholder(dtype=tf.float32, shape=(None,args.cont_dim))

			#forward prop
			self.G_sample = self.Generator(self.z, self.cat, self.cont)
			tf.summary.image('G_sample', self.G_sample)

			self.D_real = self.Discriminator(self.X)
			self.D_fake = self.Discriminator(self.G_sample)
			self.Q_cat_given_x, self.Q_cont_given_x = self.Q(self.G_sample)

			#tf.summary.scalar('Q_c_given_x',self.Q_c_given_x)

			#loss functions
			#self.D_loss = tf.reduce_mean((tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_real),logits=self.D_real)) + 
			#		tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_fake),logits=self.D_fake))))

			#self.G_loss =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_fake),logits=self.D_fake))

			self.D_loss = tf.reduce_mean(self.D_fake) - tf.reduce_mean(self.D_real)
			self.G_loss = -tf.reduce_mean(self.D_fake)
			print(self.D_loss)
			# Gradient penalty
			alpha = tf.random_uniform(
			    shape=tf.shape(self.X), 
			    minval=0.,
			    maxval=1.
			)

			differences = self.G_sample - self.X
			interpolates = self.X + (alpha*differences)
			gradients = tf.gradients(self.Discriminator(interpolates), [interpolates])[0]
			slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
			gradient_penalty = tf.reduce_mean((slopes-1.)**2)
			self.D_loss += LAMBDA*gradient_penalty

			cross_ent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.cat,logits=self.Q_cat_given_x))
			#cross_ent = tf.reduce_mean(-tf.reduce_sum(tf.log(self.Q_cat_given_x + 1e-8) * self.cat, 1))
			ent = tf.reduce_mean(-tf.reduce_sum(tf.log(self.cat + 1e-6) * self.cat, 1))

			self.Q_loss_cat = cross_ent + ent
			self.Q_loss_cont = tf.reduce_mean(tf.square(self.Q_cont_given_x-self.cont))
			self.Q_loss = 0.9*self.Q_loss_cat + 0.1*self.Q_loss_cont

			self.D_plus_Q_loss = 0.5*self.D_loss + 0.5*self.Q_loss
			self.G_plus_Q_loss = 0.5*self.G_loss + 0.5*self.Q_loss

			tf.summary.scalar('D_loss',self.D_loss)
			tf.summary.scalar('G_loss',self.G_loss)
			tf.summary.scalar('Q_loss',self.Q_loss)

			#get variables corressponding to G,D,Q networks
			theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="generator")

			theta_S = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="shared")
			theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="discriminator")+theta_S
			theta_Q_cat = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="cat")
			theta_Q_cont = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="cont")
			theta_Q = theta_Q_cat+theta_Q_cont

			theta_D_all = theta_D + theta_Q

			self.clip_D = [p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in theta_D_all]

			#add regularisation
			self.D_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(5e-6), theta_D_all)
			self.G_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(5e-6), theta_G)
			#self.Q_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(5e-6), theta_Q)

			self.global_step = tf.Variable(0, name='global_step', trainable=False)

			#define optimizers
			if(args.grad_clip==-1):
				self.D_solver = tf.train.RMSPropOptimizer(args.learning_rate).minimize(self.D_plus_Q_loss+self.D_reg, var_list=theta_D+theta_Q, global_step=self.global_step)
				self.G_solver = tf.train.RMSPropOptimizer(args.learning_rate).minimize(self.G_plus_Q_loss+self.G_reg, var_list=theta_G)
				#self.Q_solver = tf.train.AdamOptimizer(args.learning_rate).minimize(self.Q_loss+self.Q_reg, var_list=theta_Q+theta_S)
			else:
				grads, _ = tf.clip_by_global_norm(tf.gradients(self.D_plus_Q_loss+self.D_reg, theta_D+theta_Q), 5)
				opti = tf.train.RMSPropOptimizer(args.learning_rate)
				self.D_solver= opti.apply_gradients(zip(grads, theta_D+theta_Q))

				grads, _ = tf.clip_by_global_norm(tf.gradients(self.G_plus_Q_loss+self.G_reg, theta_G), 5)
				opti = tf.train.RMSPropOptimizer(args.learning_rate)
				self.G_solver= opti.apply_gradients(zip(grads, theta_G))

				#grads, _ = tf.clip_by_global_norm(tf.gradients(self.Q_loss+self.Q_reg, theta_Q+theta_S),5)
				#opti = tf.train.AdamOptimizer(args.learning_rate)
				#self.Q_solver= opti.apply_gradients(zip(grads, theta_Q+theta_S))

			self.initial_op = tf.initialize_all_variables()
			self.summary_op = tf.summary.merge_all()

			#save variables
			self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)