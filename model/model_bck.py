#author:vamshiteja

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf 
import tensorflow.contrib.layers as layers

import os,sys

def conv2d(inp, num_kernels=32, kernel_size=3, stride=2,activation_fn='relu', stddev=0.02, is_train=True, keep_prob=0.5,name="conv2d"):
    with tf.variable_scope(name):
    	conv = tf.layers.conv2d(inp,filters=num_kernels,kernel_size=kernel_size,strides=[stride,stride], padding='same',use_bias=True,
    			kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),trainable=is_train)
    	conv = layers.dropout(conv,keep_prob=keep_prob)
    	if activation_fn == 'relu':
    		bn = layers.batch_norm(conv, center=True, scale=True, decay=0.9, is_training=is_train, updates_collections=None)
    		conv = tf.nn.relu(bn)
    	elif activation_fn =='tanh':
    		conv = tf.nn.tanh(conv)
    	elif activation_fn == None:
    		conv = conv
    return conv

def deconv2d(inp, num_filters, kernel_size, stride, keep_prob=0.5, is_train=True, name="deconv2d", stddev=0.02,activation_fn='relu'):
    with tf.variable_scope(name):
        deconv = layers.conv2d_transpose(inp,num_outputs=num_filters,kernel_size=[kernel_size, kernel_size], stride=[stride,stride], padding='same',
        	weights_initializer=tf.truncated_normal_initializer(stddev=stddev))
        deconv = layers.dropout(deconv,keep_prob=keep_prob)
        if activation_fn == 'relu':
            bn = tf.contrib.layers.batch_norm(deconv, center=True, scale=True, decay=0.9, is_training=is_train, updates_collections=None)
            deconv = tf.nn.relu(bn)
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

	def __init__(self,args):
		self.args = args
		self.g_reuse = False
		self.d_reuse = False
		self.q_reuse = False


	def sample_z(self, m, n):
    	 return np.random.normal(loc=0, scale=1.0, size=[m, n])

	def sample_c(self, m):
		return np.random.multinomial(1, 10*[0.1], size=m)
	
	def Generator(self,z,c):
		'''
			Generator: generates images of size 512x512x3
			inputs : z-> 90 dimensional c->10 dimensional
		'''

		inp = tf.concat(values=[z,c],axis=1)
		inp = tf.cast(inp,dtype=tf.float32)
		with tf.variable_scope("generator", reuse=self.g_reuse) as scope:
			dense = tf.layers.dense(inp, units=8*8*128,activation=tf.nn.relu,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())
			#reshape to batch_sx8x8x128 
			inp = tf.reshape(dense, shape=(self.args.batch_size,4,4,512))

			deconv1 = deconv2d(inp, num_filters=256, kernel_size=3, stride=2, name="deconv2d_1", stddev=0.02,activation_fn='relu')
			deconv2 = deconv2d(deconv1, num_filters=128, kernel_size=3, stride=2, name="deconv2d_2", stddev=0.02,activation_fn='relu')
			deconv3 = deconv2d(deconv2, num_filters=64, kernel_size=3, stride=2, name="deconv2d_3", stddev=0.02,activation_fn='relu')
			deconv4 = deconv2d(deconv3, num_filters=32, kernel_size=3, stride=2, name="deconv2d_4", stddev=0.02,activation_fn='relu')
			deconv5 = deconv2d(deconv4, num_filters=8, kernel_size=3, stride=2, name="deconv2d_5", stddev=0.02,activation_fn='relu')
			deconv6 = deconv2d(deconv5, num_filters=3, kernel_size=3, stride=2, name="deconv2d_6", stddev=0.02,activation_fn='relu')
			deconv7 = deconv2d(deconv6, num_filters=3, kernel_size=3, stride=2, name="deconv2d_7", stddev=0.02,activation_fn='tanh')
		self.g_reuse = True
		return deconv7

	def Discriminator(self,X):
		'''
			Discriminator as in vanilla GAN 
			x: [None,512,512,3] image
			returns 0 or 1
		'''
		with tf.variable_scope("discriminator",reuse=self.d_reuse):
			conv1 = conv2d(X, num_kernels=8, kernel_size=5, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_1")
			pool1 = max_pool(conv1,pool_size=2,strides=[2,2],padding='same',name="pool_1")

			conv2 = conv2d(pool1, num_kernels=16, kernel_size=5, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_2")
			pool2 = max_pool(conv2,pool_size=2,strides=[2,2],padding='same',name="pool_2")

			conv3 = conv2d(pool2, num_kernels=32, kernel_size=3, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_3")
			pool3 = max_pool(conv3,pool_size=2,strides=[2,2],padding='same',name="pool_3")

			conv4 = conv2d(pool3, num_kernels=64, kernel_size=3, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_4")
			pool4 = max_pool(conv4,pool_size=2,strides=[2,2],padding='same',name="pool_4")

			conv5 = conv2d(pool4, num_kernels=128, kernel_size=3, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_5")
			pool5 = max_pool(conv5,pool_size=2,strides=[2,2],padding='same',name="pool_5")

			conv6 = conv2d(pool5, num_kernels=256, kernel_size=3, stride=1,activation_fn='tanh', stddev=0.02, is_train=True, name="conv2d_6")
			pool6 = max_pool(conv6,pool_size=2,strides=[2,2],padding='same',name="pool_6")
			
			reshape = tf.contrib.layers.flatten(pool6,scope="flatten")

			#add fully connected layers
			n_h1 = 1024
			n_h2 = 64
			n_h3 = 1
			fc1 = tf.layers.dense(reshape, units=n_h1,activation=tf.nn.relu,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())

			fc2 = tf.layers.dense(fc1, units=n_h2,activation=tf.nn.relu,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())

			fc3 = tf.layers.dense(fc2, units=n_h3,activation=None,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())

		self.d_reuse = True
		return fc3


	def Q(self,X,c_dim=10):
		'''
			Network which learns p(c/X) distribution
			returns c
			x: [None,512,512,3] image
			returns categorical variables
		'''
		with tf.variable_scope("q",reuse=self.q_reuse):

			conv1 = conv2d(X, num_kernels=8, kernel_size=3, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_1")
			pool1 = max_pool(conv1,pool_size=2,strides=[2,2],padding='same',name="pool_1")

			conv2 = conv2d(pool1, num_kernels=16, kernel_size=3, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_2")
			pool2 = max_pool(conv2,pool_size=2,strides=[2,2],padding='same',name="pool_2")

			conv3 = conv2d(pool2, num_kernels=32, kernel_size=3, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_3")
			pool3 = max_pool(conv3,pool_size=2,strides=[2,2],padding='same',name="pool_3")

			conv4 = conv2d(pool3, num_kernels=64, kernel_size=3, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_4")
			pool4 = max_pool(conv4,pool_size=2,strides=[2,2],padding='same',name="pool_4")

			conv5 = conv2d(pool4, num_kernels=128, kernel_size=3, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_5")
			pool5 = max_pool(conv5,pool_size=2,strides=[2,2],padding='same',name="pool_5")

			conv6 = conv2d(pool5, num_kernels=256, kernel_size=3, stride=1,activation_fn='tanh', stddev=0.02, is_train=True, name="conv2d_6")
			pool6 = max_pool(conv6,pool_size=2,strides=[2,2],padding='same',name="pool_6")

			reshape = tf.contrib.layers.flatten(pool6,scope="flatten")

			#add fully connected layers
			n_h1 = 1024
			n_h2 = 128
			n_h3 = 10
			fc1 = tf.layers.dense(reshape, units=n_h1,activation=tf.nn.relu,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())

			fc2 = tf.layers.dense(fc1, units=n_h2,activation=tf.nn.relu,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())

			fc3 = tf.layers.dense(fc2, units=n_h3,activation=None,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())

		self.q_reuse = True

		return fc3

	def build_graph(self,args):
		self.graph = tf.Graph()
		with self.graph.as_default():
			#define placeholders
			self.X = tf.placeholder(dtype=tf.float32, shape=(None,args.imsize,args.imsize,args.num_channels))
			self.z = tf.placeholder(dtype=tf.float32, shape=(None,args.z_dim))
			self.c = tf.placeholder(dtype=tf.float32, shape=(None,args.c_dim))

			#forward prop
			self.G_sample = self.Generator(self.z, self.c)
			self.D_real = self.Discriminator(self.X)
			self.D_fake = self.Discriminator(self.G_sample)
			self.Q_c_given_x = self.Q(self.G_sample,args.c_dim)

			#tf.summary.scalar('Q_c_given_x',self.Q_c_given_x)

			#loss functions
			self.D_loss = tf.reduce_mean(0.5*(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_real),logits=self.D_real)) + 
					tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_fake),logits=self.D_fake))))

			self.G_loss =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_fake),logits=self.D_fake))

			cross_ent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.c,logits=self.Q_c_given_x))
			#cross_ent = tf.reduce_mean(-tf.reduce_sum(tf.log(self.Q_c_given_x + 1e-8) * self.c, 1))
			ent = tf.reduce_mean(-tf.reduce_sum(tf.log(self.c + 1e-8) * self.c, 1))
			self.Q_loss = cross_ent + ent

			tf.summary.scalar('D_loss',self.D_loss)
			tf.summary.scalar('G_loss',self.G_loss)
			tf.summary.scalar('Q_loss',self.Q_loss)


			#get variables corressponding to G,D,Q networks
			theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="discriminator")
			theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="generator")
			theta_Q = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="q")

			#add regularisation
			self.D_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), theta_D)
			self.G_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), theta_G)
			self.Q_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), theta_Q)

			self.global_step = tf.Variable(0, name='global_step', trainable=False)

			#define optimizers
			if(args.grad_clip==-1):
				self.D_solver = tf.train.AdamOptimizer(args.learning_rate).minimize(self.D_loss+self.D_reg, var_list=theta_D, global_step=self.global_step)
				self.G_solver = tf.train.AdamOptimizer(args.learning_rate).minimize(self.G_loss+self.G_reg, arv_list=theta_G)
				self.Q_solver = tf.train.AdamOptimizer(args.learning_rate).minimize(self.Q_loss+self.Q_reg, var_list=theta_G + theta_Q)
			else:
				grads, _ = tf.clip_by_global_norm(tf.gradients(self.D_loss+self.D_reg, theta_D), args.grad_clip)
				opti = tf.train.AdamOptimizer(args.learning_rate)
				self.D_solver= opti.apply_gradients(zip(grads, theta_D))

				grads, _ = tf.clip_by_global_norm(tf.gradients(self.G_loss+self.G_reg, theta_G), args.grad_clip)
				opti = tf.train.AdamOptimizer(args.learning_rate)
				self.G_solver= opti.apply_gradients(zip(grads, theta_G))

				grads, _ = tf.clip_by_global_norm(tf.gradients(self.Q_loss+self.Q_reg, theta_Q+theta_G), args.grad_clip)
				opti = tf.train.AdamOptimizer(args.learning_rate)
				self.Q_solver= opti.apply_gradients(zip(grads, theta_Q+theta_G))

			self.initial_op = tf.initialize_all_variables()
			self.summary_op = tf.summary.merge_all()

			#save variables
			self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)