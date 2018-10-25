#author: vamshi
#July 11, 2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf 
import tensorflow.contrib.layers as layers

import os,sys

def lrelu(x, alpha):
	return tf.nn.relu(x) - alpha*tf.nn.relu(-x)

def conv2d(inp, num_kernels=32, kernel_size=3, stride=2,activation_fn='relu', stddev=0.02, is_train=True, keep_prob=0.8,name="conv2d",regularisation="l2"):
	'''
		Wrapper on tf.layers.conv2d
	'''
	if(regularisation=="l2"):
		reg = tf.contrib.layers.l2_regularizer
	else:
		reg = None

	with tf.variable_scope(name):
		conv = tf.layers.conv2d(inp,filters=num_kernels,kernel_size=kernel_size,strides=[stride,stride], padding='same',use_bias=True,
				kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1),trainable=is_train, kernel_regularizer=reg)
		conv = layers.dropout(conv,keep_prob=keep_prob)
		if activation_fn == 'relu':
			conv = layers.batch_norm(conv, center=True, scale=True, decay=0.9, is_training=is_train, updates_collections=None)
			conv = lrelu(conv,0.1)
		elif activation_fn =='tanh':
			conv = tf.nn.tanh(conv)
		elif activation_fn == None:
			conv = conv
	return conv

def deconv2d(inp, num_filters, kernel_size, stride, keep_prob=0.5, is_train=True, name="deconv2d", stddev=0.02,activation_fn='relu', regularisation="l2"):
	'''
		Wrapper on tf.layers.conv2d_transpose
	'''
	if(regularisation=="l2"):
		reg = tf.contrib.layers.l2_regularizer
	else:
		reg = None

	with tf.variable_scope(name):
	    deconv = layers.conv2d_transpose(inp,num_outputs=num_filters,kernel_size=[kernel_size, kernel_size], stride=[stride,stride], padding='same',
	    	weights_initializer=tf.contrib.layers.xavier_initializer(seed=1), weights_regularizer=reg)
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



class ConvVAE:
	def __init__(self, args, batch_size=100, z_dim=2, imsize=256, learning_rate = 0.0002):

		self.g_reuse = False
		self.r_reuse = False
		self.args = args
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.z_dim = z_dim

		self.x_dim = imsize
		self.y_dim = imsize
		self.imsize = imsize
		self.n_points = self.x_dim * self.y_dim

		# tf Graph batch of image (batch_size, height, width, depth)
		self.x_raw = tf.placeholder(tf.float32, [batch_size, self.y_dim, self.x_dim, self.args.num_channels])

		# distort raw data (decided in the end to leave this task to DataLoader class)
		self.x = self.x_raw

		# Create autoencoder network
		self._create_network()
		# Define loss function based variational upper-bound and
		# corresponding optimizer
		self._create_loss_optimizer()

		# Initializing the tensor flow variables
		init = tf.initialize_all_variables()

		# Launch the session
		self.sess = tf.InteractiveSession()
		self.sess.run(init)
		self.saver = tf.train.Saver(tf.all_variables())

	def add_gaussian_noise(self,X,std=1.0):
		noise = tf.random_normal(shape = X.get_shape(), mean = 0.0, stddev = self.args.std, dtype = tf.float32)
		return X+noise

	def _create_network(self):

		self.z_mean, self.z_log_sigma_sq = self._recognition_network(self.x)

		# Draw one sample z from Gaussian distribution
		n_z = self.z_dim
		eps = tf.random_normal((self.batch_size, n_z), 0.0, 1.0, dtype=tf.float32)
		# z = mu + sigma*epsilon
		self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

		# Use generator to determine mean of
		# Bernoulli distribution of reconstructed input
		self.x_reconstr_mean = self._generator_network(self.z)


	def _recognition_network(self, image):
		# Generate probabilistic encoder (recognition network), which
		# maps inputs onto a normal distribution in latent space.
		# The transformation is parametrized and can be learned.

		with tf.variable_scope("recognition_network",reuse=self.r_reuse):
			conv1_1 = conv2d(image, num_kernels=16, kernel_size=5, stride=2,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_1_1")

			conv2_1 = conv2d(conv1_1, num_kernels=32, kernel_size=5, stride=2,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_2_1")

			conv3_1 = conv2d(conv2_1, num_kernels=16, kernel_size=5, stride=2,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_3_1")

			conv4_1 = conv2d(conv3_1, num_kernels=32, kernel_size=3, stride=2,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_4_1")

			conv5_1 = conv2d(conv4_1, num_kernels=64, kernel_size=3, stride=2,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_5_1")

			conv6_1 = conv2d(conv5_1, num_kernels=128, kernel_size=3, stride=2,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_6_1")

			conv7_1 = conv2d(conv6_1, num_kernels=256, kernel_size=3, stride=2,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_7_1")

			conv8 = conv2d(conv7_1, num_kernels=512, kernel_size=3, stride=1,activation_fn='relu', stddev=0.02, is_train=True, name="conv2d_8")
			
			reshape = tf.contrib.layers.flatten(conv8,scope="flatten")

			#fully connected layers for mean
			n_h1 = 2048
			n_h2 = self.z_dim

			z_mean_fc1 = tf.layers.dense(reshape, units=n_h1,activation=tf.nn.relu,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())
			#z_mean_fc1 = lrelu(z_mean_fc1, 0.1)

			z_mean = tf.layers.dense(z_mean_fc1, units=n_h2,activation=None,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())

			#fully connected layers for log_sigma_sq
			n_h1 = 1024
			n_h2 = self.z_dim

			z_log_sigma_sq_fc1 = tf.layers.dense(reshape, units=n_h1,activation=tf.nn.relu,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())
			#z_log_sigma_sq_fc1 = lrelu(z_log_sigma_sq_fc1, 0.1)

			z_log_sigma_sq = tf.layers.dense(z_log_sigma_sq_fc1, units=n_h2,activation=None,use_bias=True,
					kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),bias_initializer=tf.zeros_initializer())

			return (z_mean, z_log_sigma_sq)

	def _generator_network(self, z):
		# Generate probabilistic decoder (decoder network), which
		# maps points in latent space onto a Bernoulli distribution in data space.
		# The transformation is parametrized and can be learned.

		# project `z` and reshape

		with tf.variable_scope("generator", reuse=self.g_reuse) as scope:

			inp = tf.reshape(z, shape=(self.batch_size,1,1,self.z_dim))

			inp = deconv2d(inp, num_filters=768, kernel_size=5, stride=4, name="project", stddev=0.02,activation_fn='relu')

			deconv1 = deconv2d(inp, num_filters=384, kernel_size=5, stride=2, name="deconv2d_11", stddev=0.02,activation_fn='relu')
			deconv1 =   self.add_gaussian_noise(deconv1)

			deconv2 = deconv2d(deconv1, num_filters=256, kernel_size=5, stride=2, name="deconv2d_21", stddev=0.02,activation_fn='relu',regularisation=None)
			deconv2 =   self.add_gaussian_noise(deconv2)

			deconv3 = deconv2d(deconv2, num_filters=256, kernel_size=5, stride=2, name="deconv2d_31", stddev=0.02,activation_fn='relu',regularisation=None)
			deconv3 =   self.add_gaussian_noise(deconv3)

			deconv4 = deconv2d(deconv3, num_filters=192, kernel_size=5, stride=2, name="deconv2d_41", stddev=0.02,activation_fn='relu',regularisation=None)

			deconv5 = deconv2d(deconv4, num_filters=96, kernel_size=5, stride=2, name="deconv2d_51", stddev=0.02,activation_fn='relu')

			deconv6 = deconv2d(deconv5, num_filters=3, kernel_size=5, stride=2, name="deconv2d_61", stddev=0.02,activation_fn='relu')
			conv7 =   conv2d(deconv6, num_kernels=self.args.num_channels, kernel_size=5, stride=1,activation_fn='tanh', stddev=0.02, is_train=True, name="conv2d_63")
			self.g_reuse = True

			#x_reconstr_mean = tf.nn.sigmoid(conv6_3)

			return conv7

  	def _create_loss_optimizer(self):
		# The loss is composed of two terms:
		# 1.) The reconstruction loss (the negative log probability
		#     of the input under the reconstructed Bernoulli distribution
		#     induced by the decoder in the data space).
		#     This can be interpreted as the number of "nats" required
		#     for reconstructing the input when the activation in latent
		#     is given.
		# Adding 1e-10 to avoid evaluatio of log(0.0)

		orig_image = tf.contrib.layers.flatten(self.x, scope="o")
		new_image = tf.contrib.layers.flatten(self.x_reconstr_mean, scope="r")

		
		self.reconstr_loss = \
		    -tf.reduce_sum(orig_image * tf.log(1e-10 + new_image)
		                   + (1-orig_image) * tf.log(1e-10 + 1 - new_image), 1)
		
		self.reconstr_loss = self.reconstr_loss/self.n_points

		# use L2 loss instead:
		d = (orig_image - new_image)
		d2 = tf.multiply(d, d) 
		self.vae_loss_l2 = tf.reduce_sum(d2, 1)

		self.vae_loss_l2 = self.vae_loss_l2/self.n_points  # average over batch and pixel
		#self.vae_loss_l2 = tf.losses.mean_squared_error(orig_image, new_image, reduction=None)

		# 2.) The latent loss, which is defined as the Kullback Leibler divergence
		##    between the distribution in latent space induced by the encoder on
		#     the data and some prior. This acts as a kind of regularizer.
		#     This can be interpreted as the number of "nats" required
		#     for transmitting the the latent space distribution given
		#     the prior.
		self.vae_loss_kl = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
		                                   - tf.square(self.z_mean)
		                                   - tf.exp(self.z_log_sigma_sq), 1)

		self.cost = tf.reduce_mean(tf.add(self.vae_loss_l2, self.vae_loss_kl))   # average over batch

		#self.cost = tf.reduce_mean(self.vae_loss_kl + self.vae_loss_l2)
		
		self.vae_loss_kl_avg = tf.reduce_mean(self.vae_loss_kl)
		self.vae_loss_l2_avg = tf.reduce_mean(self.vae_loss_l2)
		self.reconstr_loss_avg = tf.reduce_mean(self.reconstr_loss)

		self.t_vars = tf.trainable_variables()

		# Use ADAM optimizer
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost, var_list=self.t_vars)

  	def partial_fit(self, X):
	    """Train model based on mini-batch of input data.

	    Return cost of mini-batch.
	    """

	    opt, cost, vae_loss_l2, vae_loss_kl = self.sess.run((self.optimizer, self.cost, self.vae_loss_l2_avg, self.vae_loss_kl_avg),
	                              feed_dict={self.x_raw: X})
	    return cost, vae_loss_l2, vae_loss_kl

  	def transform(self, X):
	    """Transform data by mapping it into the latent space."""
	    # Note: This maps to mean of distribution, we could alternatively
	    # sample from Gaussian distribution
	    return self.sess.run(self.z_mean, feed_dict={self.x_raw: X})

  	def generate(self, z_mu=None):
	    """ Generate data by sampling from latent space.

	    If z_mu is not None, data for this point in latent space is
	    generated. Otherwise, z_mu is drawn from prior in latent
	    space.
	    """
	    if z_mu is None:
	        z_mu = np.random.normal(size=(1, self.z_dim))
	    # Note: This maps to mean of distribution, we could alternatively
	    # sample from Gaussian distribution
	    return self.sess.run(self.x_reconstr_mean,
	                           feed_dict={self.z: z_mu})

  	def reconstruct(self, X):
	    """ Use VAE to reconstruct given data. """
	    return self.sess.run(self.x_reconstr_mean,
	                         feed_dict={self.x_raw: X})

  	def save_model(self, checkpoint_path, epoch):
	    """ saves the model to a file """
	    self.saver.save(self.sess, checkpoint_path, global_step = epoch)

  	def load_model(self, checkpoint_path):

	    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
	    print("loading model: ",ckpt.model_checkpoint_path)

	    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
