
import os,sys
import numpy as np

def load_batched_data(data,batch_size,imsize,num_channels):
	'''
		Function to load data as batches
	'''
	num_samples = len(data)
	randIxs = np.random.permutation(num_samples)
	start,end =0,batch_size
	while(end<=num_samples):
		batchInputs_img = np.zeros((batch_size, imsize, imsize, num_channels))
		for batchI, origI in enumerate(randIxs[start:end]):
			if(num_channels==3):
				batchInputs_img[batchI,:] = data[origI]		
			else:
				batchInputs_img[batchI,:] = np.expand_dims(data[origI],axis=2)		

		start += batch_size
		end += batch_size
		yield batchInputs_img
