#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:19:52 2017

@author: ayanava
"""

import tensorflow as tf
import numpy as np, sys, gzip, cPickle
import time,random
import itertools
import csv

class SOM:
	"""
	Efficient implementation of Kohonen Self-Organizing maps using Tensorflow.

	map_size_n: size of the (square) map
	num_expected_iterations: number of iterations to be used during training. This parameter is used to derive good parameters for the learning rate and the neighborhood radius.
	"""
	def __init__(self):
          pass;


	def graph_distance_and_update(self, input_shape, map_size_n, num_expected_iterations, sess, input_x, batch_size, flag ):
		if(flag==0):
	
	  	   with tf.device(dev):	
                	input_shape = tuple([i for i in input_shape if i is not None])

			self.input_shape = input_shape
			self.sigma_act = tf.constant( 2.0*(reduce(lambda x, y:x*y, self.input_shape, 1)*0.05)**2, dtype=tf.float32 )
			self.batch_size = batch_size
			self.n = map_size_n

			self.session = sess

			self.alpha = tf.constant( 0.2 )
			self.timeconst_alpha = tf.constant( 2.0*num_expected_iterations/6.0) #2/6
	
			self.sigma = tf.constant( self.n/2.0 ) #self.n/2.0
			self.timeconst_sigma = tf.constant( 2.0*num_expected_iterations/5.0 ) #2/5

	
			self.num_iterations = 0
			self.num_expected_iterations = num_expected_iterations


			# Pre-initialize neighborhood function's data for efficiency
			self.row_indices = np.zeros((self.n, self.n), dtype="float32")
			self.col_indices = np.zeros((self.n, self.n), dtype="float32")
			for r in range(self.n):
				for c in range(self.n):
					self.row_indices[r, c] = r
					self.col_indices[r, c] = c
			self.n_square=(self.n*self.n)
			self.r_x = tf.reshape(self.row_indices, [self.n_square,1])
			self.r_y= tf.reshape(self.row_indices, [1,self.n_square])
			self.X_sub=tf.subtract(self.r_x,self.r_y)
			self.X_pow=tf.pow(self.X_sub,2)
		

			self.c_x = tf.reshape(self.col_indices, [self.n_square,1])
			self.c_y = tf.reshape(self.col_indices, [1,self.n_square])
			self.Y_sub=tf.subtract(self.c_x,self.c_y)
			self.Y_pow=tf.pow(self.Y_sub,2)
		## Compute d^2/2 for each pair of units, so that the neighborhood function can be computed as exp(-dist/sigma^2)
	
			self.dist=tf.Variable(tf.add(self.X_pow,self.Y_pow)) ;
			
			self.weights = tf.Variable( tf.random_uniform((self.batch_size,self.n*self.n, )+self.input_shape, 0.0, 1.0) ) 				
			
			self.input_placeholder = tf.placeholder(tf.float32, None, None)
			
			self.current_iteration = tf.placeholder(tf.float32)
			self.sigma_tmp = self.sigma * tf.exp( - self.current_iteration/self.timeconst_sigma  )
			self.sigma2 = 2.0*tf.multiply(self.sigma_tmp, self.sigma_tmp)
			self.alpha_tmp = self.alpha * tf.exp( - self.current_iteration/self.timeconst_alpha  )
			
			self.diff= tf.subtract(self.input_placeholder,self.weights)
			self.diff_sq = tf.square(self.diff)
			self.diff_sum = tf.reduce_sum( self.diff_sq, 2)
			
			# Get the index of the best matching unit
	     		self.bmu_index = tf.argmin(self.diff_sum, 1)
			
			self.dist_sliced=tf.gather(self.dist,self.bmu_index) 
			
			## 2) The second part computes and applies the weight update. It requires 'diff_2' and 'dist_sliced' to be filled in. dist_sliced = self.dist[bmu_index, :]
			
				
			self.distances = tf.exp(-self.dist_sliced / self.sigma2 )
			self.lr_times_neigh = tf.multiply( self.alpha_tmp, self.distances )
			#self.lr_times_neigh = tf.expand_dims(self.lr_times_neigh, 2)
			self.lr_times_neigh = tf.tile(self.lr_times_neigh, (1,)+self.input_shape )
			self.lr_times_neigh = tf.reshape(self.lr_times_neigh,[self.batch_size,self.n*self.n,25])
			self.delta_w = self.lr_times_neigh * self.diff
			#self.delta_w = tf.expand_dims(tf.reduce_sum (self.delta_w, axis=0),0) ;
			
			self.update_weights = tf.assign_add(self.weights, self.delta_w)
			
			tf.global_variables_initializer().run()
			self.session.run([self.bmu_index, self.update_weights], { self.input_placeholder:input_x, self.current_iteration:self.num_iterations})
		else:
		   with tf.device(dev):
			self.session.run([self.bmu_index, self.update_weights.op], { self.input_placeholder:input_x, self.current_iteration:self.num_iterations})
			

	
	def get_weights(self):
		"""
		Returns the full list of weights as [N*N, input_shape]
		"""
		with tf.device(dev):
			return self.weights.eval()
    
        def get_array(self, i, traind):

		self.x = traind[i].reshape((28,28)) ; 
       		self.index = np.load("index_array.npy")
        
        	self.arr =  np.take(self.x, self.index)
        
        	#print "Array formed "
        	return self.arr
            
        	
'''
		with open('/home/admin/MNIST_data/mnist_train.csv', 'r') as csv_file:
        	    for data in csv.reader(csv_file):
        
        	        self.arr=np.zeros((24*24,25), dtype='int64')
        # The first column is the label
        	        #self.label = data[i]

        # The rest of columns are pixels
        	        self.pixels = data[1:]

        # Make those columns into a array of 8-bits pixels
        # This array will be of 1D with length 784
        # The pixel intensity values are integers from 0 to 255
        	        self.pixels = np.array(self.pixels, dtype='int64')
        #print (pixels[0]).dtype
        # Reshape the array into 28 x 28 array (2-dimensional array)
        	        self.pixels = self.pixels.reshape((28, 28))
        #print pixels.size
        	        self.index = np.load("index_array.npy") 
        	        self.arr =  np.take(self.pixels, self.index)
        	        print "Array formed "
        	        return self.arr
        	        break    
'''

import argparse
import json
import os
from collections import OrderedDict
from random import randint


parser = argparse.ArgumentParser()
#parser.add_argument("mnist", help="mnist path")
parser.add_argument("device", help="GPU or CPU")
parser.add_argument("iters", help="Number of iterations",type=int)
parser.add_argument("map_size", help="Size of the output layer", type= int)
#parser.add_argument("patch_size", help="Patch Size of the Convolutional Layer", type= int)
args = parser.parse_args()

with gzip.open("/home/admin/MNIST_data/mnist.pkl.gz","rb") as f:
	((traind,trainl),(vald,vall),(testd,testl)) = cPickle.load(f) ;


counter = 0
#labels = trainl ;
map_size=args.map_size
dev=(args.device)
x=[]
g1 = tf.Graph() 

with g1.as_default() as g:
    with tf.device(dev):
        sess = tf.InteractiveSession(graph=g)
        
        num_training = args.iters
        s = SOM()
        
        batch_size = 24*24          #  args.batch_size
        flag = 0
        counter=0
     		#start_time=time.time()
             
    for i in range(0,num_training):
        #counter = randint(0, 50000)
        data = s.get_array(counter, traind)
        counter = counter + 1
       
        print counter
        if i==1:
        		#print "real start!"
                start_time=time.time()
		
        x=data#[0:batch_size]
        print x.shape
        x=np.expand_dims(x, axis=1)
        print x.shape
        s.graph_distance_and_update((25,), map_size, num_training, sess, x, batch_size, flag)
        #s.graph_distance_and_update((784,), map_size, num_training, sess, x, batch_size, flag)
        flag=flag+1

final=time.time() - start_time
print final
weights  = s.get_weights()
print weights.shape
np.savez("som.npz", weights[0,:,:]) ;

# visualize to png file later with 
# python visWeights.py som.npz 1 10 10 28 3
# this creates a file w.png thatn you can view with eog

