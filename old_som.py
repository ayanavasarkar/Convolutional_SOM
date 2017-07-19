#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:38:44 2017

@author: ayanava

 now the output of the som needs to be re-packaged, 
 ie each little SOMs activbities need to be tiled into a big image again.
Lets say each little SOM has kxk neurons, and there are 24x24 patches, 
then for each patch there is a kxk matrix of distances.
If you arrange these kxk matrices in the order of the origibnal patches, you get an image of dimension (24k)x(24k).

"""

import tensorflow as tf
import numpy as np, gzip, cPickle
import time, math


class SOM:
	"""
	Efficient implementation of Kohonen Self-Organizing maps using Tensorflow.

	map_size_n: size of the (square) map
	num_expected_iterations: number of iterations to be used during training. This parameter is used to derive good parameters for the learning rate and the neighborhood radius.
	"""
	def __init__(self):
          pass;	

        """
          tInf: time at which params should be converged. Take half the # of iterations
          sigmaInf: asymptotic value of sigma at tInf. Can use 1.0 for the time b eing
          alphaInf: asymptic learning rate at tInf. Can use 0.05 for the time being
        """

	def graph_distance_and_update(self, input_shape, map_size_n, tInf, sigmaInf, alphaInf, sess, input_x,  iteration ):
		if(iteration==0):
	
	  		#print ("Time after Weight init--- %s seconds ---" % (time.time() - start_time))
                	input_shape = input_shape#tuple([i for i in input_shape if i is not None])
			self.input_x = input_x
			self.iter = iteration
			self.input_shape = input_shape
                        print "inputs", input_shape[2]
			self.sigma_act = tf.constant( 1.0) ;

			self.n = map_size_n
			
			self.session = sess

                        alpha0=0.1; 
			self.alpha0 = tf.constant( alpha0,dtype=tf.float32 ) ;
			self.alphaInf = tf.constant( alphaInf,dtype=tf.float32 ) ;
			self.tInf = tf.constant( tInf, dtype=tf.float32 ) ;

			
			self.timeconst_alpha = tf.constant( -math.log(alphaInf/alpha0)/tInf, dtype=tf.float32) #2/6
	
                        sigma0 = self.n/4. ;
                        self.sigmaInf = tf.constant(sigmaInf,dtype=tf.float32);
			self.sigma0 = tf.constant( sigma0,dtype=tf.float32 ) #self.n/2.0
			
			self.timeconst_sigma = tf.constant( -math.log(sigmaInf/sigma0)/tInf,dtype=tf.float32 )
			
                        self.bs = tf.constant(input_x.shape[0],dtype=tf.float32) ;
			
			
		  	# Pre-initialize neighborhood function's data for efficiency
                        # important: make it float32 since numpy default is float64
			self.row_indices = np.zeros((self.n, self.n), dtype="float32")
			self.col_indices = np.zeros((self.n, self.n),dtype="float32")
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

                        # this takes 12 secs for 100x100xbs1
			self.dist=tf.Variable(tf.add(self.X_pow,self.Y_pow)) ;
                        # this takes 31 secs for 100x100xbs1
			#self.dist=tf.add(self.X_pow,self.Y_pow) ;

			self.weights = tf.Variable( tf.random_uniform((1, self.n*self.n,self.input_shape[2]), 0.0, 1.0) ) 			
	
			self.input_placeholder = tf.placeholder(tf.float32, shape=(None, 1, None))
			self.current_iteration = tf.placeholder(tf.float32)
			self.sigma_tmp = tf.cond(self.current_iteration < self.tInf, 
                                                 lambda: self.sigma0,
                                                 lambda:self.sigma0 * tf.exp( - self.current_iteration*self.timeconst_sigma  ) ) ;
			self.alpha_tmp = tf.cond(self.current_iteration < self.tInf, 
                                                 lambda: self.alpha0,
                                                 lambda:self.alpha0 * tf.exp( - self.current_iteration*self.timeconst_alpha  ) ) ;
			self.sigma_tmp = tf.cond(self.sigma_tmp < self.sigmaInf, 
                                                 lambda: self.sigmaInf,
                                                 lambda: self.sigma_tmp) ;
			self.alpha_tmp = tf.cond(self.alpha_tmp < self.alphaInf, 
                                                 lambda: self.alphaInf/self.bs,
                                                 lambda: self.alpha_tmp/self.bs) ;

                
			self.diff= tf.subtract(self.input_placeholder,self.weights)
			self.diff_sq = tf.square(self.diff)
			self.diff_sum = tf.reduce_sum( self.diff_sq, axis=2)	#Take this
			#print self.diff_sum.op	#prints the operation performed and the attributes

			# Get the index of the best matching unit
	     		self.bmu_index = tf.argmin(self.diff_sum, 1) ;
                	
			self.dist_sliced=tf.gather(self.dist,self.bmu_index) 
		
			self.distances = tf.exp(-self.dist_sliced / self.sigma_tmp )
			self.lr_times_neigh = tf.multiply( self.alpha_tmp, self.distances )
			self.lr_times_neigh = tf.expand_dims(self.lr_times_neigh, 2)
		
			self.delta_w_batch = self.lr_times_neigh * self.diff
			self.delta_w = tf.expand_dims(tf.reduce_sum (self.delta_w_batch, axis=0),0) ;

			self.update_weights = tf.assign_add(self.weights, self.delta_w)
                        #print self.update_weights.op
                        	
 		if(iteration==0):
       			tf.global_variables_initializer().run()

		#return (self.session.run([self.diff_sum.op], {self.input_placeholder:input_x, self.current_iteration:iteration}))
		self.session.run([self.update_weights], {self.input_placeholder:input_x, self.current_iteration:iteration})                
		
    

	def get_weights(self):
		"""
		Returns the full list of weights as [N*N, input_shape]
		"""
		return self.weights.eval()
    
	def get_distances(self):
        	"""
		Returns the distances.
		"""
		#distances = self.session.run([self.distances], {self.input_placeholder:input_x})#, self.current_iteration:self.num_iterations
        	return self.session.run([self.diff_sum], {self.input_placeholder:self.input_x, self.current_iteration:self.iter})
        
    
	def get_array(self, i, traind):
             #self.nu = i
             #print self.nu
             #self.y = traind[i].reshape((28,28))
             #plt.imshow(self.y, interpolation='none')
             #plt.show()
	        #self.x = traind[i].reshape((784,))
		#return self.x
		self.x = traind[i]
		#print self.x.shape --- (784,)
		self.index = np.load("index_array.npy")
        
        	self.arr =  np.take(self.x, self.index)
        
        	#print "Array formed "
		return self.arr
            
            

import argparse
from random import randint

parser = argparse.ArgumentParser()
#parser.add_argument("mnist", help="mnist path")
parser.add_argument("device", help="GPU or CPU")
parser.add_argument("batch_size", help="Number of samples per iteration",type=int)
parser.add_argument("map_size", help="Size of the output layer", type= int)
args = parser.parse_args()

#path = args.mnist 

with gzip.open("/home/admin/MNIST_data/mnist.pkl.gz","rb") as f:
        ((traind,trainl),(vald,vall),(testd,testl))=cPickle.load(f)
        if vall==None and vald==None:
          vall=vald=np.zeros([1,1])

inpDim = traind.shape[1] ;
  
#data_train = traind; 
#labels = trainl ;
dev=(args.device)
map_size=args.map_size
batch_size = (args.batch_size)

#data = np.zeros((batch_size, 576 , 25))

#g1 = tf.Graph() 
#with g1.as_default() as g:
with tf.device(dev):
	#sess = tf.InteractiveSession(graph=g)
        sess = tf.InteractiveSession()
	num_training = 10
	s = SOM()

	#sess.run(tf.global_variables_initializer())
        
	total_patches = 576 #24*24
	flag = 0
	counter=0
        start_time=time.time()

	for i in range(num_training):
			        
               	counter = randint(0, 100)
		#arr = traind[counter:counter+batch_size]
		#print arr[0].shape
		#for j in range(0, batch_size):
		data = s.get_array(counter, traind)
			            
		data=np.expand_dims(data, axis=1)
		#data=np.expand_dims(data, axis=0)
		print data.shape
                #arr = traind[i:i+batch_size,np.newaxis,:] ;
		

		#change the following to arr.shape for the original SOM implementation
		s.graph_distance_and_update(data.shape, map_size, num_training/2, 1.0, 0.05, sess, data, flag)
		flag=flag+1
         	dis = s.get_distances()
		dist = np.array(dis)
		dist = np.squeeze(dist)
		#print ("Before reshape")
            	#print (dist.shape)
		dist = np.reshape(dist, (24, 24, map_size*map_size))
		#print ("After reshape")
            	#print (dist.shape)

print ("FINAL TIME--- %s seconds ---" % (time.time() - start_time))

weights  = s.get_weights()
print weights.shape
#x = np.squeeze(weights)
#print x[0]

#np.savez("som.npz", weights[0,:,:]) ;

##### Execution steps #####

# visualize to png file later with 
# python visWeights.py som.npz 1 10 10 28 3
# this creates a file w.png thatn you can view with eog
