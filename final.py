import tensorflow as tf
import numpy as np, sys, gzip, cPickle
import time,random,math


class SOM:
	"""
	Efficient implementation of Kohonen Self-Organizing maps using Tensorflow.

	map_size_n: size of the (square) map
	num_expected_iterations: number of iterations to be used during training. This parameter is used to derive good parameters for the learning rate and the neighborhood radius.
	"""

	def __init__(self, sample_size, map_size_n, bs, tConv, tInf, sigmaInf, alphaInf ):

			self.n = map_size_n
			self.sample_size = sample_size ;
                        self.tConv = tf.constant (tConv, dtype=tf.float32) ;

			self.weights = tf.Variable( tf.random_uniform((1,1,1, self.n*self.n,self.sample_size), 0.0, 0.1) ) 				
			self.input_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 1, None))
			self.current_iteration = tf.placeholder(tf.float32)
			
                        # time constants for Learning rate
                        alpha0=0.01; 
			self.alpha0 = tf.constant( alpha0,dtype=tf.float32 ) ;
			self.alphaInf = tf.constant( alphaInf,dtype=tf.float32 ) ;
			self.tInf = tf.constant( tInf, dtype=tf.float32 ) ;
			self.timeconst_alpha = tf.constant( -math.log(alphaInf/alpha0)/tInf, dtype=tf.float32) #2/6
			
			# time constants for neighbhood function
                        sigma0 = self.n/4. ;
                        self.sigmaInf = tf.constant(sigmaInf,dtype=tf.float32);
			self.sigma0 = tf.constant( sigma0,dtype=tf.float32 ) #self.n/2.0
			self.timeconst_sigma = tf.constant( -math.log(sigmaInf/sigma0)/tInf,dtype=tf.float32 ) ;

                        self.bs = tf.constant(bs,dtype=tf.float32) ;	#the batch size as float 
			
		  	# Pre-initialize neighborhood function's data for efficiency
                        # important: make it float32 since numpy default is float64
                        # this could maybe be optimized.. need to understand tf better in order to see 
                        # whether this is all executed every time som is executed

                        # construct structures in numpy
			self.row_indices = np.zeros((self.n, self.n), dtype="float32")
			self.col_indices = np.zeros((self.n, self.n),dtype="float32")
			for r in range(self.n):
			  for c in range(self.n):
			    self.row_indices[r, c] = r
			    self.col_indices[r, c] = c
		  	self.n_square=(self.n*self.n)
                        
                        # transfer to TF
			self.r_x = tf.reshape(self.row_indices, [self.n_square,1])
			self.r_y= tf.reshape(self.row_indices, [1,self.n_square])
			self.X_sub=tf.subtract(self.r_x,self.r_y)
		  	self.X_pow=tf.pow(self.X_sub,2)
			self.c_x = tf.reshape(self.col_indices, [self.n_square,1])
			self.c_y = tf.reshape(self.col_indices, [1,self.n_square])
			self.Y_sub=tf.subtract(self.c_x,self.c_y)
			self.Y_pow=tf.pow(self.Y_sub,2)
			self.dist=tf.Variable(tf.add(self.X_pow,self.Y_pow)) ;

                        # check whether we are in global ordering or beyong the decrease phase
                        # le is divided by batch size here!
			self.sigma_tmp = tf.cond(self.current_iteration < self.tConv, 
                                                 lambda: self.sigma0,
                                                 lambda:self.sigma0 * tf.exp( - self.current_iteration*self.timeconst_sigma  ) ) ;
			self.alpha_tmp = tf.cond(self.current_iteration < self.tConv, 
                                                 lambda: self.alpha0,
                                                 lambda:self.alpha0 * tf.exp( - self.current_iteration*self.timeconst_alpha  ) ) ;
			self.sigma_tmp = tf.cond(self.sigma_tmp < self.sigmaInf, 
                                                 lambda: self.sigmaInf,
                                                 lambda: self.sigma_tmp) ;
			self.alpha_tmp = tf.cond(self.alpha_tmp < self.alphaInf, 
                                                 lambda: self.alphaInf/self.bs,
                                                 lambda: self.alpha_tmp/self.bs) ;

                        # input - weights using broadcasting
			self.diff= tf.subtract(self.input_placeholder,self.weights)
			self.diff_sq = tf.square(self.diff)
			self.diff_sum = tf.reduce_sum( self.diff_sq, axis=4)

                        # is not part of the update graph since sqrt is not required
			self.dist_input_protos = tf.sqrt(self.diff_sum) ;
			
			# Get the index of the best matching unit
	     		self.bmu_index = tf.argmin(self.diff_sum, 3) ;

                        # get pre-computed squared distances in order to compute the gaussian neighbourhood                	
			self.dist_sliced=tf.gather(self.dist,self.bmu_index) 	
                        print self.dist_sliced	
			self.distances = tf.exp(-self.dist_sliced / self.sigma_tmp )

			
			# apply lr and insert a new dimension			
			self.lr_times_neigh = tf.multiply( self.alpha_tmp, self.distances ) # vecg
			self.lr_times_neigh = tf.expand_dims(self.lr_times_neigh, 4)
		
                        print "weights", self.weights
			self.delta_w_batch = self.lr_times_neigh * self.diff
			self.delta_w = tf.expand_dims(tf.reduce_sum (self.delta_w_batch, axis=(0, 1, 2)), axis=0)
			self.delta_w = tf.expand_dims(self.delta_w,axis=0	) ;
			self.delta_w = tf.expand_dims(self.delta_w,axis=0	) ;
                        print "DeltaW", self.delta_w

			self.update_weights = tf.assign_add(self.weights, self.delta_w)

       			tf.global_variables_initializer().run()


        """
          tInf: time at which params should be converged. Take half the # of iterations
          sigmaInf: asymptotic value of sigma at tInf. Can use 1.0 for the time being
          alphaInf: asymptic learning rate at tInf. Can use 0.05 for the time being
        """
	def graph_distance_and_update(self, session, input_x,  iteration):
                        	
		session.run([self.update_weights], {self.input_placeholder:input_x, self.current_iteration:iteration})                

	def graph_distance(self, session, input_x,  iteration ):
                        	
		session.run([self.dist_input_protos], {self.input_placeholder:input_x})                
                                        


	def get_weights(self):
		"""
		Returns the full list of weights as [N*N, input_shape]
		"""
		return self.weights.eval()


	def get_array(self, arr):
             #self.nu = i
             #print self.nu
             #self.y = traind[i].reshape((28,28))
             #plt.imshow(self.y, interpolation='none')
             #plt.show()
	        #self.x = traind[i].reshape((784,))
		#return self.x
		self.x = arr
		#print self.x.shape --- (784,)
		self.index = np.load("index_array.npy")
        
        	self.arr =  np.take(self.x, self.index)
        
        	#print "Array formed "
		return self.arr

import argparse
from random import randint

#parser = argparse.ArgumentParser()

#parser.add_argument("device", help="GPU or CPU")
#parser.add_argument("batch_size", help="Number of samples per iteration",type=int)
#parser.add_argument("map_size", help="Size of the output layer", type= int)
#args = parser.parse_args()

#path = args.mnist 


with gzip.open("/home/admin/MNIST_data/mnist.pkl.gz", 'rb') as f:
        ((traind,trainl),(vald,vall),(testd,testl))=cPickle.load(f)
        if vall==None and vald==None:
          vall=vald=np.zeros([1,1])
inpDim = traind.shape[1] ;
  
data = traind; labels = trainl ;

dev="/cpu:0"

start_time=time.time()

g1 = tf.Graph() 
with g1.as_default() as g:
	with tf.device(dev):
		sess = tf.InteractiveSession(graph=g)
        
		num_training = 500
        
		n = 10 ;
		flag = 0
		counter=0;
                
                _bs = 10 ; #batch Size
		_nrPatchesY = 24; _nrPatchesX = 24 ; _nrChannels = 25;	#1 1 784

		s = SOM(sample_size=_nrChannels, map_size_n=10, bs=_bs, tConv=1000, tInf=num_training/2, sigmaInf=1.0,
                        alphaInf=0.005) ;

		for i in range(num_training):
                        if i==1:
                          start_time=time.time()
                	
			#arr = np.zeros([_bs, 576, 25])
                        arr = np.zeros([_bs, _nrPatchesY, _nrPatchesX, 1, _nrChannels])
			dat = np.zeros([576, 25])
                        for j in xrange(0,_bs):
		
                        	dat = s.get_array(data[i+j]) ;
				arr [j] = np.reshape( dat, ( _nrPatchesY, _nrPatchesX, 1, _nrChannels ))
             		
			#arr = np.expand_dims(arr, axis=2)
			#print arr.shape
			#print np.average(arr)	
			
			s.graph_distance_and_update(sess, arr, flag)
			flag=flag+1

print ("FINAL TIME--- %s seconds ---" % (time.time() - start_time))

weights  = s.get_weights()
#print weights.shape
np.savez("som.npz", weights[0,0,0,:,:]) ;

# visualize to png file later with 
# python visWeights.py som.npz 1 10 10 28 3
# this creates a file w.png thatn you can view with eog

