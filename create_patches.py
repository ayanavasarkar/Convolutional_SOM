#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:26:52 2017

@author: ayanava
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse, cv2
from matplotlib import pyplot as plt
from PIL import Image


####### np.take indexing ########

patch_x = 5
patch_y = 5
stride_x = 1
stride_y = 1
patches = patch_x * patch_y
r_x = ((28-patch_x)/stride_x)+1
r_y = ((28-patch_y)/stride_y)+1
row_lim = (r_x * r_y)
print row_lim

arr = np.zeros(shape=(row_lim,patches), dtype= 'int64')

row=0
col=0
counter = 0
iteration = np.array([0,28,28*2,28*3,28*4])

index = 0

while(row<row_lim):
    col=0
    index = 0
    initial = counter+1
    while(col<patches):                             #25 for 5 X 5 patches
        arr[row][col]= (counter+iteration[index])
        #print (counter+iteration[index])
        if(counter==(initial+(patch_x-2))):         #3 for 5 X 5 patches
            #print counter
            counter = initial-1
            index = index+1
        else:
            counter= counter+1
        
        col+=1
    #break
    row+=1
    counter = initial

np.save('index_arr.npy', arr)
#print np.max(arr)#[0][0]



'''
with gzip.open("/home/admin/MNIST_data/mnist.pkl.gz","rb") as f:
  ((traind,trainl),(vald,vall),(testd,testl)) = cPickle.load(f) ;

x = traind[0].reshape((28,28)) ;



parser = argparse.ArgumentParser()
parser.add_argument("label", help="label of the MNIST dataset", type=int)
args = parser.parse_args()

i = args.label 

with open('mnist_test_10.csv', 'r') as csv_file:
    for data in csv.reader(csv_file):
        
        arr=np.zeros((24*24,25), dtype='int64')
        # The first column is the label
        #label = data[]

        # The rest of columns are pixels
        pixels = data[1:]

        # Make those columns into a array of 8-bits pixels
        # This array will be of 1D with length 784
        # The pixel intensity values are integers from 0 to 255
        pixels = np.array(pixels, dtype='int64')
        #print (pixels[0]).dtype
        # Reshape the array into 28 x 28 array (2-dimensional array)
        pixels = pixels.reshape((28, 28))
        #print pixels.size
        index = np.load("index_array.npy") 
        plt.imshow(pixels, interpolation='none')
        plt.show()
        #arr =  np.take(pixels, index)
        #break
    print "Hello"

'''
