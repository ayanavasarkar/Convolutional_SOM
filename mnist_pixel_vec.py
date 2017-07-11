#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:26:52 2017

@author: ayanava
"""

import numpy as np, csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


x = np.load('arr_7.npy')
#arr = np.zeros(shape=(576,25), dtype= 'int64')
print x.shape
print x[500]

gs = gridspec.GridSpec(24, 24, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)
c=0

for g in gs:
    
    ax = plt.subplot(g)
    a = np.reshape(x[c, :],(5,5))
    plt.imshow(a)
    c = c + 1
    plt.show()
    
    #ax.set_aspect('auto')
    

'''


#img = mimage.imread('/home/admin/Convolutional_SOM/patches/'+str(row)+'_'+str(col)+'.png')

pdf = PdfPages( 'test.pdf' )
gs = gridspec.GridSpec(120, 120, top=1., bottom=0., right=1., left=0., hspace=0.,
        wspace=0.)

c = 0

for g in gs:
    
    ax = plt.subplot(g)
    ax.imshow('~/home/admin/Convolutional_SOM/patches/'+str(c)+'.jpg')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    c = c +1

pdf.savefig()
pdf.close()



row = 0
col = 0

l = []
final = []

with open('mnist_test_10.csv', 'r') as csv_file:
    for data in csv.reader(csv_file):
        #tf.InteractiveSession()
        arr=np.zeros((24*24,25), dtype='int')
        # The first column is the label
        label = data[0]

        # The rest of columns are pixels
        pixels = data[1:]

        # Make those columns into a array of 8-bits pixels
        # This array will be of 1D with length 784
        # The pixel intensity values are integers from 0 to 255
        pixels = np.array(pixels, dtype='int64')
        #print (pixels[0]).dtype
        # Reshape the array into 28 x 28 array (2-dimensional array)
        pixels = pixels.reshape((28, 28))
        plt.imshow(pixels, interpolation='none')
        plt.show()
        #arg = tf.convert_to_tensor(pixels, dtype=tf.int64)
        
        #with tf.Session() as sess:
        patch_size = 10
        counter = 0
        while(row<24):
            col=0
            while(col<24):
                l=(pixels[row:row+5, col:col+5]).tolist()
                #plt.imshow(pixels[row:row+5, col:col+5], interpolation='none')
                #plt.savefig('/home/admin/Convolutional_SOM/patches/'+str(counter)+'.jpg')
                
                #print len(l)
                for i in range(0,5):
                    for j in range(0,5):
                                                
                        arr[counter,i+j]= l[i][j]
                                
                del l[:]
                col=col+1
                counter = counter+1
                    
            row=row+1
        
        
        np.save('arr_7.npy', arr)
        break
'''