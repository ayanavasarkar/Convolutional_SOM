#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:20:52 2017

@author: ayanava
"""

import gzip, cPickle
import matplotlib.pyplot as plt


with gzip.open("/home/admin/MNIST_data/mnist.pkl.gz","rb") as f:
  ((traind,trainl),(vald,vall),(testd,testl)) = cPickle.load(f) ;

x = traind[200].reshape((28,28)) ;
plt.imshow(x, interpolation='none')
plt.show()
'''
for i in range(0,100):
    
    x = traind[i].reshape((28,28)) ;
    plt.imshow(x, interpolation='none')
    plt.show()
    '''