#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:20:52 2017

@author: ayanava
"""

import numpy as np
import csv, gzip, cPickle
import matplotlib.pyplot as plt


with gzip.open("/home/admin/MNIST_data/mnist.pkl.gz","rb") as f:
  ((traind,trainl),(vald,vall),(testd,testl)) = cPickle.load(f) ;

for i in range(0,100):
    
    x = traind[i].reshape((28,28)) ;
    plt.imshow(x, interpolation='none')
    plt.show()