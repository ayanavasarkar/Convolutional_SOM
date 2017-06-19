#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:09:50 2017

@author: ayanava
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import random as ran

import mnist_loader

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def main():
    training_set, validation_set, test_set = mnist_loader.load_data()
    images = get_images(training_set)
   
    print np.array(training_set).size
    
def get_images(training_set):
    """ Return a list containing the images from the MNIST data
    set. Each image is represented as a 2-d numpy array."""
    flattened_images = training_set[0]
    return [np.reshape(f, (-1, 28)) for f in flattened_images]

#### Main
if __name__ == "__main__":
    main()