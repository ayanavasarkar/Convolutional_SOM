#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:33:53 2017

@author: ayanava
"""
import tensorflow as tf
import numpy as np

n = 10


param_x = tf.placeholder(dtype=tf.float32)
param_y = tf.placeholder(dtype=tf.float32)
op_x_plus_y = tf.add(param_x, param_y)

weights = tf.Variable( tf.random_uniform((1, n*n), 0.0, 1.0) )
#print (weights.eval())
x = tf.add(weights, 10)
y = tf.subtract(weights, x)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print sess.run(y)
print (sess.run(op_x_plus_y, feed_dict={param_x: 20, param_y: 1.1}))