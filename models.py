import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import cPickle as pkl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import urllib
import os
import tarfile
import skimage
import skimage.io
import skimage.transform

# FEATURE EXTRACTOR
def shared_encoder(x, name='feat_ext', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        x = (tf.cast(x, tf.float32) - pixel_mean) / 255.
        net = slim.conv2d(x, 32, [5, 5], scope = 'conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 48, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        feat = slim.flatten(net, scope='flat')
    return feat

#Private Target Encoder
def private_target_encoder(x, name='priviate_target_encoder', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        x = (tf.cast(x, tf.float32) - pixel_mean) / 255.
        net = slim.conv2d(x, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 48, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        feat = slim.flatten(net, scope='flat')
    return feat

#Private Source Encoder
def private_source_encoder(x, name='priviate_source_encoder', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        x = (tf.cast(x, tf.float32) - pixel_mean) / 255.
        net = slim.conv2d(x, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 48, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        feat = slim.flatten(net, scope='flat')
    return feat

# CLASS PREDICTION
def shared_decoder(feat,height,width,channels,reuse=False, name='shared_decoder'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        net = slim.fully_connected(feat, 400, scope='fc1')
        net = slim.fully_connected(net, 784*3, scope='fc2')
        #net = tf.reshape(net, [batch_size, height, width, channels])
        #net = tf.reshape(net, [feat.shape[0], height, width, channels])
        net = tf.reshape(net, [-1, height, width, channels])
    return net

# CLASS PREDICTION
def class_pred_net(feat, name='class_pred', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        net = slim.fully_connected(feat, 100, scope='fc1')
        net = slim.fully_connected(net, 100, scope='fc2')
        net = slim.fully_connected(net, 10, activation_fn = None, scope='out')
    return net

# DOMAIN PREDICTION
def domain_pred_net(feat, name='domain_pred', reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        feat = flip_gradient(feat, dw) # GRADIENT REVERSAL
        net = slim.fully_connected(feat, 100, scope='fc1')
        net = slim.fully_connected(net, 2, activation_fn = None, scope='out')
    return net
