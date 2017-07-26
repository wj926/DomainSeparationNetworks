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

# DIFFERENCE LOSS
def difference_loss(private_samples, shared_samples, weight=1.0, name=''):
  private_samples -= tf.reduce_mean(private_samples, 0)
  shared_samples -= tf.reduce_mean(shared_samples, 0)
  private_samples = tf.nn.l2_normalize(private_samples, 1)
  shared_samples = tf.nn.l2_normalize(shared_samples, 1)
  correlation_matrix = tf.matmul( private_samples, shared_samples, transpose_a=True)
  cost = tf.reduce_mean(tf.square(correlation_matrix)) * weight
  cost = tf.where(cost > 0, cost, 0, name='value')
  #tf.summary.scalar('losses/Difference Loss {}'.format(name),cost)
  assert_op = tf.Assert(tf.is_finite(cost), [cost])
  with tf.control_dependencies([assert_op]):
     tf.losses.add_loss(cost)
  return cost