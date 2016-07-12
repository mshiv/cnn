#IMPORTS

import glob
import os
import argparse
import gzip
import tarfile
import urllib
from os import listdir
from os.path import isfile, join

import scipy.misc

import tensorflow as tf
import numpy as np 
import re

from tensorflow.python.framework import dtypes
import base

import matplotlib.pyplot as plt

import cv2
#-------------------------------------------------

print "Locating dataset..."
#Read images into arrays
#Sat images = Input data


# The dataset has 2 classes, representing the digits 0 and 1.

# The images are always 1500x1500 pixels.
IMAGE_SIZE = 1500
NUM_CHANNELS = 1
NUM_CLASSES = 2
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS
RESIZE_FACTOR = 100

class DataSet(object):
  
  def __init__(self, images, labels, fake_data=False, one_hot=False, dtype=dtypes.float32):
    dtype = dtypes.as_dtype(dtype).base_dtype
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      # Convert shape from [num examples, rows, columns, depth]

      #Center data
      #image_mean = np.mean(images, axis=0)
      # zero center data (subtract mean image from data matrix
      images = images.astype(np.float32)
      images -= np.mean(images)

      labels = labels.astype(np.float32)
      labels = np.multiply(labels, 1.0/255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0  
    self._index_in_epoch = 0
  
  @property
  def images(self):
    return self._images
  
  @property
  def labels(self):
    return self._labels
  
  @property
  def num_examples(self):
    return self._num_examples
  
  @property
  def epochs_completed(self):
    return self._epochs_completed
  
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(245025)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir, fake_data=False, one_hot=False, dtype=dtypes.float32):
  class DataSets(object):
    pass
  #data_sets = DataSets()
  # Satellite image inputs for testing
  mypath='/Datasets/mass_roads_batch/test/sat'# % data_type
  if not os.path.exists(mypath):
    print ("Data not found!")
  onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
  sat_images_test = np.empty([len(onlyfiles),IMAGE_SIZE,IMAGE_SIZE])
  for n in range(0, len(onlyfiles)):
    sat_images_test[n] = scipy.misc.imresize(cv2.cvtColor(cv2.imread(join(mypath, onlyfiles[n])), cv2.COLOR_RGB2GRAY),RESIZE_FACTOR)
    #sat_images_test[n] = cv2.imread(join(mypath, onlyfiles[n]))   
    print ("reading test satellite image: %d of %d" % (n+1, len(onlyfiles)))
    #sat_images.reshape([len(onlyfiles), sat_images[n].shape[0],sat_images[n].shape[1],sat_images[n].shape[2]])

  # Satellite image MAPS (Labels) for testing
  mypath='/Datasets/mass_roads_batch/test/map'# % data_type
  if not os.path.exists(mypath):
    print ("Data not found!")
  onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
  map_images_test = np.empty([len(onlyfiles),IMAGE_SIZE,IMAGE_SIZE])
  for n in range(0, len(onlyfiles)):
    map_images_test[n] = scipy.misc.imresize(cv2.cvtColor(cv2.imread(join(mypath, onlyfiles[n])), cv2.COLOR_RGB2GRAY),RESIZE_FACTOR)
    print ("reading test satellite map: %d of %d" % (n+1, len(onlyfiles)))

  # Satellite image inputs for training
  mypath='/Datasets/mass_roads_batch/train/sat'# % data_type
  if os.path.exists(mypath):
    print ("Data not found!")
  onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
  sat_images_train = np.empty([len(onlyfiles),IMAGE_SIZE,IMAGE_SIZE])
  for n in range(0, len(onlyfiles)):
    sat_images_train[n] = scipy.misc.imresize(cv2.cvtColor(cv2.imread(join(mypath, onlyfiles[n])), cv2.COLOR_RGB2GRAY),RESIZE_FACTOR)
    print ("reading training satellite image: %d of %d" % (n+1, len(onlyfiles)))
  
  # Satellite image MAPS (Labels) for training
  mypath='Datasets/mass_roads_batch/train/map'# % data_type
  if os.path.exists(mypath):
    print ("Data not found!")
  onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
  map_images_train = np.empty([len(onlyfiles),IMAGE_SIZE,IMAGE_SIZE])
  for n in range(0, len(onlyfiles)):
    map_images_train[n] = scipy.misc.imresize(cv2.cvtColor(cv2.imread(join(mypath, onlyfiles[n])), cv2.COLOR_RGB2GRAY),RESIZE_FACTOR)
    print ("reading training satellite image: %d of %d" % (n+1, len(onlyfiles)))
  
  train_images = sat_images_train
  train_labels = map_images_train
  test_images = sat_images_test
  test_labels = map_images_test

  train = DataSet(train_images, train_labels, dtype=dtype)
  test = DataSet(test_images, test_labels, dtype=dtype)

  print ("Finished importing data. Returning data to conv_net_test.py..")
  return base.DataSets(train=train, test=test)
