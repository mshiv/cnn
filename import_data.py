#SYSTEM IMPORTS

print "Importing dependencies..."

import glob
import os
import argparse
import gzip
import tarfile
import urllib
from os import listdir
from os.path import isfile, join

import tensorflow as tf
import numpy as np 
import re

import matplotlib.pyplot as plt

import cv2
from imageflow import convert_images
#-------------------------------------------------

print "Locating dataset..."
#Read images into arrays
#Sat images = Input data

class DataSet(object):
  
  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns*depth] (assuming depth == 3)
      assert images.shape[3] == 3
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2]*images.shape[3])
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(np.float32)
      images = np.multiply(images, 1.0 / 255.0)

      assert labels.shape[3] == 3
      labels = labels.reshape(labels.shape[0],
      						  labels.shape[1] * labels.shape[2] * labels.shape[3])
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
      fake_image = [1.0 for _ in xrange(6750000)]
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

def read_data_sets():
  class DataSets(object):
    pass
  data_sets = DataSets()
  # Satellite image inputs for testing
  mypath='mass_roads_batch/test/sat'# %data_type (when looped - change)
  if os.path.exists(mypath):
  	print "Found datasets \n Loading files..."
  else:
	print "Data not found!"
  onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
  sat_images_test = np.empty([len(onlyfiles),1500,1500,3])
  for n in range(0, len(onlyfiles)):
  	sat_images_test[n] = cv2.imread(join(mypath, onlyfiles[n]))
  	print ("reading test satellite image: %d of %d" % (n+1, len(onlyfiles)))
  	#sat_images.reshape([len(onlyfiles), sat_images[n].shape[0],sat_images[n].shape[1],sat_images[n].shape[2]])

  # Satellite image MAPS (Labels) for testing
  mypath='mass_roads_batch/test/map'# % data_type
  if os.path.exists(mypath):
  	print "Found datasets \n Loading files..."
  else:
  	print "Data not found!"
  onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
  map_images_test = np.empty([len(onlyfiles),1500,1500,3])
  for n in range(0, len(onlyfiles)):
  	map_images_test[n] = cv2.imread(join(mypath, onlyfiles[n]))
  	print ("reading test satellite map: %d of %d" % (n+1, len(onlyfiles)))

  # Satellite image inputs for training
  mypath='mass_roads_batch/train/sat'# % data_type
  if os.path.exists(mypath):
  	print "Found datasets \n Loading files..."
  else:
  	print "Data not found!"
  onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
  sat_images_train = np.empty([len(onlyfiles),1500,1500,3])
  for n in range(0, len(onlyfiles)):
  	sat_images_train[n] = cv2.imread(join(mypath, onlyfiles[n]))
  	print ("reading training satellite image: %d of %d" % (n+1, len(onlyfiles)))
	#sat_images.reshape([len(onlyfiles), sat_images[n].shape[0],sat_images[n].shape[1],sat_images[n].shape[2]])
	
  # Satellite image MAPS (Labels) for training
  mypath='mass_roads_batch/train/map'# % data_type
  if os.path.exists(mypath):
	print "Found datasets \n Loading files..."
  else:
	print "Data not found!"
  onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
  map_images_train = np.empty([len(onlyfiles),1500,1500,3])
  for n in range(0, len(onlyfiles)):
	map_images_train[n] = cv2.imread(join(mypath, onlyfiles[n]))
	print ("reading training satellite image: %d of %d" % (n+1, len(onlyfiles)))
  
  train_images = sat_images_train
  train_labels = map_images_train
  test_images = sat_images_test
  test_labels = map_images_test

  data_sets.train = DataSet(train_images, train_labels)
  data_sets.test = DataSet(test_images, test_labels)
  
  return data_sets


'''
mypath='data/train/sat'# % data_type
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
sat_images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  sat_images[n] = cv2.imread( join(mypath,onlyfiles[n]) )

print sat_images[1].shape

plt.figure(1)
for i in range(1, len(onlyfiles)+1):
	plt.subplot(2,7,i)
	sat_plot = plt.imshow(sat_images[i-1])

	
#map images = Labels (class scores)

#for data_type in ['train', 'test', 'valid']:
mypath='data/train/map'
onlyfilesm = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
map_images = np.empty(len(onlyfilesm), dtype=object)
for n in range(0, len(onlyfilesm)):
  map_images[n] = cv2.imread(join(mypath,onlyfilesm[n]) )

print map_images.shape

plt.figure(2)
for i in range(1, len(onlyfilesm)+1):
	plt.subplot(2,7,i)
	map_plot = plt.imshow(map_images[i-1])

#Get file name ID of samples/labels in dataset
bldg_fns = glob.glob('data/train/map/*.tif')
#bldg_fns += glob.glob('data/test/map/*/tif')
#bldg_fns += glob.glob('data/valid/map/*.tif')

def get_ids(fns):
	return [re.search('/([0-9]+_[0-9]+)', fn).groups()[0] for fn in fns]

bldg_fns = get_ids(bldg_fns)
print bldg_fns

plt.show()
'''



#-----------------------------------------------------------------------------------------------------------------------------------------------------------
#print sat_images[0].shape[0]
#print sat_images

#print (np.abs(np.sum(sat_images[1] - sat_imagesl[1])))


#plt.figure(2)
#rgb = np.fliplr(sat_images[1].reshape(-1,3)).reshape(sat_images[1].shape)
#plt.imshow(rgb)

#plt.imshow(cv2.cvtColor(sat_images[1],cv2.COLOR_BGR2RGB))

#plt.imshow(sat_images[1])
#scipy.misc.imshow(rgb)

#print sat_imagesl[0]
#print sat_images[0]
