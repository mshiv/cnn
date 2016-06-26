#SYSTEM IMPORTS
import glob
import argparse
from os import listdir
from os.path import isfile, join

import tensorflow as tf
import numpy as np 

import matplotlib.pyplot as plt

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import imageflow


# Import Mass Roads/Buildings data

print "Setting up dataset.."
import import_data

massch = import_data.read_data_sets()																																																																																				
#massch.train_in, massch.test_in, massch.train_label, massch.test_label = import_data.read_data_sets()

# LINEAR MODEL
# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 5
display_step = 1

# tf Graph Input
x = tf.placeholder("float", [None, 6750000]) # data image of shape 1500*1500*3 = 6750000	
y = tf.placeholder("float", [None, 2]) # 2 classes of label - 0, 1 (0.0, 255.0 -> normalized!)

# Create model

# Set model weights
W = tf.Variable(tf.zeros([6750000,2]))
b = tf.Variable(tf.zeros([2]))
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																														
# Construct model
activation = tf.nn.softmax(tf.matmul(x, W) + b) #Softmax

# Minimize error using cross entropy
# Cross entropy
cost = -tf.reduce_sum(y*tf.log(activation)) 
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(massch.train.num_examples/batch_size)
        # Loop over all batch_size
        for i in range(total_batch):
            batch_xs, batch_ys = massch.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: massch.test.images, y: massch.test.labels})
	


'''
# CNN MODEL

# Parameters
learning_rate = 0.001
training_iters = 15
batch_size = 2
display_step = 1

# Network parameters

n_input = 6750000 # massch data input size (img shape: 1500*1500*3)
n_labels = 6750000 # massch data total map labels (0/1 non-road/road)
dropout = 0.75

# tf Graph input

x = tf.placeholder(tf.float32, shape = (None, n_input))
y = tf.placeholder(tf.float32, shape = (None, n_labels))
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create model

def conv2d(img, w, b):
	return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img,w, strides = [1, 1, 1, 1],
													padding = 'SAME'),b))

def max_pool(img, k): 
	return tf.nn.max_pool(img, ksize=[1,k,k,1], strides = [1,k,k,1], padding = 'SAME')

def conv_net(_X, _weights, _biases, _dropout):
	# Reshape input picture
	_X = tf.reshape(_X, shape=[-1,1500,1500,1])
	# Convolutional layer
	conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
	# Max Pool (down-sampling)
	conv1 = max_pool(conv1, k=2)
	# Apply Dropout
	conv1 = tf.nn.dropout(conv1, _dropout)
	#Convolution Layer
	conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
	# Max Pooling (down-sampling)
	conv2 = max_pool(conv2, k=2)
	# Apply Dropout
	conv2 = tf.nn.dropout(conv2, _dropout)
	# Fully connected layer
	# Reshape conv2 output to fit dense layer input
	dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]])
	# ReLu activation
	dense1 = tf.nn.relu(tf.add(tf.matmul(dense1,_weights['wd1']), _biases['bd1']))
	# Apply Dropout
	dense1 = tf.nn.dropout(dense1, _dropout) 
	# Output, class prediction
	out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
	return out

# Store layers weight and bias

weights = {
	# 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])), 
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])), 
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 6750000])), 
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([6750000, n_labels])) 
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([6750000																																																																																																																								])),
    'out': tf.Variable(tf.random_normal([n_labels]))
}

# Construct model

pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = massch.train.next_batch(batch_size)
        batch_ys = np.reshape(batch_xs, (-1, 1500,1500,3))
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"
    # Calculate accuracy for 256 mnist test images
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: massch.test.images[:3], 
                                                             y: massch.test.labels[:3], 
                                                             keep_prob: 1.})
'''