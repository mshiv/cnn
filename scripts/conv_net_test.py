import tensorflow as tf
import pdb
import numpy as np

import import_data_test

# Parameters
learning_rate = 0.0001
training_iters = 800
batch_size = 20
display_step = 10

# Network Parameters
n_input = 1500
n_classes = 2 
n_channels = 1
n_output = n_input * n_input * n_channels

dropout = 0.65 # Dropout, probability to keep units

# tf Graph input
with tf.name_scope('input'):
  x = tf.placeholder(tf.float32, [None, n_input, n_input, n_channels])
  y = tf.placeholder(tf.int64, [None, n_input, n_input, n_channels])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

with tf.name_scope('input_reshape'):
  image_shaped_input = tf.reshape(x, [-1, 150, 150, 1])
  tf.image_summary('input', image_shaped_input, 2)



# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, n_input, n_input, n_channels])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = tf.nn.local_response_normalization(conv1)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = tf.nn.local_response_normalization(conv2)
    conv2 = maxpool2d(conv2, k=3)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = tf.nn.local_response_normalization(conv3)
    conv3 = maxpool2d(conv3, k=1)

    # return conv3
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    return tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    # Output, class prediction
    # output = []
    # for i in xrange(2):
    #     # output.append(tf.nn.softmax(tf.add(tf.matmul(fc1, weights['out']), biases['out'])))
    #     output.append((tf.add(tf.matmul(fc1, weights['out']), biases['out'])))
    #
    # return output

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([250*250*128, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_output]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
#pdb.set_trace()
flattened_y = tf.cast(tf.reshape(y, [-1, n_output]), tf.float32)
# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, flattened_y))
tf.scalar_summary('cross_entropy/', loss)
#cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(temp_pred, temp_y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Evaluate model
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
temp_pred2 = tf.reshape(pred, [-1,n_input,n_input, n_channels])
correct_pred = tf.equal(tf.cast(y,tf.float32),tf.sub(temp_pred2,tf.cast(y,tf.float32)))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


print ("initializing all variables now...")
# Initializing the variables
init = tf.initialize_all_variables()
print ("FINISHED INITIALIZING all variables!!!")
print("launching the graph now:")
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    merged = tf.merge_all_summaries()
    summ = tf.train.SummaryWriter('/tmp/logdir/', sess.graph_def)
    step = 1
    #from tensorflow.contrib.learn.python.learn.datasets.scroll import scroll_data
    data = import_data_test.read_data_sets('Dataset')
    print "train images shape:", data.train.images.shape
    print "train labels shape:", data.train.labels.shape
    print "test images shape:", data.test.images.shape
    print "test labels shape:", data.test.labels.shape
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = data.train.next_batch(batch_size)
        # Run optimization op (backprop)
        #print ("running optimization op (backprop)...for step %d" % step)
        batch_x = batch_x.reshape(batch_size, n_input, n_input, n_channels)
        batch_y = batch_y.reshape(batch_size, n_input, n_input, n_channels)
        batch_y = np.int64(batch_y)
        #pdb.set_trace()
        #print("run the optimizer! provide feed_dict values as batch data and dropout: for step %d" % step)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            #pdb.set_trace()
            loss_val, acc = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss_val) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    # Calculate accuracy for test images
    test_x = data.test.images.reshape(-1,n_input, n_input, n_channels)
    test_y = data.test.labels.reshape(-1,n_input, n_input, n_channels)
    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_x,
                                      y: test_y,
                                      keep_prob: 1.})
