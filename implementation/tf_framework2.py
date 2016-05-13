import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cbook as cbook
import time
import scipy
from PIL import Image
import tensorflow as tf

# own stuff
import tf_alexnet2
import tf_lstm
import UCF_reader


# img = Image.open("/home/mmoreaux/work/perso/implementation/images_test/IMG_20160420_164136.jpg")
# img = img.resize([227,227], Image.ANTIALIAS)


# Initialze batch readers
mUCFVideos = UCF_reader.UCF_videos()
batchGenerator = mUCFVideos.next_batch(batchSize=1)



# Initalize alexnet & LSTM
graph = tf.Graph()
mAlexnet = tf_alexnet2.Alexnet()

with graph.as_default():
  mAlexnet.get_graph()
  
  # variables
  wEnd   = tf.Variable(tf.truncated_normal([4096, 101], -0.1, 0.1))
  bEnd   = tf.Variable(tf.zeros([101]))
  labels = tf.placeholder(tf.float32, shape=[1,101])
  
  # Predictions.
  train_prediction = tf.nn.xw_plus_b(mAlexnet.fc6, wEnd, bEnd)
  train_prediction = tf.nn.softmax(train_prediction)
  
  # Loss
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_prediction, labels))
  
  # Optimizer.
  # optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
  optimizer = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)


def print_training_info(cumulated_loss, step):
    cumulated_loss = cumulated_loss / 500
    # The mean loss is an estimate of the loss over the last few batches.
    print('Average loss at step %d: %f' % (step, cumulated_loss))





num_steps = 150000

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  cumulated_loss = 0
  for step in range(num_steps):
    ########################################
    ###  Create input variables to network
    ########################################
    batch = next(batchGenerator)
    feed_dict = dict()
    feed_dict[mAlexnet.mInput] = batch[0][0][0:1]
    label_vec = np.zeros((1,101))
    label_vec[0,batch[1]-1] = 1
    feed_dict[labels] = label_vec
    
    ########################################
    ###  Run predictions 
    ### & print debug info every 1000 steps
    ########################################
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    cumulated_loss += l
    if step % 500 == 0 and step > 0:
      print_training_info(cumulated_loss, step)
      cumulated_loss = 0
      








