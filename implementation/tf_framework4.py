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
import KTH_reader



# Initalize alexnet, LSTM & batches
mAlexnet = tf_alexnet2.Alexnet(15)
mLSTMs = tf_lstm.MY_LSTM()
mKTH = KTH_reader.KTH_videos()
batch_gen = mKTH.train_batch_generator(batch_size=20)



nbCells = 64
nbInputs = 4096
nbOutputs = 101
batchSize = 1
nbFrames = 15
nbUnrollings = 10





graph = tf.Graph()
with graph.as_default():
  mAlexnet.get_graph()
  ''' Return the LSTM composed by <nbCells> and given the tf input <tensor_input>'''
  # Define variables 
  # - Weights & bias on cells
  wCells = tf.Variable(tf.truncated_normal([nbInputs+nbCells, nbCells*4], -0.1, 0.1))
  bCells = tf.Variable(tf.zeros([1, nbCells*4]))
  # - Memory of previous cell's values
  saved_output = tf.Variable(tf.truncated_normal([batchSize, nbCells], -0.1, 0.1), trainable=False)
  saved_state  = tf.Variable(tf.truncated_normal([batchSize, nbCells], -0.1, 0.1), trainable=False)
  # - Output classifier
  wClassif = tf.Variable(tf.truncated_normal([nbCells, nbOutputs], -0.1, 0.1))
  bClassif = tf.Variable(tf.zeros([nbOutputs]))
  # - Label placeholder
  train_labels = tf.placeholder(tf.float32, shape=[nbFrames,nbOutputs])
  
  
  def lstm_cell(i, o, state):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    # previous state and the gates."""
    allVar      = tf.matmul(  tf.concat(1,[i,o]), wCells ) + bCells
    input_gate  = tf.sigmoid( tf.slice(allVar, [0,nbCells*0], [-1, nbCells]) )
    forget_gate = tf.sigmoid( tf.slice(allVar, [0,nbCells*1], [-1, nbCells]) )
    output_gate = tf.sigmoid( tf.slice(allVar, [0,nbCells*2], [-1, nbCells]) )
    update      = tf.tanh(    tf.slice(allVar, [0,nbCells*3], [-1, nbCells]) )  
    state = forget_gate * state + input_gate * update
    return output_gate * tf.tanh(state), state
  
  
  
  # - LSTM unrolling's placeholder
  # LSTM_inputs = list()
  # for _ in range(nbFrames):
  #   LSTM_inputs.append(
  #     tf.placeholder(tf.float32, shape=[batchSize,nbInputs]))
  
  # Propagate all the images into the LSTM cells
  # outputs = list()
  # for i in LSTM_inputs:
  #   saved_output, saved_state = lstm_cell(mAlexnet.fc6[i], saved_output, saved_state)
  #   outputs.append(saved_output)
  # # Outputs list to tf_variable
  # outputs = tf.concat(0, outputs)
  
  
  
  LSTM_inputs = mAlexnet.fc6
  outputs = list()
  for idx in range(nbFrames):
    saved_output, saved_state = lstm_cell(
                                  tf.slice(mAlexnet.fc6, [idx,0], [idx,-1]),  ## NEED TO GET VECTOR OUT OF MATRICE
                                  saved_output, saved_state)
  
  
  # Classifier.    
  logits = tf.nn.xw_plus_b(outputs, wClassif, bClassif)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
      logits, train_labels))
  
  # Predictions.
  train_prediction = tf.nn.softmax(logits)
  
  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    50.0, global_step, 5000, 0.8, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(
    zip(gradients, v), global_step=global_step)








mKTH = KTH_reader.KTH_videos()
batch_gen = mKTH.train_batch_generator(batch_size=10)
num_steps = 150000

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  cumulated_loss = 0
  for step in range(num_steps):
    ########################################
    ###  Create input variables to network
    ########################################
    batch = next(batch_gen)
    feed_dict = dict()
    feed_dict[mAlexnet.mInput] = batch[0][0][0:15]
    labels = np.zeros((15,101))
    labels[:,batch[1][0,2]] = 1
    # label_idxs = batch[1][0:15,2]
    # labels[np.arange(15),label_idxs] = 1
    # feed_dict[mLSTMs.train_labels] = labels
    
    ########################################
    ###  Run predictions 
    ### & print debug info every 1000 steps
    ########################################
    out = session.run(outputs, feed_dict=feed_dict)
    print(out)

    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    cumulated_loss += l
    if step % 500 == 0 :#&& step > 0:
      print_training_info(cumulated_loss, step, lr, batch, predictions)
      cumulated_loss = 0
      


