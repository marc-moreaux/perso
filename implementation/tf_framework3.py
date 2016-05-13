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
mAlexnet = tf_alexnet2.Alexnet(10)
mLSTMs = tf_lstm.MY_LSTM()
mKTH = KTH_reader.KTH_videos()
batch_gen = mKTH.train_batch_generator(batch_size=20)



graph = tf.Graph()
with graph.as_default():
  mAlexnet.get_graph()
  # mLSTMs.get_graph(mAlexnet.fc6)



def print_training_info(cumulated_loss, step, lr, batches, predictions):
    cumulated_loss = cumulated_loss / 500
    # The mean loss is an estimate of the loss over the last few batches.
    print('Average loss at step %d: %f learning rate: %f' % (step, cumulated_loss, lr))
    #
    # Not enought elements in a batch to do this 
    # Could be done with few iterations
    #
    # labels = np.concatenate(list(batches)[1:])
    # print('Minibatch perplexity: %.2f' 
    #       % float(np.exp(logprob(predictions, labels))))
    ########################################
    ###  Measure validation set perplexity.
    ########################################
    # reset_sample_state.run()
    # valid_logprob = 0
    # for _ in range(valid_size):
    #   b = valid_batches.next()
    #   predictions = sample_prediction.eval({sample_input: b[0]})
    #   valid_logprob = valid_logprob + logprob(predictions, b[1])
    # print('Validation set perplexity: %.2f' 
    #       % float(np.exp(valid_logprob / valid_size)))





batch_gen = mKTH.train_batch_generator(batch_size=20)
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
    feed_dict[mAlexnet.mInput] = batch[0][0][0:10]
    label_vec = np.zeros((1,101))
    label_vec[0,batch[1][0][2]-1] = 1
    # feed_dict[mLSTMs.train_labels] = label_vec
    
    ########################################
    ###  Run predictions 
    ### & print debug info every 1000 steps
    ########################################
    out = session.run(mAlexnet.fc6, feed_dict=feed_dict)
    print(out)
    
    _, l, predictions, lr = session.run([mLSTMs.optimizer, mLSTMs.loss, mLSTMs.train_prediction, mLSTMs.learning_rate], feed_dict=feed_dict)
    cumulated_loss += l
    if step % 500 == 0 :#&& step > 0:
      print_training_info(cumulated_loss, step, lr, batch, predictions)
      cumulated_loss = 0
      


























# with tf.Session(graph=graph) as session:
#   tf.initialize_all_variables().run()
#   print('Initialized')
#   mean_loss = 0
#   for step in range(num_steps):
#     batch = next(batchGenerator)
#     feed_dict = dict()
#     feed_dict[train_data] = batch # feed AlexNet with inputs
#     _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
#     mean_loss += l
#     if step % summary_frequency == 0:
#       if step > 0:
#         mean_loss = mean_loss / summary_frequency
#       # The mean loss is an estimate of the loss over the last few batches.
#       print(
#         'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
#       mean_loss = 0
#       labels = np.concatenate(list(batches)[1:])
#       print('Minibatch perplexity: %.2f' % float(
#         np.exp(logprob(predictions, labels))))
#       if step % (summary_frequency * 10) == 0:
#         # Generate some samples.
#         print('=' * 80)
#         for _ in range(5):
#           feed = sample(random_distribution())
#           sentence = characters(feed)[0]
#           reset_sample_state.run()
#           for _ in range(79):
#             prediction = sample_prediction.eval({sample_input: feed})
#             feed = sample(prediction)
#             sentence += characters(feed)[0]
#           print(sentence)
#         print('=' * 80)
#       # Measure validation set perplexity.
#       reset_sample_state.run()
#       valid_logprob = 0
#       for _ in range(valid_size):
#         b = valid_batches.next()
#         predictions = sample_prediction.eval({sample_input: b[0]})
#         valid_logprob = valid_logprob + logprob(predictions, b[1])
#       print('Validation set perplexity: %.2f' % float(np.exp(
#         valid_logprob / valid_size)))
