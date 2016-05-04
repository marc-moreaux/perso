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


graph = tf.Graph()

# Initalize alexnet & LSTM
mAlexnet = tf_alexnet2.Alexnet()
mLSTMs = tf_lstm.MY_LSTM()


with graph.as_default():
  mAlexnet.get_graph()
  mLSTMs.get_graph(mAlexnet.fc6)


# with tf.Session(graph=graph) as session:
#   tf.initialize_all_variables().run()
  
#   image = np.zeros((1, 227,227,3)).astype(np.float32)
#   image[0,:,:,:] = img
#   label = np.zeros((1,101))
#   label[0,10] = 1
#   feed_dict = {mAlexnet.mInput: image, mLSTMs.train_labels: label}
#   output = session.run(mLSTMs.myVar, feed_dict=feed_dict)
#   print(output)


mUCFVideos = UCF_reader.UCF_videos()
batchGenerator = mUCFVideos.next_batch(batchSize=1)
# y = next(b)


with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batch = next(batchGenerator)
    feed_dict = dict()
    feed_dict[train_data] = batch
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print(
        'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches)[1:])
      print('Minibatch perplexity: %.2f' % float(
        np.exp(logprob(predictions, labels))))
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = sample(random_distribution())
          sentence = characters(feed)[0]
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input: feed})
            feed = sample(prediction)
            sentence += characters(feed)[0]
          print(sentence)
        print('=' * 80)
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size):
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, b[1])
      print('Validation set perplexity: %.2f' % float(np.exp(
        valid_logprob / valid_size)))
