

# AlexNet implementation + weights in TensorFlow:
# http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/


# Alexnet classes:
# http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/caffe_classes.py

# Alexnet Images:
# ImageNet LSVRC-2010

################################################################################
#Michael Guerzhoy, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
from os.path import isfile, join
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.ndimage import filters
import urllib
from numpy import random
from PIL import Image
import numpy as np


import tensorflow as tf

from caffe_classes import class_names

train_x = np.zeros((1, 227,227,3)).astype(np.float32)
train_y = np.zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]



################################################################################
#Read Image

# img = Image.open('/home/mmoreaux/rosbag/goodBags1/image12345_2016-02-29-16-35-56/frame0326.jpg')
# img = img.resize((1500,1500))

# train_x = np.zeros((10, 227,227,3)).astype(np.float32)
# i=0
# for x0 in range(600,1000, 100):
#   for y0 in range(800,1100, 100):
#     crop = img.crop((x0, y0, x0+227, y0+227))
#     imgplot = plt.imshow(crop)
#     train_x[i%10,:,:,:] = crop
#     i+=1
#     # plt.show()


# img_paths=[
# "/home/mmoreaux/Téléchargements/images.jpg",
# "/home/mmoreaux/Téléchargements/téléchargement.jpg",
# "/home/mmoreaux/Téléchargements/mug.jpg",
# "/home/mmoreaux/Téléchargements/mug2.jpg",
# "/home/mmoreaux/Téléchargements/mug3.jpg",
# "/home/mmoreaux/Téléchargements/images4.jpg",
# "/home/mmoreaux/Téléchargements/mug5.jpg",
# "/home/mmoreaux/Téléchargements/mug6.jpg",
# "/home/mmoreaux/Téléchargements/mug7.jpg",
# ]


pngPath = "/home/mmoreaux/work/perso/implementation/images_test"
# pngPath = "/home/mmoreaux/rosbag/goodBags1/image12345_2016-02-29-16-33-26/"
img_paths = [f for f in os.listdir(pngPath) if os.path.isfile(os.path.join(pngPath, f))]
img_paths = [os.path.join(pngPath, f) for f in img_paths]

# import my_rosbags
# img_paths = my_rosbags.imagesOf("goodBags1", "12345")
# img_paths


def subSampleImg(path):
  subSamples = []
  img = Image.open(path)

  crop = img.resize([227,227], Image.ANTIALIAS)
  crop = crop-np.mean(crop)
  subSamples.append(crop)

  img.thumbnail((227*3, 227*3), Image.ANTIALIAS)
  for x0 in range(0,227*3-227, 100):
    for y0 in range(0,227*3-227, 100):
      crop = img.crop((x0, y0, x0+227, y0+227))
      crop = crop-np.mean(crop)
      subSamples.append(crop)

  return subSamples

  crop = img.crop((0, 0, 227, 227))
  crop = crop-np.mean(crop)
  subSamples.append(crop)

  zoom = img.resize((img.size[0]*2,img.size[1]*2))
  for x0 in range(0,img.size[0]*2-227, 100):
    for y0 in range(0,img.size[1]*2-227, 100):
      crop = zoom.crop((x0, y0, x0+227, y0+227))
      crop = crop-np.mean(crop)
      subSamples.append(crop)
  
  zoom = img.resize((img.size[0]*4,img.size[1]*4))
  for x0 in range(0,img.size[0]*4-227, 150):
    for y0 in range(0,img.size[1]*4-227, 150):
      crop = zoom.crop((x0, y0, x0+227, y0+227))
      crop = crop-np.mean(crop)
      subSamples.append(crop)
  
  return subSamples









# x_dummy = (random.random((1,)+ xdim)/255.).astype(float32)
# image = x_dummy.copy()
# image[0,:,:,:] = (imread("test_mug.jpg")[:,:,:3]).astype(float32)
# image = image-mean(image)

################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))


net_data = load("bvlc_alexnet.npy", encoding="bytes").item()

graph = tf.Graph()
with graph.as_default():

  def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
      '''From https://github.com/ethereon/caffe-tensorflow
      '''
      c_i = input.get_shape()[-1]
      assert c_i%group==0
      assert c_o%group==0
      convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
      
      
      if group==1:
          conv = convolve(input, kernel)
      else:
          input_groups = tf.split(3, group, input)
          kernel_groups = tf.split(3, group, kernel)
          output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
          conv = tf.concat(3, output_groups)
      return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
  # x = tf.Variable(i)
  mInput = tf.placeholder(tf.float32, shape=(1, 227, 227, 3))
  
  #conv1
  #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
  k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
  conv1W = tf.Variable(net_data["conv1"][0])
  conv1b = tf.Variable(net_data["conv1"][1])
  conv1_in = conv(mInput, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
  conv1 = tf.nn.relu(conv1_in)
  
  #lrn1
  #lrn(2, 2e-05, 0.75, name='norm1')
  radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
  lrn1 = tf.nn.local_response_normalization(conv1,
                                            depth_radius=radius,
                                            alpha=alpha,
                                            beta=beta,
                                            bias=bias)
  
  #maxpool1
  #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
  k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
  maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
  
  
  #conv2
  #conv(5, 5, 256, 1, 1, group=2, name='conv2')
  k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
  conv2W = tf.Variable(net_data["conv2"][0])
  conv2b = tf.Variable(net_data["conv2"][1])
  conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
  conv2 = tf.nn.relu(conv2_in)
  
  
  #lrn2
  #lrn(2, 2e-05, 0.75, name='norm2')
  radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
  lrn2 = tf.nn.local_response_normalization(conv2,
                                            depth_radius=radius,
                                            alpha=alpha,
                                            beta=beta,
                                            bias=bias)
  
  #maxpool2
  #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
  k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
  maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
  
  #conv3
  #conv(3, 3, 384, 1, 1, name='conv3')
  k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
  conv3W = tf.Variable(net_data["conv3"][0])
  conv3b = tf.Variable(net_data["conv3"][1])
  conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
  conv3 = tf.nn.relu(conv3_in)
  
  #conv4
  #conv(3, 3, 384, 1, 1, group=2, name='conv4')
  k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
  conv4W = tf.Variable(net_data["conv4"][0])
  conv4b = tf.Variable(net_data["conv4"][1])
  conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
  conv4 = tf.nn.relu(conv4_in)
  
  #conv5
  #conv(3, 3, 256, 1, 1, group=2, name='conv5')
  k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
  conv5W = tf.Variable(net_data["conv5"][0])
  conv5b = tf.Variable(net_data["conv5"][1])
  conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
  conv5 = tf.nn.relu(conv5_in)
  
  #maxpool5
  #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
  k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
  maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
  
  #fc6
  #fc(4096, name='fc6')
  fc6W = tf.Variable(net_data["fc6"][0])
  fc6b = tf.Variable(net_data["fc6"][1])
  fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
  
  #fc7
  #fc(4096, name='fc7')
  fc7W = tf.Variable(net_data["fc7"][0])
  fc7b = tf.Variable(net_data["fc7"][1])
  fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
  
  #fc8
  #fc(1000, relu=False, name='fc8')
  fc8W = tf.Variable(net_data["fc8"][0])
  fc8b = tf.Variable(net_data["fc8"][1])
  fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
  
  #prob
  #softmax(name='prob'))
  prob = tf.nn.softmax(fc8)



with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  for i in range(len(img_paths)):
    for img in subSampleImg(img_paths[i]):
      image = np.zeros((1, 227,227,3)).astype(np.float32)
      image[0,:,:,:] = img
      feed_dict = {mInput: image}
      output = session.run(prob, feed_dict=feed_dict)
    
      # show coffe mug prediction (504)
      if(output[0,504] > 0.01):
        plt.imshow(img)
        plt.savefig("resultats/coffee_"+str(output[0,504])+".jpg")

      if(output[0,968] > 0.01):
        plt.imshow(img)
        plt.savefig("resultats/cup_"+str(output[0,504])+".jpg")
  
      if(output[0,487] > 0.01):
        plt.imshow(img)
        plt.savefig("resultats/phone"+str(output[0,487])+".jpg")




################################################################################

#Output:

# inds = argsort(output)[0,:]
# for i in range(10):
#   print(class_names[inds[-1-i]], output[0, inds[-1-i]])


