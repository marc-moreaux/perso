import numpy as np
import tensorflow as tf



class Alexnet(object):
  def __init__(self):
    self.net_data = np.load("bvlc_alexnet.npy", encoding="bytes").item()

  def get_graph(self):
    # x = tf.Variable(i)
    net_data = self.net_data
    mInput = tf.placeholder(tf.float32, shape=(1, 227, 227, 3))

    conv1W = tf.Variable(net_data["conv1"][0], trainable=False)
    conv1b = tf.Variable(net_data["conv1"][1], trainable=False)
    conv2W = tf.Variable(net_data["conv2"][0], trainable=False)
    conv2b = tf.Variable(net_data["conv2"][1], trainable=False)
    conv3W = tf.Variable(net_data["conv3"][0], trainable=False)
    conv3b = tf.Variable(net_data["conv3"][1], trainable=False)
    conv4W = tf.Variable(net_data["conv4"][0], trainable=False)
    conv4b = tf.Variable(net_data["conv4"][1], trainable=False)
    conv5W = tf.Variable(net_data["conv5"][0], trainable=False)
    conv5b = tf.Variable(net_data["conv5"][1], trainable=False)

    fc6W = tf.Variable(net_data["fc6"][0], trainable=False)
    fc6b = tf.Variable(net_data["fc6"][1], trainable=False)
    fc7W = tf.Variable(net_data["fc7"][0], trainable=False)
    fc7b = tf.Variable(net_data["fc7"][1], trainable=False)
    fc8W = tf.Variable(net_data["fc8"][0], trainable=False)
    fc8b = tf.Variable(net_data["fc8"][1], trainable=False)


    #conv1 - conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1_in = conv(mInput, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1 - lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    #maxpool1 - max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv2 - conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)

    #lrn2 - lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    #maxpool2 - max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3 - conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4 - conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    #conv5 - conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    #maxpool5 - max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #fc6 - fc(4096, name='fc6')
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

    #fc7 - fc(4096, name='fc7')
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

    #fc8 - fc(1000, relu=False, name='fc8')
    fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

    #prob - softmax(name='prob'))
    prob = tf.nn.softmax(fc8)

    self.mInput = mInput
    self.fc6  = fc6
    self.fc7  = fc7
    self.fc8  = fc8
    self.prob = prob


def conv(mInput, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
  '''From https://github.com/ethereon/caffe-tensorflow
  '''
  c_i = mInput.get_shape()[-1]
  assert c_i%group==0
  assert c_o%group==0
  convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
  
  if group==1:
      conv = convolve(mInput, kernel)
  else:
      input_groups = tf.split(3, group, mInput)
      kernel_groups = tf.split(3, group, kernel)
      output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
      conv = tf.concat(3, output_groups)
  return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())



