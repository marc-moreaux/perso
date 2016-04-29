import numpy as np
import tensorflow as tf







class MY_LSTM:
  def __init__(self):
    # init variables
    self.nbCells = 64
    self.nbInputs = 4096
    self.nbOutputs = 101
    self.batchSize = 20
    self.nbFrames = 15
    self.nbUnrollings = 10
  
  def lstm_cell(self, i, o, state):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    # previous state and the gates."""
    allVar      = tf.matmul(  tf.concat(1,[i,o]), self.wCells ) + self.bCells
    input_gate  = tf.sigmoid( tf.slice(allVar, [0,self.nbCells*0], [-1, self.nbCells]) )
    forget_gate = tf.sigmoid( tf.slice(allVar, [0,self.nbCells*1], [-1, self.nbCells]) )
    output_gate = tf.sigmoid( tf.slice(allVar, [0,self.nbCells*2], [-1, self.nbCells]) )
    update      = tf.tanh(    tf.slice(allVar, [0,self.nbCells*3], [-1, self.nbCells]) )
    
    state = forget_gate * state + input_gate * update
    return output_gate * tf.tanh(state), state
  
  def get_graph(self):
    #
    # Define variables 
    #  - weights & bias on cells
    #  - memory of previous cell's values
    #  - output classifier
    #
    self.wCells = tf.Variable(tf.truncated_normal([self.nbInputs+self.nbCells, self.nbCells*4], -0.1, 0.1))
    self.bCells = tf.Variable(tf.zeros([1, self.nbCells*4]))
    saved_output = tf.Variable(tf.zeros([self.batchSize, self.nbCells]), trainable=False)
    saved_state  = tf.Variable(tf.zeros([self.batchSize, self.nbCells]), trainable=False)
    wClassif = tf.Variable(tf.truncated_normal([self.nbCells, self.nbOutputs], -0.1, 0.1))
    bClassif = tf.Variable(tf.zeros([self.nbOutputs]))
    
    
    # Feed <nbInputs> inputs to <nbCells> LSTM cells
    # which have <self.nbFrames> consecutive LSTMs
    LSTM_inputs = list()
    for _ in range(self.nbFrames):
      LSTM_inputs.append(
        tf.placeholder(tf.float32, shape=[self.batchSize,self.nbInputs]))
    train_labels = tf.placeholder(tf.float32, shape=[self.batchSize,self.nbOutputs])
    
    # Propagate images into LSTM cells
    for fc6 in LSTM_inputs:
      saved_output, saved_state = self.lstm_cell(fc6, saved_output, saved_state)
    
    # State saving across unrollings. 
    # control_dependencies => must be true to continue
    # with tf.control_dependencies([saved_output.assign(output),
    #                               saved_state.assign(state)]):
    
    # Classifier.
    logits = tf.nn.xw_plus_b(saved_output, wClassif, bClassif)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
        logits, tf.concat(0, train_labels)))
    
    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
      10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
      zip(gradients, v), global_step=global_step)
    
    # Predictions.
    train_prediction = tf.nn.softmax(logits)





