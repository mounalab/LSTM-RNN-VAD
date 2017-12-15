from __future__ import unicode_literals

import glob
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import csv
import time
import json
import yaml
import logging

from datetime import datetime
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import BasicLSTMCell, DropoutWrapper, MultiRNNCell, GRUCell

plt.style.use('ggplot')

class VADModel(object):
  """ Building the Recurrent Neural Network model for Voice Activity Detection
  """

  @classmethod
  def build(cls, 
    param_dir):
    """
    Restore a previously trained model
    """
    with open(cls._parameters_file(param_dir)) as f:
      parameters = json.load(f)

      # Encapsulate training parameters
      training_parameters = TrainingParameters(parameters["training_epochs"])

      # Encapsulate model hyperparameters
      model_parameters = ModelParameters(
        parameters["learning_rate"],
        parameters["momentum"],
        parameters["model"],
        parameters["input_keep_probability"],
        parameters["output_keep_probability"],
        parameters["sequence_length"],
        parameters["input_dimension"],
        parameters["batch_size"], 
        parameters["state_size"], 
        parameters["n_layers"],
        parameters["n_classes"])

      # Encapsulate directories name
      directories = Directories(parameters["log_dir"],
        parameters["checkpoint_dir"])

      model = cls(
        model_parameters,
        training_parameters,
        directories)

    return model

  @classmethod
  def restore(cls, 
    session, 
    param_dir):
    """
    Restore a previously trained model and its session
    """
    with open(cls._parameters_file(param_dir)) as f:
      parameters = json.load(f)

      # Encapsulate training parameters
      training_parameters = TrainingParameters(parameters["training_epochs"])

      # Encapsulate model hyperparameters
      model_parameters = ModelParameters(
        parameters["learning_rate"],
        parameters["momentum"],
        parameters["model"],
        parameters["input_keep_probability"],
        parameters["output_keep_probability"],
        parameters["sequence_length"],
        parameters["input_dimension"],
        parameters["batch_size"], 
        parameters["state_size"], 
        parameters["n_layers"],
        parameters["n_classes"])

      # Encapsulate directories name
      directories = Directories(parameters["log_dir"],
        parameters["checkpoint_dir"])

      model = cls(
        model_parameters,
        training_parameters,
        directories)

      # Load the saved meta graph and restore variables
      checkpoint_file = tf.train.latest_checkpoint(directories.checkpoint_dir)
      print("restoring graph from {} ...".format(checkpoint_file))
      # Restore an empty computational graph
      #saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
      
      # Restore an already existing graph
      saver = tf.train.Saver()
      saver.restore(session, checkpoint_file)

    return model

  @staticmethod
  def _parameters_file(param_dir):
    return os.path.join(param_dir, "parameters.json")

  @staticmethod
  def _model_file(model_dir):
    return os.path.join(model_directory, "model")


  def __init__(self,
    model_parameters,
    training_parameters,
    directories, 
    **kwargs):

    """ Initialization of the RNN Model as TensorFlow computational graph
    """

    self.model_parameters = model_parameters
    self.training_parameters = training_parameters
    self.directories = directories

    # Define model hyperparameters Tensors
    with tf.name_scope("Parameters"):
      self.learning_rate = tf.placeholder(tf.float32, 
        name="learning_rate")
      self.momentum = tf.placeholder(tf.float32, 
        name="momentum")
      self.input_keep_probability = tf.placeholder(tf.float32, 
        name="input_keep_probability")
      self.output_keep_probability = tf.placeholder(tf.float32, 
        name="output_keep_probability")

    # Define input, output and initialization Tensors
    with tf.name_scope("Input"):
      self.inputs = tf.placeholder("float", [None, 
        self.model_parameters.sequence_length, 
        self.model_parameters.input_dimension], 
        name='input_placeholder')

      self.targets = tf.placeholder("float", [None, 
        self.model_parameters.sequence_length, 
        self.model_parameters.n_classes], 
        name='labels_placeholder')

      self.init = tf.placeholder(tf.float32, shape=[None, 
        self.model_parameters.state_size], 
        name="init")

    # Define the TensorFlow RNN computational graph
    with tf.name_scope("RNN"):
      cells = []

      # Define the layers
      for _ in range(self.model_parameters.n_layers):
        if self.model_parameters.model == 'rnn':
          cell = BasicRNNCell(self.model_parameters.state_size)
        elif self.model_parameters.model == 'gru':
          cell = GRUCell(self.model_parameters.state_size)
        elif self.model_parameters.model == 'lstm':
          cell = BasicLSTMCell(self.model_parameters.state_size, state_is_tuple=True)
        elif self.model_parameters.model == 'nas':
          cell = NASCell(self.model_parameters.state_size)
        else:
          raise Exception("model type not supported: {}".format(self.model_parameters.model))

        if (self.model_parameters.output_keep_probability < 1.0 
          or self.model_parameters.input_keep_probability < 1.0):

          if self.model_parameters.output_keep_probability < 1.0 :
            cell = DropoutWrapper(cell,
              output_keep_prob=self.output_keep_probability)

          if self.model_parameters.input_keep_probability < 1.0 :
            cell = DropoutWrapper(cell,
              input_keep_prob=self.input_keep_probability)

        cells.append(cell)
      cell = MultiRNNCell(cells)

      # Simulate time steps and get RNN cell output
      self.outputs, self.next_state = tf.nn.dynamic_rnn(cell, self.inputs, dtype = tf.float32)


    # Define cost Tensors
    with tf.name_scope("Cost"):

      # Flatten to apply same weights to all time steps
      self.flattened_outputs = tf.reshape(self.outputs, [-1, 
        self.model_parameters.state_size], 
        name="flattened_outputs")

      self.softmax_w = tf.Variable(tf.truncated_normal([
        self.model_parameters.state_size, 
        self.model_parameters.n_classes], stddev=0.01), 
        name="softmax_weights")

      self.softmax_b = tf.Variable(tf.constant(0.1, shape=[self.model_parameters.n_classes]), 
        name="softmax_biases")

      # Softmax activation layer, using RNN inner loop last output
      # logits and labels must have the same shape [batch_size, num_classes]
      self.logits = tf.matmul(self.flattened_outputs, self.softmax_w) + self.softmax_b
      self.unshaped_predictions = tf.nn.softmax(self.logits, 
        name="unshaped_predictions")

      tf.summary.histogram('logits', self.logits)

      # Return to the initial predictions shape
      self.predictions = tf.reshape(self.unshaped_predictions, 
        [-1, self.model_parameters.sequence_length, 
        self.model_parameters.n_classes], 
        name="predictions")

      self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.targets * tf.log(self.predictions), 
        reduction_indices=[2]))

      # Get the most likely label for each input
      self.label_prediction = tf.argmax(self.predictions,2, 
        name="label_predictions")

      # Compare predictions to labels
      self.correct_prediction = tf.equal(tf.argmax(self.predictions,2), tf.argmax(self.targets,2), 
        name="correct_predictions")
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), 
        name="accuracy")


    # Define Training Tensors
    with tf.name_scope("Train"):
      #self.validation_perplexity = tf.Variable(dtype=tf.float32, initial_value=float("inf"), 
        #trainable=False,
        #name="validation_perplexity")

      #self.validation_accuracy = tf.Variable(dtype=tf.float32, initial_value=float("inf"), 
        #trainable=False,
        #name="validation_accuracy")

      #tf.scalar_summary(self.validation_perplexity.op.name, self.validation_perplexity)
      #tf.scalar_summary(self.validation_accuracy.op.name, self.validation_accuracy)

      #self.training_epoch_perplexity = tf.Variable(dtype=tf.float32, initial_value=float("inf"), 
        #trainable=False,
        #name="training_epoch_perplexity")

      #self.training_epoch_accuracy = tf.Variable(dtype=tf.float32, initial_value=float("inf"), 
        #trainable=False,
        #name="training_epoch_accuracy")

      #tf.scalar_summary(self.training_epoch_perplexity.op.name, self.training_epoch_perplexity)
      #tf.scalar_summary(self.training_epoch_accuracy.op.name, self.training_epoch_accuracy)

      #self.iteration = tf.Variable(0, dtype=tf.int64, name="iteration", trainable=False)

      # Momentum optimisation
      self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, 
        momentum=self.momentum, 
        name="optimizer")

      self.train_step = self.optimizer.minimize(self.cross_entropy, 
        name="train_step")

      # Initializing the variables
      self.initializer = tf.global_variables_initializer()
      #self.summary = tf.merge_all_summaries()


  @property
  def batch_size(self):
    return self.inputs.get_shape()[0].value

  @property
  def sequence_length(self):
    return self.inputs.get_shape()[1].value

  @property
  def input_dimension(self):
    return self.inputs.get_shape()[2].value

  @property
  def n_classes(self):
    return self.targets.get_shape()[2].value

  @property
  def state_size(self):
    return self.init.get_shape()[1].value

  @staticmethod
  def perplexity(cost, iterations):
    return np.exp(cost / iterations)

  @staticmethod
  def plot_training_losses(cost_history):
    fig = plt.figure(figsize=(15,10))
    plt.plot(cost_history)
    plt.axis([0,training_epochs,0,np.max(cost_history)])
    plt.show()

  @staticmethod
  def plot_predictions(predictions):
    fig = plt.figure(figsize=(15,10))
    plt.plot(predictions, color="blue")
    plt.axis([0,len(predictions),0,1.2])
    plt.show()


  def _get_batch(self,
    X_train, 
    Y_train):
    """
    Formatting our raw data s.t. [batch_size, sequence_length, input_dimension]
    :param X_train: dataset features matrix
    :type 2-D Numpy array
    :param Y_train: dataset one-hot encoded labels matrix
    :type 2-D Numpy array
    :return: Iteratot over training batches
    :rtype: Iterator
    """

    raw_data_length = len(X_train)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = raw_data_length // self.model_parameters.batch_size
    data_x = np.zeros([self.model_parameters.batch_size, 
      batch_partition_length, 
      self.model_parameters.input_dimension], 
      dtype=np.float32)

    data_y = np.zeros([self.model_parameters.batch_size, 
      batch_partition_length, 
      self.model_parameters.n_classes], 
      dtype=np.float32)
    #data_y = np.zeros([batch_size, n_classes], dtype=np.int32)
    
    for i in range(self.model_parameters.batch_size):
        data_x[i] = X_train[batch_partition_length * i:batch_partition_length * (i + 1), :]
        data_y[i] = Y_train[batch_partition_length * i:batch_partition_length * (i + 1),:]
    
    # further divide batch partitions into sequence_length for truncated backprop
    epoch_size = batch_partition_length // self.model_parameters.sequence_length

    for i in range(epoch_size):
        x = data_x[:, i * self.model_parameters.sequence_length:(i + 1) * self.model_parameters.sequence_length,:]
        y = data_y[:, i * self.model_parameters.sequence_length:(i + 1) * self.model_parameters.sequence_length,:]
        yield (x, y)



  def _get_epochs(self, 
    n, 
    X_train, 
    Y_train):
    """
    Generate iterator over training epochs
    :param n: max number of training epochs
    :type int
    :param X_train: dataset features matrix
    :type 2-D Numpy array
    :param Y_train: dataset one-hot encoded labels matrix
    :type 2-D Numpy array
    :return: Iteratot over training epochs
    :rtype: Iterator
    """
    for i in range(n):
        yield self._get_batch(X_train, Y_train)


  def train(self,
    session,
    X_train,
    Y_train,
    checkpoint_every=1000,
    log_dir = 'log',
    display_step=5,
    verbose=True):

    """ Training the network
    :param X_train: features matrix
    :type 2-D Numpy array of float values
    :param Y_train: one-hot encoded labels matrix
    :type 2-D Numpy array of int values
    :param checkpoint_every: RNN model checkpoint frequency 
    :type int 
    :param log_dir: TensorBoard log directory
    :type string
    :param display_step: number of traing epochs executed before logging messages
    :type int
    :param verbose: display log mesages on screen at each training epoch
    :type boolean
    :returns: Cost history of each training epoch
              and the training Perplexity
    :rtype float, float
    :raises: -
    """

    print("\nTraining the network...\n")

    try:
    #with tf.Session() as session:

      # instrument for tensorboard
      summaries = tf.summary.merge_all()
      writer = tf.summary.FileWriter(os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
      writer.add_graph(session.graph)

      session.run(self.initializer)
      saver = tf.train.Saver(tf.global_variables())

      cost_history = np.empty(shape=[1], dtype=float)
      perplexity_history = np.empty(shape=[1], dtype=float)
      accuracy_history = np.empty(shape=[1], dtype=float)

      epoch_cost=0
      epoch_accuracy=0
      epoch_iteration=0

      for epoch_idx, epoch in enumerate(
        self._get_epochs(
          self.training_parameters.training_epochs, 
          X_train, 
          Y_train)):

        current_epoch = epoch_idx

        avg_cost = 0.

        #training_state = np.zeros((batch_size, state_size))

        current_iteration = 0
        done = False

        for batch_step, (batch_x, batch_y) in enumerate(epoch):
          
          current_iteration = batch_step

          # Run optimization op (backprop) and cost op (to get loss value)
          _train_step, _cost, _prediction_series = session.run(
            [self.train_step, self.cross_entropy, self.predictions], 
            feed_dict={ 
            self.inputs:batch_x, 
            self.targets:batch_y,
            self.learning_rate : self.model_parameters.learning_rate,
            self.momentum : self.model_parameters.momentum,
            self.input_keep_probability : self.model_parameters.input_keep_probability,
            self.output_keep_probability : self.model_parameters.output_keep_probability})

          # Compute average loss 
          avg_cost += _cost / self.model_parameters.batch_size
          tf.summary.scalar('train_loss', avg_cost)

          if (epoch_idx * self.model_parameters.batch_size + batch_step) % checkpoint_every == 0 or (
            epoch_idx == self.training_parameters.training_epochs-1 and 
            batch_step == self.model_parameters.batch_size-1):

            # Save for the last result
            checkpoint_path = os.path.join(self.directories.checkpoint_dir, 'model.ckpt')
            saver.save(session, checkpoint_path, global_step=epoch_idx * self.model_parameters.batch_size + batch_step)
            print("model saved to {}".format(checkpoint_path))


          epoch_cost += _cost
          epoch_iteration += self.model_parameters.batch_size

          # Display logs per epoch step
          if epoch_idx % display_step == 0:
            if verbose and not done:
              # Calculate batch accuracy
              epoch_accuracy = session.run(
                self.accuracy, {
                self.inputs: batch_x, 
                self.targets: batch_y})
              time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
              done = True

              print(str(time),
                ": Epoch:", '%04d' % (epoch_idx),
                "cost=", "{:.9f}".format(avg_cost),
                ", Accuracy= ", "{:.5f}".format(epoch_accuracy))

        
        cost_history = np.append(cost_history,avg_cost)

    except KeyboardInterrupt:
      pass

    print("Stop training at epoch %d, iteration %d" % (current_epoch, current_iteration),
                "Perplexity=", "{:.4f}".format(self.perplexity(epoch_cost, epoch_iteration)),
                ", Accuracy= ", "{:.5f}".format(epoch_accuracy))

    #logger.info("Stop training at epoch %d, iteration %d" % (current_epoch, current_iteration))
    #summary.close()
    """
    if self.directories.checkpoint_dir is not None:
      checkpoint_file = tf.train.latest_checkpoint(self.directories.checkpoint_dir)
      tf.train.Saver().save(session, checkpoint_file)
      self._write_model_parameters(str(self.directories.param_dir))
    print("Saved model in %s " % self.directories.checkpoint_dir)
    """
    #logger.info("Saved model in %s " % self.directories.checkpoint_dir)

    print("Optimization Finished!")
    #self.plot_training_losses(cost_history)
    return cost_history, epoch_accuracy, self.perplexity(epoch_cost, epoch_iteration)


  def _write_model_parameters(self, param_dir):
    """ Store parameters in a JSON file
    :param param_dir: parameter save directory
    :type string
    """
    parameters = {
    "training_epochs" : self.training_parameters.training_epochs,
    "learning_rate" : self.model_parameters.learning_rate,
    "momentum" : self.model_parameters.momentum,
    "model" : self.model_parameters.model,
    "input_keep_probability" : self.model_parameters.input_keep_probability,
    "output_keep_probability" : self.model_parameters.output_keep_probability,
    "sequence_length" : self.model_parameters.sequence_length,
    "input_dimension" : self.model_parameters.input_dimension,
    "batch_size" : self.model_parameters.batch_size,
    "state_size" : self.model_parameters.state_size,
    "n_layers" : self.model_parameters.n_layers,
    "n_classes" : self.model_parameters.n_classes,
    "log_dir" : self.directories.log_dir,
    "checkpoint_dir" : self.directories.checkpoint_dir,
    }

    with open(self._parameters_file(param_dir), "w") as f:
      json.dump(parameters, f, indent=4)


  def evaluate(self,
    session,
    X_test, 
    Y_test,
    allow_soft_placement=True,
    log_device_placement=False):
    """ Evaluating the network
    :param X_test: features matrix
    :type 2-D Numpy array of float values
    :param Y_test: one-hot encoded labels matrix
    :type 2-D Numpy array of int values
    :returns: -
    :raises: -
    """

    print("\nEvaluating the network...\n")
    
    raw_data_length = len(X_test)
    sequence_size = raw_data_length // self.model_parameters.sequence_length
    epoch_cost = epoch_iteration = 0
    
    data_x_test = np.zeros([
      sequence_size,
      self.model_parameters.sequence_length,
      self.model_parameters.input_dimension],
      dtype=np.float32)

    data_y_test = np.zeros([
      sequence_size,
      self.model_parameters.sequence_length,
      self.model_parameters.n_classes],
      dtype=np.float32)
        
    for i in range(sequence_size):
      data_x_test[i] = X_test[self.model_parameters.sequence_length * i:self.model_parameters.sequence_length * (i + 1), :]
      data_y_test[i] = Y_test[self.model_parameters.sequence_length * i:self.model_parameters.sequence_length * (i + 1), :]
                
    cost, all_predictions = session.run(
      [self.cross_entropy, 
      self.predictions],
      feed_dict={
      self.inputs: data_x_test,
      self.targets: data_y_test,
      self.input_keep_probability : self.model_parameters.input_keep_probability,
      self.output_keep_probability : self.model_parameters.output_keep_probability})

    epoch_cost += cost
    epoch_iteration += sequence_size
            
    label_predictions = session.run(
      self.label_prediction,
      feed_dict={
      self.inputs: data_x_test,
      self.targets: data_y_test,
      self.input_keep_probability : self.model_parameters.input_keep_probability,
      self.output_keep_probability : self.model_parameters.output_keep_probability})

    #self.plot_predictions(label_predictions)

    # Print accuracy if test label set is provided
    if Y_test is not None:
        evaluation_accuracy = round(session.run(
            self.accuracy, 
            feed_dict={
            self.inputs: data_x_test, 
            self.targets: data_y_test,
            self.input_keep_probability : self.model_parameters.input_keep_probability,
            self.output_keep_probability : self.model_parameters.output_keep_probability}) , 3)
    
    evaluation_perplexity=self.perplexity(epoch_cost, epoch_iteration)

    print("Total number of test examples: {}".format(len(Y_test)))
    print("Accuracy: ",evaluation_accuracy)
    
    # Save the results in a CSV output file
    out_path = "prediction.csv"
    print("Saving evaluation to {0}".format(out_path))
    
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(label_predictions)

    return evaluation_accuracy, evaluation_perplexity




  @staticmethod
  def summary_writer(summary_directory, session):
    class NullSummaryWriter(object):
      def add_summary(self, *args, **kwargs):
        pass

      def flush(self):
        pass

      def close(self):
        pass

    if summary_directory is not None:
      return tf.train.SummaryWriter(summary_directory, session.graph)
    else:
      return NullSummaryWriter()



# Objects used to store parameters

class TrainingParameters(object):
    def __init__(self, 
      training_epochs):
      """ Encapsulation of RNN training parameters
      :param training_epochs: number of the training epochs
      :type int
      """
      self.training_epochs = training_epochs


class ModelParameters(object):
    def __init__(self,
      learning_rate,
      momentum=None,
      model='lstm',
      input_keep_probability=1.0,
      output_keep_probability=1.0,
      sequence_length=None,
      input_dimension=None,
      batch_size=None, 
      state_size=None, 
      n_layers=None,
      n_classes=None,):

      """ Encapsulation of RNN model hyperparameters
      :param learning_rate: gradient descent optimization learning rate
      :type float between 0..1
      :param momentum: momentum optimization parameter
      :type float between 0..1
      :param input_keep_probability:
      :type float between 0..1
      :param output_keep_probability: 
      :type float between 0..1
      :param sequence_length: number of truncated backpropagation steps
      :type int
      :param input_dimension: number of features
      :type int
      :param batch_size: size of the training batch
      :type int
      :param state_size: number of LSTM memory cells inside each memory block
      :type int
      :param n_layers: number of layers in the RNN
      ;type int
      :param n_classes: number of class labels
      :type int
      """

      self.learning_rate = learning_rate
      self.momentum = momentum
      self.model=model
      self.input_keep_probability = input_keep_probability
      self.output_keep_probability = output_keep_probability
      self.sequence_length=sequence_length
      self.input_dimension=input_dimension
      self.batch_size=batch_size
      self.state_size=state_size
      self.n_layers=n_layers
      self.n_classes=n_classes


class Directories(object):
    def __init__(self, 
      log_dir, 
      checkpoint_dir):
      """ Encaplsulation of the directories names
      :param log_dir: TensorBoard log directory
      :type string
      :param checkpoint_dir: TensorFlow checkpoint directory
      :type string
      """
      self.log_dir = log_dir
      self.checkpoint_dir = checkpoint_dir



