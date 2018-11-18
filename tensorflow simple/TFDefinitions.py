import tensorflow as tf
import numpy as np

def createPlaceholders(input_size, output_size):
    
    X = tf.placeholder(tf.float32, shape = (input_size, None), name = "X")
    Y = tf.placeholder(tf.float32, shape = (output_size, None), name = "Y")
    
    return X, Y

def initialize_parameters(input_size, output_size):
    
    
    W1 = tf.get_variable("W1", [25, input_size], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())
    
    W2 = tf.get_variable("W2", [40, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    b2 = tf.get_variable("b2", [40, 1], initializer = tf.zeros_initializer())
    
    W3 = tf.get_variable("W3", [output_size, 40], initializer = tf.contrib.layers.xavier_initializer(seed =0))
    b3 = tf.get_variable("b3", [output_size, 1], initializer = tf.zeros_initializer())
    
    parameters = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2,
                  "W3" : W3,
                  "b3" : b3,}
    
    return parameters

def forwardIteration(X, parameters):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    
    return Z3

def computeCost(Z, Y):
    
    labels = tf.transpose(Y)
    logits = tf.transpose(Z)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits))
    
    return cost

