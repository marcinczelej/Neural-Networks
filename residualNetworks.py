import tensorflow as tf
import numpy as np
        
def definePlaceholders(in_h, in_w, in_channels, out_elements):
        
    X = tf.placeholder(tf.float32, [None, in_h, in_w, in_channels], name = "X" )
    Y = tf.placeholder(tf.float32, [None, out_elements], name = "Y")
    is_train = tf.placeholder(tf.bool, name= "is_train")
        
    return X, Y, is_train
        
def identityBlock(X, parameters):
        
        #input = [batch, in_height, in_width, in_channels]
        
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
        
    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = "SAME")
    B1 = tf.layers.batch_normalization(Z1, training=is_train)
    A1 = tf.nn.relu(B1)
        
    Z2 = tf.nn.conv2d(A1, W2, strides = [1,1,1,1], padding = "SAME")
    B2 = tf.layers.batch_normalization(Z2, training = is_train)
    A2 = tf.nn.relu(B2)
        
    Z3 = tf.nn.conv2d(A2, W3, strides = [1,1,1,1], padding = "SAME")
    B3 = tf.layers.batch_normalization(Z3, training = is_train)
        
    A3 = tf.nn.relu(B3+X)
        
    return A3
    

        