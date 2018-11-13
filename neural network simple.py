import numpy as np
import keras as ks
import tensorflow as tf

def initialize_weights(layers_dimensions):
    # wejscie tablica z wymiarami poszczegolnych warstw sieci
    # wyjscie parametry sieci
    
    parameters = {}
    
    np.random.seed(3)
    
    for layer in range(1, len(layers_dimensions)):
        parameters["W" + str(layer)] = np.random.randn(layers_dimensions[layer], layers_dimensions[layer-1])*np.sqrt(1/np.power(layers_dimensions[layer-1], layer))
        parameters["b" + str(layer)] = np.random.randn(layers_dimensions[layer], 1)
    
    return parameters

def forward_function(A_prev, W, b):
    # wyjscie : Z, cache z A_prev, W , b
    
    Z = np.dot(W, A_prev) + b
    
    cache = (A_prev, W, b)
    
    return Z, cache

def forward_activation(A_prev, W, b, activation_function):
    
    if activation_function== "relu":
        print("relu")
        Z, linear_cache = forward_function(A_prev, W, b)
        A = np.maximum(0, Z)
    
    if activation_function == "sigmoid":
        print("sigmoid")
        Z, linear_cache = forward_function(A_prev, W, b)
        A = np.divide(1, 1 + np.exp(-Z))
    
    cache = (Z, linear_cache)
    
    return A, cache

def compute_cost(A, Y):
    
    cost = -1/m*(Y*np.log(A) + (1-Y)*np.log(1-A))
    
    return cost

def complete_forward(X, parameters):
    
    LEN = len(parameters) // 2
    A_prev = X
    
    caches = []
    
    for l in range(1, LEN):
        print(l)
        A_prev, cache= forward_activation(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)
        
    A_last, cache = forward_activation(A_prev, parameters["W" + str(LEN)], parameters["b" + str(LEN)], "sigmoid")
    caches.append(cache)
    
    return A_last, cache

def linear_backward(dA, cache):
    
    (A_prev, W, b) = cache
    m = len(b)
    
    dW = 1/m*np.dot(dZ, A_prev)
    db = 1/m*np.sum(dZ, axis = 1, keepdims = true)
    dA_prev = np.dot(dZ, W.T)
    
    return dA_prev, dW, db

