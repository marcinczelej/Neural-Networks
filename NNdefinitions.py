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
        parameters["b" + str(layer)] = np.zeros((layers_dimensions[layer], 1))
    
    return parameters

#---------------------------------------forward_part----------------------------------------

def forward_linear(A_prev, W, b):
    # wyjscie : Z, cache z A_prev, W , b
    
    Z = np.dot(W, A_prev) + b
    
    cache = (A_prev, W, b)
    
    return Z, cache

def forward_activation(A_prev, W, b, activation_function):
    
    if activation_function== "relu":
        print("relu")
        Z, linear_cache = forward_linear(A_prev, W, b)
        A = np.maximum(0, Z)
    
    if activation_function == "sigmoid":
        print("sigmoid")
        Z, linear_cache = forward_linear(A_prev, W, b)
        A = np.divide(1, 1 + np.exp(-Z))
    
    cache = Z
    
    caches = (cache, linear_cache)
    
    return A, caches

def compute_cost(A, Y):
    
    cost = -1/m*(Y*np.log(A) + (1-Y)*np.log(1-A))
    
    return cost

def stablesoftmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

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

#---------------------------------------backward_part----------------------------------------

def relu_backward(dA, activation_cache):
    
    Z = activation_cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    
    return dZ

def sigmoid_backward(dA, activation_cache):
    
    Z = activation_cache
    s = 1/(1 + np.exp(-Z))
    dZ =dA* s*(1-s)
            
    return dZ

def linear_backward(dZ, cache):
    
    (A_prev, W, b) = cache
    m = A_prev.shape[1]
    
    dW = 1/m*np.dot(dZ, A_prev.T)
    db = 1/m*np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def activation_backward(dA, activation_cache, activation_function):
    
    if activation_function == "relu":
        dZ = relu_backward(dA, activation_cache)
    
    if activation_function == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    return dZ

def layer_backward(dA, cache, activation_function):
    
    (activation_cache, linear_cache) = cache
    
    dZ = activation_backward(dA, activation_cache, activation_function)
    print(" dZ = " + str(dZ))
    A_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return A_prev, dW, db

def backward_propagation(A_last, Y, caches):
    
    dA = -(np.divide(Y, A_last) - np.divide((1-Y), (1-A_last)))
    
    changes = {}
    
    LEN = len(caches)
    
    print(" LEN = " + str(LEN))
    print(caches)
    current_cache = caches[LEN-1]
    Y = Y.reshape(A_last.shape)
    
    changes["dA" + str(LEN-1)], changes["dW" + str(LEN)], changes["db" + str(LEN)] = layer_backward(dA, current_cache, "sigmoid")
    dA_prev = changes["dA" + str(LEN-1)]
    print(" grads[dA + str(L-1)] = " + str(changes["dA" + str(LEN-1)]))
    print(" grads[dW + str(L)] = " + str(changes["dW" + str(LEN)]))
    print(" grads[db + str(L)] = " + str(changes["db" + str(LEN)]))
    
    
    for l in reversed(range(LEN-1)):
        print(" l = " + str(l))
        current_cache = caches[l]
        changes["dA" + str(l)], changes["dW" + str(l+1)], changes["db" + str(l+1)] = layer_backward(dA_prev, current_cache, "relu")
        dA_prev = changes["dA" + str(l)]
        
    return changes

def update_parameters(gradients, parameters, learning_rate):
    
    LEN = len(parameters) // 2
    
    for l in range(LEN):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*gradients["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*gradients["db" + str(l+1)]
    
    return parameters
