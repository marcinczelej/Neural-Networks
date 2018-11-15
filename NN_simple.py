import NNdefinitions
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle as pc
import time

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

layer_dims = [784, 10]

def one_hot_matrix(labels, C):
    C = tf.constant(C, name = "C")
    
    one_hot_matrix = tf.one_hot(labels, C, axis = 0)
    
    with tf.Session() as sess:
        one_hot = sess.run(one_hot_matrix)
    
    sess.close()
    
    return one_hot

def prediction(X, y, parameters):
    
    m = X.shape[1]
    p = np.zeros((1,m))
    wrong_count =0
    
    # Forward propagation
    probas, _ = NNdefinitions.complete_forward(X, parameters, 1)
    
    for i in range(0, probas.shape[1]):
        if np.argmax(probas[:, i]) == y[i]:
            p[0,i] = 1
        else:
            wrong_count += 1
            p[0,i] = 0
    
    accuracy = np.sum((p)/m)
    #print results
        
    return p, accuracy, wrong_count

def fullNeuralNetwork(X, Y, learning_rate, layer_dims, epochs):
    
    start = time.time()
    
    parameters = NNdefinitions.initialize_weights(layer_dims)
    
    costs = []
    last_cost = 0
    
    mini_bacth_size = 32
        
    miniBatchesAmount = int(len(X[0, :])/mini_bacth_size)
    print(" miniBatchesAmount = " + str(miniBatchesAmount))
    
    for epoch in range(epochs):
        for batch in range(miniBatchesAmount):
            
            X_batch = X[:, batch*mini_bacth_size: (batch+1)*mini_bacth_size]
            Y_batch = Y[:, batch*mini_bacth_size: (batch+1)*mini_bacth_size]
        
            A_last, cache = NNdefinitions.complete_forward(X_batch, parameters, 0.5)
    
            cost = NNdefinitions.compute_cost(A_last, Y_batch)
            last_cost = cost
    
            gradients = NNdefinitions.backward_propagation(A_last, Y_batch, cache)
            parameters = NNdefinitions.update_parameters(gradients, parameters, learning_rate, batch)
            if  epoch % 20 == 0 and batch == 1:
                print ("Cost after iteration %i: %f" %(epoch, cost))
                costs.append(cost)
        
    print("cost = " + str(cost))
     # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    end = time.time()
    
    print( " elapsed = " + str(end - start))
    
    return parameters

train_x_flatten = x_train.reshape(x_train.shape[0], -1).T 
train_x = train_x_flatten/255.

test_x_flatten = x_test.reshape(x_test.shape[0], -1).T 
x_test = test_x_flatten/255.

train_y = y_train
b = train_y.T

y_train = one_hot_matrix(b, 10)

parameters = fullNeuralNetwork(train_x, y_train, 0.01, layer_dims, 1500)

p, accuracy, wrong_count = prediction(train_x, train_y, parameters)
p2, accuracy2, wrong_count2 = prediction(x_test, y_test, parameters)

print(" train = " + str(accuracy) + " wrongly recognized : " + str(wrong_count))
print(" test = " + str(accuracy2) + " wrongly recognized : " + str(wrong_count2))