import NNdefinitions
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

layer_dims = [784, 3,4, 10]

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    
    ### START CODE HERE ###
    
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name = "C")
    
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels, C, axis = 0)
    
    # Create the session (approx. 1 line)
    with tf.Session() as sess:
    
    # Run the session (approx. 1 line)
        one_hot = sess.run(one_hot_matrix)
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    
    ### END CODE HERE ###
    
    return one_hot

def fullNeuralNetwork(X, Y, learning_rate, layer_dims, epochs):
    
    parameters = NNdefinitions.initialize_weights(layer_dims)
    
    costs = []
    
    mini_bacth_size = 30000
        
    miniBatchesAmount = int(len(X[0, :])/mini_bacth_size)
    print(" miniBatchesAmount = " + str(miniBatchesAmount))
    
    for epoch in range(epochs):
        for batch in range(miniBatchesAmount):
            
            X_batch = X[:, batch*mini_bacth_size: (batch+1)*mini_bacth_size]
            Y_batch = Y[:, batch*mini_bacth_size: (batch+1)*mini_bacth_size]
        
            A_last, cache = NNdefinitions.complete_forward(X_batch, parameters)
    
            cost = NNdefinitions.compute_cost(A_last, Y_batch)
            costs.append(cost)
    
            gradients = NNdefinitions.backward_propagation(A_last, Y_batch, cache)
    
            parameters = NNdefinitions.update_parameters(gradients, parameters, learning_rate, epoch)
            print(" epoch : " + str(epoch) + " cost : " + str(cost))
        
     # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

train_x_flatten = x_train.reshape(x_train.shape[0], -1).T 
train_x = train_x_flatten/255.

train_y = y_train
print(train_y.shape)
b = train_y.T
print(b.shape)

y_train = one_hot_matrix(b, 10)
print (y_train[:, 0])

print( " train_x = " + str(train_x.shape))
print( " y_train = " + str(y_train.shape))
fullNeuralNetwork(train_x, y_train, 0.01, layer_dims, 1000)
