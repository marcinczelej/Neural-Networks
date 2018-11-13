import NNdefinitions
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

layer_dims = [784, 20, 7, 5, 10]

def fullNeuralNetwork(X, Y, learning_rate, layer_dims, epochs):
    
    parameters = NNdefinitions.initialize_weights(layer_dims)
    
    costs = []
    
    for epoch in range(epochs):
        A_last, cache = NNdefinitions.complete_forward(X, parameters)
    
        print(A_last.shape)
        cost = NNdefinitions.stablesoftmax(A_last)
        costs.append(cost)
    
        gradients = NNdefinitions.backward_propagation(A_last, Y, cache)
    
        parameters = NNdefinitions.update_parameters(gradients, parameters, learning_rate)
        
        if epoch//100 != 0:
            print(" epoch : " + str(epoch) + " cost : " + str(cost))
            
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

train_x_flatten = x_train.reshape(x_train.shape[0], -1).T 
train_x = train_x_flatten/255.

train_y = y_train.reshape((1, y_train.shape[0]))

print( " train_y = " + str(train_y.shape))
fullNeuralNetwork(train_x, train_y, 0.1, layer_dims, 1000)