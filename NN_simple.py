import NNdefinitions
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()


def fullNeuralNetwork(X, Y, learning_rate, layer_dims, epochs):
    
    parameters = NNdefinitions.initialize_weights(layer_dims)
    
    costs = []
    
    for epoch in range(epochs):
        A_last, cache = NNdefinitions.complete_forward(X, parameters)
    
        cost = NNdefinitions.compute_cost(A_last, Y)
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

    