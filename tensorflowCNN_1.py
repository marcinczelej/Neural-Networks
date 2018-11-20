import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
"""
 - Conv2D: stride 1, padding is "SAME"
 - ReLU
 - Max pool: Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
 - Conv2D: stride 1, padding is "SAME"
 - ReLU
 - Max pool: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
 - Flatten the previous output.
 - FULLYCONNECTED (FC) layer: Apply a fully connected layer without an non-linear activation function. Do not call the softmax here. This will result in 6 neurons in the output layer, which then get passed later to a softmax. In TensorFlow, the softmax and cost function are lumped together into a single function, which you'll call in a different function when computing the cost. 
  W1 : [4, 4, 1, 8]
                        W2 : [2, 2, 8, 16]
 
 """
def getData():

    mnist = tf.keras.datasets.fashion_mnist.load_data()

    (x_train, y_train),(x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    return x_train, y_train, x_test, y_test


def createPlaceholders(input_h, input_w, channels, output_size):
    
    X = tf.placeholder(tf.float32, [None, input_h, input_w, channels], name = "X")
    Y = tf.placeholder(tf.float32, [None, output_size], name = "Y")
    is_train = tf.placeholder(tf.bool, name = "is_train")
    
    return X, Y, is_train 

def initializeVariables():
    
    W1 = tf.get_variable("W1", [4,4,1,8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [2,2,8,16], initializer = tf.contrib.layers.xavier_initializer(seed =0))
    W3 = tf.get_variable("W3", [5,5,16,16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    
    parameters = {"W1" : W1,
                  "W2" : W2,
                  "W3" : W3}
    
    return parameters

def  forwardIteration(X, parameters):
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')
    Z1 = tf.nn.relu(A1)
    X = tf.nn.max_pool(Z1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    
    A2 = tf.nn.conv2d(X, W2, strides = [1,1,1,1], padding = 'SAME')
    Z2 = tf.nn.relu(A2)
    X = tf.nn.max_pool(Z2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    
    flat = tf.contrib.layers.flatten(X)
    
    fullyConnected = tf.contrib.layers.fully_connected(flat, 10, activation_fn = None)
    
    return fullyConnected

def identityBlock(X, parameters, is_train):
        
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
    
    flat = tf.contrib.layers.flatten(A3)
    
    fullyConnected = tf.contrib.layers.fully_connected(flat, 10, activation_fn = None)
        
    return fullyConnected

def forwardIterationResNet(X, parameters, is_train):
    
    return identityBlock(X, parameters, is_train)


def costFunction(lastLayer, Y):
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = lastLayer))
    
    return cost
    
def model(x_train, y_train, x_test, y_test, learningRate = 0.0001, batchSize = 256, epochs = 10):
    
    tf.reset_default_graph()
    
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)
    
    with tf.Session() as sess:
        y_train = sess.run(y_train)
        y_test = sess.run(y_test)
    
    sess.close()
    
    print("x_train = " + str(x_train.shape))
    print("y_train = " + str(y_train.shape))
    
    print("x_test = " + str(x_test.shape))
    print("y_test = " + str(y_test.shape))
    
    x_train = np.resize(x_train, (60000, 28, 28, 1))
    print("x_train = " + str(x_train.shape))
    x_test = np.resize(x_test, (60000, 28, 28, 1))
    print("x_test = " + str(x_test.shape))
    
    (examples, input_h, input_w, channels) = x_train.shape
    (examples, outputFeatures) = y_train.shape
    
    X, Y, is_train = createPlaceholders(input_h, input_w, 1, 10)
    
    print("examples = " + str(examples))
    print(X.shape)
    print(Y.shape)
    print(is_train)
    
    parameters = initializeVariables()
    
    #Z3 = forwardIteration(X, parameters)
    Z3 = forwardIterationResNet(X, parameters, is_train)
    
    cost = costFunction(Z3, Y)
    
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    costs = []
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess: 
        sess.run(init)
        for epoch in range(epochs):
        
            minibatchCost = 0.
            minibatchAmount = int(int(examples) / batchSize)
        
            for batch in range(minibatchAmount):
                XBatch = x_train[batch*batchSize:(batch+1)*batchSize, :, :, :]
                YBatch = y_train[batch*batchSize:(batch+1)*batchSize, :]
                print("minibatch " + str(batch))
            
                _, tempCost = sess.run([optimizer, cost], feed_dict = {X : XBatch, Y : YBatch, is_train:True})
    
                minibatchCost += tempCost/minibatchAmount
            
            if epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatchCost))
            if epoch % 1 == 0:
                costs.append(minibatchCost)
    
    # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learningRate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: x_train, Y: y_train, is_train:False})
        test_accuracy = accuracy.eval({X: x_test, Y: y_test, is_train:False})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        
        sess.close()