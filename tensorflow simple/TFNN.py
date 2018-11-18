import TFDefinitions as TFD
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()


def prepareData(x_train, y_train, x_test, y_test):
    
    train_x_flatten = x_train.reshape(x_train.shape[0], -1).T 
    train_x = train_x_flatten/255.

    test_x_flatten = x_test.reshape(x_test.shape[0], -1).T 
    x_test = test_x_flatten/255.
    
    hotMatrixTrain = tf.one_hot(y_train.T, 10, axis = 0)
    hotMatrixTest = tf.one_hot(y_test.T, 10, axis = 0)
    
    with tf.Session() as sess:
        hotTrain = sess.run(hotMatrixTrain)
        hotTest = sess.run(hotMatrixTest)
    
    sess.close()
    
    return hotTrain, hotTest, train_x, x_test

def completedNetwork(x_train, x_test, y_train, y_test, learningRate, batchSize, epochs):
    tf.reset_default_graph() 
    YTrain, YTest, XTrain, XTest = prepareData(x_train, y_train, x_test, y_test)
    
    print(XTrain.shape)
    
    X, Y = TFD.createPlaceholders(XTrain.shape[0], YTrain.shape[0])
    
    parameters = TFD.initialize_parameters(XTrain.shape[0], YTrain.shape[0])
    
    Z3 = TFD.forwardIteration(X, parameters)
    
    cost = TFD.computeCost(Z3, Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(cost)
    
    paramsInit = tf.global_variables_initializer()
        
    costs = []
    
    with tf.Session() as sess:
        sess.run(paramsInit)
        
        for epoch in range(epochs):
        
            epoch_cost = 0.
            batchesAmount = int(XTrain.shape[1]/batchSize)
        
            for batch in range(batchesAmount):
                
                miniBatch = XTrain[:, batch*batchSize: (batch+1)*batchSize]
                _, temp_cost = sess.run([optimizer, cost], feed_dict = {X: miniBatch, Y: YTrain[:, batch*batchSize: (batch+1)*batchSize]})
                epoch_cost += temp_cost / batchesAmount
            
            if epoch % 10 == 0:
                print ("        Cost after epoch %i: %f" % (epoch, epoch_cost))
            if epoch % 5 == 0:
                costs.append(epoch_cost)
    
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
        print ("Train Accuracy:", accuracy.eval({X: XTrain, Y: YTrain}))
        print ("Test Accuracy:", accuracy.eval({X: XTest, Y: YTest}))
    
        sess.close()
    print(X)
    print(Y)
    
    return 0