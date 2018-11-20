import tensorflow as tf
import keras
import numpy as np

def getData():

    mnist = tf.keras.datasets.fashion_mnist.load_data()

    (x_train, y_train),(x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    return x_train, y_train, x_test, y_test

"""
First component of main path:

    The first CONV2D has F1F1 filters of shape (1,1) and a stride of (1,1). Its padding is "valid" and its name should be conv_name_base + '2a'. Use 0 as the seed for the random initialization.
    The first BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '2a'.
    Then apply the ReLU activation function. This has no name and no hyperparameters.

Second component of main path:

    The second CONV2D has F2F2 filters of shape (f,f)(f,f) and a stride of (1,1). Its padding is "same" and its name should be conv_name_base + '2b'. Use 0 as the seed for the random initialization.
    The second BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '2b'.
    Then apply the ReLU activation function. This has no name and no hyperparameters.

Third component of main path:

    The third CONV2D has F3F3 filters of shape (1,1) and a stride of (1,1). Its padding is "valid" and its name should be conv_name_base + '2c'. Use 0 as the seed for the random initialization.
    The third BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '2c'. Note that there is no ReLU activation function in this component.

Final step:

    The shortcut and the input are added together.
    Then apply the ReLU activation function. This has no name and no hyperparameters.

"""    
def identityBlock(X, f, filters, stage, inputShape = (28, 28, 1)):

    F1, F2, F3 = filters
    
    filterName = "identityBlock_" + str(stage) + "_filter_"
    batchName = "identityBlock_" + str(stage) + "_batch_"
    
    XShortcut = X
    #print(X.shape)
    X = keras.layers.Conv2D(F1, kernel_size= (1,1), strides = (1, 1), input_shape=inputShape, padding = "valid", name = filterName + str(1), kernel_initializer = keras.initializers.glorot_uniform(seed = 0))(X)
    X = keras.layers.BatchNormalization(axis = 3, name = batchName + "1")(X)
    X = keras.layers.ReLU()(X)
    #print(X.shape)
    
    X = keras.layers.Conv2D(F2, kernel_size = (f, f), strides = (1, 1),input_shape=inputShape,  padding = "same", name = filterName + str(2), kernel_initializer = keras.initializers.glorot_uniform(seed = 0))(X)
    X= keras.layers.BatchNormalization(axis = 3, name = batchName + "2")(X)
    X = keras.layers.ReLU()(X)
    #print(X.shape)
    
    X = keras.layers.Conv2D(F3, kernel_size = (1, 1), strides = (1, 1),input_shape=inputShape,  padding = "valid", name = filterName + str(3), kernel_initializer = keras.initializers.glorot_uniform(seed = 0))(X)
    X = keras.layers.BatchNormalization(axis = 3, name = batchName + "3")(X)
    
    X = keras.layers.Add()([X, XShortcut])
    X = keras.layers.ReLU()(X)
    #print(X.shape)
    
    return X


def convultionalResBlock(X, f, filters, stage, s = 2, inputShape = (28, 28, 1)):
    """
    First component of main path:

    The first CONV2D has F1F1 filters of shape (1,1) and a stride of (s,s). Its padding is "valid" and its name should be conv_name_base + '2a'.
    The first BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '2a'.
    Then apply the ReLU activation function. This has no name and no hyperparameters.

    Second component of main path:

    The second CONV2D has F2F2 filters of (f,f) and a stride of (1,1). Its padding is "same" and it's name should be conv_name_base + '2b'.
    The second BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '2b'.
    Then apply the ReLU activation function. This has no name and no hyperparameters.

    Third component of main path:

    The third CONV2D has F3F3 filters of (1,1) and a stride of (1,1). Its padding is "valid" and it's name should be conv_name_base + '2c'.
    The third BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '2c'. Note that there is no ReLU activation function in this component.

    Shortcut path:

    The CONV2D has F3F3 filters of shape (1,1) and a stride of (s,s). Its padding is "valid" and its name should be conv_name_base + '1'.
    The BatchNorm is normalizing the channels axis. Its name should be bn_name_base + '1'.

    Final step:

    The shortcut and the main path values are added together.
    Then apply the ReLU activation function. This has no name and no hyperparameters.
    """
    
    F1, F2, F3 = filters
    
    filterName = "convultionalResBlock_" +  str(stage) + "_filter_"
    batchName = "convultionalResBlock_" +  str(stage) + "_batch_"
    
    XShortcut = X
    #print(X.shape)
    X = keras.layers.Conv2D(F1, kernel_size = (1, 1), strides = (s, s), input_shape= inputShape, padding = "valid", name = filterName + str(1), kernel_initializer = keras.initializers.glorot_uniform(seed = 0))(X)
    X = keras.layers.BatchNormalization(axis = 3, name = batchName + str(1))(X)
    X = keras.layers.ReLU()(X)
    #print(X.shape)
    
    X = keras.layers.Conv2D(F2, kernel_size = (f, f), strides = (1, 1), input_shape= inputShape, padding = "same", name = filterName + str(2), kernel_initializer = keras.initializers.glorot_uniform(seed = 0))(X)
    X = keras.layers.BatchNormalization(axis = 3, name = batchName + str(2))(X)
    X = keras.layers.ReLU()(X)
    #print(X.shape)
    
    X = keras.layers.Conv2D(F3, kernel_size = (1, 1), strides = (1, 1), input_shape= inputShape, padding = "valid", name = filterName + str(3), kernel_initializer = keras.initializers.glorot_uniform(seed = 0))(X)
    X = keras.layers.BatchNormalization(axis = 3, name = batchName +str(3))(X)
    
    XShortcut = keras.layers.Conv2D(F3, kernel_size = (1, 1), strides = (s, s), input_shape= inputShape, padding = "valid", name = filterName + str("Shortcut"), kernel_initializer = keras.initializers.glorot_uniform(seed = 0))(XShortcut)
    XShortcut = keras.layers.BatchNormalization(axis = 3, name = batchName +str("Shortcut"))(XShortcut)
    
    X = keras.layers.Add()([X, XShortcut])
    X = keras.layers.ReLU()(X)
    #print(X.shape)
    
    return X

"""
The details of this ResNet-50 model are:

    Zero-padding pads the input with a pad of (3,3)
    Stage 1:
        The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). Its name is "conv1".
        BatchNorm is applied to the channels axis of the input.
        MaxPooling uses a (3,3) window and a (2,2) stride.
    Stage 2:
        The convolutional block uses three set of filters of size [64,64,256], "f" is 3, "s" is 1 and the block is "a".
        The 2 identity blocks use three set of filters of size [64,64,256], "f" is 3 and the blocks are "b" and "c".
    Stage 3:
        The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
        The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
    Stage 4:
        The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
        The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
    Stage 5:
        The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
        The 2 identity blocks use three set of filters of size [512, 512, 2048], "f" is 3 and the blocks are "b" and "c".
    The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
    The flatten doesn't have any hyperparameters or name.
    The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation. Its name should be 'fc' + str(classes).
"""

def ResNet_50(inputShape = (28, 28, 6), outputClasses = 10):
    
    XInput = keras.Input(inputShape)
    X = keras.layers.ZeroPadding2D(padding = (3, 3))(XInput)

    #stage 1
    X = keras.layers.Conv2D(64, kernel_size = (7, 7), strides = (2, 2), input_shape= inputShape, name = "conv_1", kernel_initializer = keras.initializers.glorot_uniform(seed = 0))(X)
    X = keras.layers.BatchNormalization(axis = 3, name = "batchNorm_1")(X)
    X = keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), name = "maxPool")(X)

    #stage 2
    X = convultionalResBlock(X, f = 3, filters = [64, 64, 256], stage = 2, s = 1)
    X =identityBlock(X, f = 3, filters = [64, 64, 256], stage = 2)
    X =identityBlock(X, f = 3, filters = [64, 64, 256], stage = 2)
    
    #stage 3
    X = convultionalResBlock(X, f = 3, filters = [128,128,512], stage = 3, s = 2)
    X =identityBlock(X, f = 3, filters = [128,128,512], stage = 3)
    X =identityBlock(X, f = 3, filters = [128,128,512], stage = 3)
    X =identityBlock(X, f = 3, filters = [128,128,512], stage = 3)
    
    #stage 4
    X = convultionalResBlock(X, f = 3, filters = [256, 256, 1024], stage = 4, s = 2)
    X =identityBlock(X, f = 3, filters = [256, 256, 1024], stage = 4)
    X =identityBlock(X, f = 3, filters = [256, 256, 1024], stage = 4)
    X =identityBlock(X, f = 3, filters = [256, 256, 1024], stage = 4)
    
    #stage 5
    X = convultionalResBlock(X, f = 3, filters = [512, 512, 2048], stage = 5, s = 2)
    X =identityBlock(X, f = 3, filters = [512, 512, 2048], stage = 5)
    X =identityBlock(X, f = 3, filters = [512, 512, 2048], stage = 5)
    
    X = keras.layers.AveragePooling2D(pool_size = (2, 2), name = "avg_pool")(X)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(outputClasses, activation = "softmax", name = "fc" + str(outputClasses), kernel_initializer = keras.initializers.glorot_uniform(seed = 0))(X)
    
    model = keras.Model(inputs = XInput, outputs = X, name = "ResNet_50")
    
    return model
    
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def startModel():
    
    x_train, y_train, x_test, y_test = getData()
    
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(y_train, 10).T
    Y_test = convert_to_one_hot(y_test, 10).T
    
    ResNet_50((28, 28, 1), 10)
    
    print("starting model")
    model = ResNet_50(input_shape = (64, 64, 1), outputClasses = 10)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(x_train, Y_train, epochs = 2, batch_size = 32)
    
    preds = model.evaluate(x_test, Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

#---------------------------------------------TESTS-------------------------------------------------

def testIdentityBlock():
    
    tf.reset_default_graph()

    with tf.Session() as test:
        np.random.seed(1)
        A_prev = tf.placeholder("float", [3, 4, 4, 6])
        X = np.random.randn(3, 4, 4, 6)
        A = identityBlock(A_prev, f = 2, filters = [2, 4, 6], stage = 1, inputShape = (28, 28, 6))
        test.run(tf.global_variables_initializer())
        out = test.run([A], feed_dict={A_prev: X, keras.backend.learning_phase(): 0})
        print("out = " + str(out[0][1][1][0]))    
    
    test.close()

def testConvResBlockBlock():
    
    tf.reset_default_graph()

    with tf.Session() as test:
        np.random.seed(1)
        A_prev = tf.placeholder("float", [3, 4, 4, 6])
        X = np.random.randn(3, 4, 4, 6)
        A = convultionalResBlock(A_prev, f = 2, filters = [2, 4, 6], stage = 1, inputShape = (28, 28, 6))
        test.run(tf.global_variables_initializer())
        out = test.run([A], feed_dict={A_prev: X, keras.backend.learning_phase(): 0})
        print("out = " + str(out[0][1][1][0]))    
    
    test.close()

