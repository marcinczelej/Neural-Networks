import numpy as np

def zero_padding(A_prev, pad_amount):
    
    #input (examples, x, y, layers)
    
    print("  shape = " + str(A_prev.shape))
    
    X_padded = np.pad(A_prev, ((0,0), (pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)), 'constant')
    return X_padded

def conv_single(input_slice, W, b):
    
    s = np.multiply(W, input_slice)+ b
    Z = np.sum(s)
    
    return Z 

def convolution(A_prev, W, b, parameters):
    
    #input  (examples, x, y, layers)
    #W -> filter (x, y, layers, filter_amount)
    # parameters (padding, stride)
    
    (f, f, f_layers, f_amount) = W.shape
    (examples, input_h, input_w, input_layers) = A_prev.shape
    
    pad = parameters["pad"]
    stride = parameters["stride"]
    
    output_h = int(np.floor((input_h + 2*pad - f)/stride)) + 1
    output_w = int(np.floor((input_w + 2*pad - f)/stride)) + 1
    
    Z = np.zeros((examples, output_h, output_w, f_amount))
    
    input_padded = zero_padding(A_prev, pad)
    
    for ex in range(examples):
        currentInput = input_padded[ex]
        for h in range(output_h):
            for w in range(output_w):
                for fltr_nr in range(f_amount):
                    min_h = stride*h
                    max_h = min_h + f
                    min_w = stride*w
                    max_w = min_w + f
                    
                    input_slice = currentInput[min_h: max_h, min_w:max_w, :]
                    
                    Z[ex, h, w, fltr_nr] = conv_single(input_slice, W[:, :, :, fltr_nr], b[:, :, : , fltr_nr])
    
    
    cache = (A_prev, W, b, parameters)
    return Z, cache

def pooling(X, parameters, mode = "max"):

    #input  (examples, x, y, layers)
    # parameters (f, stride)

    f = parameters["f"]
    stride = parameters["stride"]
    
    (examples, input_h, input_w, input_layers) = X.shape
    
    output_h = int(np.floor(input_h - f)/stride) +1
    output_w = int(np.floor(input_w - f)/stride) +1
    
    Z = np.zeros((examples, output_h, output_w, input_layers))
    
    for ex in range(examples):
        currentX = X[ex]
        for h in range(output_h):
            for w in range(output_w):
                for layer in range(input_layers):
                    min_h = stride*h
                    max_h = min_h + f
                    min_w = stride*w
                    max_w = min_w + f
                    
                    inputSplice =  currentX[min_h:max_h, min_w:max_w, layer]
                
                    if mode == "max":
                        Z[ex, h, w, layer] = np.max(inputSplice)
                    if mode == "average":
                        Z[ex, h, w, layer] = np.mean(inputSplice)
                
    cache = (X, parameters)
    return Z, cache

def checkConv():
    np.random.seed(1)
    A_prev = np.random.randn(10,4,4,3)
    W = np.random.randn(2,2,3,8)
    b = np.random.randn(1,1,1,8)
    hparameters = {"pad" : 2,
               "stride": 1}

    Z, cache_conv = convolution(A_prev, W, b, hparameters)
    print("Z's mean =", np.mean(Z))
    print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
    
def checkPooling():
    np.random.seed(1)
    A_prev = np.random.randn(2, 4, 4, 3)
    hparameters = {"stride" : 1, "f": 4}

    A, cache = pooling(A_prev, hparameters)
    print("mode = max")
    print("A =", A)
    print()
    A, cache = pooling(A_prev, hparameters, mode = "average")
    print("mode = average")
    print("A =", A)