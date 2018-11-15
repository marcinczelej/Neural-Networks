# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 23:12:40 2018

@author: Marcin
"""
200, 32, [784, 150, 100, 10]

 train = 0.8443500000000002
 test = 0.8519000000000002
 
 300, 32, [784, 150, 10]
 
 train = 0.8466333333333335 wrongly recognized : 9202
 test = 0.8545000000000003 wrongly recognized : 1455
 
 600, 32, [784, 150, 10]
 
  train = 0.8467000000000001 wrongly recognized : 9198
 test = 0.8547000000000003 wrongly recognized : 1453
￼

[784, 10] fullNeuralNetwork(train_x, y_train, 0.0001, layer_dims, 600) 32

 train = 0.9059666666666669 wrongly recognized : 5642
 test = 0.9124000000000002 wrongly recognized : 876
 
[784, 10] fullNeuralNetwork(train_x, y_train, 0.0001, layer_dims, 1500) 32 
 
 train = 0.9164333333333335 wrongly recognized : 5014
 test = 0.9185000000000003 wrongly recognized : 815
 cost 0.1296205990308535
 
 
 [784, 10] fullNeuralNetwork(train_x, y_train, 0.001, layer_dims, 1500) 32 
 
 train = 0.9314500000000002 wrongly recognized : 4113
 test = 0.9253000000000002 wrongly recognized : 747
 cost = 0.08998270528234092
 
 ADAM