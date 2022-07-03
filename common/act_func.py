import numpy as np

def step_function(x):
    y = x>0 # y is boolean array
    return y.astype(np.int) # true =>1 false =>0

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def identity_function(x): # identity function, regression
    return x

def softmax(x): # softmax function, classification
    c = np.max(x) # overflow 대책
    return np.exp(x-c)/np.sum(np.exp(x-c))