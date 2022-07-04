import sys, os
import pickle
import numpy as np
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
from common.functions import sigmoid, softmax

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    A1 = np.dot(x,W1)+b1
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1,W2)+b2
    Z2 = sigmoid(A2)
    A3 = np.dot(Z2,W3)+b3
    Y = softmax(A3)

    return Y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]: 
        accuracy_cnt+=1

print("Accuracy: "+str(float(accuracy_cnt)/len(x)))