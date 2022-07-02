import numpy as np

# linear perceptron
def AND(x1, x2):
    Input = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(Input*w)+b
    if tmp > 0:
        return 1
    else :
        return 0

def NAND(x1, x2):
    Input = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(Input*w)+b
    if tmp > 0:
        return 1
    else :
        return 0

def OR(x1, x2):
    Input = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.4
    tmp = np.sum(Input*w)+b
    if tmp > 0:
        return 1
    else :
        return 0

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y