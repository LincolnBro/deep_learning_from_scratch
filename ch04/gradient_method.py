import numpy as np
from gradient_2d import numerical_gradient

def gradient_descent(f, init_x, lr=10, step_num=100):
    x = init_x
    for i in range(step_num):
        grad=numerical_gradient(f,x)
        x-=lr*grad
    return x

def function_2(x):
    return np.sum(x**2)

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x, lr=10, step_num=100)) # [-6.11110793e-10  8.14814391e-10] => [0,0]에 근사