import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()

x = np.array([0.6, 0.9])
p = net.predict(x)
t = np.array([0, 0, 1])

f = lambda w: net.loss(x, t) # def f(w): return net.loss(x, t), w는 더미
dw = numerical_gradient(f, net.W)
print(dw)