from tkinter import Y
import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**3 + 0.1*x 

def tanget_line(f, x):
    d = numerical_diff(f, x)
    y = f(x)-d*x # 접선 방정식의 상수항 표현
    return lambda t: d*t+y

x=np.arange(-20.0,20.0,0.1)
y=function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tanget_line(function_1, 5)
y2 = tf(x)

plt.plot(x,y)
plt.plot(x,y2)
plt.show()