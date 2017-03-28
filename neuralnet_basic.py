# -*- coding : utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append("../deep-learning-from-scratch")
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

def numerical_diff (f,x):
    h = 1e-4
    return (f(x+h) - f(x-h))/(2*h)

def function_part_diff(x):
    return x**2 + 4.0**2

def function_part_diff2(y):
    return 3.0**2 + y**2

def function_1(x):
    return 0.01*x**2+0.1*x


def gradient_descent(f, init_x, lr = 0.01, step_num= 100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad

    return x

def plt_func():
    x,y = np.arange(-3.0,3.0,0.03), np.arange(-3.0,3.0,0.03)
    x,y =np.meshgrid(x,y)
    #    print (x)
    #    print (x.shape)
    #    print (y)
    #    print (y.shape)
    #    z = function_2(x,y)
    z = function_3(np.array([x,y]))
    fig = plt.figure(figsize=plt.figaspect(0.7))
    #    print (z)
    #    print (z.shape)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_wireframe(x, y, z, rstride=10, cstride=10)
    plt.show()

def function_3(x):
    return x[0]**2 + x[1]**2

def function_2(x,y):
    return x**2 + y**2

def numerical_gradient_local (f, x):
    h= 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        tv = x[i]
        x[i] = tv + h
        fxh1 = f(x)
        x[i] = tv - h
        fxh2 = f(x)
        grad [i] = (fxh1 - fxh2) / (2*h)
        x[i] = tv

    return grad

class simplenet:
    def __init__(self):
        self.S = np.random.randn(2,3)

    def predict (self, x):
        return np.dot(x, self.S)

    def loss (self, x, t):
        z= self.predict(x)
        y= softmax(z)
        loss= cross_entropy_error(y,t) # t is one-hot encoded answer label
        return loss


def main():
    #yy = gradient_descent(function_3, np.array([-3.0,4.0]), 1e-10)
    yy=simplenet()
    xx = np.array([0.6,0.9])
    p =yy.predict(xx)

    t = (p>=p.max()).astype(np.int)

    def fnc(W):
        return yy.loss(xx,t)

    dW= numerical_gradient(fnc,yy.S)
    print (dW)


if __name__ == "__main__":
    main()
