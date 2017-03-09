# -*- coding : utf-8 -*-

import numpy as np

x= np.array([1.0,2.0,3.0])
print (x)
y= np.array([4.0,6.0,9.0])
print (x+y)

a= np.array([[1,3],[4,5]])
b= np.array([[5,6],[8,0]])
print (a)
print (b)
print (a+b)

for row in a:
    print (row)

x = np.array([[51,13],[20,12],[14,0]])
y= x.flatten()
print(y)

# boolean filter
z = y > 13
z1 = np.array ([False, False, True, True, False, False])

print (y[z])
print (y[z1])

import matplotlib.pyplot as plt

xx = np.arange(0, 6, 0.1)
yy1 = np.sin(xx)
yy2 = np.cos(xx)

plt.plot(xx, yy1, label='sin')
plt.plot(xx, yy2, linestyle='--', label='cos')
plt.xlabel('x')
plt.ylabel('y')
plt.title("sin & cos")
plt.legend()
plt.show()

from matplotlib.image import imread

img = imread("gt.jpeg")
plt.imshow(img)
plt.show()