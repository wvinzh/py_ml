#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Yangel'
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

def Sigmoid(z):
    return 1/(1 + np.exp(-z));

def Gradient(theta,x,y):
    m , n = x.shape
    theta = theta.reshape((n,1));
    y = y.reshape((m,1))
    sigmoid_x_theta = Sigmoid(x.dot(theta));
    grad = ((x.T).dot(sigmoid_x_theta-y))/m;
    return grad.flatten();

def CostFunc(theta,x,y):
    m,n = x.shape;
    theta = theta.reshape((n,1));
    y = y.reshape((m,1));
    term1 = np.log(Sigmoid(x.dot(theta)));
    term2 = np.log(1-Sigmoid(x.dot(theta)));
    term1 = term1.reshape((m,1))
    term2 = term2.reshape((m,1))
    term = y * term1 + (1 - y) * term2;
    J = -((np.sum(term))/m);
    return J;

# intialize X and y
# X = np.array([[1,2,3],[1,3,4]]);
# y = np.array([[1],[0]]);

#加载数据
x1,x2,y1 = np.loadtxt("ex2data1.txt",delimiter=',',usecols=(0,1,2),unpack='true')

m = np.size(y1)
#初始化
ones = np.ones(m)
X = np.c_[x1,x2]
X = np.c_[ones,X]
m , n = X.shape;
initial_theta = np.zeros(n);

Result = op.minimize(fun = CostFunc,x0 = initial_theta,args = (X, y1), method = 'TNC',jac=Gradient)
optimal_theta = Result.x;
print(optimal_theta)

#绘制图
plt.figure(figsize=(8,4))

for i in range(m):
    if(y1[i]==1):
        plt.plot(x1[i],x2[i],'k+',color='red')
    else:
        plt.plot(x1[i],x2[i],'ko',color='blue')
plt.xlabel('first score');
plt.ylabel('second score');

#绘制分类边界
x_min = min(x1)-5;
x_max = max(x1)+5;
plt.plot([x_min,x_max],[-(optimal_theta[0]+optimal_theta[1]*x_min)/optimal_theta[2],-(optimal_theta[0]+optimal_theta[1]*x_max)/optimal_theta[2]])
plt.show()
