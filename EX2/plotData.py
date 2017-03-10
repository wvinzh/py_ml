#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Yangel'
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(z):
    g = np.zeros(np.size(z))
    g = 1/(1+np.exp(-z))
    return g

def J(X,y,theta):
    m = len(y)
    JJ= -1 * sum( y * np.log( sigmoid(X*theta) ) + (1 - y ) *np.log( (1 - sigmoid(X*theta)) ) ) / m ;
    return JJ;

def gradientDescent(X, y, theta, alpha, iterations):
    theta_temp = theta
    m = len(y)
    n = len(theta)
    while 1==1:
        for j in range(n):
            xj = X[:,j]
            theta[j] = theta_temp[j] - alpha/m*sum((sigmoid(np.dot(X,theta_temp))-y)*xj)
        if(sum((theta-theta_temp)*(theta-theta_temp))/3<= 10^(-1)):
            break
        else:
            print(theta)
            theta_temp = theta
    return theta


#加载数据
x1,x2,y = np.loadtxt("ex2data1.txt",delimiter=',',usecols=(0,1,2),unpack='true')


#创建figure
plt.figure(figsize=(8,4))
# plt.ylim(30,100)
# plt.xlim(20,100)
m = np.size(y)
#初始化
ones = np.ones(m)
X = np.c_[x1,x2]
X = np.c_[ones,X]

theta = np.zeros(3);
alpha = 0.03
iteration = 1000
print(gradientDescent(X,y,theta,alpha,iteration))

# for i in range(m):
#     if(y[i]==1):
#         plt.plot(x1[i],x2[i],'k+',color='red')
#     else:
#         plt.plot(x1[i],x2[i],'ko',color='blue')
# plt.xlabel('first score');
# plt.ylabel('second score');
# plt.show()