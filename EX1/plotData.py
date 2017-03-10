#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Yangel'

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
from scipy import  optimize

def h_theta(x,theta):
    h = np.dot(x,theta)
    # print(h)
    return h

def computeCost(theta,y,x):
    j=0
    m=len(y)
    J = (np.dot(x,theta)-y)
    return j

def gradientDescent(X, y, theta, alpha, iterations):
    m=len(y);
    theta_temp = theta
    itera = iterations
    print (X[:,1])
    for i in range(iterations):
        theta[0]=theta_temp[0]-alpha/m*sum(np.dot(X,theta_temp)-y)
        zh=(np.dot(X,theta_temp)-y)
        zh2 = X[:,1]
        theta[1]=theta_temp[1]-alpha/m*sum(zh*zh2)
        theta_temp=theta
    return theta
#载入数据
x,y=np.loadtxt("ex1data1.txt",delimiter=',',usecols=(0,1), unpack=True);
#绘制散点图
plt.ylim(-5,25)
plt.xlim(4,24)
plt.plot(x,y, 'rx', 10);
# plt.plot([0,20],[-3.63,20])
plt.ylabel('Profit in $10,000s');
plt.xlabel('Population of City in 10,000s');

#增加一列x0
m=len(x);
o1=np.ones(m);
x=np.c_[o1,x]
print(x)
print(y)
#=初始化，θ初始化为0 ，迭代1500次
theta=np.zeros(2)
print(theta)
iterations = 1500;
alpha = 0.01;

# plst = leastsq(computeCost,theta,args=(y,x))
theta = optimize.fmin_cg(computeCost,theta,args=(y,x))
#计算theta
# theta = gradientDescent(x,y,theta,alpha,iterations)

print(theta)
#绘制直线
minX = min(x[:,1])
maxX = max(x[:,1])
plt.plot([minX,maxX],[minX*theta[1]+theta[0],maxX*theta[1]+theta[0]]);
plt.show();