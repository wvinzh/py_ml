#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Yangel'

import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import scipy.io as scio
import random

def displayData(X,example_width):
    m = np.size(X,0);
    n = np.size(X,1);
    print(m,n);
    #数字的长宽（pixel）
    example_width = example_width;#一个数字的像素宽
    example_height = n/example_width;#像素长
    #显示的长宽（个）
    display_width = int(np.floor(np.sqrt(m)));
    display_height = int(np.ceil(m/display_width));
    print(display_width,display_height);
    pad = 1;#数字间的间距
    #初始化显示矩阵
    display_arr = np.ones((pad+display_width*(example_width+pad),pad+display_height*(example_height+pad)));
    #填充像素
    curr_index = 0;
    for i in range(0,display_height):
        for j in range(0,display_width):
            if curr_index >= m:
                break;
            #填充第i行，第j列
            yy = pad + i*(pad+example_height);
            xx = pad + j*(pad+example_width);
            # temp_arr = display_arr[yy:yy+example_height,xx:xx+example_width];
            max_val = max(abs(X[curr_index, :]));
            display_arr[yy:(yy+example_height),xx:(xx+example_width)] =\
               np.reshape(X[curr_index,:],(example_height,example_width))/max_val;
            curr_index = curr_index + 1;
        if curr_index >= m:
            break;
    # fig = plt.figure();
    fig, ax = plt.subplots();
    ax.imshow(display_arr.T, cmap=plt.cm.gray);
    # plt.show(display_arr,cmap=plt.cm.gray);
    plt.show()

def g(z):
    r = 1/(1+np.exp(-z))
    return r;

def Sigmoid(z):
    return 1/(1 + np.exp(-z));

def costFuction(theta, x, y):
    m,n = x.shape;
    # print(theta.shape,y.shape)
    theta = theta.reshape((n,1));
    # y = y.reshape((m,1));
    # m = len(y);
    z = X.dot(theta);
    p1 = -(y*np.log(g(z)));
    p2 = -((1-y)*np.log(1-g(z)));
    return (np.sum(p1+p2))/m;

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

def gradient(theta, x, y):
    m , n = x.shape
    theta = theta.reshape((n,1));
    y = y.reshape((m,1))
    z = X.dot(theta);
    h = g(z);
    r = ((X.T).dot(h-y))/m;
    return r.flatten();
def Gradient(theta,x,y):
    m , n = x.shape
    theta = theta.reshape((n,1));
    y = y.reshape((m,1))
    sigmoid_x_theta = Sigmoid(x.dot(theta));
    grad = ((x.T).dot(sigmoid_x_theta-y))/m;
    return grad.flatten();

def predict(all_theta, X):
    m = len(X);
    # p = np.zeros(m);
    p =np.argmax(Sigmoid( X.dot(all_theta.T)),axis=1)+1 ;
    return p;

data=scio.loadmat("ex3data1.mat")
X = data['X']
y = data['y']

list1 = np.arange(np.size(X,0));
# print(list1)
random.shuffle(list1) ;
list2 = list1[100:200];
# print(list2)
displayData(X[list2,:],20)
#计算X的行列
m = np.size(X,0);
n = np.size(X,1);

X = np.c_[np.ones(m),X]
#初始化 theta
theta = np.zeros((n+1,1));
print(theta.shape)
#标签数 10
labels = 10;
#所有的theta
all_theta = np.zeros((labels,n+1))

for c in range(1,10):
    Result = op.minimize(fun = costFuction,x0 = theta,args = (X, y==c), method = 'TNC',jac=gradient)
    # Result = op.minimize(fun = CostFunc,x0 = theta,args = (X, y==c), method = 'TNC',jac=Gradient)
    all_theta[c-1,:]=Result.x
pred = predict(all_theta,X);

temp = y.flatten()
print(pred,temp)
print('\nTraining Set Accuracy: %f\n', np.mean((pred == temp)) * 100);


