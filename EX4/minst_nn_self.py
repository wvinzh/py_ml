#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Yangel'

import displaydata.displayData as dsp
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import scipy.io as scio
import random
import os

def Sigmoid(z):
    return 1/(1+np.exp(-z));

def  h_theta_x(Theta1, Theta2, X):
    m = np.size(X,0);
    n = np.size(X,1);
    num_labels = np.size(Theta2,0);
    a1 = X;
    a1 = np.c_[np.ones(m),a1];
    a2 = Sigmoid(a1.dot(Theta1.T));
    a2 = np.c_[np.ones(m),a2];
    a3 = Sigmoid(a2.dot(Theta2.T));
    return a3

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lamda):
    #computing Theta1 Theta2
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1)));
    Theta2 = np.reshape(nn_params[((hidden_layer_size * (input_layer_size + 1))):],(num_labels, (hidden_layer_size + 1)));
    print(Theta1.shape, Theta2.shape)
    m = np.size(X,0);
    J = 0;
    Theta1_grad = np.zeros(Theta1.shape);
    Theta2_grad = np.zeros(Theta2.shape);
    # print(Theta1_grad.shape, Theta2_grad.shape)
    #将y向量化
    E = np.eye(num_labels);
    Y = np.reshape(E[y-1],(m,num_labels));
    a3 = h_theta_x(Theta1,Theta2,X);
    # print(a3[0],Y[0])
    temp1 = Theta1[::,1:];   #% 先把theta(1)拿掉，不参与正则化
    temp2 = Theta2[::,1:];
    # print(temp1.shape,temp2.shape)
    temp1 = sum(temp1**2);     #% 计算每个参数的平方，再就求和
    temp2 = sum(temp2**2);

    cost = Y * np.log(a3) + (1 - Y ) * np.log((1 - a3)); # % cost是m*K(5000*10)的结果矩阵  sum(cost(:))全部求和
    J= -1  * sum(sum(cost[::,::]))/ m + lamda * ( sum(temp1[:])+ sum(temp2[:]))/(2*m);
    print(J)

input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   #25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10

print('Loading and Visualizing Data ...\n')
data = scio.loadmat('ex4data1.mat');
x = data['X']
y = data['y']
# print(X.shape, y.shape)
m = np.size(x,0);#行，训练集大小
n = np.size(x,1);#列，特征数量
#相当于随机100个数字
sel = np.arange(m);
random.shuffle(sel);
sel = sel[100:200];
#选择100个数据显示
dsp.displayData(x[sel,:],20);
###########part2#################
print('\nLoading Saved Neural Network Parameters ...\n')

#Load the weights into variables Theta1 and Theta2
th = scio.loadmat('ex4weights.mat');
Theta1 = th['Theta1']
Theta2 = th['Theta2']

list_theta1 = Theta1.flatten();
list_theta2 = Theta2.flatten();
nn_params = np.concatenate((list_theta1,list_theta2));
# print(nn_params.shape)
nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,x,y,1)
# print(Theta1.shape, Theta2.shape)