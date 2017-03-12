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

def predict(Theta1, Theta2, X):
    # print(Theta1.shape, Theta2.shape,X.shape)
    m = np.size(X,0);
    n = np.size(X,1);
    num_labels = np.size(Theta2,0);
    #computing
    a1 = X;
    a1 = np.c_[np.ones(m),a1];
    a2 = Sigmoid(a1.dot(Theta1.T));
    a2 = np.c_[np.ones(m),a2];
    a3 = Sigmoid(a2.dot(Theta2.T));
    p = np.argmax(a3,axis=1)+1;
    return p;

print('Loading and Visualizing Data ...\n')
#loading data
data = scio.loadmat('ex4data1.mat');
x = data['X'];
y = data['y'];
m = np.size(x,0);#行，训练集大小
n = np.size(x,1);#列，特征数量
#相当于随机100个数字
sel = np.arange(m);
random.shuffle(sel);
sel = sel[100:200];
#选择100个数据显示
dsp.displayData(x[sel,:],20);

# os.system('pause');

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
theta = scio.loadmat('ex4weights.mat');
Theta1 = theta['Theta1'];
Theta2 = theta['Theta2'];
print(Theta1.shape,Theta2.shape)
theta_temp = np.load("nn_theta.npy");
theta_temp1 = theta_temp[:25*401];
theta_temp2 = theta_temp[25*401:];
print(theta_temp1.shape,theta_temp2.shape)
theta_temp1 = theta_temp1.reshape((25,401));
theta_temp2 = theta_temp2.reshape((10,26));

# p = predict(Theta1,Theta2,x);
p = predict(theta_temp1,theta_temp2,x);
temp = y.flatten()
print('\nTraining Set Accuracy: %f\n', np.mean((p == temp)) * 100);

for i in range(m):
    index = int(random.random()*m);
    # print(index)
    dsp.displayData(x[index:index+1,:],20)
    pred = predict(theta_temp1,theta_temp2,x[index:index+1,:]);
    print(pred[0]);