#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Yangel'

import numpy as np

a = np.array([0,1,2,3])
b = np.array([[2,5,6,3],
              [4,8,6,9]])
# c = np.array([2])
print(a.shape)
print(b[0][3])
c = np.eye(5,5)
print(b)
X,y=np.loadtxt("ex1data1.txt",delimiter=',',usecols=(0,1), unpack=True);
m=len(y);
d=np.ones(m);
e=np.c_[d,X];
l=[x[1] for x in e]
# e=np.matrix(e)
f=e[:1]
print(e[:,1])
