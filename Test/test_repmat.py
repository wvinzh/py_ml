#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Yangel'

import numpy as np
y=[1,1,2,1,6,8,9,10,5,5,3,4,3,7]
Y=[];
E = np.eye(10);
for i in range(10):
    Y0 = np.where(y==i);
    print(Y0)
    Y[Y0,:]=np.tile(E[i,:],(np.size(Y0,0),1));
print(Y)