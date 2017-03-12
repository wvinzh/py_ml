#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Yangel'

import numpy as np

a = np.array([0,1,2,3])
b = np.array([[2,5,6,3],
              [4,8,6,9]])
print(b)
np.save("barr",b);
c = np.load("barr.npy");
print(c)
