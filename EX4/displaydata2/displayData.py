#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Yangel'
import numpy as np
import matplotlib.pyplot as plt


def displayData(X,example_width):
    m = np.size(X,0);
    n = np.size(X,1);
    # print(m,n);
    #数字的长宽（pixel）
    example_width = example_width;#一个数字的像素宽
    example_height = n/example_width;#像素长
    # print(example_height,example_width)
    #显示的长宽（个）
    display_width = int(np.floor(np.sqrt(m)));
    display_height = int(np.ceil(m/display_width));
    # print(display_width,display_height);
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

