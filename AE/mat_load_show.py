# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   mat_load_show
   Author:      crd
   date:        2018/4/27
-------------------------------------------------
"""

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import autocode

data = sio.loadmat('Indian_pines.mat')
data_corrected = sio.loadmat('Indian_pines_corrected.mat')
data_gt = sio.loadmat('Indian_pines_gt.mat')
indian_pines = data['indian_pines']
indian_pines_gt = data_gt['indian_pines_gt']
indian_pines_corrected = data_corrected['indian_pines_corrected']

data = np.empty((145*145,200),dtype=float)
label = np.empty((145*145,1),dtype=int)
i = 0
for row in range(145):
    for col in range(145):
        data[i,:] = indian_pines_corrected[row,col,:]
        label[i] = indian_pines_gt[row,col]
        i += 1
index = np.where(label != 0)
x = np.empty((index[0].size,200))
i = 0
for rows in index[0]:
    x[i,:] = data[rows,:]
    i += 1
y = label[label != 0]
y.reshape((y.size,1))

