# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   mat_load_show
   Author:      crd
   date:        2018/4/27
-------------------------------------------------
"""

import numpy as np
import scipy.io as sio
import sklearn.preprocessing as prep
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

data = sio.loadmat('Indian_pines.mat')
data_corrected = sio.loadmat('Indian_pines_corrected.mat')
data_gt = sio.loadmat('Indian_pines_gt.mat')
indian_pines = data['indian_pines']
indian_pines_gt = data_gt['indian_pines_gt']
indian_pines_corrected = data_corrected['indian_pines_corrected']

data = np.empty((145 * 145, 200), dtype=float)
label = np.empty((145 * 145, 1), dtype=int)
i = 0
for row in range(145):
    for col in range(145):
        data[i, :] = indian_pines_corrected[row, col, :]
        label[i] = indian_pines_gt[row, col]
        i += 1
index = np.where(label != 0)
x = np.empty((index[0].size, 200))
i = 0
for rows in index[0]:
    x[i, :] = data[rows, :]
    i += 1
y = label[label != 0]
y = y.reshape((y.size, 1))
kf = KFold(n_splits=100)
kf.get_n_splits(x)
for train_index, test_index in kf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]



def standard_scale(X_train, X_test, Y_train, Y_test):
    """
    生成TensorFlow合适的样本。
    :param X_train:
    :param X_test:
    :param Y_train:
    :param Y_test:
    :return:
    """

    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    x_train = X_train.reshape(-1, 1, 1, 200)
    x_test = X_test.reshape(-1, 1, 1, 200)
    y_train = OneHotEncoder().fit_transform(Y_train).todense()
    y_test = OneHotEncoder().fit_transform(Y_test).todense()
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = standard_scale(X_train, X_test, y_train, y_test)
