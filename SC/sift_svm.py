# -*- coding: utf-8 -*-
"""
Created on Sat May  5 22:29:39 2018

@author: Administrator
"""

from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy as np
import glob
import os
import time
from config import *
from sklearn.cluster import KMeans

if __name__ == "__main__":
    t0 = time.time()
    clf_type = 'LIN_SVM'
    fds = []
    labels = []
    num = 0
    total = 0

    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
        data = joblib.load(feat_path)
        fds.append(data[:-1])
        labels.append(data[-1])

    fds = np.array(fds)  # sift特征存储为list列表，此处转换为矩阵
    fds =np.reshape(fds,[50000,fds.shape[1]])
    if clf_type is 'LIN_SVM':
        clf = SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
        print ("Training a nonLinear SVM Classifier.")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        # if not os.path.isdir(os.path.split(model_path)[0]):
        #     os.makedirs(os.path.split(model_path)[0])

        joblib.dump(clf, model_path1)
        clf = joblib.load(model_path1)

        print ("Classifier saved to {}".format(model_path1))
        for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
            total += 1
            data_test = joblib.load(feat_path)
            data_test_feat = data_test[:-1].reshape((1, -1))
           # print(data_test_feat.shape)
            result = clf.predict(data_test_feat)
            if int(result) == int(data_test[-1]):
                num += 1
        rate = float(num) / total
        t1 = time.time()
        print('The classification accuracy is %f' % rate)
        print('The cast of time is :%f' % (t1 - t0))