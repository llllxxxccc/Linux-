#!/usr/bin/env python
#encoding:utf-8
"""
@author:
@time:2017/3/19 11:08
"""
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import numpy as np#数组操作
import glob
import os
import time
from config import *
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

K = np.zeros((10),dtype=np.int)
if __name__ == "__main__":
    t0 = time.time()
    clf_type = 'LIN_SVM'#选择线性分类器
    fds = []
    labels = []
    num = 0
    total = 0

    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):#在config.cfg改
        data = joblib.load(feat_path)
        fds.append(data[:-1])#因为最后一个是label，去掉倒数第一个
        labels.append(data[-1])


    if clf_type is 'LIN_SVM':
        #clf = LinearSVC(C=1.5)
        #clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
        #clf = neighbors.KNeighborsClassifier(n_neighbors=7)
        #clf = tree.DecisionTreeClassifier()
        clf= GaussianNB()
        print ("Training a  SVM Classifier.")
        clf.fit(fds, labels)#输入数据
        # If feature directories don't exist, create them
        # if not os.path.isdir(os.path.split(model_path)[0]):
        #     os.makedirs(os.path.split(model_path)[0])

        joblib.dump(clf, model_path_2)#路径在config保存
        clf = joblib.load(model_path_2)#加载

        print ("Classifier saved to {}".format(model_path_5))
        for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):#开始测试
            total += 1#到一万截至
            data_test = joblib.load(feat_path)#载入测试特征
            data_test_feat = data_test[:-1].reshape((1, -1))#载入成一行，去掉标签1是行，-1，烈数不确定
           # print(data_test_feat.shape)
            result = clf.predict(data_test_feat)
            if int(result) == int(data_test[-1]):#和标签对比
                num += 1
                k = int(data_test[-1])
                K[k] = K[k]+1;#测试每类识别率
        rate = float(num) / total
        t1 = time.time()
        print('The classification accuracy is %f' % rate)
        print('The cast of time is :%f' % (t1 - t0))
        print(K)