#!/usr/bin/env python
#encoding:utf-8
"""
@author:
@time:2017/3/18 14:33
"""
# Import the functions to calculate feature descriptions
from skimage.feature import hog
import numpy as np
from sklearn.externals import joblib
# To read image file and save image feature descriptions
import os
import time
import glob
import pickle as pk
from config import *
from  matplotlib import pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
        return dict

def getData(filePath):
    TrainData = []
    for childDir in os.listdir(filePath):#记得移除数据库之外的所有文件
        if childDir != 'test_batch':
            f = os.path.join(filePath, childDir)
            data = unpickle(f)#逆向打包，加载到字典，数据都出
           # print(data)
           # print(data.keys())
            train = np.reshape(data['data'], (10000, 3, 32 * 32))#32像素，3通道红绿蓝
            labels = np.reshape(data['labels'], (10000, 1))#类别
            fileNames = np.reshape(data['filenames'], (10000, 1))#图片名
            datalebels = zip(train, labels, fileNames)
            TrainData.extend(datalebels)#加入总字典
        else:
            f = os.path.join(filePath, childDir)
            data = unpickle(f)
            test = np.reshape(data['data'], (10000, 3, 32 * 32))
            labels = np.reshape(data['labels'], (10000, 1))
            fileNames = np.reshape(data['filenames'], (10000, 1))
            TestData = zip(test, labels, fileNames)
    return TrainData, TestData

def hog(image,stride = 8, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(2, 2)):
    cx, cy = pixels_per_cell
    bx, by = cells_per_block
    sx, sy = image.shape
    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y
    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    gx = np.zeros((sx, sy), dtype=np.double)
    gy = np.zeros((sx, sy), dtype=np.double)
    eps = 1e-5
    grad = np.zeros((sx, sy, 2), dtype=np.double)
    for i in range(1, sx - 1):
        for j in range(1, sy - 1):
            gx[i, j] = image[i, j - 1] - image[i, j + 1]
            gy[i, j] = image[i + 1, j] - image[i - 1, j]
            grad[i, j, 0] = np.arctan(gy[i, j] / (gx[i, j] + eps)) * 180 / np.math.pi
            if gx[i, j] < 0:
                grad[i, j, 0] += 180
            grad[i, j, 0] = (grad[i, j, 0] + 360) % 360
            grad[i, j, 1] = np.sqrt(gy[i, j] ** 2 + gx[i, j] ** 2)
    normalised_blocks = np.zeros((n_blocksy, n_blocksx, by * bx * orientations))
    for y in range(n_blocksy):
        for x in range(n_blocksx):
            block = grad[y * stride:y * stride + 16, x * stride:x * stride + 16]
            hist_block = np.zeros(32, dtype='double')
            eps = 1e-5
            for k in range(by):
                for m in range(bx):
                    cell = block[k * 8:(k + 1) * 8, m * 8:(m + 1) * 8]
                    hist_cell = np.zeros(8, dtype='double')
                    for i in range(cy):
                        for j in range(cx):
                            n = int(cell[i, j, 0] / 45)
                            hist_cell[n] += cell[i, j, 1]
                    hist_block[(k * bx + m) * orientations:(k * bx + m + 1) * orientations] = hist_cell[:]
            normalised_blocks[y, x, :] = hist_block / np.sqrt(hist_block.sum() ** 2 + eps)
    return normalised_blocks.ravel()

def getFeat(TrainData, TestData):

    for data in TestData:
        image = np.reshape(data[0].T, (32, 32, 3))#顺序改变
        gray = rgb2gray(image)/255.0
        #print(gray.shape)#(32,32)
        fd = hog(image=gray)
        #print(fd.shape)#特征列向量(288,)
        '''#输出特征到文件

        #fd=hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualise=False, transform_sqrt=False,feature_vector=True, normalise=None)
        fd = np.concatenate((fd, data[1]))
        filename = list(data[2])
        fd_name = filename[0].split('.')[0]+'.feat'
        fd_path = os.path.join('./data/features/test/', fd_name)
        joblib.dump(fd, fd_path)
        '''

    print ("Test features are extracted and saved.")

    for data in TrainData:
        image = np.reshape(data[0].T, (32, 32, 3))
        '''
        gray = rgb2gray(image)/255.0
        fd = hog(image=gray)

      #  fd = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys',
       #          visualise=False, transform_sqrt=False, feature_vector=True, normalise=None)
        fd = np.concatenate((fd, data[1]))#把标签加到特征向量后面
        filename = list(data[2])#对应名字
        fd_name = filename[0].split('.')[0]+'.feat'#为了存储
        fd_path = os.path.join('./data/features/train/', fd_name)#路径
        joblib.dump(fd, fd_path)#写到路径
        '''
    print ("Train features are extracted and saved.")
def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140#012代表三个通道，灰度化
    return gray


if __name__ == '__main__':
    t0 = time.time()
    filePath = r'D:\zsm\JIA\HOG+SVM classifer\cifar-10-batches-py'#要用根目录，读入原始数据
    TrainData, TestData = getData(filePath)
    getFeat(TrainData, TestData)
    t1 = time.time()
    print ("Features are extracted and saved.")
    print ('The cast of time is:%f'%(t1-t0))


