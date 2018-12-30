#!/usr/bin/env python
#encoding:utf-8
"""
@author:

"""
# Import the functions to calculate feature descriptions
from skimage.feature import hog
import numpy as np
from sklearn.externals import joblib
import cv2
# To read image file and save image feature descriptions
import os
import time
from PIL import Image, ImageTk
import glob
import pickle as pk
from config import *
from  matplotlib import pyplot as plt
from skimage.feature import greycomatrix
from skimage.feature import greycoprops


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
        return dict

def getData(filePath):
    TrainData = []
    for childDir in os.listdir(filePath):
        if childDir != 'test_batch':
            f = os.path.join(filePath, childDir)
            data = unpickle(f)
           # print(data)
           # print(data.keys())
            train = np.reshape(data['data'], (10000, 3, 32 * 32))
            labels = np.reshape(data['labels'], (10000, 1))
            fileNames = np.reshape(data['filenames'], (10000, 1))
            datalebels = zip(train, labels, fileNames)
            TrainData.extend(datalebels)
        else:
            f = os.path.join(filePath, childDir)
            data = unpickle(f)
            test = np.reshape(data['data'], (10000, 3, 32 * 32))
            labels = np.reshape(data['labels'], (10000, 1))
            fileNames = np.reshape(data['filenames'], (10000, 1))
            TestData = zip(test, labels, fileNames)
    return TrainData, TestData


def extra_glcm(img):
    glcm = greycomatrix(img, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, normed=True, symmetric=True)
    contrast = greycoprops(glcm, 'contrast')
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    homogeneity = greycoprops(glcm, 'homogeneity')
    energy = greycoprops(glcm, 'energy')
    correlation = greycoprops(glcm, 'correlation')
    ASM = greycoprops(glcm, 'ASM')
    # entropy = greycoprops(glcm, 'entropy')

    avg_feature = np.array([np.average(contrast), np.average(dissimilarity), np.average(homogeneity),
                            np.average(correlation), np.average(ASM), np.var(contrast), np.var(dissimilarity),
                            np.var(homogeneity),
                            np.var(correlation), np.var(ASM)])
    return avg_feature


def getFeat(TrainData, TestData):

    for data in TestData:
        image = np.reshape(data[0].T, (32, 32, 3))
        #print(image)
        r=image[:,:,0]
        g=image[:,:,1]
        b=image[:,:,2]

        image = cv2.merge([b, g,r])
        #print(image)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = img_gray
        fd = extra_glcm(image)
       #print(fd.length)  # (288,)
       # fd = hog(gray, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
        #fd=hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualise=False, transform_sqrt=False,feature_vector=True, normalise=None)
        fd = np.concatenate((fd, data[1]))
        filename = list(data[2])
        fd_name = filename[0].split('.')[0]+'.feat'
        fd_path = os.path.join('./data/features/test3/', fd_name)
        joblib.dump(fd, fd_path)


    print ("Test features are extracted and saved.")

    for data in TrainData:
        image = np.reshape(data[0].T, (32, 32, 3))
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        image = cv2.merge([b, g, r])
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = img_gray
        fd = extra_glcm(image)
       # fd = hog(image=gray)
      #  fd = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys',
       #          visualise=False, transform_sqrt=False, feature_vector=True, normalise=None)
        fd = np.concatenate((fd, data[1]))
        filename = list(data[2])
        fd_name = filename[0].split('.')[0]+'.feat'
        fd_path = os.path.join('./data/features/train3/', fd_name)
        joblib.dump(fd, fd_path)
    print ("Train features are extracted and saved.")

def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray


if __name__ == '__main__':
    t0 = time.time()
    filePath = r'D:\zsm\JIA\HOG+SVM classifer\cifar-10-batches-py'
    TrainData, TestData = getData(filePath)
    getFeat(TrainData, TestData)
    t1 = time.time()
    print ("Features are extracted and saved.")
    print ('The cast of time is:%f'%(t1-t0))