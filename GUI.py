from tkinter import *
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"#防止显卡占用
import sys
import json
import time
import numpy as np
from keras.models import Sequential
from tkinter import filedialog
from PIL import Image, ImageTk
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from sklearn import neighbors
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.models import load_model
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras import metrics
from keras.optimizers import SGD,RMSprop,Adam
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import matplotlib.pyplot as plt


class CNNNet:

    @staticmethod
    def createNet(input_shapes,nb_class):

        feature_layers = [
        BatchNormalization(input_shape=input_shapes),
        Conv2D(64,3,3,border_mode="same"),
        Activation("relu"),
        BatchNormalization(),
        Conv2D(64,3,3,border_mode="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2,2),strides=(2,2)),
        BatchNormalization(),
        Conv2D(128,3,3,border_mode="same"),
        Activation("relu"),
        BatchNormalization(),
        Dropout(0.5),
        Conv2D(128,3,3,border_mode="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2,2),strides=(2,2)),
        BatchNormalization(),
        Dropout(0.5),
        Conv2D(128,3,3,border_mode="same"),
        Activation("relu"),
        Dropout(0.5),
        Conv2D(128,3,3,border_mode="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2,2),strides=(2,2)),
        BatchNormalization()
        ]

        classification_layer=[
        Flatten(),
        Dense(512),
        Activation("relu"),
        Dropout(0.5),
        Dense(nb_class),
        Activation("softmax")
        ]
        model = Sequential(feature_layers+classification_layer)
        return model
#parameters
NB_EPOCH = 40
BATCH_SIZE = 128
VERBOSE = 1
VALIDATION_SPLIT = 0.2
IMG_ROWS=32
IMG_COLS = 32
NB_CLASSES = 10
INPUT_SHAPE =(IMG_ROWS,IMG_COLS,3)

def rgb2gray(im):
    gray = im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140
    return gray


##照片分析
def hog(image, stride=8, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
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


def extr_sift(image):
    sift = image.SIFT()
    return sift

def extra_lbp(img):
    radius = 1
    n_point = radius * 8
    lbp = local_binary_pattern(img, n_point, radius)

    max_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    return hist

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

def selectPath():
    str = filedialog.askopenfilename()  # 打开任意路径文件

    emotion_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 加载json并且创建模型拱门
    model = joblib.load('C:\\Users\\郏紫宇\\Desktop\\HOG+SVM classifer\\data\\models\\svm.model')  # 加载模型，必须是绝对路径

    # 加载权重到新的模型中

    def predict_emotion(face_image_gray):  # a single cropped face

        result = model.predict(face_image_gray)
        if int(result) == 0:
            return 'airplane'
        elif int(result) == 1:
            return 'automobile'
        elif int(result) == 2:
            return 'bird'
        elif int(result) == 3:
            return 'cat'
        elif int(result) == 4:
            return 'deer'
        elif int(result) == 5:
            return 'dog'
        elif int(result) == 6:
            return 'frog'
        elif int(result) == 7:
            return 'horse'
        elif int(result) == 8:
            return 'ship'
        elif int(result) == 9:
            return 'trunk'

    # img = cv2.imread("E:/灵山游/013.JPG", cv2.IMREAD_GRAYSCALE)

    im = Image.open(str)  # 打开图片
    im2 = im  # 保存元图片
    im = im.resize((32, 32))  # 先处理大小
    frame = np.array(im)  ##将image读取到的照片转换为数组格式

    frame2 = np.array(im2)
    r, g, b = cv2.split(frame2)  # 分割rgb
    frame2 = cv2.merge([b, g, r])  # 转置转杯bgr

    '''
    r, g, b = cv2.split(frame)
    frame = cv2.merge([b, g, r])###cv2遵循[B,G,R],而python的PIL读取到的图片为[R,G,B],所以进行数组转换
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, 1)  # 转化为灰度图片

    img_gray=img_gray/255.0
    '''
    print(frame.shape)
    img_gray = rgb2gray(frame) / 255.0
    print(img_gray.shape)
    # img = cv2.resize(img_gray, (32, 32),interpolation=cv2.INTER_AREA)
    #plt.imshow(img_gray, cmap='gray')
    #plt.show()
    # print(img.shape)
    fd = hog(image=img_gray)

    fd = fd[:].reshape((1, -1))  # 分类器要求行向量1×288
    print(fd.shape)
    emotion = predict_emotion(fd)

    cv2.putText(frame2, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # 给图片打标签。显示文字
    # cv2.putText(I,'there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
    # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色bgr/字体粗细
    # 呈现结果框

    frame2 = cv2.resize(frame2, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)  # 4x4像素邻域的双三次插值,担心图片太大，到百分之六十
    cv2.imshow('Photo', frame2)


def selectPath_1():
    str = filedialog.askopenfilename()  # 打开任意路径图片

    emotion_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 加载json并且创建模型拱门
    model1 = joblib.load('C:\\Users\\郏紫宇\\Desktop\\HOG+SVM classifer\\data\\models\\svm.model_2')

    # 加载权重到新的模型中

    def predict_emotion(face_image_gray):  # a single cropped face

        result = model1.predict(face_image_gray)
        if int(result) == 0:
            return 'airplane'
        elif int(result) == 1:
            return 'automobile'
        elif int(result) == 2:
            return 'bird'
        elif int(result) == 3:
            return 'cat'
        elif int(result) == 4:
            return 'deer'
        elif int(result) == 5:
            return 'dog'
        elif int(result) == 6:
            return 'frog'
        elif int(result) == 7:
            return 'horse'
        elif int(result) == 8:
            return 'ship'
        elif int(result) == 9:
            return 'trunk'

    # img = cv2.imread("E:/灵山游/013.JPG", cv2.IMREAD_GRAYSCALE)

    im = Image.open(str)
    im2 = im
    im = im.resize((32, 32))
    frame = np.array(im)  ##将image读取到的照片转换为数组格式

    frame2 = np.array(im2)
    r, g, b = cv2.split(frame2)
    frame2 = cv2.merge([b, g, r])

    print(frame.shape)
    img_gray = rgb2gray(frame) / 255.0  # 灰度函数然后归一化
    print(img_gray.shape)

    fd = hog(image=img_gray)

    fd = fd[:].reshape((1, -1))
    print(fd.shape)
    emotion = predict_emotion(fd)

    cv2.putText(frame2, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(I,'there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
    # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    # 呈现结果框

    frame2 = cv2.resize(frame2, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)  # 4x4像素邻域的双三次插值
    cv2.imshow('Photo', frame2)

def selectPath_2():
    str = filedialog.askopenfilename()  # 打开任意路径图片

    emotion_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 加载json并且创建模型拱门
    model1 = joblib.load('C:\\Users\\郏紫宇\\Desktop\\HOG+SVM classifer\\data\\models\\svm.model_3')

    # 加载权重到新的模型中

    def predict_emotion(face_image_gray):  # a single cropped face

        result = model1.predict(face_image_gray)
        if int(result) == 0:
            return 'airplane'
        elif int(result) == 1:
            return 'automobile'
        elif int(result) == 2:
            return 'bird'
        elif int(result) == 3:
            return 'cat'
        elif int(result) == 4:
            return 'deer'
        elif int(result) == 5:
            return 'dog'
        elif int(result) == 6:
            return 'frog'
        elif int(result) == 7:
            return 'horse'
        elif int(result) == 8:
            return 'ship'
        elif int(result) == 9:
            return 'trunk'

    # img = cv2.imread("E:/灵山游/013.JPG", cv2.IMREAD_GRAYSCALE)

    im = Image.open(str)
    im2 = im
    im = im.resize((32, 32))
    frame = np.array(im)  ##将image读取到的照片转换为数组格式

    frame2 = np.array(im2)
    r, g, b = cv2.split(frame2)
    frame2 = cv2.merge([b, g, r])

    print(frame.shape)
    img_gray = rgb2gray(frame) / 255.0  # 灰度函数然后归一化
    print(img_gray.shape)

    fd = hog(image=img_gray)

    fd = fd[:].reshape((1, -1))
    print(fd.shape)
    emotion = predict_emotion(fd)

    cv2.putText(frame2, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(I,'there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
    # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    # 呈现结果框

    frame2 = cv2.resize(frame2, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)  # 4x4像素邻域的双三次插值
    cv2.imshow('Photo', frame2)
def selectPath_3():
    str = filedialog.askopenfilename()  # 打开任意路径图片

    emotion_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 加载json并且创建模型拱门
    model1 = joblib.load('C:\\Users\\郏紫宇\\Desktop\\HOG+SVM classifer\\data\\models\\svm.model_4')

    # 加载权重到新的模型中

    def predict_emotion(face_image_gray):  # a single cropped face

        result = model1.predict(face_image_gray)
        if int(result) == 0:
            return 'airplane'
        elif int(result) == 1:
            return 'automobile'
        elif int(result) == 2:
            return 'bird'
        elif int(result) == 3:
            return 'cat'
        elif int(result) == 4:
            return 'deer'
        elif int(result) == 5:
            return 'dog'
        elif int(result) == 6:
            return 'frog'
        elif int(result) == 7:
            return 'horse'
        elif int(result) == 8:
            return 'ship'
        elif int(result) == 9:
            return 'trunk'

    # img = cv2.imread("E:/灵山游/013.JPG", cv2.IMREAD_GRAYSCALE)

    im = Image.open(str)
    im2 = im
    im = im.resize((32, 32))
    frame = np.array(im)  ##将image读取到的照片转换为数组格式

    frame2 = np.array(im2)
    r, g, b = cv2.split(frame2)
    frame2 = cv2.merge([b, g, r])

    print(frame.shape)
    img_gray = rgb2gray(frame) / 255.0  # 灰度函数然后归一化
    print(img_gray.shape)

    fd = hog(image=img_gray)

    fd = fd[:].reshape((1, -1))
    print(fd.shape)
    emotion = predict_emotion(fd)

    cv2.putText(frame2, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(I,'there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
    # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    # 呈现结果框

    frame2 = cv2.resize(frame2, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)  # 4x4像素邻域的双三次插值
    cv2.imshow('Photo', frame2)

def selectPath_4():
    str = filedialog.askopenfilename()  # 打开任意路径图片

    emotion_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 加载json并且创建模型拱门
    model1 = joblib.load('C:\\Users\\郏紫宇\\Desktop\\HOG+SVM classifer\\data\\models\\svm.model_5')

    # 加载权重到新的模型中

    def predict_emotion(face_image_gray):  # a single cropped face

        result = model1.predict(face_image_gray)
        if int(result) == 0:
            return 'airplane'
        elif int(result) == 1:
            return 'automobile'
        elif int(result) == 2:
            return 'bird'
        elif int(result) == 3:
            return 'cat'
        elif int(result) == 4:
            return 'deer'
        elif int(result) == 5:
            return 'dog'
        elif int(result) == 6:
            return 'frog'
        elif int(result) == 7:
            return 'horse'
        elif int(result) == 8:
            return 'ship'
        elif int(result) == 9:
            return 'trunk'

    # img = cv2.imread("E:/灵山游/013.JPG", cv2.IMREAD_GRAYSCALE)

    im = Image.open(str)
    im2 = im
    im = im.resize((32, 32))
    frame = np.array(im)  ##将image读取到的照片转换为数组格式

    frame2 = np.array(im2)
    r, g, b = cv2.split(frame2)
    frame2 = cv2.merge([b, g, r])

    print(frame.shape)
    img_gray = rgb2gray(frame) / 255.0  # 灰度函数然后归一化
    print(img_gray.shape)

    fd = hog(image=img_gray)

    fd = fd[:].reshape((1, -1))
    print(fd.shape)
    emotion = predict_emotion(fd)

    cv2.putText(frame2, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(I,'there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
    # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    # 呈现结果框

    frame2 = cv2.resize(frame2, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)  # 4x4像素邻域的双三次插值
    cv2.imshow('Photo', frame2)


def selectPath1():
    str = filedialog.askopenfilename()  # 打开任意路径图片

    emotion_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 加载json并且创建模型拱门
    model1 = joblib.load('C:\\Users\\郏紫宇\\Desktop\\HOG+SVM classifer\\data\\models\\svm.model1')

    # 加载权重到新的模型中

    def predict_emotion(face_image_gray):  # a single cropped face

        result = model1.predict(face_image_gray)
        if int(result) == 0:
            return 'airplane'
        elif int(result) == 1:
            return 'automobile'
        elif int(result) == 2:
            return 'bird'
        elif int(result) == 3:
            return 'cat'
        elif int(result) == 4:
            return 'deer'
        elif int(result) == 5:
            return 'dog'
        elif int(result) == 6:
            return 'frog'
        elif int(result) == 7:
            return 'horse'
        elif int(result) == 8:
            return 'ship'
        elif int(result) == 9:
            return 'trunk'

    # img = cv2.imread("E:/灵山游/013.JPG", cv2.IMREAD_GRAYSCALE)

    im = Image.open(str)
    im2 = im
    im = im.resize((32, 32))
    frame = np.array(im)  ##将image读取到的照片转换为数组格式

    frame2 = np.array(im2)
    r, g, b = cv2.split(frame2)
    frame2 = cv2.merge([b, g, r])

    print(frame.shape)
    img_gray = rgb2gray(frame) / 255.0  # 灰度函数然后归一化
    print(img_gray.shape)
    # img = cv2.resize(img_gray, (32, 32),interpolation=cv2.INTER_AREA)
    #plt.imshow(img_gray, cmap='gray')
    #plt.show()
    # print(img.shape)
    fd = extr_sift(image=img_gray)

    fd = fd[:].reshape((1, -1))
    print(fd.shape)
    emotion = predict_emotion(fd)

    cv2.putText(frame2, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(I,'there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
    # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    # 呈现结果框

    frame2 = cv2.resize(frame2, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)  # 4x4像素邻域的双三次插值
    cv2.imshow('Photo', frame)



def selectPath2():
    str = filedialog.askopenfilename()  # 打开任意路径图片

    emotion_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 加载json并且创建模型拱门
    model1 = joblib.load('C:\\Users\\郏紫宇\\Desktop\\HOG+SVM classifer\\data\\models\\svm.model2_2')

    # 加载权重到新的模型中

    def predict_emotion(face_image_gray):  # a single cropped face

        result = model1.predict(face_image_gray)
        if int(result) == 0:
            return 'airplane'
        elif int(result) == 1:
            return 'automobile'
        elif int(result) == 2:
            return 'bird'
        elif int(result) == 3:
            return 'cat'
        elif int(result) == 4:
            return 'deer'
        elif int(result) == 5:
            return 'dog'
        elif int(result) == 6:
            return 'frog'
        elif int(result) == 7:
            return 'horse'
        elif int(result) == 8:
            return 'ship'
        elif int(result) == 9:
            return 'trunk'

    # img = cv2.imread("E:/灵山游/013.JPG", cv2.IMREAD_GRAYSCALE)

    im = Image.open(str)
    im2 = im
    im = im.resize((32, 32))
    frame = np.array(im)  ##将image读取到的照片转换为数组格式

    frame2 = np.array(im2)
    r, g, b = cv2.split(frame2)
    frame2 = cv2.merge([b, g, r])

    print(frame.shape)
    img_gray = rgb2gray(frame) / 255.0  # 灰度函数然后归一化
    print(img_gray.shape)

    image = img_gray
    fd = extra_lbp(image)

    fd = fd[:].reshape((1, -1))
    print(fd.shape)
    emotion = predict_emotion(fd)

    cv2.putText(frame2, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(I,'there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
    # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    # 呈现结果框

    frame2 = cv2.resize(frame2, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)  # 4x4像素邻域的双三次插值
    cv2.imshow('Photo', frame2)

def selectPath2_1():
    str = filedialog.askopenfilename()  # 打开任意路径图片

    emotion_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 加载json并且创建模型拱门
    model1 = joblib.load('C:\\Users\\郏紫宇\\Desktop\\HOG+SVM classifer\\data\\models\\svm.model2')

    # 加载权重到新的模型中

    def predict_emotion(face_image_gray):  # a single cropped face

        result = model1.predict(face_image_gray)
        if int(result) == 0:
            return 'airplane'
        elif int(result) == 1:
            return 'automobile'
        elif int(result) == 2:
            return 'bird'
        elif int(result) == 3:
            return 'cat'
        elif int(result) == 4:
            return 'deer'
        elif int(result) == 5:
            return 'dog'
        elif int(result) == 6:
            return 'frog'
        elif int(result) == 7:
            return 'horse'
        elif int(result) == 8:
            return 'ship'
        elif int(result) == 9:
            return 'trunk'

    # img = cv2.imread("E:/灵山游/013.JPG", cv2.IMREAD_GRAYSCALE)

    im = Image.open(str)
    im2 = im
    im = im.resize((32, 32))
    frame = np.array(im)  ##将image读取到的照片转换为数组格式

    frame2 = np.array(im2)
    r, g, b = cv2.split(frame2)
    frame2 = cv2.merge([b, g, r])

    print(frame.shape)
    img_gray = rgb2gray(frame) / 255.0  # 灰度函数然后归一化
    print(img_gray.shape)

    image = img_gray
    fd = extra_lbp(image)

    fd = fd[:].reshape((1, -1))
    print(fd.shape)
    emotion = predict_emotion(fd)

    cv2.putText(frame2, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(I,'there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
    # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    # 呈现结果框

    frame2 = cv2.resize(frame2, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)  # 4x4像素邻域的双三次插值
    cv2.imshow('Photo', frame2)

def selectPath2_2():
    str = filedialog.askopenfilename()  # 打开任意路径图片
    emotion_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # 加载json并且创建模型拱门
    model1 = joblib.load('C:\\Users\\郏紫宇\\Desktop\\HOG+SVM classifer\\data\\models\\svm.model2_3')

    # 加载权重到新的模型中

    def predict_emotion(face_image_gray):  # a single cropped face

        result = model1.predict(face_image_gray)
        if int(result) == 0:
            return 'airplane'
        elif int(result) == 1:
            return 'automobile'
        elif int(result) == 2:
            return 'bird'
        elif int(result) == 3:
            return 'cat'
        elif int(result) == 4:
            return 'deer'
        elif int(result) == 5:
            return 'dog'
        elif int(result) == 6:
            return 'frog'
        elif int(result) == 7:
            return 'horse'
        elif int(result) == 8:
            return 'ship'
        elif int(result) == 9:
            return 'trunk'

    # img = cv2.imread("E:/灵山游/013.JPG", cv2.IMREAD_GRAYSCALE)

    im = Image.open(str)
    im2 = im
    im = im.resize((32, 32))
    frame = np.array(im)  ##将image读取到的照片转换为数组格式

    frame2 = np.array(im2)
    r, g, b = cv2.split(frame2)
    frame2 = cv2.merge([b, g, r])

    print(frame.shape)
    img_gray = rgb2gray(frame) / 255.0  # 灰度函数然后归一化
    print(img_gray.shape)

    image = img_gray
    fd = extra_lbp(image)

    fd = fd[:].reshape((1, -1))
    print(fd.shape)
    emotion = predict_emotion(fd)

    cv2.putText(frame2, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(I,'there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
    # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    # 呈现结果框

    frame2 = cv2.resize(frame2, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)  # 4x4像素邻域的双三次插值
    cv2.imshow('Photo', frame2)

def selectPath2_3():
    str = filedialog.askopenfilename()  # 打开任意路径图片

    emotion_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 加载json并且创建模型拱门
    model1 = joblib.load('C:\\Users\\郏紫宇\\Desktop\\HOG+SVM classifer\\data\\models\\svm.model2_4')

    # 加载权重到新的模型中

    def predict_emotion(face_image_gray):  # a single cropped face

        result = model1.predict(face_image_gray)
        if int(result) == 0:
            return 'airplane'
        elif int(result) == 1:
            return 'automobile'
        elif int(result) == 2:
            return 'bird'
        elif int(result) == 3:
            return 'cat'
        elif int(result) == 4:
            return 'deer'
        elif int(result) == 5:
            return 'dog'
        elif int(result) == 6:
            return 'frog'
        elif int(result) == 7:
            return 'horse'
        elif int(result) == 8:
            return 'ship'
        elif int(result) == 9:
            return 'trunk'

    # img = cv2.imread("E:/灵山游/013.JPG", cv2.IMREAD_GRAYSCALE)

    im = Image.open(str)
    im2 = im
    im = im.resize((32, 32))
    frame = np.array(im)  ##将image读取到的照片转换为数组格式

    frame2 = np.array(im2)
    r, g, b = cv2.split(frame2)
    frame2 = cv2.merge([b, g, r])

    print(frame.shape)
    img_gray = rgb2gray(frame) / 255.0  # 灰度函数然后归一化
    print(img_gray.shape)

    image = img_gray
    fd = extra_lbp(image)

    fd = fd[:].reshape((1, -1))
    print(fd.shape)
    emotion = predict_emotion(fd)

    cv2.putText(frame2, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(I,'there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
    # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    # 呈现结果框

    frame2 = cv2.resize(frame2, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)  # 4x4像素邻域的双三次插值
    cv2.imshow('Photo', frame2)

def selectPath2_4():
    selectPath()

def selectPath3():
    str = filedialog.askopenfilename()  # 打开任意路径图片

    emotion_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 加载json并且创建模型拱门
    model1 = joblib.load('C:\\Users\\郏紫宇\\Desktop\\HOG+SVM classifer\\data\\models\\svm.model3')

    # 加载权重到新的模型中

    def predict_emotion(face_image_gray):  # a single cropped face

        result = model1.predict(face_image_gray)
        if int(result) == 0:
            return 'airplane'
        elif int(result) == 1:
            return 'automobile'
        elif int(result) == 2:
            return 'bird'
        elif int(result) == 3:
            return 'cat'
        elif int(result) == 4:
            return 'deer'
        elif int(result) == 5:
            return 'dog'
        elif int(result) == 6:
            return 'frog'
        elif int(result) == 7:
            return 'horse'
        elif int(result) == 8:
            return 'ship'
        elif int(result) == 9:
            return 'trunk'

    # img = cv2.imread("E:/灵山游/013.JPG", cv2.IMREAD_GRAYSCALE)

    im = Image.open(str)
    im2 = im
    im = im.resize((32, 32))
    frame = np.array(im)  ##将image读取到的照片转换为数组格式

    frame2 = np.array(im2)
    r, g, b = cv2.split(frame2)
    frame2 = cv2.merge([b, g, r])

    #print(frame.shape)
    img_gray = rgb2gray(frame) / 255.0  # 灰度函数然后归一化
    #print(img_gray.shape)

    image = img_gray
    fd = extra_glcm(image)

    fd = fd[:].reshape((1, -1))
    print(fd.shape)
    emotion = predict_emotion(fd)

    cv2.putText(frame2, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(I,'there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
    # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    # 呈现结果框

    frame2 = cv2.resize(frame2, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)  # 4x4像素邻域的双三次插值
    cv2.imshow('Photo', frame)

def selectPath4():
    str = filedialog.askopenfilename()  # 打开任意路径图片

    emotion_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 加载json并且创建模型拱门
    model1 = CNNNet.createNet(input_shapes=(32, 32, 3), nb_class=NB_CLASSES)
    #print(model1)
    model1.load_weights('my_model.h5')

    # 加载权重到新的模型中

    def predict_emotion(face_image_gray):  # a single cropped face

        result = model1.predict(face_image_gray)
        result = np.argmax(result)
        if int(result) == 0:
            return 'airplane'
        elif int(result) == 1:
            return 'automobile'
        elif int(result) == 2:
            return 'bird'
        elif int(result) == 3:
            return 'cat'
        elif int(result) == 4:
            return 'deer'
        elif int(result) == 5:
            return 'dog'
        elif int(result) == 6:
            return 'frog'
        elif int(result) == 7:
            return 'horse'
        elif int(result) == 8:
            return 'ship'
        elif int(result) == 9:
            return 'trunk'

    # img = cv2.imread("E:/灵山游/013.JPG", cv2.IMREAD_GRAYSCALE)

    im = Image.open(str)
    im2 = im
    im = im.resize((32, 32))
    frame = np.array(im)  ##将image读取到的照片转换为数组格式
    frame2 = np.array(im2)
    r, g, b = cv2.split(frame2)
    frame2 = cv2.merge([b, g, r])
    #print(frame.shape)
    #img_gray = rgb2gray(frame) / 255.0  # 灰度函数然后归一化
    #print(img_gray.shape)
    #image = img_gray
    fd = frame
    fd = fd[:].reshape(1,32,32,3)
    print(fd.shape)
    emotion = predict_emotion(fd)

    cv2.putText(frame2, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.putText(I,'there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
    # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    # 呈现结果框

    frame2 = cv2.resize(frame2, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)  # 4x4像素邻域的双三次插值
    cv2.imshow('Photo', frame2)

# 识别系统窗口建立
win = Tk()
win.title("图像实时识别系统")
win.geometry('1000x600+100+20')  # 窗口大小和偏移度
win.resizable(0, 0)  # 不可调整串口大小

def hello():
   print('hello')

def about():
    text = Entry(win, background='white',width = 45,insertborderwidth = 5, borderwidth=5, font=('Helvetica', '14', 'bold'))
    text.pack()
    text.insert(0, "作者：郏紫宇"+"       ")

    text.insert(20, 'E-mail:31414119@njau.edu.cn')


menubar = Menu(win)

#创建下拉菜单File，然后将其加入到顶级的菜单栏中
filemenu = Menu(menubar,tearoff=0)
filemenu.add_command(label="SVM线性分类器", font = ("Arial, 14"),command=selectPath)
filemenu.add_command(label="SVM非线性分类器",font = ("Arial, 14"), command=selectPath_1)
filemenu.add_command(label="KNN分类器",font = ("Arial, 14"), command=selectPath_2)
filemenu.add_command(label="决策树分类器",font = ("Arial, 14"), command=selectPath_3)
filemenu.add_command(label="贝叶斯分类器",font = ("Arial, 14"), command=selectPath_4)

filemenu.add_separator()
menubar.add_cascade(label="HOG特征提取",font = ("Arial, 14"), menu=filemenu)

#创建另一个下拉菜单Edit
editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="SVM线性分类器",font = ("Arial, 14"), command=selectPath2)
editmenu.add_command(label="SVM非线性分类器", font = ("Arial, 14"),command=selectPath2_1)
editmenu.add_command(label="KNN分类器", font = ("Arial, 14"),command=selectPath2_2)
editmenu.add_command(label="决策树分类器", font = ("Arial, 14"),command=selectPath2_3)
editmenu.add_command(label="贝叶斯分类器", font = ("Arial, 14"),command=selectPath2_4)
menubar.add_cascade(label="LBP特征提取",font = ("Arial, 14"),menu=editmenu)

editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="SVM线性分类器",font = ("Arial, 14"), command=selectPath)
menubar.add_cascade(label="GLCM特征提取",font = ("Arial, 14"),menu=editmenu)

editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="稀疏编码+SVM", font = ("Arial, 14"),command=selectPath)
menubar.add_cascade(label="SIFT特征提取",font = ("Arial, 14"),menu=editmenu)

editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="CNN",font = ("Arial, 14"), command=selectPath4)
menubar.add_cascade(label="神经网络",font = ("Arial, 14"),menu=editmenu)

#创建下拉菜单Help
helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="About",font = ("Arial, 14"), command=about)
helpmenu.add_command(label="Exit", font = ("Arial, 14"),command=win.quit)
menubar.add_cascade(label="Help", font = ("Arial, 14"),menu=helpmenu)


#显示菜单
win.config(menu=menubar)

win.mainloop()  # 循环点击
