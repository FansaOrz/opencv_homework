#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import copy
import numpy as np
import math
import os

Mo0 = Mo1 = Mo2 = Mo3 = 0
# 声明三个窗口
cv2.namedWindow("Moravec", cv2.WINDOW_NORMAL)
cv2.namedWindow("Harris", cv2.WINDOW_NORMAL)
cv2.namedWindow("goodFeaturesToTrack", cv2.WINDOW_NORMAL)


# 读取图像的函数
def read_img():
    img = cv2.imread("/home/jiashi/Pictures/building.jpg")
    return img


# 生成高斯矩阵　size为矩阵大小, sigma为参数
def Gauss_Kernel(size, sigma):
    gaus = np.ones((size, size), np.double)
    PI = 4.0 * math.atan(1.0)
    center = (size - 1) / 2
    sum = 0
    for i in range(0, size):
        for j in range(0, size):
            gaus[i][j] = (1 / (2 * PI * sigma * sigma)) * math.exp(-((i - center) * (i - center) + (j - center) * (j - center)) / float((2 * sigma * sigma)))
            sum = sum + gaus[i][j]
    for i in range(0, size):
        for j in range(0, size):
            gaus[i][j] = gaus[i][j] / sum
    return gaus


# 对矩阵进行高斯滤波
def Gau_fliter(img):
    gaus = Gauss_Kernel(3, 1)
    res = np.ones((rows, cols), np.double)
    sum = 0
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            temp = img[i - 1:i + 2, j - 1:j + 2]

            for k in range(0, 3):
                for l in range(0, 3):
                    sum = sum + temp[k, l] * gaus[k, l]
            res[i][j] = int(sum)
            sum = 0
    return res


# 非极大值抑制
def non_maximum(img):
    temp = img
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            img0 = temp[i - 1: i + 1, j - 1: j + 1]
            # 不是最大
            if temp[i, j] < img0.max():
                img[i, j] = 0
    return img


# 画圆
def draw_circle(img, in_img):
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if img[i, j] != 0:
                cv2.circle(in_img, (j, i), 3, (0, 0, 255))

# Moravec算法中计算四个方向的角点响应函数
def cal_four_gradient(img0, img1):
    temp = np.zeros((3,3), np.int32)
    for t in range(0, 3):
        for p in range(0, 3):
            temp[t, p] = int(img1[t, p]) - int(img0[t, p])
    return (temp ** 2).sum()

# Moravec算法
def Moravec(img):
    res = np.ones((rows, cols), np.int64)
    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            # 横向
            Mo0 = cal_four_gradient(img[i - 1: i + 2, j - 1: j + 2], img[i - 1: i + 2, j: j + 3])
            # 右下
            Mo1 = cal_four_gradient(img[i - 1: i + 2, j - 1: j + 2], img[i: i + 3, j: j + 3])
            # 纵向
            Mo2 = cal_four_gradient(img[i - 1: i + 2, j - 1: j + 2], img[i: i + 3, j - 1: j + 2])
            # 左下
            Mo3 = cal_four_gradient(img[i - 1: i + 2, j - 1: j + 2], img[i: i + 3, j - 2: j + 1])
            # 求最小值
            res[i, j] = min(Mo0,Mo1,Mo2,Mo3)

    cv2.createTrackbar("max/1000*", "Moravec", 1000, 1000, Mora_print)
    return res


def Mora_print(arg):
    T = float(arg)/1000.0 * Mora.max()
    print arg, Mora.max(),T
    temp = copy.copy(Mora)
    for i in range(0, rows):
        for j in range(0, cols):
            # 角点响应函数值小于阈值
            if Mora[i][j] <= T:
                temp[i][j] = 0
    # 非极大值抑制和画圈
    res = non_maximum(temp)
    img_Mor = read_img()
    draw_circle(res, img_Mor)
    cv2.imshow("Moravec", img_Mor)


# 滑动条的回调函数
def Harr_print(arg):
    T = float(arg) / 1000.0 * Harr.max()
    #print T,arg,Harr.max()
    temp = copy.copy(Harr)
    for i in range(0, rows):
        for j in range(0, cols):
            # 角点响应函数值小于阈值
            if Harr[i][j] <= T:
                temp[i][j] = 0
    # 非极大值抑制和画圈
    res = non_maximum(temp)
    img_Harr = read_img()

    draw_circle(res, img_Harr)
    cv2.imshow("Harris", img_Harr)

# Harris算法
def Harris(img, k):
    # 生成一个等同大小的矩阵，类型为double
    R_img = np.ones(img.shape, np.double)
    # 计算x和y方向上的导数
    img_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    img_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    # 生成等同大小的矩阵，类型为double，分别对应M矩阵中的A B C
    A_img = np.zeros(img.shape, np.double)
    B_img = np.zeros(img.shape, np.double)
    C_img = np.zeros(img.shape, np.double)
    # 计算M矩阵中的A B C
    for i in range(0, rows):
        for j in range(0, cols):
            A_img[i][j] = img_x[i][j] ** 2
            B_img[i][j] = img_y[i][j] ** 2
            C_img[i][j] = img_x[i][j] * img_y[i][j]
    # 进行高斯滤波
    A_img2 = Gau_fliter(A_img)
    B_img2 = Gau_fliter(B_img)
    C_img2 = Gau_fliter(C_img)

    for i in range(0, rows):
        for j in range(0, cols):
            R_img[i][j] = A_img2[i][j] * B_img2[i][j] - C_img2[i][j] ** 2 - k * (A_img2[i][j] + B_img2[i][j]) ** 2

    cv2.createTrackbar("max/1000*", "Harris", 1000, 1000, Harr_print)

    return R_img


if __name__ == '__main__':

    # 载入图片
    img_in = read_img()

    rows = img_in.shape[0]
    cols = img_in.shape[1]

    img_Mor = img_in.copy()
    img_Harr = img_in.copy()
    print "opencv_version:"+cv2.__version__

    # 没有载入图像，跳出
    if img_in is None:
        os._exit(0)

    # 转换成灰度图
    im = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    im_res = im.copy()

    # Moravec和Harris算法
    Mora = Moravec(im)
    Harr = Harris(im, 0.04)

    # opencv自带的goodFeaturesToTrack函数
    im_corners = cv2.goodFeaturesToTrack(im_res, maxCorners=100, qualityLevel=0.05, minDistance= 5.0, blockSize=3, useHarrisDetector=True, k=0.04)
    # 画goodFeaturesToTrack的特征点
    for i in range(0, np.shape(im_corners)[0]):
        cv2.circle(img_in, (im_corners[i][0][0],im_corners[i][0][1]), 3, (0, 0, 255))

    cv2.imshow("Moravec", img_Mor)
    cv2.imshow("goodFeaturesToTrack",img_in)
    cv2.imshow("Harris",img_Harr)
    cv2.waitKey()