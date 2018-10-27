# -*- coding:utf-8 -*-
"""
（一）作业要求
    结合Sobel边缘检测算子和Canny边缘检测算法中相关步骤，
    实现“有方向边缘数据的非最大抑制”算法
    和“边缘检测算子输出的滞后过滤”算法，实现边缘图像的阈值化。
（二）实现步骤
     1 将图像f与尺度为σ的二维高斯函数h做卷积以消除噪声

        方法：opencv提供了GaussianBlur()函数对图形进行高斯滤波

     2 对g中的每个像素计算梯度的大小和方向

        方法：①调用opencv函数void Sobel(InputArray src, OutputArray dst, int ddepth,
                            int dx, int dy, int ksize=3, double scale=1, double delta=0,
                            intborderType=BORDER_DEFAULT )
                实现计算x和y方向的Sobel算子
             ②cv2.magnitude和cv2.phase分别计算大小和方向

     3 根据像素梯度方向，获取该像素沿梯度的邻接像素

        方法：遍历角度矩阵，按下图分成八份，每一份PI/4，判断每一个梯度方向处在哪个范围，然后获取邻接像素
                               \ | /
                              ——像素——
                               / | \

     4 非极大值抑制：遍历，若某个像素的灰度值与其梯度方向上前后两个像素的灰度值相比并非最大，则该像素不是边缘

        方法：比较梯度方向上的三个像素灰度值大小，如果不是最大，就置成0（黑色）（255-白色是边缘）

     5 滞后阈值化处理：设定高低阈值，凡是大于高阈值的一定是边缘；
             凡是小于低阈值的一定不是边缘；
             检测结果在高低阈值之间的，看其周边8个像素中是否有超过高阈值的边缘像素，
             有则为边缘，否则不是边缘
        方法：……

（三）作者

    智能科学与技术 张家释 1611462

（四）开发环境

    Ubuntu16.04 + Python + Opencv2.4.9 + Pycharm

"""
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

PI = 3.14159
max_name = "max"
min_name = "min"
win1_name = "miss_Canny_opencv_function"
win2_name = "miss_Canny_myself_function"
win3_name = "road_Canny_opencv_function"
win4_name = "road_Canny_myself_function"
win5_name = "building_Canny_opencv_function"
win6_name = "building_Canny_myself_function"



def result_show():

    plt.figure("image")

    plt.subplot(321),plt.imshow(miss_canny_func, cmap = 'gray')
    plt.title(win1_name),plt.xticks([]),plt.yticks([])
    plt.subplot(322),plt.imshow(miss_canny_myself, cmap = 'gray')
    plt.title(win2_name),plt.xticks([]),plt.yticks([])

    plt.subplot(323),plt.imshow(road_canny_func, cmap = 'gray')
    plt.title(win3_name),plt.xticks([]),plt.yticks([])
    plt.subplot(324),plt.imshow(road_canny_myself, cmap = 'gray')
    plt.title(win4_name),plt.xticks([]),plt.yticks([])

    plt.subplot(325),plt.imshow(building_canny_func, cmap = 'gray')
    plt.title(win5_name),plt.xticks([]),plt.yticks([])
    plt.subplot(326),plt.imshow(building_canny_myself, cmap = 'gray')
    plt.title(win6_name),plt.xticks([]),plt.yticks([])

    plt.show()



def combine_sobel_xy(x, y, rows, cols, if_magni):

    res = x
    if if_magni:
        # by myself
        for i in range(0, rows):
            for j in range(0, cols):
                res[i, j] = np.sqrt(np.square(x[i, j]) + np.square(y[i, j]))
    else:
        # opencv function
        res = cv2.magnitude(x, y)
    return res


def cal_sobel_and_angle(img):
    # calculate sobel
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3, borderType=cv2.BORDER_REPLICATE)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 3, borderType=cv2.BORDER_REPLICATE)

    # combine x and y
    sp_sobel_x = sobel_x.shape
    sobel = combine_sobel_xy(sobel_x, sobel_y, sp_sobel_x[0], sp_sobel_x[1], if_magni= True)

    sobel = cv2.convertScaleAbs(sobel)
    angle = cal_angle(sobel_x, sobel_y)

    return sobel, angle


def cal_angle(x,  y):
    # from 0 to 2PI
    angle = cv2.phase(x, y, True)
    return angle


def if_edge(pixel1, pixel2, pixel3):
    # judge if biggest
    if pixel1 > pixel2 and pixel1 > pixel3:
        return True
    else:
        return False


def Canny_bymyself(img, rows, cols, angle, value_min, value_max):
    # 非极大抑制
    print value_min, value_max
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):

            #  横向-左右
            if angle[i, j] > 0 and angle[i, j] < PI / 8 or angle[i, j] > 15 * PI / 8 and angle[i, j] < 2 * PI or angle[i, j] > 7 * PI / 8 and angle[i, j] < 9 * PI / 8 :
                # 不是最大
                if not if_edge(img[i, j], img[i, j - 1], img[i, j + 1]):
                    # 不是边缘
                     img[i, j] = 0

            #  竖直-上下
            elif angle[i, j] > 3 * PI / 8 and angle[i, j] < 5 * PI / 8 or angle[i, j] > 11 * PI / 8 and angle[i, j] < 13 * PI / 8 :
                if not if_edge(img[i, j], img[i - 1, j], img[i + 1, j]):
                    img[i, j] = 0

            #  斜向右上-左下
            elif angle[i, j] > PI / 8 and angle[i, j] < 3 * PI / 8 or angle[i, j] > 9 * PI / 8 and angle[i, j] < 11 * PI / 8:
                if not if_edge(img[i, j], img[i - 1, j + 1], img[i + 1, j - 1]):
                    img[i, j] = 0

            #  斜向右下-左上
            elif angle[i, j] > 5 * PI / 8 and angle[i, j] < 7 * PI / 8 or angle[i, j] > 13 * PI / 8 and angle[i, j] < 15 * PI / 8:
                if not if_edge(img[i, j], img[i + 1, j + 1], img[i - 1, j - 1]):
                    img[i, j] = 0

    #双阈值
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # 大于高阈值
            if img[i, j] >= value_max:
                img[i, j] = 255

            # 小于低阈值
            elif img[i, j] <= value_min:
                img[i, j] = 0
            # 两者之间
            elif img[i, j] < value_max and img[i, j] > value_min:
                # 周围八个有一个大于大阈值， 就是边缘
                if img[i, j + 1] > value_max or img[i + 1, j - 1] > value_max or \
                        img[i + 1, j] > value_max or img[i + 1, j + 1] > value_max or \
                        img[i - 1, j - 1] > value_max or img[i - 1, j] > value_max or \
                        img[i - 1, j + 1] > value_max or img[i, j - 1] > value_max:
                    img[i, j] = 255
                else:
                    img[i, j] = 0
    cv2.namedWindow("res",cv2.WINDOW_NORMAL)
    cv2.imshow("res",img)
    cv2.waitKey()
    return img


if __name__ == '__main__':

    # 载入图片
    miss = cv2.imread("/home/jiashi/Pictures/miss.bmp")
    road = cv2.imread("/home/jiashi/Pictures/road.jpg")
    building = cv2.imread("/home/jiashi/Pictures/building.jpg")

    print "opencv_version:"+cv2.__version__
    # 没有载入图像，跳出
    if miss is None:
        os._exit(0)

    # 转换成灰度图
    miss = cv2.cvtColor(miss, cv2.COLOR_BGR2GRAY)
    road = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    building = cv2.cvtColor(building, cv2.COLOR_BGR2GRAY)

    # 高斯滤波
    miss = cv2.GaussianBlur(miss, (3,3), 0.5)  # 0 0.5 1 2 3
    road = cv2.GaussianBlur(road, (3,3), 0.5)  # 0 0.5 1 2 3
    building = cv2.GaussianBlur(building, (3,3), 0.5)  # 0  0.5 1 2 3

    # Sobel
    miss_sobel, miss_sobel_angle = cal_sobel_and_angle(miss)
    road_sobel, road_sobel_angle = cal_sobel_and_angle(road)
    building_sobel, building_sobel_angle = cal_sobel_and_angle(building)

    # 调用自己写的函数
    sp_miss = miss.shape
    miss_canny_func = cv2.Canny(miss, 50, 100)
    miss_canny_myself = Canny_bymyself(miss_sobel, sp_miss[0], sp_miss[1], miss_sobel_angle, 50, 100)

    sp_road = road.shape
    road_canny_func = cv2.Canny(road, 50, 100)
    road_canny_myself = Canny_bymyself(road_sobel, sp_road[0], sp_road[1], road_sobel_angle, 50, 100)

    sp_building = building.shape
    building_canny_func = cv2.Canny(building, 50, 100)
    building_canny_myself = Canny_bymyself(building_sobel, sp_building[0], sp_building[1], building_sobel_angle, 50, 100)
    # 显示结果
    result_show()


