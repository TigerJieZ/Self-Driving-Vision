# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pickle
from DrivingZoneDetection.CalculateOffset import line, calibration
import matplotlib.pyplot as plt


def sobel_xy(image, kernel):
    return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel), \
           cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel)


def direction_gradient(image, kernel, threshold):
    sobel_x, sobel_y = sobel_xy(image, kernel)
    abs_gradient_dir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    binary_output = threshold_filter(abs_gradient_dir, threshold)
    return binary_output.astype(np.uint8)


def m_sobel(image, threshold, transverse_axis='x'):
    if transverse_axis is 'x':
        sobel_image = abs(cv2.Sobel(image, cv2.CV_64F, 1, 0))
    elif transverse_axis is 'y':
        sobel_image = abs(cv2.Sobel(image, cv2.CV_64F, 0, 1))
    else:
        print('Please enter the correct Sobel direction')
        return None
    scaled_sobel = np.uint8(255 * sobel_image / np.max(sobel_image))

    temp = threshold_filter(scaled_sobel, threshold)

    # show_img(temp, 10000, 'sobel_'+transverse_axis)

    return temp


def magnitude_gradient(image, kernel, threshold):
    sobel_x, sobel_y = sobel_xy(image, kernel)
    # 计算梯度量级
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # 转换成unit8
    scale_factor = np.max(gradient_magnitude) / 255
    gradient_magnitude = (gradient_magnitude / scale_factor).astype(np.uint8)

    # 不满足阈值为0
    temp = threshold_filter(gradient_magnitude, threshold)

    # show_img(temp, 10000, 'mag')

    return temp


def gradient_combine(image, threshold_x, threshold_y, threshold_mag, threshold_dir):
    sobel_x = m_sobel(image, threshold_x, 'x')
    sobel_y = m_sobel(image, threshold_y, 'y')
    mag_img = magnitude_gradient(image, 3, threshold_mag)
    dir_img = direction_gradient(image, 15, threshold_dir)

    # 结合梯度测量
    gradient_comb = np.zeros_like(dir_img).astype(np.uint8)
    # gradient_comb[((sobel_x > 1) & (mag_img > 1) & (dir_img > 1)) | ((sobel_x > 1) & (sobel_y > 1))] = 255
    gradient_comb[((mag_img > 1) & (dir_img > 1)) | ((sobel_y > 1))] = 255

    # show_img(gradient_comb, 10000, 'gradient')

    return gradient_comb


def threshold_filter(ch, threshold=(80, 255)):
    binary = np.zeros_like(ch)
    # cv2.imshow('fda',binary)
    # cv2.waitKey(10000)
    binary[(ch > threshold[0]) & (ch <= threshold[1])] = 255
    return binary


def hls_combine(img, threshold_h, threshold_l, threshold_s):
    # HLS颜色空间的转换
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_data = hls[:, :, 0]
    l_data = hls[:, :, 1]
    s_data = hls[:, :, 2]

    h_img = threshold_filter(h_data, threshold_h)
    l_img = threshold_filter(l_data, threshold_l)
    s_img = threshold_filter(s_data, threshold_s)

    # 两种情况——阴影中的车道线
    hls_comb = np.zeros_like(s_img).astype(np.uint8)
    hls_comb[((s_img > 1) & (l_img == 0)) | ((s_img == 0) & (h_img > 1) & (l_img > 1))] = 255  # | (R > 1)] = 255

    return hls_comb


def warp_image(img, src, dst, size):
    """ 透视变换 """
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    # show_img(warp_img, 10000, 'warp')

    return warp_img, M, Minv


def gaussian_blur(image):
    # 高斯模糊处理
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return blur_gray


def store_data(warp_point, is_save=False):
    """
    计算透视参数和畸变系数
    :param warp_point: 透视的四个顶点
    :param is_save: 是否保存数据至文件中
    :return:
    """
    warp_top_left, warp_top_right = warp_point[0], warp_point[1]
    warp_bottom_left, warp_bottom_right = warp_point[2], warp_point[3]

    src = np.float32([warp_bottom_left, warp_top_left, warp_top_right, warp_bottom_right])
    dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

    mtx, dist = calibration.calib()

    if is_save:
        f = open('temp.pkl', 'wb')
        pickle.dump(mtx, f)
        pickle.dump(dist, f)
        pickle.dump(src, f)
        pickle.dump(dst, f)

        f.close()

    return mtx, dist, src, dst


def load_data():
    f = open('temp.pkl', 'rb')
    mtx = None
    pickle.dump(mtx, f)
    dist = None
    pickle.dump(dist, f)
    src = None
    pickle.dump(src, f)
    dst = None
    pickle.load(dst, f)
    return mtx, dist, src, dst


def find_lines(im, line_left=line.Line(), line_right=line.Line(), window_num=9):
    # 车道线图像的shape
    shape = im.shape

    # 扫描窗口的高度
    window_height = shape[0] / window_num

    # 计算图像下1/4部分的直方图（降低运算规模，并减少噪声干扰，因为一般情况噪声出现在图像的靠上部分）
    histogram = np.sum(im[int(shape[0] / 4 * 3):, :], axis=0)

    if line_left.detected:
        '''如果上一帧检测到了车道线'''

        pass
    else:
        '''如果上一帧未检测到车道线'''

        # 以直方图中最大值的x点为左边窗口检测起点
        start_left_X = np.argmax(histogram[:int(shape[1] / 2)])
        start_right_X = np.argmax(histogram[int(shape[1] / 2):]) + int(shape[1] / 2)


def print_vehicle_data(image, left_line=line.Line(), right_line=line.Line()):
    '''打印车辆位置和车道半径信息'''

    cv2.putText(image, 'Radius of Curvature = ' + str(round(left_line.radius_of_curvature, 3)) + '(m)',
                (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    center_lane = (right_line.startX - left_line.startX) / 2 + left_line.startX
    lane_width = right_line.startX - left_line.startX
    xm_per_pix = 5.6 * (720 / 1280) / lane_width  # 像素/米
    center_car = 640 / 2
    excursion = round(abs(center_lane - center_car) * xm_per_pix, 3)
    if center_lane > center_car:
        deviation = 'Vehicle is ' + str(
            excursion) + 'm left of center'
    elif center_lane < center_car:
        deviation = 'Vehicle is ' + str(
            excursion) + 'm right of center'
        excursion = 0 - excursion
    else:
        deviation = 'Center'
    left_line.deviation = deviation
    cv2.putText(image, deviation, (20, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    return image, excursion


def print_line(left_line, img):
    """
    打印左车道线到俯视图中
    :param left_line:
    :param img:
    :return:
    """
    # allx = (x for x in left_line.allx)
    # ally = (y for y in left_line.ally)
    allx = tuple(list(left_line.allx))
    ally = tuple(list(left_line.ally))
    plt.imshow(img)
    plt.plot(allx, ally, label='$sin(x)$', color='red', linewidth=3)
    plt.show()


def show_warpp(roi_img, warp_point):
    import pylab
    pylab.imshow(roi_img)
    pylab.plot([warp_point[0][0], warp_point[1][0], warp_point[2][0], warp_point[3][0]],
               [warp_point[0][1], warp_point[1][1], warp_point[2][1], warp_point[3][1]],
               'r*')
    pylab.show()


def combine_lane_img(lane_img, img, roi_top_y, roi_bottom_y):
    """

    :param lane_img:
    :param img:
    :param roi_top_y:
    :param roi_bottom_y:
    :return:
    """
    if type(lane_img) == 'str':
        lane_img = cv2.imread(lane_img)
    if type(img) == 'str':
        img = cv2.imread(img)

    if len(lane_img.shape) != len(img.shape):
        print('请传入格式相同的img')
        return
    mask = np.zeros_like(img)
    mask[roi_top_y:img.shape[0] - roi_bottom_y, 0:img.shape[1]] = lane_img

    new = cv2.addWeighted(img, 1, mask, 0.3, 0)

    return new
