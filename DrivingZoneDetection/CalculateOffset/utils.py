# -*- coding: utf-8 -*-

import cv2
import numpy as np


def show_img(image, time, windowName):
    cv2.imshow(winname=windowName, mat=image)
    cv2.waitKey(time)


def warp_image(img, src, dst, size):
    """ 透视变换 """
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    # show_img(warp_img, 10000, 'warp')

    return warp_img, M, Minv


def rad_of_curvature(left_line, right_line):
    """ 测量曲率半径  """

    ploty = left_line.ally
    leftx, rightx = left_line.allx, right_line.allx

    leftx = leftx[::-1]  # 在y上反向匹配上下
    rightx = rightx[::-1]  # 在y上反向匹配上下

    # 在x和y中定义从像素空间到米的转换
    width_lanes = abs(right_line.startX - left_line.startX)
    ym_per_pix = 30 / 720  # 像素/米
    xm_per_pix = 3.7 * (720 / 1280) / width_lanes  # 像素/米
    # 定义的值，这里我们想曲率半径
    # 最大值，对应于图像的底部
    y_eval = np.max(ploty)

    # 新多项式在世界空间中的x、y拟合
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # 计算新的曲率半径
    left_curvature = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curvature = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # 曲率半径结果
    left_line.radius_of_curvature = left_curvature
    right_line.radius_of_curvature = right_curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    left_line.curvature = left_curverad


def smoothing(lines, pre_lines=3):
    # 收集线和打印平均线
    lines = np.squeeze(lines)
    avg_line = np.zeros(720)

    for ii, line in enumerate(reversed(lines)):
        if ii == pre_lines:
            break
        avg_line += line
    avg_line = avg_line / pre_lines

    return avg_line


def blind_search_l(img, line_left):
    """
    盲目搜索-第一帧/迷失车道线
    使用直方图和滑动窗口
    """
    # 获取图像底半部的直方图
    histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)

    # 创建输出图像以绘制并可视化结果。
    output = np.dstack((img, img, img)) * 255
    # show_img(output, 10000, 'output')

    # 找到直方图的左右两边的顶点。
    # 这将是左右线的起点。
    start_left_x = np.argmax(histogram[:int(histogram.shape[0] / 2)])

    # 选择滑动窗口的个数
    num_windows = 9
    # 设置窗口高度
    window_height = int(img.shape[0] / num_windows)

    # 识别图像中所有非零像素的x和y位置。
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # 每个窗口更新的当前位置
    current_left_x = start_left_x

    ###################
    # 设置最小像素数发现唤醒窗口
    min_num_pixel = 50

    # 创建空列表来接收左、右行像素索引
    win_left_lane = []

    window_margin = line_left.window_margin

    # 一步一步地滑过窗口
    for window in range(num_windows):
        # 标识x和y（右和左）的窗口边界
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_leftx_min = current_left_x - window_margin
        win_leftx_max = current_left_x + window_margin

        # 在可视化图像上绘制窗口
        cv2.rectangle(output, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)

        # 标识窗口中x和y中的非零值像素。
        left_window_inds = ((nonzero_y >= win_y_low) & (nonzero_y <= win_y_high) & (nonzero_x >= win_leftx_min) & (
                nonzero_x <= win_leftx_max)).nonzero()[0]
        # 将这些索引附加到列表中
        win_left_lane.append(left_window_inds)

        # 如果你发现> minpix像素，唤醒下一个窗口在其平均位置
        if len(left_window_inds) > min_num_pixel:
            current_left_x = np.int(np.mean(nonzero_x[left_window_inds]))

    # 连接索引的数组
    win_left_lane = np.concatenate(win_left_lane)

    # 提取左、右行像素位置
    left_x, left_y = nonzero_x[win_left_lane], nonzero_y[win_left_lane]

    output[left_y, left_x] = [255, 0, 0]

    # 将二阶多项式拟合到每个
    left_fit = np.polyfit(left_y, left_x, 2)

    line_left.current_fit = left_fit

    # 生成用于绘图的x和y值。
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

    # ax^2 + bx + c
    left_plot_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

    line_left.prevX.append(left_plot_x)

    if len(line_left.prevX) > 10:
        left_avg_line = smoothing(line_left.prevX, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plot_x = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        line_left.current_fit = left_avg_fit
        line_left.allx, line_left.ally = left_fit_plot_x, ploty
    else:
        line_left.current_fit = left_fit
        line_left.allx, line_left.ally = left_plot_x, ploty

    line_left.startX = line_left.allx[- 1]
    line_left.endX = line_left.allx[0]

    line_left.detected = True
    # 打印曲率半径
    # rad_of_curvature(line_left)
    return output


def blind_search(img, line_left, line_right):
    """
    盲目搜索-第一帧/迷失车道线
    使用直方图和滑动窗口
    """
    # 获取图像底半部的直方图
    histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)

    # 创建输出图像以绘制并可视化结果。
    output = np.dstack((img, img, img)) * 255
    # show_img(output, 10000, 'output')

    # 找到直方图的左右两边的顶点。
    # 这将是左右线的起点。
    start_left_x = np.argmax(histogram[:int(histogram.shape[0] / 2)])
    start_right_x = np.argmax(histogram[int(histogram.shape[0] / 2):]) + int(histogram.shape[0] / 2)

    # 选择滑动窗口的个数
    num_windows = 9
    # 设置窗口高度
    window_height = int(img.shape[0] / num_windows)

    # 识别图像中所有非零像素的x和y位置。
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # 每个窗口更新的当前位置
    current_left_x = start_left_x
    current_right_x = start_right_x

    ###################
    # 设置最小像素数发现唤醒窗口
    min_num_pixel = 50

    # 创建空列表来接收左、右行像素索引
    win_left_lane = []
    win_right_lane = []

    window_margin = line_left.window_margin

    # 一步一步地滑过窗口
    for window in range(num_windows):
        # 标识x和y（右和左）的窗口边界
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_leftx_min = current_left_x - window_margin
        win_leftx_max = current_left_x + window_margin
        win_rightx_min = current_right_x - window_margin
        win_rightx_max = current_right_x + window_margin

        # 在可视化图像上绘制窗口
        cv2.rectangle(output, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(output, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)

        # 标识窗口中x和y中的非零值像素。
        left_window_inds = ((nonzero_y >= win_y_low) & (nonzero_y <= win_y_high) & (nonzero_x >= win_leftx_min) & (
                nonzero_x <= win_leftx_max)).nonzero()[0]
        right_window_inds = ((nonzero_y >= win_y_low) & (nonzero_y <= win_y_high) & (nonzero_x >= win_rightx_min) & (
                nonzero_x <= win_rightx_max)).nonzero()[0]
        # 将这些索引附加到列表中
        win_left_lane.append(left_window_inds)
        win_right_lane.append(right_window_inds)

        # 如果你发现> minpix像素，唤醒下一个窗口在其平均位置
        if len(left_window_inds) > min_num_pixel:
            current_left_x = np.int(np.mean(nonzero_x[left_window_inds]))
        if len(right_window_inds) > min_num_pixel:
            current_right_x = np.int(np.mean(nonzero_x[right_window_inds]))

    # 连接索引的数组
    win_left_lane = np.concatenate(win_left_lane)
    win_right_lane = np.concatenate(win_right_lane)

    # 提取左、右行像素位置
    left_x, left_y = nonzero_x[win_left_lane], nonzero_y[win_left_lane]
    right_x, right_y = nonzero_x[win_right_lane], nonzero_y[win_right_lane]

    output[left_y, left_x] = [255, 0, 0]
    output[right_y, right_x] = [0, 0, 255]

    # 将二阶多项式拟合到每个
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    line_left.current_fit = left_fit
    line_right.current_fit = right_fit

    # 生成用于绘图的x和y值。
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

    # ax^2 + bx + c
    left_plot_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plot_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    line_left.prevX.append(left_plot_x)
    line_right.prevX.append(right_plot_x)

    if len(line_left.prevX) > 10:
        left_avg_line = smoothing(line_left.prevX, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plot_x = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        line_left.current_fit = left_avg_fit
        line_left.allx, line_left.ally = left_fit_plot_x, ploty
    else:
        line_left.current_fit = left_fit
        line_left.allx, line_left.ally = left_plot_x, ploty

    if len(line_right.prevX) > 10:
        right_avg_line = smoothing(line_right.prevX, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plot_x = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        line_right.current_fit = right_avg_fit
        line_right.allx, line_right.ally = right_fit_plot_x, ploty
    else:
        line_right.current_fit = right_fit
        line_right.allx, line_right.ally = right_plot_x, ploty

    line_left.startX, line_right.startX = line_left.allx[- 1], line_right.allx[- 1]
    line_left.endX, line_right.endX = line_left.allx[0], line_right.allx[0]

    line_left.detected, line_right.detected = True, True
    # 打印曲率半径
    rad_of_curvature(line_left, line_right)
    return output


def blind_search_r(img, line_right):
    """
    盲目搜索-第一帧/迷失车道线
    使用直方图和滑动窗口
    """
    # 获取图像底半部的直方图
    histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)

    # 创建输出图像以绘制并可视化结果。
    output = np.dstack((img, img, img)) * 255
    # show_img(output, 10000, 'output')

    # 这将是左右线的起点。
    start_right_x = np.argmax(histogram[int(histogram.shape[0] / 2):]) + int(histogram.shape[0] / 2)

    # 选择滑动窗口的个数
    num_windows = 9
    # 设置窗口高度
    window_height = int(img.shape[0] / num_windows)

    # 识别图像中所有非零像素的x和y位置。
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # 每个窗口更新的当前位置
    current_right_x = start_right_x

    ###################
    # 设置最小像素数发现唤醒窗口
    min_num_pixel = 50

    # 创建空列表来接收左、右行像素索引
    win_right_lane = []

    window_margin = line_right.window_margin

    # 一步一步地滑过窗口
    for window in range(num_windows):
        # 标识x和y（右和左）的窗口边界
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_rightx_min = current_right_x - window_margin
        win_rightx_max = current_right_x + window_margin

        # 在可视化图像上绘制窗口
        cv2.rectangle(output, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)

        # 标识窗口中x和y中的非零值像素。
        right_window_inds = ((nonzero_y >= win_y_low) & (nonzero_y <= win_y_high) & (nonzero_x >= win_rightx_min) & (
                nonzero_x <= win_rightx_max)).nonzero()[0]
        # 将这些索引附加到列表中
        win_right_lane.append(right_window_inds)

        # 如果你发现> minpix像素，唤醒下一个窗口在其平均位置
        if len(right_window_inds) > min_num_pixel:
            current_right_x = np.int(np.mean(nonzero_x[right_window_inds]))

    # 连接索引的数组
    win_right_lane = np.concatenate(win_right_lane)

    right_x, right_y = nonzero_x[win_right_lane], nonzero_y[win_right_lane]

    output[right_y, right_x] = [0, 0, 255]

    # 将二阶多项式拟合到每个
    right_fit = np.polyfit(right_y, right_x, 2)

    line_right.current_fit = right_fit

    # 生成用于绘图的x和y值。
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

    # ax^2 + bx + c
    right_plot_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    line_right.prevX.append(right_plot_x)

    if len(line_right.prevX) > 10:
        right_avg_line = smoothing(line_right.prevX, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plot_x = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        line_right.current_fit = right_avg_fit
        line_right.allx, line_right.ally = right_fit_plot_x, ploty
    else:
        line_right.current_fit = right_fit
        line_right.allx, line_right.ally = right_plot_x, ploty

    line_right.startX = line_right.allx[- 1]
    line_right.endX = line_right.allx[0]

    line_right.detected = True
    # 打印曲率半径
    # rad_of_curvature(line_left, line_right)
    return output


def prev_window_refer_l(b_img, left_line):
    """
    在上一帧中检测车道线后，参考前面的窗口信息
    """
    # 创建输出图像以绘制并可视化结果。
    output = np.dstack((b_img, b_img, b_img)) * 255

    # 识别图像中所有非零像素的x和y位置。
    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # 设置窗口的边距
    window_margin = left_line.window_margin

    left_line_fit = left_line.current_fit
    leftx_min = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] - window_margin
    leftx_max = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] + window_margin

    # 标识窗口中x和y中的非零值像素。
    left_inds = ((nonzerox >= leftx_min) & (nonzerox <= leftx_max)).nonzero()[0]

    # 提取左、右行像素位置
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]

    output[lefty, leftx] = [255, 0, 0]

    # 将二阶多项式拟合到每个
    left_fit = np.polyfit(lefty, leftx, 2)

    # 生成用于绘图的x和y值。
    plot_y = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    # ax^2 + bx + c
    left_plot_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]

    # left_x_avg = np.average(left_plot_x)
    # right_x_avg = np.average(right_plot_x)

    left_line.prevX.append(left_plot_x)

    if len(left_line.prevX) > 10:
        left_avg_line = smoothing(left_line.prevX, 10)
        left_avg_fit = np.polyfit(plot_y, left_avg_line, 2)
        left_fit_plot_x = left_avg_fit[0] * plot_y ** 2 + left_avg_fit[1] * plot_y + left_avg_fit[2]
        left_line.current_fit = left_avg_fit
        left_line.allx, left_line.ally = left_fit_plot_x, plot_y
    else:
        left_line.current_fit = left_fit
        left_line.allx, left_line.ally = left_plot_x, plot_y

    left_line.startX = left_line.allx[len(left_line.allx) - 1]
    left_line.endX = left_line.allx[0]

    # 打印曲率半径
    # rad_of_curvature(left_line)
    return output


def prev_window_refer(b_img, left_line, right_line):
    """
    在上一帧中检测车道线后，参考前面的窗口信息
    """
    # 创建输出图像以绘制并可视化结果。
    output = np.dstack((b_img, b_img, b_img)) * 255

    # 识别图像中所有非零像素的x和y位置。
    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # 设置窗口的边距
    window_margin = left_line.window_margin

    left_line_fit = left_line.current_fit
    right_line_fit = right_line.current_fit
    leftx_min = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] - window_margin
    leftx_max = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] + window_margin
    rightx_min = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] - window_margin
    rightx_max = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] + window_margin

    # 标识窗口中x和y中的非零值像素。
    left_inds = ((nonzerox >= leftx_min) & (nonzerox <= leftx_max)).nonzero()[0]
    right_inds = ((nonzerox >= rightx_min) & (nonzerox <= rightx_max)).nonzero()[0]

    # 提取左、右行像素位置
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    output[lefty, leftx] = [255, 0, 0]
    output[righty, rightx] = [0, 0, 255]

    # 将二阶多项式拟合到每个
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # 生成用于绘图的x和y值。
    plot_y = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    # ax^2 + bx + c
    left_plot_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_plot_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

    # left_x_avg = np.average(left_plot_x)
    # right_x_avg = np.average(right_plot_x)

    left_line.prevX.append(left_plot_x)
    right_line.prevX.append(right_plot_x)

    if len(left_line.prevX) > 10:
        left_avg_line = smoothing(left_line.prevX, 10)
        left_avg_fit = np.polyfit(plot_y, left_avg_line, 2)
        left_fit_plot_x = left_avg_fit[0] * plot_y ** 2 + left_avg_fit[1] * plot_y + left_avg_fit[2]
        left_line.current_fit = left_avg_fit
        left_line.allx, left_line.ally = left_fit_plot_x, plot_y
    else:
        left_line.current_fit = left_fit
        left_line.allx, left_line.ally = left_plot_x, plot_y

    if len(right_line.prevX) > 10:
        right_avg_line = smoothing(right_line.prevX, 10)
        right_avg_fit = np.polyfit(plot_y, right_avg_line, 2)
        right_fit_plot_x = right_avg_fit[0] * plot_y ** 2 + right_avg_fit[1] * plot_y + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plot_x, plot_y
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plot_x, plot_y

    # 去blind_search如果车道线的标准值高。
    standard = np.std(right_line.allx - left_line.allx)

    if standard > 80:
        left_line.detected = False

    left_line.startX, right_line.startX = left_line.allx[len(left_line.allx) - 1], right_line.allx[
        len(right_line.allx) - 1]
    left_line.endX, right_line.endX = left_line.allx[0], right_line.allx[0]

    # 打印曲率半径
    rad_of_curvature(left_line, right_line)
    return output


def prev_window_refer_r(b_img, right_line):
    """
    在上一帧中检测车道线后，参考前面的窗口信息
    """
    # 创建输出图像以绘制并可视化结果。
    output = np.dstack((b_img, b_img, b_img)) * 255

    # 识别图像中所有非零像素的x和y位置。
    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # 设置窗口的边距
    window_margin = right_line.window_margin

    right_line_fit = right_line.current_fit
    rightx_min = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] - window_margin
    rightx_max = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] + window_margin

    # 标识窗口中x和y中的非零值像素。
    right_inds = ((nonzerox >= rightx_min) & (nonzerox <= rightx_max)).nonzero()[0]

    # 提取左、右行像素位置
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    output[righty, rightx] = [0, 0, 255]

    # 将二阶多项式拟合到每个
    right_fit = np.polyfit(righty, rightx, 2)

    # 生成用于绘图的x和y值。
    plot_y = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    # ax^2 + bx + c
    right_plot_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

    # left_x_avg = np.average(left_plot_x)
    # right_x_avg = np.average(right_plot_x)

    right_line.prevX.append(right_plot_x)

    if len(right_line.prevX) > 10:
        right_avg_line = smoothing(right_line.prevX, 10)
        right_avg_fit = np.polyfit(plot_y, right_avg_line, 2)
        right_fit_plot_x = right_avg_fit[0] * plot_y ** 2 + right_avg_fit[1] * plot_y + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plot_x, plot_y
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plot_x, plot_y

    right_line.startX = right_line.allx[
        len(right_line.allx) - 1]
    right_line.endX = right_line.allx[0]

    # 打印曲率半径
    # rad_of_curvature(left_line, right_line)
    return output


def find_l_lines(binary_img, left_line):
    """
    查找左、右行，隔离左、右行
    盲目搜索-第一帧/迷失车道线
    上一个窗口——检测前一帧中的车道线
    """

    # 如果没有车道线信息
    if left_line.detected is False:
        return blind_search_l(binary_img, left_line)
    # 如果有车道线信息
    else:
        return prev_window_refer_l(binary_img, left_line)


def find_r_lines(binary_img, right_lines):
    """
    查找左、右行，隔离左、右行
    盲目搜索-第一帧/迷失车道线
    上一个窗口——检测前一帧中的车道线
    """

    # 如果没有车道线信息
    if right_lines.detected is False:
        return blind_search_r(binary_img, right_lines)
    # 如果有车道线信息
    else:
        return prev_window_refer_l(binary_img, right_lines)


def find_lr_lines(binary_img, left_line, right_line):
    """
    查找左、右行，隔离左、右行
    盲目搜索-第一帧/迷失车道线
    上一个窗口——检测前一帧中的车道线
    """

    # 如果没有车道线信息
    if left_line.detected is False:
        return blind_search(binary_img, left_line, right_line)
    # 如果有车道线信息
    else:
        return prev_window_refer(binary_img, left_line, right_line)


def draw_lane_l(img, left_line, lane_color=(0, 0, 255)):
    """ 绘制车道线和当前驱动空间 """
    window_img = np.zeros_like(img)

    window_margin = left_line.window_margin
    left_plot_x = left_line.allx
    plot_y = left_line.ally

    # 生成一个多边形来显示搜索窗口区域。
    # 重铸的x和y坐标为fillpoly() CV2可用的格式。
    left_pts_l = np.array([np.transpose(np.vstack([left_plot_x - window_margin / 5, plot_y]))])
    left_pts_r = np.array([np.flipud(np.transpose(np.vstack([left_plot_x + window_margin / 5, plot_y])))])
    left_pts = np.hstack((left_pts_l, left_pts_r))

    # 在扭曲的空白图像上画出车道
    cv2.fillPoly(window_img, np.int_([left_pts]), lane_color)
    cv2.circle(window_img, (int(left_plot_x[-1]), int(plot_y[-1])), 1, (255, 0, 0))

    return window_img


def draw_lane_r(img, right_line, lane_color=(0, 0, 255)):
    """ 绘制车道线和当前驱动空间 """
    window_img = np.zeros_like(img)

    window_margin = right_line.window_margin
    left_plot_x = right_line.allx
    plot_y = right_line.ally

    # 生成一个多边形来显示搜索窗口区域。
    # 重铸的x和y坐标为fillpoly() CV2可用的格式。
    left_pts_l = np.array([np.transpose(np.vstack([left_plot_x - window_margin / 5, plot_y]))])
    left_pts_r = np.array([np.flipud(np.transpose(np.vstack([left_plot_x + window_margin / 5, plot_y])))])
    left_pts = np.hstack((left_pts_l, left_pts_r))

    # 在扭曲的空白图像上画出车道
    cv2.fillPoly(window_img, np.int_([left_pts]), lane_color)
    cv2.circle(window_img, (int(left_plot_x[-1]), int(plot_y[-1])), 1, (255, 0, 0))

    return window_img


def draw_lane(img, left_line, right_line, lane_color=(0, 0, 255), road_color=(0, 255, 0)):
    """ 绘制车道线和当前驱动空间 """
    window_img = np.zeros_like(img)

    window_margin = left_line.window_margin
    left_plot_x, right_plot_x = left_line.allx, right_line.allx
    plot_y = left_line.ally

    # 生成一个多边形来显示搜索窗口区域。
    # 重铸的x和y坐标为fillpoly() CV2可用的格式。
    left_pts_l = np.array([np.transpose(np.vstack([left_plot_x - window_margin / 5, plot_y]))])
    left_pts_r = np.array([np.flipud(np.transpose(np.vstack([left_plot_x + window_margin / 5, plot_y])))])
    left_pts = np.hstack((left_pts_l, left_pts_r))
    right_pts_l = np.array([np.transpose(np.vstack([right_plot_x - window_margin / 5, plot_y]))])
    right_pts_r = np.array([np.flipud(np.transpose(np.vstack([right_plot_x + window_margin / 5, plot_y])))])
    right_pts = np.hstack((right_pts_l, right_pts_r))

    # 在扭曲的空白图像上画出车道
    cv2.fillPoly(window_img, np.int_([left_pts]), lane_color)
    cv2.fillPoly(window_img, np.int_([right_pts]), lane_color)

    # 重铸的X和Y点为fillpoly() CV2可用的格式。
    pts_left = np.array([np.transpose(np.vstack([left_plot_x + window_margin / 5, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plot_x - window_margin / 5, plot_y])))])
    pts = np.hstack((pts_left, pts_right))

    # 绘制行驶区域
    cv2.fillPoly(window_img, np.int_([pts]), road_color)

    return window_img
