# -*- coding: utf-8 -*-

import numpy as np


class Line:
    def __init__(self):
        # 在最后一次迭代中是否检测到该行
        self.detected = False
        # 窗口的宽度
        self.window_margin = 56
        # 在最后n次迭代中拟合线的x值
        self.prevX = []
        # 最近拟合多项式系数
        self.current_fit = [np.array([False])]
        # 某些单位线的曲率半径
        self.radius_of_curvature = None
        # starting x_value
        self.startX = None
        # ending x_value
        self.endX = None
        # 检测到的行像素的x值
        self.allx = None
        # y values for detected line pixels
        # 检测到的行像素的y值
        self.ally = None

        # 道路的偏转方向，正值代表向右转，负值代表向左转
        self.road_inf = None
        # 车道线的曲率
        self.curvature = None
        # 车辆偏移百分比，负值代表向左偏移，正值代表向右偏移，0表示车道处于行驶区域的中心
        self.deviation = None
