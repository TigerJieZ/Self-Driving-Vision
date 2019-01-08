from DrivingZoneDetection.CalculateOffset.utils import *
from DrivingZoneDetection.CalculateOffset import m_utils
import cv2
import os
from DrivingZoneDetection.CalculateOffset.line import Line
import matplotlib.pyplot as plt
from DrivingZoneDetection.config import Conf


class Lane:
    def __init__(self):
        self.conf = Conf()

        # 默认车道宽度3米
        self.LENGTH_ROAD = int(self.conf.get_attribute('offset', 'road_width_real'))
        self.LENGTH_ROAD_IMG = int(self.conf.get_attribute('offset', 'road_width_img'))

        # roi区域的截取大小
        self.roi_y_top = int(self.conf.get_attribute('offset', 'roi_y_top'))
        self.roi_y_bottom = int(self.conf.get_attribute('offset', 'roi_y_bottom'))

        # 车道线数据存储器
        self.left_line = Line()
        self.right_line = Line()

        # 用于透视变换的四个顶点
        self.warp_point = None
        # 透视参数和畸变系数
        self.mtx, self.dist, self.src, self.dst = None, None, None, None

    def init_warp(self, shape):
        self.warp_point = self.auto_warp_p(shape)

        # 如果相机纠正数据和透视数据文件不存在则计算数据并记录否则加载数据
        if not os.path.exists('temp_carla.pkl'):
            # mtx, dist, src, dst = m_utils.store_data([640, 128])
            self.mtx, self.dist, self.src, self.dst = m_utils.store_data(self.warp_point)
        else:
            self.mtx, self.dist, self.src, self.dst = m_utils.load_data()

    @staticmethod
    def auto_warp_p(shape):
        """
        根据shape自动计算透视变换的四点顶点
        :param shape:(h,w)
        :return:
        """

        top_left_p = [shape[1] / 2 - shape[1] / 6, 10]
        top_right_p = [shape[1] / 2 + shape[1] / 6, 10]
        bottom_left_p = [shape[1] / 16, shape[0] - 10]
        bottom_right_p = [shape[1] / 16 * 15, shape[0] - 10]

        return [top_left_p, top_right_p, bottom_left_p, bottom_right_p]

    def get_excursion_l_simple(self, image):
        """
        计算车辆偏移中心线的值
        :param image:
        :return: 偏移中心线的值/cm,<0:right,>0:left
        """
        # 截取出roi区域（图片中含有道路）
        shape = image.shape

        roi_image = image[280:shape[0] - 12, 0:shape[1], 2]
        roi_color_image = image[280:shape[0] - 12, 0:shape[1]]
        # show_img(roi_color_image, 10, 'roi')

        # 色值过滤
        new_img = roi_color_image[:, :, 2]

        new_img = new_img > 145
        new_img = new_img * 255

        new_img = np.array(new_img).astype(np.uint8)
        # show_img(new_img, 10000, 'new_img')

        # import pylab
        # pylab.imshow(new_img)
        #
        # pylab.plot([200, 300, 600, 500], [150, 10, 150, 10], 'r*')
        # # pylab.plot([110, 296, 530, 344], [180, 5, 180, 5], 'r*')
        # pylab.show()

        # 透视图像使图像呈俯视角度
        perspective_image, M, Minv = warp_image(new_img, src=self.src, dst=self.dst, size=(720, 720))
        perspective_image_color = np.dstack((perspective_image, perspective_image, perspective_image))

        # zoom_perspective_image = cv2.resize(perspective_image, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA)
        # pylab.imshow(perspective_image)
        # pylab.show()
        # exit()
        # show_img(perspective_image, 10000, 'perspective')
        # plt.imshow(perspective_image)
        # plt.show()

        # 寻找车道线
        try:
            find_l_lines(perspective_image, left_line=self.left_line)
        except Exception as e:
            self.left_line.detected = False
            print(e)

        if self.left_line.detected:
            # 将左车道线识别结果绘制到透视后的图像上
            w_color_result = draw_lane_l(perspective_image_color, self.left_line)

            # 将绘制车道线的俯视角度透视成原角度
            color_road = cv2.warpPerspective(w_color_result, Minv, (new_img.shape[1], new_img.shape[0]))
            # plt.imshow(color_road)
            # plt.show()
            # show_img(color_road, 10000, 'color_road')

            # 获取左车道线底部的点坐标
            a = color_road[:, :, 0] > 0
            left_line_bottom = np.where(a == True)
            left_line_bottom = (left_line_bottom[0][0] + 280, left_line_bottom[1][0])
            print('左车道线底部点位：', left_line_bottom)
            '''
            left_bottom(429,253),right_bottom=(429,670)
            车道宽度为417像素，一半为208
            '''

            # 将透视回原角度的车道线标注叠加到原图像中
            mask = np.zeros_like(image)
            mask[280:shape[0] - 12, 0:shape[1]] = color_road
            road_image = cv2.addWeighted(image, 1, mask, 0.3, 0)

            road_center = (left_line_bottom[1] + 208, left_line_bottom[0])

            # 打印当前帧的车道中心点
            cv2.circle(road_image, road_center, 5, (255, 255, 0))

            # 计算偏移
            py_img = road_center[0] - shape[1] / 2
            # print('偏移像素：', py_img)
            py_real = self.LENGTH_ROAD / 417.0 * float(py_img)
            # print('偏移厘米：', py_real)

            if py_real > 5:
                cv2.putText(road_image, 'Left ' + str(py_real) + ' CM', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255),
                            2)
            elif py_real < -5:
                cv2.putText(road_image, 'Right ' + str(-py_real) + ' CM', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 2)
            else:
                cv2.putText(road_image, 'Road Center', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            return py_real

    def get_excursion_r_fcn(self, image):
        """
        计算车辆偏移中心线的值
        :param image:
        :return: 偏移中心线的值/cm,<0:right,>0:left
        """

        # 截取出roi区域（图片中含有道路）
        shape = image.shape

        if len(shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # roi截取linux
        roi_image = image[self.roi_y_top:shape[0] - self.roi_y_bottom, 0:shape[1]]

        if self.warp_point is None:
            self.init_warp(roi_image.shape)

        # m_utils.show_warpp(roi_image, self.warp_point)

        # 透视图像使图像呈俯视角度
        perspective_image, M, Minv = warp_image(roi_image, src=self.src, dst=self.dst, size=(720, 720))
        perspective_image_color = np.dstack((perspective_image, perspective_image, perspective_image))

        # 寻找车道线
        try:
            find_r_lines(perspective_image, right_lines=self.right_line)
        except Exception as e:
            self.right_line.detected = False
            print("find wrong", e)

        if self.right_line.detected:
            # 将左车道线识别结果绘制到透视后的图像上
            w_color_result = draw_lane_r(perspective_image_color, self.right_line)

            # 将绘制车道线的俯视角度透视成原角度
            color_road = cv2.warpPerspective(w_color_result, Minv, (roi_image.shape[1], roi_image.shape[0]))
            plt.imshow(color_road)
            plt.show()
            # show_img(color_road, 10000, 'color_road')

            # 获取左车道线底部的点坐标
            a = color_road[:, :, 0] > 0
            right_line_bottom = np.where(a == True)
            right_line_bottom = (right_line_bottom[0][0] + self.roi_y_top, right_line_bottom[1][0])
            print('右车道线底部点位：', right_line_bottom)
            '''
            carla
            left_bottom(429,253),right_bottom=(429,670)
            车道宽度为417像素，一半为208
            '''
            '''
            mine
            left_bottom(80,700),right_bottom=(1074,700)
            车道宽度为994像素，一半为497
            '''

            # 将透视回原角度的车道线标注叠加到原图像中
            result = m_utils.combine_lane_img(color_road,
                                              cv2.imread('/media/n6-301/workspace/CslgSelfCar_vision/data/img.jpg'),
                                              self.roi_y_top, self.roi_y_bottom)

            road_center = (right_line_bottom[1] - int(self.LENGTH_ROAD_IMG/2), right_line_bottom[0])

            # 打印当前帧的车道中心点
            cv2.circle(result, road_center, 5, 255)

            # 计算偏移
            py_img = road_center[0] - shape[1] / 2
            # print('偏移像素：', py_img)
            py_real = self.LENGTH_ROAD / self.LENGTH_ROAD_IMG * float(py_img)
            # print('偏移厘米：', py_real)

            if py_real > 5:
                cv2.putText(result, 'Left ' + str(py_real) + ' CM', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                            255,
                            2)
            elif py_real < -5:
                cv2.putText(result, 'Right ' + str(-py_real) + ' CM', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                            255, 2)
            else:
                cv2.putText(result, 'Road Center', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            plt.imshow(result)
            plt.show()
            print(py_real)
            return py_real


if __name__ == '__main__':
    img = cv2.imread('/root/PycharmProjects/CslgSelfCar_Vision/data/line.png')
    lane = Lane()
    lane.get_excursion_r_fcn(img)
