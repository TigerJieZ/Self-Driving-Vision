import configparser
import os


class Conf:
    def __init__(self):
        self.__file = 'settings.conf'
        self.__conf = configparser.RawConfigParser()

        # 如果配置文件存在读取配置文件，若不存在初始化配置文件
        if os.path.exists(self.__file):
            self.__conf.read(self.__file)
        else:
            self.init_all_atrribute()

    def __str__(self):
        return "《常熟理工车联网实验室自动驾驶》--行驶区域感知（车辆偏移计算）-配置模块"

    def init_all_atrribute(self):
        # fcn
        section_fcn = 'fcn'
        self.__conf.add_section(section_fcn)
        # 存放vgg16权重文件的路径
        self.__conf.set(section_fcn, 'vgg16_npy_path', '/media/n6-301/workspace/CslgSelfCar_vision/data/vgg16.npy')
        self.__conf.set(section_fcn, 'cp_file', '/media/n6-301/workspace/CslgSelfCar_vision/data/model.ckpt')

        # 基于fcn的右道路边界识别结果的车辆位置偏移中心线的偏移量计算
        section_offset = 'offset'
        self.__conf.add_section(section_offset)
        # roi区域的上下边界
        self.__conf.set(section_offset, 'roi_y_top', 380)
        self.__conf.set(section_offset, 'roi_y_bottom', 12)
        # 道路的实际宽度
        self.__conf.set(section_offset, 'road_width_real', 546)
        # 图像中道路的偏移量计算采用的y所对应的道路像素宽度
        self.__conf.set(section_offset, 'road_width_img', 994)
        # 计算摄像头畸变系数的棋盘图像
        self.__conf.set(section_offset, 'calib_img_path',
                        '/media/n6-301/workspace/CslgSelfCar_vision/DrivingZoneDetection/CalculateOffset/camera_cal/calibration*.jpg')

        # 写入文件
        with open(self.__file, 'w') as config_file:
            self.__conf.write(config_file)

    def get_attribute(self, section, atb):
        """
        获取对应的section中的attribute
        :param section:
        :param atb:
        :return:
        """
        op = self.__conf.get(section=section, option=atb)
        return op

    def show_all(self):
        pass
