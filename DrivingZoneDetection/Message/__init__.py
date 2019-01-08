import logging
import os
import time


class Message:

    def __init__(self):
        self.logger = logging.getLogger()

    def init_log(self, log_dir='/log/'):
        """
        初始化日志模块
        :return:
        """
        if not os.path.exists(log_dir):
            print('日志目录不存在，创建日志目录')
            self.logger.info('日志目录不存在，创建日志目录')
            os.mkdir(log_dir)

        self.logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_name = log_dir + rq + '.log'
        logfile = log_name
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        self.logger.addHandler(fh)

    @staticmethod
    def print_excursion(excursion):
        """
        打印
        :param excursion:
        :return:
        """
        if excursion > 5:
            print("车辆左偏中心线" + str(excursion) + ' CM')
        elif excursion < -5:
            print("车辆右偏中心线" + str(abs(excursion)) + ' CM')
        else:
            print("车辆处于中心线上")

    @staticmethod
    def print_std_input(text):
        print("请传入正确的参数！" + "正确的参数为：" + text)

    def print_log(self, text, _type='info'):
        """
        根据type来以对应的级别写入日志文件中，并将信息打印在console中
        :param text:
        :param _type:
        :return:
        """
        if _type == 'info':
            self.logger.info(text)
        elif _type == 'debug':
            self.logger.debug(text)

        print(text)
