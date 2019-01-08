from DrivingZoneDetection.RoadBoundaryDetection.fcn_pred import FCNRoad
from DrivingZoneDetection.RoadBoundaryDetection.filter_lane import FilterLane
from DrivingZoneDetection.CalculateOffset.lane_offset import Lane
from DrivingZoneDetection.Message import Message
import cv2


class AppLane:
    def __init__(self):
        """
        init the app
        """

        # fcn network to predict the img's road
        self.fcn = FCNRoad()
        # filter the road's right boundary
        self.filter_lane = FilterLane()
        # find the right road's boundary-line and calculate the offset(car-road_center)
        self.lane_offset = Lane()
        # communication module
        self.message = Message()

    def __str__(self):
        return "《常熟理工车联网实验室自动驾驶》--行驶区域感知（车辆偏移计算）"

    def main(self, img=None, video_type=None):
        """

        :param img:
        :param video_type:
        :return:
        """

        if img is not None:
            pred_img = self.fcn.predict_img(img_bgr=img)
            line_img = self.filter_lane.boundary(pred=pred_img)
            import matplotlib.pyplot as plt
            plt.imshow(line_img)
            plt.show()
            excursion = self.lane_offset.get_excursion_r_fcn(line_img)
            self.message.print_excursion(excursion)
        elif video_type is not None:
            video = cv2.VideoCapture(video_type)
            _, frame = video.read()
            while frame is not None:
                pred_img = self.fcn.predict_img(img_bgr=frame)
                line_img = self.filter_lane.boundary(pred=pred_img)
                excursion = self.lane_offset.get_excursion_r_fcn(line_img)
                self.message.print_excursion(excursion)

                _, frame = video.read()
        else:
            self.message.print_std_input("(img:bgr_img / video_type:(camera_path/video_path))")


if __name__ == '__main__':
    app = AppLane()
    app.main(cv2.imread('/media/n6-301/workspace/CslgSelfCar_vision/data/img.jpg'))
