import cv2
import numpy as np
from skimage import draw


class FilterLane:
    """
    detect the road's boundary from fcn-prediction
    """

    def __init__(self):
        pass

    @staticmethod
    def filter_lines(lines):
        """
        filter the lines where abs(slope)<1/3
        :return:
        """
        new_lines = []
        for y1, x1, y2, x2 in lines[:]:
            slope = (x2 - x1) / (y2 - y1)
            if abs(slope) > 0.3:
                new_lines.append([y1, x1, y2, x2])
        return new_lines

    def boundary(self, pred):
        """
        road's boundary detection
        :param pred:
        :return:
        """
        pred *= 255
        # pred=pred.transpose((1,2,0))
        pred = pred[0]
        # pred.dtype='uint8'
        pred = pred.astype(np.uint8)
        up_color = cv2.Canny(pred, 50, 130)

        lines = cv2.HoughLinesP(up_color, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=10)

        lines = lines[:, 0, :]
        img = np.zeros_like(pred)
        lines = self.filter_lines(lines)
        for y1, x1, y2, x2 in lines[:]:
            #     cv2.line(pred, (x1, y1), (x2, y2), (100), 1)
            slope = (x2 - x1) / (y2 - y1)
            rr, cc = draw.line(x1, y1, x2, y2)
            img[rr, cc] = 150
        return img
