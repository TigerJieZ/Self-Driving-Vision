import copy
import time
import cv2
import matplotlib as mpl
import matplotlib.cm
import matplotlib.colors
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf
from DrivingZoneDetection.RoadBoundaryDetection import fcn8_vgg
from DrivingZoneDetection.config import Conf


class FCNRoad:
    """
    road semantic analysis
    """

    def __init__(self, cp_file=None,
                 vgg16_npy_path=None):
        """
        restore model to analysis road
        :param cp_file:
        """
        self.conf = Conf()

        # create network
        self.x_image = tf.placeholder(tf.float32, [1, None, None, 3])
        if vgg16_npy_path is not None:
            self.vgg_fcn = fcn8_vgg.FCN8VGG(vgg16_npy_path=vgg16_npy_path)
        else:
            self.vgg_fcn = fcn8_vgg.FCN8VGG(vgg16_npy_path=self.conf.get_attribute('fcn', 'vgg16_npy_path'))
        self.vgg_fcn.build(self.x_image, debug=True, num_classes=2)

        # restore network
        self.model_file = cp_file
        saver = tf.train.Saver()
        self.sess = tf.Session()
        if cp_file is not None:
            saver.restore(self.sess, cp_file)
        else:
            saver.restore(self.sess, self.conf.get_attribute('fcn', 'cp_file'))

    @staticmethod
    def color_image(image, num_classes=20):
        norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
        mycm = mpl.cm.get_cmap('Set1')
        return mycm(norm(image))

    def predict_img(self, img_bgr):
        """
        predict the img 's road
        :param img:
        :return:
        """
        img = np.expand_dims(img_bgr, axis=0)
        pred = self.sess.run(self.vgg_fcn.pred_up, feed_dict={self.x_image: img})
        return pred

    def run(self, video_type):
        """
        predict the video by video path(video_type)
        :param video_type:
        :return:
        """
        video = cv2.VideoCapture(video_type)
        _, frame = video.read()
        i = 0
        start = time.time()
        while frame is not None:
            _, frame = video.read()
            frame = np.array(frame)
            img = np.expand_dims(frame, axis=0)
            print('-----used time:', time.time() - start)
            start = time.time()
            pred = self.sess.run(self.vgg_fcn.pred_up, feed_dict={self.x_image: img})
            print('>>>>>used time:', time.time() - start)
            start = time.time()
            print('+++++used time:', time.time() - start)
            start = time.time()

    def save_output(self, index, training_image, prediction):
        prediction_label = 1 - prediction[0]
        output_image = copy.copy(training_image)

        # Save prediction
        up_color = self.color_image(prediction[0], 2)
        scp.misc.imsave('output/decision_%d.png' % index, up_color)

        # Merge true positive with training images' green channel
        true_positive = prediction_label
        merge_green = (1 - true_positive) * training_image[..., 1] + true_positive * 255
        output_image[..., 1] = merge_green

        # Merge false positive with training images' red channel
        false_positive = prediction_label
        merge_red = (1 - false_positive) * training_image[..., 0] + false_positive * 255
        output_image[..., 0] = merge_red

        # Merge false negative with training images' blue channel
        false_negative = (1 - prediction_label)
        merge_blue = (1 - false_negative) * training_image[..., 2] + false_negative * 255
        output_image[..., 2] = merge_blue

        import cv2

        cv2.imshow('img', output_image[0, :, :, :])
        cv2.waitKey(100)
        cv2.destroyAllWindows()

        # Save images
        scp.misc.imsave('merge/fcn_%d.png' % index, output_image[0, :, :, :])
