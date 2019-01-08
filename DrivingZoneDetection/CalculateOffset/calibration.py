# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.image as mpimg
import glob
import matplotlib.pyplot as plt
from DrivingZoneDetection.config import Conf


def calib():
    """
    To get an undistorted image, we need camera matrix & distortion coefficient
    Calculate them with 9*6 20 chessboard images
    """
    conf = Conf()
    # Read in and make a list of calibration images
    images = glob.glob(conf.get_attribute('offset', 'calib_img_path'))

    # Array to store object points and image points from all the images

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Prepare object points
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # x,y coordinates

    for fname in images:

        img = cv2.imread(fname)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If corners are found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            continue

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist


def undistort(img, mtx, dist):
    """ undistort image """
    return cv2.undistort(img, mtx, dist, None, mtx)


def undistort_calibration(mtx, dist):
    # 将棋盘图像矫正
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[10, 6])
    image = []
    calibration = []
    for fname in images:
        img = mpimg.imread(fname)
        image.append(img)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        calibration.append(cv2.undistort(gray, mtx, dist, None, mtx))

    ax1.imshow(image[0])
    ax1.set_title('Distorted image')
    ax1.set_axis_off()

    ax2.imshow(calibration[0])
    ax2.set_title('Undistorted image')
    ax2.set_axis_off()

    ax3.imshow(image[1])
    ax3.set_title('Distorted image')
    ax3.set_axis_off()

    ax4.imshow(calibration[1])
    ax4.set_title('Undistorted image')
    ax4.set_axis_off()

    plt.show()


if __name__ == '__main__':
    mtx, dist = calib()
    undistort_calibration(mtx, dist)
