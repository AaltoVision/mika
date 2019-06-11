#!/usr/bin/env python
from __future__ import print_function
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image

FRAME_HEIGHT = 180
FRAME_WIDTH = 320
ORIGINAL_HEIGHT = 720
ORIGINAL_WIDTH = 1280

# Camera calibration stuff: K = projection matrix, D = distortion coefficients
K = np.array([[1.03333490e+03, 0.00000000e+00, 6.06973807e+02],
        [0.00000000e+00, 1.03016824e+03, 3.40045442e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]) / 4.0
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]) / 8.0

K[2, 2] = 1.0
D = np.array([[-0.41524183,  0.2372752 ,  0.00153846,  0.00116556, -0.08571556]]) 

class CameraCalibration(object):
    def __init__(self, K=K, D=D, alpha=0.0, height=FRAME_HEIGHT, width=FRAME_WIDTH):
        dim = (width, height)
        self.newK = cv2.getOptimalNewCameraMatrix(K, D, dim, alpha, dim)[0]
        self.K = K
        self.D = D
        self.umap1, self.umap2 = cv2.initUndistortRectifyMap(K, D, None, self.newK, dim, cv2.CV_16SC2)

    def undistort(self, frame):
        return cv2.remap(frame, self.umap1, self.umap2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

class CameraNode(object):
    def __init__(self):
        rospy.init_node('camera')
        self.lens_correction = rospy.get_param('driver/lens_correction', True)
        self.camera_calibration = CameraCalibration()
        self._setup_camera()
        self.frame_pub = rospy.Publisher('camera_frames', Image, queue_size=1)
        self.rate = rospy.Rate(30)

    def _setup_camera(self):
        self.camera = cv2.VideoCapture(0)

    def _capture_frame(self):
        ret, frame = self.camera.read()
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
        frame = np.rot90(frame, 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.lens_correction:
            frame = self.camera_calibration.undistort(frame)
        return ret, frame

    def _create_frame_message(self, frame):
        msg = Image()
        msg.height = FRAME_HEIGHT
        msg.width = FRAME_WIDTH
        msg.encoding = 'rgb8'
        msg.data = np.ravel(frame).tolist()
        return msg

    def loop(self):
        try:
            while not rospy.is_shutdown():
                ret, frame = self._capture_frame()
                msg = self._create_frame_message(frame)
                self.frame_pub.publish(msg)
                self.rate.sleep()
        finally:
            if self.camera is not None and self.camera.isOpened():
                self.camera.release()

if __name__ == "__main__":
    node = CameraNode()
    node.loop()

