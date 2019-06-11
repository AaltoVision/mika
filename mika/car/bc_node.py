#!/usr/bin/env python
from __future__ import print_function
import time
import tensorflow as tf
import rospy
import sys
import time
import os
import csv
import cv2
import numpy as np
from std_msgs.msg import MultiArrayDimension, UInt8MultiArray, Float32MultiArray, Bool
from sensor_msgs.msg import Image
from argparse import ArgumentParser
from multiprocessing import Process, Queue

FRAME_HEIGHT = 180
FRAME_WIDTH = 320
ORIGINAL_HEIGHT = 720
ORIGINAL_WIDTH = 1280

def frame_writer(queue, image_dir):
    while True:
        index, command, frame = queue.get()
        if index == -1:
            return 0
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(image_dir, '{0:06d}.png'.format(index)), frame)
        path = os.path.join(image_dir, "{0:06d}.cmd".format(index))
        with open(path, 'wt') as f:
            f.write(" ".join(map(str, command)))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class BCNode(object):
    def __init__(self, path):
        rospy.init_node('agent')
        self.frame_queue = Queue(128)
        self.pub = rospy.Publisher('commands', Float32MultiArray, queue_size=1)
        self.sub = rospy.Subscriber('start_stop', Bool, self._start_stop_callback)
        self.action_sub = rospy.Subscriber('manual_commands', Float32MultiArray, self._manual_command_callback)
        agent_model = rospy.get_param('agent/load_model')

        self._load_model(agent_model)
        path_dir = os.path.join('.', path)
        os.mkdir(path_dir)
        print(path)
        self.path_dir = path_dir

        self.image_process = Process(target=frame_writer, args=(self.frame_queue, path_dir))
        self.image_process.start()
        rospy.on_shutdown(self._shutdown)

        self.cam_sub = rospy.Subscriber('camera_frames', Image, self._camera_callback)
        self.is_active = False
        self.frames_processed = 0
        self.t0 = time.time()
        self.latest_command = [0, 0, 0]

    def _manual_command_callback(self, msg):
        self.latest_command = msg.data

    def _camera_callback(self, msg):
        frame = np.ndarray(shape=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8, buffer=msg.data)
        if self.is_active:
            speed = self.latest_command[0]
            frame = (frame / 127.5) - 1.0
            with self.graph.as_default():
                command = self.model.predict([frame[None], np.array([[speed]], dtype=np.float32)])[0]
            msg = self._create_action_message(command)
            self.pub.publish(msg)
        else:
            self.frame_queue.put((self.frames_processed, self.latest_command, frame))
        self.frames_processed += 1
        # if self.frames_processed % 10 == 0:
        #     fps = 10.0/(time.time() - self.t0)
        #     self.t0 = time.time()
        #     print("FPS: {:.2f}".format(fps))

    def _load_model(self, agent_model):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf.keras.backend.set_session(self.sess)
        self.model = tf.keras.models.load_model(agent_model, {
            "GlorotUniform": tf.keras.initializers.glorot_uniform()
        })
        self.graph = tf.get_default_graph()

    def _start_stop_callback(self, msg):
        self.is_active = msg.data

    def _shutdown(self):
        print("terminating process")
        self.frame_queue.put((-1, None, None))
        self.image_process.join()

    def _create_action_message(self, action):
        dim = MultiArrayDimension(label='action', size=2, stride=1)
        msg = Float32MultiArray(data=action.tolist())
        msg.layout.dim.append(dim)
        return msg


def main(args):
    try:
        os.makedirs('/tmp/images')
    except OSError:
        pass
    path = os.path.join('/tmp/images/', time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
    node = BCNode(path)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)

