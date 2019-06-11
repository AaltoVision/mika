
# Mika
Authors: [Kenneth Blomqvist](http://github.com/kekeblom), [Oskari Mantere](https://github.com/omantere), [Aleksi Tella](https://github.com/jormeli)

## Overview

This repository contains code for an autonomous radio controlled car using an Nvidia Jetson TX1 computer. The software is built on top of ROS kinetic and JetPack 3.3.

The Jetson is hooked up to an Arduino Nano which is used to read pulse-width modulated signals from the RC radio receiver. The arduino is hooked up to the HAL sensor of the brushless motor of the car and measures the speed of the motor. The arduino communicates with the Jetson through an I2C bus.

We make use of three channels of the RC radio. The first one is the steering channel, the second is throttle and the third is a switch which flips between manual and autonomous mode.

Hardware used:
- Jetson TX1
- PCA9685 I2C PWM servo driver
- Arduino Nano
- Pololu U3V70A step-up converter to power the Jetson
- Pololu D24V50F5 to power the Arduino and other 5v components
- 3-channel 2.4GHz rc radio system
- Brushless ESC
- Brushless motor
- Servo
- usb RGB camera
- 7.4V battery
- 1/10th scale radio controlled touring car kit

A simple behavioral cloning agent is implemented which takes data recorded on the car and learns a model to predict actions taken by a human driver.

The actual car itself runs four different ROS nodes:
- `mika/car/camera_node.py` which captures images from the camera, optionally undistorts them and publishes them to a rostopic.
- `mika/car/bc_node.py` this will load a learned model, read camera images and publish predictions made by the model to a topic.
- `mika/car/control_node.py`this node reads data from the arduino, commands published by the `bc_node` and writes them to the I2C servo driver.

The `launch/bc.launch` is a roslaunch file which can be used to launch ros and the required nodes all at once.


## Setup

Install JetPack 3.3 using the Nvidia installer.

Install ROS kinetic on the Jetson. This step can be automated using the ansible script found in `ops/setup_ros.yml`.

Flash the arduino code tachometer and PWM reader code found under `arduino/tachometer_pwm_reader/` on the arduino. Connect the arduino to the Jetson through the I2C channel and the arduino to the RC receiver.

Connect the PCA9685 driver onto the I2C bus. Install [this python library](https://github.com/kekeblom/JHPWMDriver) which wraps the JetsonHacks PCA9685 c++ library into a python library.


## Training a model

Record training data by driving the car manually. The data will be recorded under `/tmp/images/`. Transfer the data off the car for training.

First we need to create a tfrecords file with the training data for training. Split the data into a trainingand test set. Run `python mika/data_wrangler.py --from-dir <path-to-images-dir> --to-dir <some-path>` for both datasets. The `--to-dir` parameter specifies where the tfrecords file will be created. The images will also be copied to this directory.

To train a model, run `python mika/train.py --dataset <path-to-tfrecords> --validation <path-to-validation-tfrecrods>`. The model will be saved under `models/bc`.

Transfer the `model_weigthts.h5` onto the car and update `launch/bc.launch` to point to it.


