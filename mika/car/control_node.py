#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import time
import numpy as np
import smbus2 as smbus
import rospy
from PWMDriver import PWMDriver
from std_msgs.msg import Float32MultiArray, Bool, MultiArrayDimension

ARDUINO_ADDR = 0x06

SERVO_HZ = 50
SERVO_PERIOD = 1000000 / SERVO_HZ # 1,000,000us / frequency
NEUTRAL_PW = 1525
PW_RANGE = 350
STEERING_BIAS = 30
MIN_PW = NEUTRAL_PW - PW_RANGE
MAX_PW = NEUTRAL_PW + PW_RANGE
PW_RANGE = MAX_PW - MIN_PW
SERVO_MIN = int(MIN_PW * 4096 / SERVO_PERIOD)  # SERVO_MIN / 4096 is the mininimum pulse width.
SERVO_MAX = int(MAX_PW * 4096 / SERVO_PERIOD)
PW_ACCURACY = 10
OLD_STATE_FRACTION = 0.90

MIN_AUTONOMOUS_THROTTLE_PW = NEUTRAL_PW - 50
MAX_AUTONOMOUS_THROTTLE_PW = NEUTRAL_PW + 50

def to_int(byte_list):
    number = 0
    for i in reversed(range(len(byte_list))):
        number = (number ^ byte_list[i]) << (i * 8)
    return number

def parse_bytes(byte_list):
    # Reads 16 bit integers.
    bytes_total = len(byte_list) * 8
    numbers = bytes_total / 16
    out = np.zeros(numbers, dtype=np.float32)
    for i in range(numbers):
        out[i] = to_int(byte_list[i*2:(i+1)*2])
    return out

class ControlNode(object):
    def __init__(self, bus, pwm_driver):
        rospy.init_node("control_node")
        self.command_sub = rospy.Subscriber("commands", Float32MultiArray, self._message_callback)
        self.command_pub = rospy.Publisher("manual_commands", Float32MultiArray, queue_size=1)
        self.start_stop = rospy.Publisher("start_stop", Bool, queue_size=1)
        self.bus = bus
        self.pwm_driver = pwm_driver
        self.constant_throttle = self._pulse_width_to_command(1700.0)
        self.autonomous_mode = False
        self.manual_throttle = False
        self.constant_throttle = 0.1
        self._state = np.zeros(4, dtype=np.uint16)
        self._autonomous_command = np.zeros(2, dtype=np.float32)
        self._filtered_autonomous_command = np.zeros(2, dtype=np.float32)

    def step(self):
        state = self._update_state()
        speed, pwm1, pwm2, pwm3 = state

        self._update_status(state)

        if self.autonomous_mode:
            self._filter_autonomous_command()
            self._act_autonomous()
        else:
            self._act_manual(state)
        #print("speed: {} pwm 0: {} pwm 1: {}".format(speed, pwm1, pwm2), end='\r')
            
    def _filter_autonomous_command(self):
        self._filtered_autonomous_command = 0.5 * self._filtered_autonomous_command + 0.5 * self._autonomous_command

    def _message_callback(self, message):
        # Commands are sent in the range -1, 1.
        # The first command is steering and the second is throttle.
        command = message.data
        assert len(command) == 2
        self._autonomous_command = np.array(command)

    def _command_to_pulse_width(self, command):
        return NEUTRAL_PW + command * PW_RANGE

    def _pulse_width_to_command(self, pulse_width, bias=0):
        return (pulse_width - NEUTRAL_PW - bias) / PW_RANGE

    def _calculate_throttle(self, steering_command):
        throttle_command = self.constant_throttle * (1 - 0.70 * np.abs(steering_command) ** 2)
        return self._command_to_pulse_width(throttle_command)

    def _act_autonomous(self):
        steering_pw, throttle_pw = self._command_to_pulse_width(self._filtered_autonomous_command)
        #throttle_pw = np.clip(throttle_pw, MIN_AUTONOMOUS_THROTTLE_PW, MAX_AUTONOMOUS_THROTTLE_PW)
        if self.manual_throttle:
            throttle_pw = self._state[2]
            steering_pw = self._state[1]
        else:
            steer_cmd = self._pulse_width_to_command(steering_pw, bias=STEERING_BIAS)
            throttle_pw = self._calculate_throttle(steer_cmd) + 5
        #print("steering", steering_pw, "throttle", throttle_pw, end='\r')
        self._set_pwm(0, steering_pw + STEERING_BIAS)
        self._set_pwm(1, throttle_pw)

    def _act_manual(self, state):
        self._set_pwm(0, state[1] + STEERING_BIAS)
        self._set_pwm(1, state[2])
        msg = self._bc_create_state_message()
        self.command_pub.publish(msg)
    
    def _bc_create_state_message(self):
        dim = MultiArrayDimension(label='action', size=3, stride=1)
        msg = Float32MultiArray(data=[self._pulse_width_to_command(x) for x in self._state])
        msg.layout.dim.append(dim)
        return msg

    def _set_pwm(self, channel, pulse_width):
        pulse_width = self._round(pulse_width)
        period_part = pulse_width * 4096 / SERVO_PERIOD
        if channel == 0:
            # Steering limits.
            period_part = np.clip(period_part, SERVO_MIN, SERVO_MAX)
        # Pulse width as fraction of 4096 (12 bits)
        self.pwm_driver.setPWM(channel, 0, period_part)

    def _round(self, pulse_width):
        # Round the number to get rid of some noise.
        return int(PW_ACCURACY * round(float(pulse_width) / PW_ACCURACY))

    def _update_state(self):
        try:
            self.bus.write_byte(ARDUINO_ADDR, 1)
            received = self.bus.read_i2c_block_data(ARDUINO_ADDR, 0, 8)
            # speed, channel 1, channel 2, channel 3 pwm
            state = parse_bytes(received)
            self._state = OLD_STATE_FRACTION * self._state + (1.0 - OLD_STATE_FRACTION) * state.astype(np.float32)
            return self._state
        except IOError:
            return self._state

    def _update_status(self, state):
        if self.autonomous_mode and state[3] < 1500:
            self._stop_autonomous_mode()
        elif not self.autonomous_mode and state[3] > 1500:
            self._start_autonomous_mode()

    def _start_autonomous_mode(self):
        print("\nAutonomous mode turned on.")
        self.autonomous_mode = True
        msg = Bool(data=True)
        self.start_stop.publish(msg)
        
    def _stop_autonomous_mode(self):
        print("\nAutonomous mode turned off.")
        self.autonomous_mode = False
        msg = Bool(data=False)
        self.start_stop.publish(msg)

def main():
    pwm_driver = PWMDriver()
    pwm_driver.openPCA9685()
    pwm_driver.reset()
    pwm_driver.setPWMFrequency(SERVO_HZ)

    bus = smbus.SMBus(0)
    driver = ControlNode(bus, pwm_driver)
    
    rate = rospy.Rate(100)

    try:
        while not rospy.is_shutdown():
            driver.step()
            rate.sleep()
    finally:
        pwm_driver.reset()
        pwm_driver.setAllPWM(0, 0)
        pwm_driver.closePCA9685()


if __name__ == "__main__":
    main()

