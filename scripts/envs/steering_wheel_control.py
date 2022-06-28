import carla
import pygame

import random
import numpy as np
import math
import queue

import sys
if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    from ConfigParser import RawConfigParser as ConfigParser

from pygame.locals import K_r, K_w, K_s, K_a, K_d, K_SPACE


class SteeringWheelControl:
    def __init__(self, is_wheel=False):
        self._control = carla.VehicleControl()
        # initialize steering wheel
        if is_wheel:
            pygame.joystick.init()
            joystick_count = pygame.joystick.get_count()
            if joystick_count > 1:
                raise ValueError("Please Connect Just One Joystick")

            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()

            self._parser = ConfigParser()
            self._parser.read('wheel_config.ini')
            self._steer_idx = int(
                self._parser.get('G29 Racing Wheel', 'steering_wheel'))
            self._throttle_idx = int(
                self._parser.get('G29 Racing Wheel', 'throttle'))
            self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
            self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
            self._handbrake_idx = int(
                self._parser.get('G29 Racing Wheel', 'handbrake'))
        self.is_wheel = is_wheel

    def parseKey(self):
        ready_pressed = False
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if event.key == K_r:
                    print("Keyboard ready pressed")
                    ready_pressed = True
        return ready_pressed

    def parseVehicleKey(self, milliseconds):
        keys = pygame.key.get_pressed()
        if keys[K_w]:
            self._control.throttle = 1.0#min(self._control.throttle + 0.01, 1)
        else:
            self._control.throttle = 0.0
        if keys[K_s]:
            self._control.brake = 1.0#min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

        return self._control

    def parseButton(self):
        ready_pressed = False
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                print(event.button)
                if event.button == 0:
                    ready_pressed = True
        return ready_pressed

    def parseVehicleWheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        # toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

        return self._control
