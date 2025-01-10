from collections import deque
import math
import sys
import os

FILE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, FILE_DIR)
# import ipdb
import numpy as np
import carla
import random
from Scenarios.Tinyscenarios import *
from Logger import *

from .controller import VehiclePIDController


class PIDcontroller(VehiclePIDController):
    def __init__(self, vehicle, args_lateral, args_longitudinal, acc_lim1, acc_lim2):
        self.vehicle = vehicle
        self.acc_lim1 = acc_lim1
        self.acc_lim2 = acc_lim2
        self.v_o = None
        self.dis_o = None
        super(PIDcontroller, self).__init__(vehicle, args_lateral, args_longitudinal)
        # self.adjust_max_brake(1.0)

    def max_throttle_control(self, decount, target_velocity):
        if decount < target_velocity / 10 * 3:
            cur_max_throttle = 0.2 + (decount / 15) * 0.5
            # message(cur_max_throttle) 
            self.adjust_max_throttle(cur_max_throttle)

    def increase(self, ego_wp, acceleration, target_velocity, distance=6):
        target_velocity = target_velocity + 3
        if abs(acceleration.x) < self.acc_lim1 and abs(acceleration.y) < self.acc_lim1:
            control = self.run_step(target_velocity,
                                    move_waypoint_forward(ego_wp, distance))
            # ego.apply_control(control)
            # message(f'1c {self.PID.max_throt},speed:{ego_speed}km/h, throttle: {control.throttle},max: {self.PID.max_throt}')
        elif abs(acceleration.x) < self.acc_lim2 or abs(acceleration.y) < self.acc_lim2:
            control = self.run_step(target_velocity,
                                    move_waypoint_forward(ego_wp, distance))
            control.throttle = control.throttle * 0.8
            # message(f'2count:{self.default_settings["de_count"]}, speed:{ego_speed}km/h, throttle: {control.throttle}')
        else:
            control = self.run_step(target_velocity,
                                    move_waypoint_forward(ego_wp, distance + 2))
            control.throttle = control.throttle * 0.6
        return control

    def quick_stop(self, end_wp):
        control = self.run_step(0,
                                end_wp, immidiate=0)
        control.throttle = 0.0
        control.brake = 1.0
        # cur_loc = self.vehicle.get_transform().location
        # dis = cur_loc.distance(end_wp.transform.location)
        vel = [self.vehicle.get_velocity().x, self.vehicle.get_velocity().y, self.vehicle.get_velocity().z]
        # acc = [self.vehicle.get_velocity().x ** 2 / (2 * dis),
        #        self.vehicle.get_velocity().y ** 2 / (2 * dis),
        #        self.vehicle.get_velocity().z ** 2 / (2 * dis)]
        acc = [self.vehicle.get_acceleration().x, self.vehicle.get_acceleration().y, self.vehicle.get_acceleration().z]
        self.vehicle.apply_control(control)
        return acc, control.brake, control.throttle, vel

    def decrease(self, end_wp, target_speed):
        # max_brake = 0.001
        self.adjust_max_brake(0.000)
        ego_speed = round(math.sqrt(self.vehicle.get_velocity().x ** 2 + self.vehicle.get_velocity().y ** 2) * 3.6, 2)
        t_speed = target_speed + random.randint(2, 5)

        if self.v_o is None:
            if ego_speed - target_speed <= 3:
                control = self.increase(end_wp, self.vehicle.get_acceleration(), target_speed)
                self.vehicle.apply_control(control)
                acc = [self.vehicle.get_acceleration().x * 0.4, self.vehicle.get_acceleration().y * 0.4, self.vehicle.get_acceleration().z * 0.4]
                vel = [self.vehicle.get_velocity().x, self.vehicle.get_velocity().y, self.vehicle.get_velocity().z]
                return acc, control.brake, control.throttle, vel
            self.v_o = [self.vehicle.get_velocity().x, self.vehicle.get_velocity().y, self.vehicle.get_velocity().z]
            self.dis_o = self.vehicle.get_transform().location.distance(end_wp.transform.location)

        control = self.run_step(t_speed, end_wp, immidiate=1)
        if ego_speed <= 0.05:
            control.brake = 1.0
            control.throttle = 0.0
            self.vehicle.apply_control(control)
            acc = [self.vehicle.get_acceleration().x, self.vehicle.get_acceleration().y, self.vehicle.get_acceleration().z]
            vel = [0.0, 0.0, 0.0]
            return acc, control.brake, control.throttle,vel
        control.throttle = 0.0
        control.brake = 0.0

        self.vehicle.apply_control(control)

        vel = [self.vehicle.get_velocity().x, self.vehicle.get_velocity().y, self.vehicle.get_velocity().z]
        acc = [self.vehicle.get_acceleration().x * 0.2,
               self.vehicle.get_acceleration().y * 0.2,
               self.vehicle.get_acceleration().z * 0.2]

        # cur_loc = self.vehicle.get_transform().location
        # dis = cur_loc.distance(end_wp.transform.location)
        # yaw = self.vehicle.get_transform().rotation.yaw
        # pitch = self.vehicle.get_transform().rotation.pitch
        #
        # vx = t_speed / 3.6 * math.cos(math.radians(yaw))
        # vy = t_speed / 3.6 * math.sin(math.radians(yaw))
        # vz = t_speed / 3.6 * math.sin(math.radians(pitch))
        #
        # acc = [(vx ** 2 - self.v_o[0] ** 2) / (2 * self.dis_o),
        #        (vy ** 2 - self.v_o[1] ** 2) / (2 * self.dis_o),
        #        (vz ** 2 - self.v_o[2] ** 2) / (2 * self.dis_o)]

        # vx_int = 1 if self.vehicle.get_velocity().x >= 0 else -1
        # vy_int = 1 if self.vehicle.get_velocity().y >= 0 else -1
        # vz_int = 1 if self.vehicle.get_velocity().z >= 0 else -1
        #
        # vel = [vx_int * np.sqrt(vx ** 2 - 2 * acc[0] * dis),
        #        vy_int * np.sqrt(vy ** 2 - 2 * acc[1] * dis),
        #        vz_int * np.sqrt(vz ** 2 - 2 * acc[2] * dis)]
        #
        if ego_speed <= target_speed:
            self.v_o = None
            self.dis_o = None
        return acc, control.brake, control.throttle, vel
