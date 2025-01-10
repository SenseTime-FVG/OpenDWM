# -*- coding: utf-8 -*-
# @Time    : 2023/11/12 下午6:45
# @Author  : Hcyang
# @File    : carla_desc.py
# @Desc    : xxx


import sys
import os
import argparse
import pickle
import json
from .trafficlight_desc import TrafficlightDesc
from .vehicle_desc import VehicleDesc
from .weather_desc import WeatherDesc
from .obstacle_desc import ObstacleDesc
from .pedestrian_desc import PedestrianDesc


class CarlaDesc(object):
    def __init__(self, world, ego):
        self.world = world
        self.ego = ego

        self.vehicle_desc = VehicleDesc(world, ego)
        self.weather_desc = WeatherDesc(world, ego)
        self.trafficlight_desc = TrafficlightDesc(world, ego)
        self.obstacle_desc = ObstacleDesc(world, ego)
        self.pedestrian_desc = PedestrianDesc(world, ego)

        self.last_env_desc = ''
        self.last_reason = None
        self.freeze_sign = False
        self.last_freeze_sign = False

    def freeze(self):
        # Desc: 冻结当前环境描述和决策原因
        if not self.freeze_sign:
            print(f'+++++++++++ Freeze')
            self.freeze_sign = True

    def unfreeze(self):
        if self.freeze_sign:
            print(f'========== Unfreeze')
            self.freeze_sign = False
            self.last_freeze_sign = False

    def get_env_description(self, scenario=''):
        if self.freeze_sign:
            if self.last_freeze_sign:
                return self.last_env_desc

        env_desc = ''

        weather = self.weather_desc.get_env_desc(short=False, long=True)
        if len(weather) > 0:
            env_desc += '. '.join(weather) + '. '

        traffic_light = self.trafficlight_desc.get_env_desc(short=False, long=True)
        if len(traffic_light) > 0:
            env_desc += '. '.join(traffic_light) + '. '

        obstacle = self.obstacle_desc.get_env_desc(short=False, long=True)
        if len(obstacle) > 0:
            env_desc += '. '.join(obstacle) + '. '

        pedestrian = self.pedestrian_desc.get_env_desc(short=False, long=True, scenario=scenario)
        if len(pedestrian) > 0:
            env_desc += '. '.join(pedestrian) + '. '

        vehicle = self.vehicle_desc.get_env_desc(short=True, long=False)
        if len(vehicle) > 0:
            env_desc += '. '.join(vehicle) + '. '

        self.last_env_desc = env_desc.strip()

        return self.last_env_desc

    def get_decision_reason(self, decision, debug=False):
        if self.freeze_sign:
            if self.last_freeze_sign:
                return self.last_reason
            else:
                self.last_freeze_sign = True

        path_decision = decision['path'][0]
        path_reason_keys = decision['path'][1]

        path_reasons = []
        for reason_key in ['weather', 'traffic_light', 'vehicle']:
            if reason_key.startswith('weather'):
                path_reasons.extend(self.weather_desc.get_path_reason(path_decision))
            elif reason_key.startswith('traffic_light'):
                path_reasons.extend(self.trafficlight_desc.get_path_reason(path_decision))
            elif reason_key.startswith('vehicle'):
                path_reasons.extend(self.vehicle_desc.get_path_reason(path_decision, short=False, long=True))
            else:
                raise ValueError("Unknown reason key: {}".format(reason_key))

        if len(path_reasons) > 0:
            path_reason = f'Because ' + ' and '.join(path_reasons)
        else:
            path_reason = f''

        speed_decision = decision['speed'][0]
        speed_reason_keys = decision['speed'][1]

        speed_reasons = []
        for reason_key in ['weather', 'traffic_light', 'vehicle']:
            if reason_key.startswith('weather'):
                speed_reasons.extend(self.weather_desc.get_speed_reason(speed_decision))
            elif reason_key.startswith('traffic_light'):
                speed_reasons.extend(self.trafficlight_desc.get_speed_reason(speed_decision))
            elif reason_key.startswith('vehicle'):
                speed_reasons.extend(self.vehicle_desc.get_speed_reason(speed_decision, short=False, long=True))
            else:
                raise ValueError("Unknown reason key: {}".format(reason_key))

        if len(speed_reasons) > 0:
            speed_reason = f'Because ' + ' and '.join(speed_reasons)
        else:
            speed_reason = f''

        self.last_reason = (path_reason, speed_reason)

        return self.last_reason

