# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 下午8:27
# @Author  : Hcyang
# @File    : trafficlight_desc.py
# @Desc    : xxx


import sys
import os
import argparse
import pickle
import json
# # import ipdb
from .basic_desc_utils import *
from Scenarios.Tinyscenarios.functions import *


class TrafficlightDesc(BasicDesc):
    def __init__(self, world, ego):
        super(TrafficlightDesc, self).__init__(world, ego)
        self.wmap = world.get_map()
        self.all_traffic_light = [item for item in world.get_actors().filter('traffic.traffic_light')]

    def _gen_env_desc(self, short=True, long=False):
        # Desc: 生成环境描述
        # Special: [短描述，长描述，反面描述]
        essential_atomic_keys = ['traffic_light']
        if short:
            desc = [self.atomic_desc[key][0] for key in essential_atomic_keys]
        elif long:
            desc = [self.atomic_desc[key][1] for key in essential_atomic_keys]
        else:
            raise ValueError("short and long cannot be both False")
        return [item.capitalize() for item in desc if item.strip()]

    def _gen_atomic_desc(self):
        """
        Desc: 生成交通信号灯的描述和原因
        """
        front_traffic_light, distance = get_vehicle_front_trafficlight(self.all_traffic_light, self.ego, self.wmap, distance=30)
        if front_traffic_light is None:
            desc = ["", ""]
        else:
            light_state = front_traffic_light.state
            if light_state == carla.TrafficLightState.Green:
                # desc = ["green light", "the traffic light is green"]
                desc = ["", ""]
            else:
                if light_state == carla.TrafficLightState.Red:
                    desc = ["red light", "the traffic light is red"]
                else:
                    desc = ["yellow light", "the traffic light is yellow"]

                if distance < 8:
                    self.add_speed_reason(STOP, 'traffic_light')
                else:
                    self.add_speed_reason(DEC, 'traffic_light')

        return {'traffic_light': desc}

