# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 下午9:21
# @Author  : Hcyang
# @File    : vehicle_desc.py
# @Desc    : xxx


import sys
import os
import argparse
import pickle
import json

# # import ipdb

# # import ipdb
from .basic_desc_utils import *
from Scenarios.Tinyscenarios.functions import *
from .vehicle_desc import VEHICLE_TYPE


class PedestrianDesc(BasicDesc):
    def __init__(self, world, ego):
        super(PedestrianDesc, self).__init__(world, ego)
        self.wmap = world.get_map()
        self.azimuth_keys = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    def _gen_env_desc(self, short=True, long=False):
        # Desc: 生成环境描述
        # Special: [短描述，长描述，反面描述]
        essential_atomic_keys = self.azimuth_keys
        if short:
            desc = [self.atomic_desc[key][0] for key in essential_atomic_keys]
        elif long:
            desc = [self.atomic_desc[key][1] for key in essential_atomic_keys]
        else:
            raise ValueError("short and long cannot be both False")
        vehicle_desc = [item for item in desc if item.strip()]
        if len(vehicle_desc) > 0:
            return vehicle_desc
        else:
            return []

    def _gen_atomic_desc(self, scenario=''):
        self.ego_loc = self.ego.get_location()
        self.ego_wp = self.wmap.get_waypoint(self.ego_loc)

        azimuth_pedestrians = get_nearby_pedestrians_by_azimuth(self.world, self.ego, normal_radius=20.0, fb_radius=40)

        atomic_desc = {key: [] for key in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']}
        for azimuth_key in atomic_desc:
            pedestrians = azimuth_pedestrians[azimuth_key]
            pedestrians = _process_scenario_pedestrian(self.wmap, azimuth_key, pedestrians, scenario)
            azimuth_pedestrian_desc = _gen_pedestrian_desc(azimuth_key, pedestrians)
            atomic_desc[azimuth_key] = [azimuth_pedestrian_desc, azimuth_pedestrian_desc]

        """
        Examples of atomic_desc:
        {
            'NE': [
                'There is an accident on the right front'，
                'There is an accident on the right front'
            ],
            ...
        }
        """

        return atomic_desc


def _process_scenario_pedestrian(wmap, azimuth_key, pedestrians, scenario):
    scenario = scenario.lower()
    if scenario == '':
        pass
    elif scenario == 'ghosta':
        if azimuth_key in ['NE', 'N']:
            new_pedes = []
            for pedestrian in pedestrians:
                p_loc = pedestrian['instance'].get_location()
                p_wp = wmap.get_waypoint(p_loc)
                if p_wp.transform.location.distance(p_loc) < p_wp.lane_width * 1.3:
                    new_pedes.append(pedestrian)
            pedestrians = new_pedes
    else:
        raise ValueError('unknown scenario')

    return pedestrians


def _gen_pedestrian_desc(azimuth_key, pedestrians):
    azimuth_desc = _gen_azimuth_desc(azimuth_key)
    if len(pedestrians) == 0:
        pedestrian_desc = ''
    else:
        if azimuth_key == 'N':
            azimuth_desc = 'in front of ego'
        elif azimuth_key == 'S':
            azimuth_desc = 'behind ego'
        else:
            azimuth_desc = f'on the {azimuth_desc}'

        if len(pedestrians) == 1:
            pedestrian_desc = f'there is a pedestrian {azimuth_desc}'
        else:
            pedestrian_desc = f'there are pedestrians {azimuth_desc}'
    return pedestrian_desc.capitalize()


def _gen_azimuth_desc(azimuth_key):
    if azimuth_key == 'N':
        azimuth_desc = 'front'
    elif azimuth_key == 'NE':
        azimuth_desc = 'front right'
    elif azimuth_key == 'E':
        azimuth_desc = 'right'
    elif azimuth_key == 'SE':
        azimuth_desc = 'right behind'
    elif azimuth_key == 'S':
        azimuth_desc = 'behind'
    elif azimuth_key == 'SW':
        azimuth_desc = 'left behind'
    elif azimuth_key == 'W':
        azimuth_desc = 'left'
    elif azimuth_key == 'NW':
        azimuth_desc = 'front left'
    else:
        raise ValueError('azimuth_key should be in N, NE, E, SE, S, SW, W, NW')
    return azimuth_desc
