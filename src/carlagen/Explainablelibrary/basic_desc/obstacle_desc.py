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
from .basic_desc_utils import *
from Scenarios.Tinyscenarios.functions import *
from .vehicle_desc import VEHICLE_TYPE


class ObstacleDesc(BasicDesc):
    def __init__(self, world, ego):
        super(ObstacleDesc, self).__init__(world, ego)
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
            # return ['There are no obstacles around']
            return []

    def _gen_atomic_desc(self):
        self.ego_loc = self.ego.get_location()
        self.ego_wp = self.wmap.get_waypoint(self.ego_loc)

        azimuth_obstacles = get_nearby_obstacles_by_azimuth(self.world, self.ego, normal_radius=20.0, fb_radius=40)

        atomic_desc = {key: [] for key in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']}
        for azimuth_key in atomic_desc:
            obstacles = azimuth_obstacles[azimuth_key]
            azimuth_obstacle_desc = _gen_obstacle_desc(azimuth_key, obstacles)
            atomic_desc[azimuth_key] = [azimuth_obstacle_desc, azimuth_obstacle_desc]

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


def _gen_obstacle_desc(azimuth_key, obstacles):
    azimuth_desc = _gen_azimuth_desc(azimuth_key)
    if len(obstacles) == 0:
        # obstacle_desc = f'no obstacles on the {azimuth_desc}'
        obstacle_desc = ''
    else:
        assert len(obstacles) == 1
        obstacle_type_desc = _gen_obstacle_type_desc(obstacles[0]['instance'])
        if obstacle_type_desc.startswith('o') or obstacle_type_desc.startswith('a'):
            obstacle_desc = f'there is an {obstacle_type_desc} on the {azimuth_desc}'
        else:
            obstacle_desc = f'there is a {obstacle_type_desc} on the {azimuth_desc}'
    return obstacle_desc.capitalize()


def _gen_obstacle_type_desc(actor):
    actor_type_id = actor.type_id.lower()
    if 'vehicle' in actor_type_id:
        type_desc = 'accident'
    elif 'static' in actor_type_id:
        # Special: 暂时不区分box和obstacle
        # if 'static.prop.box' in actor_type_id:
        #     type_desc = 'box'
        # else:
        type_desc = 'obstacle'
    else:
        raise ValueError('actor_type_id should be in vehicle, static')

    return type_desc


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
