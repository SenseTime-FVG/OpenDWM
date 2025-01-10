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
from .basic_desc_utils import *
from Scenarios.Tinyscenarios.functions import *


VEHICLE_TYPE = {
    'truck': ['vehicle.carlamotors.carlacola', 'vehicle.carlamotors.european_hgv', 'vehicle.tesla.cybertruck'],
    'van': ['vehicle.mercedes.sprinter', 'vehicle.volkswagen.t2', 'vehicle.volkswagen.t2_2021'],
    'bus': ['vehicle.mitsubishi.fusorosa'],
    'motorcycle': ['vehicle.harley-davidson.low_rider', 'vehicle.kawasaki.ninja', 'vehicle.vespa.zx125', 'vehicle.yamaha.yzf'],
    'bike': ['vehicle.bh.crossbike', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets'],
    'fire truck': ['vehicle.carlamotors.firetruck'],
    'ambulance': ['vehicle.ford.ambulance'],
    'police car': ['vehicle.dodge.charger_police', 'vehicle.dodge.charger_police_2020'],
    'taxi': ['vehicle.ford.crown']
}


class VehicleDesc(BasicDesc):
    def __init__(self, world, ego):
        super(VehicleDesc, self).__init__(world, ego)
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
            return ['There are no vehicles around']

    def _gen_atomic_desc(self):
        self.ego_loc = self.ego.get_location()
        self.ego_wp = self.wmap.get_waypoint(self.ego_loc)

        azimuth_vehicles = get_nearby_vehicles_by_azimuth(self.world, self.ego, normal_radius=20.0, fb_radius=40)

        atomic_desc = {key: [] for key in self.azimuth_keys}
        for azimuth_key in atomic_desc:
            # Special: 默认每个方位角的车辆都是按照距离从小到大排列的
            vehicles = azimuth_vehicles[azimuth_key]

            coarse_assist = 0
            fine_descriptions = []
            for vehicle in vehicles:
                # Special: 粗略描述辅助
                coarse_assist += 1

                # Special: 细致描述辅助
                fine_desc = _gen_fine_item_desc(vehicle)
                fine_descriptions.append(fine_desc)

            # Special: 生成粗略描述
            coarse_desc = _gen_coarse_desc(azimuth_key, coarse_assist)
            # Special: 生成细致描述
            fine_desc = _gen_fine_desc(azimuth_key, fine_descriptions)

            atomic_desc[azimuth_key].extend([fine_desc, coarse_desc])

        self._gen_possible_reason()

        # TODO: direction, relative still
        """
        Section: Vehicle Description
        Desc1: 先描述方位角
        Desc2: 描述所有车的详细信息（方向）
        Desc3: 描述距离
        E.g. 
        D1: On the left front, 
        D2: A black truck <v1> and a white ambulance <v2> are in the same direction to us. A red bus <v3> is in the opposite direction to us. TODO: A blue car <v4> is driving laterally to us. A yellow taxi <v5> is static.
        D3: The distance to the black truck <v1> and the red bus <v3> is decreasing. The distance to the white ambulance <v2> is increasing. 
        
        Section: Obstacle Description
        Desc1: 描述障碍物的详细信息, 描述方位角
        Desc2: 描述距离
        E.g. 
        D1: There is a cone <o1> and a barrier <o2> in front of the ego car, and there is a barrier <o3> and a cone <o4> behind it.
        D2: The traffic cone <o1>, the barrier <o2> and the barrier <o3> is closing in on the ego car. The cone <o4> is receding from the ego car.
        
        Section: Pedestrian
        Desc1: 朝向比较独特 
        Desc2: In front of the ego car, there are a few pedestrians ready to cross. There are a few other pedestrians in the sight as well.
        
        右转左侧来车让行场景
        
        Examples of atomic_desc:
        {
            'NE': [
                'on the right front, a black truck is approaching, a red car is leaving, and a white ambulance is approaching'，
                'there are vehicles from the front right'
            ],
            'NW': [
                'On the left front, there are a black truck, a red car. The distance between the ego and them becomes smaller. There is also a white ambulance. The distance between the ego and it becomes larger.'
                'On the left front, there are a black truck, a red car and a white ambulance. The black truck and the white ambulance are slower than me. The red car is faster than me.'
                'On the left front, there are three vehicles. A black truck and a white ambulance are getting closer to me. A red car is moving farther away from me.'
            ]
            ...
        }
        """

        return atomic_desc

    def _gen_possible_reason(self):
        # Desc: 前方/右前方/右侧 有车，向左变道
        self.add_path_reason(LEFT_C, 'N')
        self.add_path_reason(LEFT_C, 'NE')
        self.add_path_reason(LEFT_C, 'E')
        # Desc: 前方/右前方/右侧 有车，向左借道
        self.add_path_reason(LEFT_B, 'N')
        self.add_path_reason(LEFT_B, 'NE')
        self.add_path_reason(LEFT_B, 'E')
        # Desc: 前方/左前方/左侧 有车，向右变道
        self.add_path_reason(RIGHT_C, 'N')
        self.add_path_reason(RIGHT_C, 'NW')
        self.add_path_reason(RIGHT_C, 'W')
        # Desc: 前方/左前方/左侧 有车，向右借道
        self.add_path_reason(RIGHT_B, 'N')
        self.add_path_reason(RIGHT_B, 'NW')
        self.add_path_reason(RIGHT_B, 'W')

        # Desc: 前方有车，减速/停车
        self.add_speed_reason(DEC, 'N')
        self.add_speed_reason(STOP, 'N')


def _gen_coarse_desc(azimuth_key, coarse_assist):
    # Desc: 生成粗略描述，并判断是否影响决策
    azimuth_desc = _gen_azimuth_desc(azimuth_key)
    if coarse_assist == 0:
        # coarse_desc = f'there are no vehicles from the {azimuth_desc}'
        coarse_desc = ''
    else:
        if coarse_assist == 1:
            coarse_desc = f'there is a vehicle from the {azimuth_desc}'
        elif coarse_assist >= 2:
            coarse_desc = f'there are multiple vehicles from the {azimuth_desc}'
        else:
            raise ValueError('coarse_assist should be in 0, 1, 2, inf')
    return coarse_desc.capitalize()


def _gen_fine_desc(azimuth_key, fine_descriptions):
    azimuth_desc = _gen_azimuth_desc(azimuth_key)
    if len(fine_descriptions) == 0:
        # fine_desc = f'there are no vehicles from the {azimuth_desc}'
        fine_desc = ''
    else:
        if azimuth_key == 'N':
            fine_desc = f'in front of ego, '
        elif azimuth_key == 'S':
            fine_desc = f'behind ego, '
        else:
            fine_desc = f'on the {azimuth_desc}, '
        if len(fine_descriptions) > 1:
            fine_descriptions[len(fine_descriptions) - 1] = 'and ' + fine_descriptions[len(fine_descriptions) - 1]
        fine_desc += ', '.join(fine_descriptions)
    return fine_desc.capitalize()


def _gen_fine_item_desc(azimuth_data):
    actor = azimuth_data['instance']
    color_desc = _gen_color_desc(actor)
    car_type_desc = _gen_car_type_desc(actor)
    direction_desc = azimuth_data['direction']
    fine_desc = f'a {color_desc} {car_type_desc} is {direction_desc}'
    return fine_desc


def _gen_color_desc(actor):
    if "color" not in actor.attributes:
        if actor.type_id == 'vehicle.tesla.cybertruck':
            rgb = "0,0,0"
        else:
            rgb = "255,255,255"
    else:
        rgb = actor.attributes['color']
    r = float(rgb.split(',')[0])
    g = float(rgb.split(',')[1])
    b = float(rgb.split(',')[2])
    color_desc = color_cls([r, g, b])
    return color_desc


def _gen_car_type_desc(actor):
    actor_type_id = actor.type_id.lower()
    type_desc = 'car'
    for key, value in VEHICLE_TYPE.items():
        if actor_type_id in value:
            type_desc = key
            break
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
