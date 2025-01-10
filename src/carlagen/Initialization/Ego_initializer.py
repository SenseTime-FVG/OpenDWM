# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 下午5:22
# @Author  : Hcyang
# @File    : Ego_initializer.py
# @Desc    : TODO:

import os
import sys
import json
# import ipdb
import carla
import pickle
import random
import argparse

# from tqdm import tqdm
from .Basic_initializer import BasicInitializer
from Logger import *


class EgoInitializer(BasicInitializer):
    def __init__(self, world, ego_config):
        self.world = world
        self.ego_config = ego_config
        self.blueprint_library = self.world.get_blueprint_library()

    def run(self, spawn_transform):
        vehicle_type = self.ego_config["type"]
        return self.spawn_actor(vehicle_type, spawn_transform)

    def spawn_actor(self, actor_type, spawn_transform, retry=3):
        if isinstance(spawn_transform, carla.Waypoint):
            spawn_transform = spawn_transform.transform

        possible_bps = self.blueprint_library.filter(actor_type)
        if len(possible_bps) == 0:
            error(f'Blueprint {actor_type} not found!')
            return None

        bp = random.choice(possible_bps)
        if self.ego_config["is_hero"]:
            bp.set_attribute('role_name', 'hero')

        _spawn_transform = carla.Transform(spawn_transform.location, spawn_transform.rotation)
        _spawn_transform.location.x = spawn_transform.location.x
        _spawn_transform.location.y = spawn_transform.location.y
        _spawn_transform.location.z = spawn_transform.location.z + 0.5
        actor = self.world.try_spawn_actor(bp, _spawn_transform)

        if actor is None:
            if retry == 0:
                warning(f'Spawn actor {actor_type} failed!')
                return None
            else:
                return self.spawn_actor(actor_type, spawn_transform, retry=retry-1)
        else:
            actor.set_simulate_physics(True)
            success(f'Spawn actor {actor.type_id} success!')
            return actor

