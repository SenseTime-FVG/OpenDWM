# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 下午5:03
# @Author  : Hcyang
# @File    : World_initializer.py
# @Desc    : TODO:

import os
import sys
import json
# import ipdb
import carla
import pickle
import argparse

from time import sleep

# from tqdm import tqdm
from .Basic_initializer import BasicInitializer
from Logger import *


class WorldInitializer(BasicInitializer):
    def __init__(self, client, traffic_manager, scenario_config, world_config):
        self.client = client
        self.traffic_manager = traffic_manager
        self.world_config = world_config
        self.scenario_config = scenario_config
        self.world = None

    def run(self, *args, **kwargs):
        town = self.scenario_config['town']
        world = self.load_word(town, force=True)
        for layer in self.world_config["unload_map_layers"]:
            world.unload_map_layer(getattr(carla.MapLayer, layer))

        self.world = world
        sleep(3)
        return world

    def load_word(self, town_name, force=False):
        print(town_name)
        old_world = self.client.get_world()

        if old_world is not None and not force:
            old_map_name = old_world.get_map().name
            if '/' in old_map_name:
                old_map_name = old_map_name.split('/')[-1]
            if old_map_name == town_name:
                print(f'use old world')
                self.world = old_world
            else:
                self.world = self.client.load_world(town_name)
        else:
            self.world = self.client.load_world(town_name)
        self._clean()
        return self.world

    def _clean(self):
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()

    def get_word(self):
        return self.world

    def set_sync_mode(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.world_config["delta_s"]  
        self.traffic_manager.set_synchronous_mode(True)
        self.world.apply_settings(settings)

    def set_async_mode(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(False)

    def world_tick(self):
        if self.world is not None:
            self.world.tick()
        else:
            error('world is None, can not tick')
