# -*- coding: utf-8 -*-
# @Time    : 2024/3/8 下午1:43
# @Author  : Hcyang
# @File    : Mainexecution.py
# @Desc    : TODO:
import time

import carla
# import ipdb

from .Basicexecution import BasicExecution
from Initialization.Standardimport.scenariomanager.carla_data_provider import CarlaDataProvider
from Initialization.Standardimport.scenariomanager.timer import GameTime
from Logger import *
from time import sleep
from Scenarios.Tinyscenarios import *


class MainExecution(BasicExecution):
    def __init__(self, args, main_initialization):
        self.args = args
        self.main_initialization = main_initialization

        self.ego = CarlaDataProvider.get_ego() 
        self.spec = None
        self.spec_transform = None
        self._init_view_follower()

        self.world = CarlaDataProvider.get_world()
        self.start_save_count = 0

    def _init_view_follower(self):
        special(f'>> 已开启视角跟随')
        self.spec = find_spectator(CarlaDataProvider.get_world())
        self.spec_transform = self.spec.get_transform()

    def run(self):
        scenario = self.main_initialization.scenario_manager.scenario
        behavior = self.main_initialization.scenario_manager.behavior
        num_frame = self.main_initialization.num_frame

        tick_num = 0

        while tick_num < num_frame:
            # start_time = time.time()
            self.view_follow(tick_num)
            CarlaDataProvider.on_carla_tick()
            timestamp = self.world.get_snapshot().timestamp
            GameTime.on_carla_tick(timestamp)
            behavior.tick_once()
            tick_num += 1
            if behavior.status.name == 'SUCCESS':
                success(f'场景已通过')
                break
            elif behavior.status.name == 'FAILURE':
                error(f'场景未通过')
                raise NotImplementedError

            frame_data = scenario.tick_autopilot()
            self.main_initialization.run_call(True, frame_data)
            self.main_initialization.world_tick()
        
        self.main_initialization.done_call()

    def view_follow(self, tick_num):
        ego_loc = self.ego.get_location()
        if tick_num == 0 or ego_loc.distance(self.spec_transform.location) >= 30:
            ego_forward_vector = self.ego.get_transform().get_forward_vector()
            ego_up_vector = self.ego.get_transform().get_up_vector()
            visual_loc = self.ego.get_location() + ego_forward_vector * 5 + ego_up_vector * 60
            ego_rot = self.ego.get_transform().rotation
            visual_rot = carla.Rotation(yaw=ego_rot.yaw, pitch=-90)
            self.spec_transform = carla.Transform(visual_loc, visual_rot)
            self.spec.set_transform(self.spec_transform)

def find_spectator(world):
    spec = None
    for item in world.get_actors():
        if item.type_id == 'spectator':
            spec = item
            break

    if spec is None:
        error('look_actor: can not find spectator')
        return None
    return spec
