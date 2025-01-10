# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 下午4:59
# @Author  : Hcyang
# @File    : Main_initializer.py
# @Desc    : 初始化所有模块

import os
import sys
import json
# import ipdb
import carla
import json
import pickle
import argparse

# from tqdm import tqdm
from time import sleep

from .Basic_initializer import BasicInitializer
from .World_initializer import WorldInitializer
from .Ego_initializer import EgoInitializer
from .Scenario_initializer import ScenarioInitializer
from .Sensor_initializer import SensorInitializer
from .Weather_initializer import WeatherInitializer
from Datasave.Autopilot_saver import AutopilotSaver
from Logger import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Datasave.Saveutils.save_static import save_static

def load_config_from_json(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)

    return config


class MainInitializer(BasicInitializer):
    def __init__(self, args, scenario_config):
        self.args = args
        self.scenario_config = scenario_config
        self.simulator_config = load_config_from_json(args.simulator_config)
        self.sensors_config = load_config_from_json(args.sensor_config)
        self.saver_config = load_config_from_json(args.saver_config)

        self.client = None
        self.traffic_manager = None

        self.world_initializer = None
        self.world = None

        self.ego_initializer = None
        self.ego = None
        self.saver = None

        self.sensor_initializer = None

        self.weather_initializer = None

        self.scenario_manager = None

        self.num_frame = None

    def run(self):
        retry = 3
        while retry > 0:
            try:
                self.client = carla.Client(self.args.ip, self.args.port)
                self.client.set_timeout(20.0)
                self.traffic_manager = self.client.get_trafficmanager(self.args.traffic_port)
                success(f'>> carla server连接成功: {self.args.ip}:{self.args.port} trafficManager: {self.args.traffic_port}')
                break
            except Exception as e:
                retry -= 1
                warning(f'>> carla server连接失败: {self.args.ip}:{self.args.port} trafficManager: {self.args.traffic_port} 重试中...')

        self.world_initializer = WorldInitializer(self.client, self.traffic_manager, self.scenario_config, self.simulator_config["world"])
        self.world = self.world_initializer.run()
        self.num_frame = self.simulator_config["num_frame"]
        save_static(self.world, self.scenario_config['town'])
        message(f'>> world加载成功: {self.scenario_config["town"]}')

        trigger_point = self.scenario_config['trigger_point']
        trigger_wp = self.world.get_map().get_waypoint(
            location=carla.Location(
                x=float(trigger_point['x']),
                y=float(trigger_point['y']),
                z=float(trigger_point['z']),
            )
        )

        self.ego_initializer = EgoInitializer(self.world, self.simulator_config["ego"])
        self.ego = self.ego_initializer.run(trigger_wp)
        self.traffic_manager.set_global_distance_to_leading_vehicle(7)
        message(f'>> 自车生成成功: {trigger_wp.transform}')
        sleep(0.5)

        self.weather_initializer = WeatherInitializer(self.world, self.ego, self.simulator_config["weather"])
        self.weather_initializer.run()
        # self.weather_initializer.set_daylight()

        self.set_sync_mode()
        message('>> 同步模式已设置')
        # self.saver = AutopilotSaver(self.scenario_config, debug=self.args.debug, aug=self.args.aug)#如果加相机扰动的话 
        self.saver = AutopilotSaver(self.scenario_config, self.sensors_config, self.saver_config, debug=self.args.debug) 
        message(f'>> 数据保存器初始化成功')
        message(f'>> 数据保存位置：{self.saver.get_save_path()}')
        self.world_tick()

        self.scenario_manager = ScenarioInitializer(self.args.scenario, self.client, self.world, self.traffic_manager, self.ego, self.scenario_config, self.args.traffic_port)
        self.scenario_manager.run(trigger_wp.transform)
        message(f'>> 场景加载成功: {self.args.scenario}')

        self.world_tick()
        self.sensor_initializer = SensorInitializer(self.ego, self.saver)
        self.sensor_initializer.run()
        message('>> 自车传感器生成成功，监听中...')
        self.world_tick()

    def set_sync_mode(self):
        self.world_initializer.set_sync_mode()

    def set_async_mode(self):
        self.world_initializer.set_async_mode()

    def done_call(self):        
        self.sensor_initializer.stop_all()

    def run_call(self, data_save_sign, frame_data):
        self.saver(data_save_sign=data_save_sign, frame_data=frame_data, sensors=self.sensor_initializer.sensors_list)

    def world_tick(self):
        if self.world_initializer is not None:
            self.world_initializer.world_tick()
        else:
            error('>> 未初始化world initializer')
