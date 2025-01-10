from __future__ import print_function

import random

import carla
# import ipdb
import py_trees
import operator
import numpy as np
import math
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, \
    ScenarioTimeoutTest
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance,
                                                                                                     StandStill,
                                                                                                     InTriggerDistanceToVehicle,
                                                                                                     TimeoutRaiseException,
                                                                                                     InTriggerDistanceToNextIntersection)
from Initialization.Standardimport.basic_scenario import BasicScenario

from Explainablelibrary.explainable_utils import *
from Explainablelibrary.basic_desc import *
from Scenarios.Tinyscenarios import *
from Logger import *
from Scenarios.condition_utils import *


class OpenDoor01Condition(AtomicCondition):
    def __init__(self, debug=False):
        self.scenario_name = 'OpenDoor01'
        self.min_distance_to_junction = 120

        super(OpenDoor01Condition, self).__init__(debug)

    def check_condition_by_waypoint(self, wp):
        lane_info = get_lane_info(wp)
        if lane_info is None:
            self.dprint(f'条件1：获取车道信息失败')
            return False

        if lane_info.r2l > 1:
            self.dprint(f'条件2：不位于最右侧车道')
            return False

        if lane_info.num <= 1:
            self.dprint(f'条件3：车道数小于2')
            return False

        if distance_to_next_junction(wp) < self.min_distance_to_junction:
            self.dprint(
                f'条件4：到前方最近路口的距离为{distance_to_next_junction(wp)}，不处于限制区间[{self.min_distance_to_junction}, inf]')
            return False

        return True


class OpenDoor01(BasicScenario):
    """
    Desc: 正在行驶的车道及右侧车道有违规/正常停车，道路为双向车道以上，自车向左变道绕行，变道后不变回原车道
    Special: 补充TODO的数据
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180, uniad=False, interfuser=False, assign_distribution=None):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout
        self.ego = ego_vehicles[0]
        self.starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self.distance_to_junction = distance_to_next_junction(self.starting_wp)
        self.actor_desc = []
        self.traffic_manager = None

        # For: tick autopilot
        self.front_car = None
        self.front_car_wp = None
        self.front_car_loc = None
        self.initialized = False

        self.data_tags = {'door': 'unknown'}
        self.parking_st = None

        self.drive_plan_func = self.sample_drive_plan()
        self.carla_vars = {
            'world': world,
            'blueprint_library': world.get_blueprint_library(),
            'map': self._map,
        }
        self.stage_values = {}

        self.ego_last_wp = None
        self.ego_drive_distance = 0
        self.start_save_sign = False
        self.navi_command = random.choice([0, 2])  # 0:right, 2: straight
        self.start_save_speed_seed = random.random()

        super().__init__("OpenDoor01", ego_vehicles, config, world, randomize, debug_mode,
                         criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser,
                         assign_distribution=assign_distribution)

    def get_data_tags(self):
        return self.data_tags

    def is_success(self):
        # Desc: 非法行为会抛出异常
        return True

    def sample_drive_plan(self):
        distance_to_junc = distance_to_next_junction(self.starting_wp)
        if distance_to_junc > 150:
            move_distance = random.randint(25, 50)
            self.parking_st = move_waypoint_forward(self.starting_wp, move_distance)
            self.data_tags['scenario_length'] = move_distance
            return self.drive_planA1
        else:
            move_distance = random.randint(10, 25)
            self.parking_st = move_waypoint_forward(self.starting_wp, move_distance)
            self.data_tags['scenario_length'] = move_distance
            return self.drive_planB

    def drive_planA1(self):
        # Desc: 驾驶方案A
        # Desc: 一开始高速行驶（40-50之间），距离前方施工区域30米时减速，减速到15之间，距离到10m左右时向左变道，变道后变回原车道，变完道加速到30-40之间，而后恢复到40-50
        # Special: 由于变道回去，导航指令为右转或直行
        ego = self.ego_vehicles[0]
        ego_loc = ego.get_location()
        ego_wp = self._map.get_waypoint(ego_loc)
        frame_data = {'navi_command': self.navi_command}

        if self.initialized is False:
            init_speed = random.randint(40, 50)
            self.stage_values.update({
                'init_speed': init_speed,
                'stage': 1,
                'decrease_start_point': {
                    'distance': random.randint(35, 55),
                    'speed': init_speed
                },
                'decrease_end_point': {
                    'distance': random.randint(28, 35),
                    'speed': 15
                },
                'lane_change_sign': False,
                'left_lane': self.front_car_wp.get_left_lane(),
                'middle_area_speed': random.randint(18, 25),
                # 使用sigmoid函数变道
                'lane_change_start_point': {
                    'distance': random.randint(22, 28),
                    'distance2': random.randint(12, 16),
                },
                'lane_change_end_point': {
                    'distance': random.randint(7, 10),
                    'distance2': random.randint(20, 24),
                },
                'lane_change_curvature': 0.01,
                'lane_change_sigmoid': {
                    'a': None,
                    'a2': None,
                    'b': float(ego_wp.lane_width),
                }
            })
            self.stage_values['lane_change_sigmoid']['a'] = (-1 * math.log(
                (self.stage_values['lane_change_sigmoid']['b'] / self.stage_values['lane_change_curvature']) - 1)) / ((
                          self.stage_values['lane_change_start_point']['distance'] -
                          self.stage_values['lane_change_end_point']['distance']) / 2.0 * -1)
            self.stage_values['lane_change_sigmoid']['a2'] = (-1 * math.log(
                            (self.stage_values['lane_change_sigmoid']['b'] / self.stage_values['lane_change_curvature']) - 1)) / ((
                                      self.stage_values['lane_change_end_point']['distance2'] -
                                      self.stage_values['lane_change_start_point']['distance2']) / 2.0 * -1)

            self._tf_set_ego_route([self.navi_command])
            self._tf_set_ego_speed(self.stage_values['init_speed'])
            self._tf_disable_ego_auto_lane_change()
            self._set_ego_autopilot()
            self.initialized = True

            # 可控变量
            self.data_tags['init_speed'] = init_speed
            self.data_tags['decrease_length'] = self.stage_values['decrease_start_point']['distance'] - \
                                                self.stage_values['decrease_end_point']['distance']
            self.data_tags['lane_change_length'] = self.stage_values['lane_change_start_point']['distance'] - \
                                                   self.stage_values['lane_change_end_point']['distance']

        if not self.start_save_sign:
            ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)
            if self.start_save_speed_seed < 0.5:
                if ego_speed > self.stage_values['init_speed'] - random.randint(2, 7):
                    self.start_save_sign = True
                    success(f'开始保存数据，自车速度：{round(ego_speed, 2)}km/h')
            else:
                if ego_speed > self.stage_values['init_speed'] - 2:
                    self.start_save_sign = True
                    success(f'开始保存数据，自车速度：{round(ego_speed, 2)}km/h')

        # 更新ego_drive_distance
        if self.ego_last_wp is not None:
            self.ego_drive_distance += ego_wp.transform.location.distance(self.ego_last_wp.transform.location)
        self.ego_last_wp = ego_wp

        if self.stage_values['stage'] == 1:
            # 还未变道
            dis2parking_start = distance_between_waypoints(ego_wp, self.front_car_wp)
            # print(dis2cone_start)

            if dis2parking_start > self.stage_values['decrease_start_point']['distance']:
                self._tf_set_ego_speed(self.stage_values['init_speed'])
            elif self.stage_values['decrease_start_point']['distance'] >= dis2parking_start > \
                    self.stage_values['decrease_end_point']['distance']:
                # 线性减速
                p1 = (self.stage_values['decrease_start_point']['distance'],
                      self.stage_values['decrease_start_point']['speed'])
                p2 = (
                    self.stage_values['decrease_end_point']['distance'],
                    self.stage_values['decrease_end_point']['speed'])
                target_speed = calculate_y_for_given_x(dis2parking_start, p1, p2)
                self._tf_set_ego_speed(target_speed)
            if self.stage_values['lane_change_start_point']['distance'] >= dis2parking_start > \
                    self.stage_values['lane_change_end_point']['distance']:
                # sigmoid变道
                sigmoid_x = self.stage_values['lane_change_start_point']['distance'] - dis2parking_start - (
                        self.stage_values['lane_change_start_point']['distance'] -
                        self.stage_values['lane_change_end_point']['distance']) / 2.0
                ego_offset = self.stage_values['lane_change_sigmoid']['b'] / (
                        1 + math.exp(-self.stage_values['lane_change_sigmoid']['a'] * sigmoid_x))
                self._tf_set_ego_offset(-ego_offset)
                self._tf_set_ego_force_go(100)
            elif self.stage_values['lane_change_end_point']['distance'] >= dis2parking_start:
                self.stage_values['stage'] = 2
                self.ego_drive_distance = 0
                self._tf_set_ego_offset(-self.stage_values['lane_change_sigmoid']['b'])
        elif self.stage_values['stage'] == 2:
            # 已经向左变道，变道回去
            dis2parking_start = distance_between_waypoints(ego_wp, self.front_car_wp)
            if self.stage_values['lane_change_start_point']['distance2'] <= dis2parking_start < \
                    self.stage_values['lane_change_end_point']['distance2']:
                sigmoid_x = self.stage_values['lane_change_start_point']['distance2'] - dis2parking_start - (
                        self.stage_values['lane_change_start_point']['distance2'] -
                        self.stage_values['lane_change_end_point']['distance2']) / 2.0
                ego_offset = self.stage_values['lane_change_sigmoid']['b'] / (
                        1 + math.exp(self.stage_values['lane_change_sigmoid']['a2'] * sigmoid_x))
                self._tf_set_ego_offset(ego_offset-self.stage_values['lane_change_sigmoid']['b'])
                self._tf_set_ego_force_go(100)

            elif self.stage_values['lane_change_end_point']['distance2'] <= dis2parking_start:
                self.stage_values['stage'] = 3
                self._tf_set_ego_offset(0)


        elif self.stage_values['stage'] == 3:
            forward_vector = self.stage_values['left_lane'].transform.get_forward_vector()
            if position_with_respect_to_vector_2d(ego_loc, forward_vector) == 'back':
                self._tf_set_ego_speed(self.stage_values['middle_area_speed'])
            else:
                self._tf_set_ego_speed(self.stage_values['init_speed'])

        return self.start_save_sign, frame_data

    def drive_planB(self):
        # Desc: 驾驶方案B
        # Desc: 一开始低速行驶（20-30之间），然后减速到15，距离前方施工区域15米时向左变道，变道后变回原车道，变完道保持车速30-40
        # Special: 由于变道回去，导航指令为右转或直行
        ego = self.ego_vehicles[0]
        ego_loc = ego.get_location()
        ego_wp = self._map.get_waypoint(ego_loc)
        frame_data = {'navi_command': self.navi_command}

        if self.initialized is False:
            init_speed = random.randint(20, 30)
            self.stage_values.update({
                'init_speed': init_speed,
                'stage': 1,
                'decrease_start_point': {
                    'distance': random.randint(45, 50),
                    'speed': init_speed
                },
                'decrease_end_point': {
                    'distance': random.randint(20, 35),
                    'speed': 15
                },
                'lane_change_sign': False,
                'left_lane': self.front_car_wp.get_left_lane(),
                'middle_area_speed': random.randint(15, 20),
                # 使用sigmoid函数变道
                'lane_change_start_point': {
                    'distance': random.randint(18, 20),
                    'distance2': random.randint(10, 12),
                },
                'lane_change_end_point': {
                    'distance': random.randint(7, 9),
                    'distance2': random.randint(20, 24),
                },
                'lane_change_curvature': 0.05,
                'lane_change_sigmoid': {
                    'a': None,
                    'a2': None,
                    'b': float(ego_wp.lane_width),
                }
            })
            self.stage_values['lane_change_sigmoid']['a'] = (-1 * math.log(
                (self.stage_values['lane_change_sigmoid']['b'] / self.stage_values['lane_change_curvature']) - 1)) / ((
                          self.stage_values['lane_change_start_point']['distance'] -
                          self.stage_values['lane_change_end_point']['distance']) / 2.0 * -1)
            self.stage_values['lane_change_sigmoid']['a2'] = (-1 * math.log(
                (self.stage_values['lane_change_sigmoid']['b'] /self.stage_values['lane_change_curvature']) - 1)) / ((
                          self.stage_values['lane_change_end_point']['distance2'] -
                          self.stage_values['lane_change_start_point']['distance2']) / 2.0 * -1)


            self._tf_set_ego_route([self.navi_command])
            self._tf_set_ego_speed(self.stage_values['init_speed'])
            self._tf_disable_ego_auto_lane_change()
            self._set_ego_autopilot()
            self.initialized = True

            # 可控变量
            self.data_tags['init_speed'] = init_speed
            self.data_tags['decrease_length'] = self.stage_values['decrease_start_point']['distance'] - \
                                                self.stage_values['decrease_end_point']['distance']
            self.data_tags['lane_change_length'] = self.stage_values['lane_change_start_point']['distance'] - \
                                                   self.stage_values['lane_change_end_point']['distance']
            self.data_tags['lane_change_length2'] = self.stage_values['lane_change_start_point']['distance2'] - \
                                                   self.stage_values['lane_change_end_point']['distance2']
            self.data_tags['middle_area_speed'] = self.stage_values['middle_area_speed']

        if not self.start_save_sign:
            ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)
            if self.start_save_speed_seed < 0.5:
                if ego_speed > self.stage_values['init_speed'] - random.randint(2, 7):
                    self.start_save_sign = True
                    success(f'开始保存数据，自车速度：{round(ego_speed, 2)}km/h')
            else:
                if ego_speed > self.stage_values['init_speed'] - 2:
                    self.start_save_sign = True
                    success(f'开始保存数据，自车速度：{round(ego_speed, 2)}km/h')

        # 更新ego_drive_distance
        if self.ego_last_wp is not None:
            self.ego_drive_distance += ego_wp.transform.location.distance(self.ego_last_wp.transform.location)
        self.ego_last_wp = ego_wp

        if self.stage_values['stage'] == 1:
            # 还未变道
            dis2parking_start = distance_between_waypoints(ego_wp, self.front_car_wp)

            if dis2parking_start > self.stage_values['decrease_start_point']['distance']:
                self._tf_set_ego_speed(self.stage_values['init_speed'])
            elif self.stage_values['decrease_start_point']['distance'] >= dis2parking_start > \
                    self.stage_values['decrease_end_point']['distance']:
                # 线性减速
                p1 = (self.stage_values['decrease_start_point']['distance'],
                      self.stage_values['decrease_start_point']['speed'])
                p2 = (
                    self.stage_values['decrease_end_point']['distance'],
                    self.stage_values['decrease_end_point']['speed'])
                target_speed = calculate_y_for_given_x(dis2parking_start, p1, p2)
                self._tf_set_ego_speed(target_speed)

            if self.stage_values['lane_change_start_point']['distance'] >= dis2parking_start > \
                    self.stage_values['lane_change_end_point']['distance']:
                # sigmoid变道
                sigmoid_x = self.stage_values['lane_change_start_point']['distance'] - dis2parking_start - (
                        self.stage_values['lane_change_start_point']['distance'] -
                        self.stage_values['lane_change_end_point']['distance']) / 2.0
                ego_offset = self.stage_values['lane_change_sigmoid']['b'] / (
                        1 + math.exp(-self.stage_values['lane_change_sigmoid']['a']* sigmoid_x))
                self._tf_set_ego_offset(-ego_offset)
                self._tf_set_ego_force_go(100)
            elif self.stage_values['lane_change_end_point']['distance'] >= dis2parking_start:
                self.stage_values['stage'] = 2
                self.ego_drive_distance = 0
                self._tf_set_ego_offset(-self.stage_values['lane_change_sigmoid']['b'])
        elif self.stage_values['stage'] == 2:
            # 已经向左变道，变道回去
            dis2parking_start = distance_between_waypoints(ego_wp, self.front_car_wp)
            if self.stage_values['lane_change_start_point']['distance2'] <= dis2parking_start < \
                    self.stage_values['lane_change_end_point']['distance2']:
                sigmoid_x = self.stage_values['lane_change_start_point']['distance2'] - dis2parking_start - (
                        self.stage_values['lane_change_start_point']['distance2'] -
                        self.stage_values['lane_change_end_point']['distance2']) / 2.0
                ego_offset = self.stage_values['lane_change_sigmoid']['b'] / (
                        1 + math.exp(self.stage_values['lane_change_sigmoid']['a2'] *sigmoid_x))
                self._tf_set_ego_offset(ego_offset-self.stage_values['lane_change_sigmoid']['b'])
                self._tf_set_ego_force_go(100)

            elif self.stage_values['lane_change_end_point']['distance2'] <= dis2parking_start:
                self.stage_values['stage'] = 3
                self._tf_set_ego_offset(0)


        elif self.stage_values['stage'] == 3:
            forward_vector = self.stage_values['left_lane'].transform.get_forward_vector()
            if position_with_respect_to_vector_2d(ego_loc, forward_vector) == 'back':
                self._tf_set_ego_speed(self.stage_values['middle_area_speed'])
            else:
                self._tf_set_ego_speed(self.stage_values['init_speed'])

        return self.start_save_sign, frame_data

    def interfuser_tick_autopilot(self):
        if self.front_car is None:
            self.front_car = self.other_actors[self.front_index]  #[self.actor_desc.index('front')]
            if 'park' not in self.actor_desc[self.front_index]:
                error(f'场景未通过(无停车辆)')
                raise NotImplementedError
            self.front_car_wp = self._map.get_waypoint(self.front_car.get_location())
            self.front_car_loc = self.front_car.get_location()
        return self.drive_plan_func()

    def _initialize_actors(self, config):
        # Depend: 场景前方生成停放车辆
        lane_info = get_lane_info(self.starting_wp)
        if lane_info['num'] < 2:
            error(f'场景未通过(车道不支持向左变道)')
            raise NotImplementedError
        cfg = None
        default_settings = {
            'params': {
                'tidy_min': 8,  # 纵向停车最小连续长度
                'tidy_max': 20,  # 纵向停车最大连续长度
                'tidy_dis_min': 11,  # 纵向停车最小间距，最小值9
                'tidy_dis_max': 14,  # 纵向停车最大间距
                'total': 80,  # 场景总长度
            },
            'distributions': {
                'tidy_length': 'default',  # 纵向停车连续长度概率
                # 'scene': 'default',  # FL//FL+RL概率 ({0:,1:}) 有的车只有前门，省略为两种场景
                'space': 'default',  # 空场景概率（{0:,1:}）
                'straight_length': 'default',  # 遇到开门的车之前经过了总场景长的百分之多少（{0.1:，0.2:}）
            },
            'probs': {
                'tidy': 0.6,  # 生成纵向车辆概率
            }
        }
        if isinstance(cfg, dict):
            default_settings['params'].update(cfg.get('params', {}))
            default_settings['distributions'].update(cfg.get('distributions', {}))
            default_settings['probs'].update(cfg.get('probs', {}))

        wp = self.parking_st

        self.data_tags['tidy_parking_length'] = 0
        self.data_tags['tidy_dis_min'] = default_settings['params']['tidy_dis_min']
        self.data_tags['tidy_dis_max'] = default_settings['params']['tidy_dis_max']

        if default_settings['distributions']['straight_length'] == 'default':
            door_dis = float(default_settings['params']['total']+self.data_tags['scenario_length']) / 10.0 * random.randint(3, 5)
        else:
            door_dis = do_sample(default_settings['distributions']['straight_length'])
        open_flag = 0
        try_time = 10
        while distance_to_next_junction(wp) >= 50 and self.data_tags['scenario_length'] <= default_settings['params']['total']:
            if self.data_tags['scenario_length'] >= door_dis and open_flag == 0:
                open_flag = 1
            if default_settings['distributions']['space'] == 'default':
                space = do_sample({0: 0.9, 1: 0.1})
            else:
                space = do_sample(default_settings['distributions']['space'])
            if space and open_flag > 1:
                dis = random.randint(2, 5)
                wp = move_waypoint_forward(wp, dis)
                self.data_tags['scenario_length'] += dis
            else:
                if try_time>0:
                    if open_flag & 1:
                        tp = len(self.other_actors)
                        right_parking_open_door_scenario(
                            wp, self.other_actors, self.actor_desc, self.data_tags,
                            scene_cfg={'filters': '+door',
                                       'idp': 1.0, 'length': 10},
                            gen_cfg={'name_prefix': 'parkdoor'}
                        )
                        if len(self.other_actors) != tp:

                            self.front_index = random.randint(tp, len(self.other_actors)-1)
                            wp = move_waypoint_forward(wp, 10)
                            self.data_tags['scenario_length'] += 10
                            self.data_tags['tidy_parking_length'] += 10
                            self.data_tags['door'] = 'Both'

                            self.other_actors[self.front_index].open_door(carla.VehicleDoor.FL)
                            self.other_actors[self.front_index].open_door(carla.VehicleDoor.RL)

                            open_flag = 2
                        else:
                            try_time-=1
                            wp = move_waypoint_forward(wp, 1)
                            self.data_tags['scenario_length'] += 1
                        continue
                else:
                    break
                if default_settings['distributions']['tidy_length'] == 'default':
                    length = random.randint(default_settings['params']['tidy_min'],
                                            default_settings['params']['tidy_max'])
                else:
                    length = do_sample(default_settings['distributions']['tidy_length'])
                right_parking_open_door_scenario(
                    wp, self.other_actors, self.actor_desc, self.data_tags,
                    scene_cfg={'filters': '+car+suv-moto-electric-bicycle-truck-police',
                               'idp': default_settings['probs']['tidy'], 'length': length},
                    gen_cfg={'name_prefix': 'tidypark'}
                )
                wp = move_waypoint_forward(wp, length)
                self.data_tags['scenario_length'] += length
                self.data_tags['tidy_parking_length'] += length


        # 如果离路口太近，或者没生成（虽然前面会一直尝试生成但以防万一），就生成一辆车
        if len(self.other_actors) < 1 or self.data_tags['door'] == 'unknown':
            self.data_tags['door'] = 'Both'
            # 以防撞车的万一
            for actor in self.world.get_actors().filter('*vehicle*'):
                if actor.id == self.ego.id:
                    continue
                actor.destroy()
            dis = distance_to_next_junction(self.parking_st) // 2
            wp = move_waypoint_forward(self.parking_st, dis)
            self.data_tags['scenario_length'] += dis
            # message(f"inplace {self.data_tags['scenario_length']}")
            right_parking_open_door_scenario(
                wp, self.other_actors, self.actor_desc, self.data_tags,
                scene_cfg={'filters': '+door',
                           'idp': 1.0, 'length': 1},
                gen_cfg={'name_prefix': 'doorpark'}
            )
            if len(self.other_actors) == 0:  # but whyyyyyy
                raise NotImplementedError("未生成开门车辆")
            self.front_index = len(self.actor_desc)-1
            self.other_actors[self.front_index].open_door(carla.VehicleDoor.FL)
            self.other_actors[self.front_index].open_door(carla.VehicleDoor.RL)


        # # For: 左侧交通流
        left_traffic_flow_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.5, 'lanes_num': 4, 'skip_num': 1,
                       'forward_num': random.randint(5, 8), 'backward_num': random.randint(0, 20)},
            gen_cfg={'name_prefix': 'left'}
        )
        # For: 对向交通流
        opposite_traffic_flow_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.4, 'forward_num': random.randint(5, 8),
                       'backward_num': random.randint(0, 20)},
            gen_cfg={'name_prefix': 'opposite'}
        )

        self.traffic_manager = CarlaDataProvider.get_trafficmanager()
        for a_index, (actor, actor_desc) in enumerate(zip(self.other_actors, self.actor_desc)):
            if actor.type_id.startswith('vehicle'):
                if 'park' in actor_desc:
                    actor.apply_control(carla.VehicleControl(brake=1.0))
                    continue
                actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
                self.traffic_manager.update_vehicle_lights(actor, True)
                self.traffic_manager.set_desired_speed(actor, random.randint(15, 30))
                if 'left' in actor_desc:
                    self.traffic_manager.auto_lane_change(actor, False)
        # spectator = self.world.get_spectator()
        # spectator.set_transform(self.ego.get_transform())
        # ipdb.set_trace()

    def _create_behavior(self):
        root = py_trees.composites.Parallel(name="OpenDoor01",
                                            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(TimeoutRaiseException(300))
        root.add_child(CollisionTest(self.ego_vehicles[0], terminate_on_failure=True))
        root.add_child(InTriggerDistanceToNextIntersection(self.ego_vehicles[0], 8))
        root.add_child(DriveDistance(self.ego_vehicles[0], self.data_tags['scenario_length'] + 25))
        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        pass
