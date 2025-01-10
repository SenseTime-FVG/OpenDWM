from __future__ import print_function

import random

import carla
import py_trees
import operator
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance, StandStill, InTriggerDistanceToVehicle, TimeoutRaiseException, InTriggerDistanceToNextIntersection)
from Initialization.Standardimport.basic_scenario import BasicScenario

from Scenarios.Tinyscenarios import *
from Logger import *
from time import sleep
from Scenarios.condition_utils import *
from easydict import EasyDict as Edict
import sys
from agents.pid_func import PIDcontroller
# import ipdb

class pidFollowingStopCondition(AtomicCondition):
    def __init__(self, debug=False):

        self.scenario_name = 'pidFollowingStop'

        self.min_distance_to_junction = 50  #

        super(pidFollowingStopCondition, self).__init__(debug)

    def check_condition_by_waypoint(self, wp):
        lane_info = get_lane_info(wp)
        if lane_info is None:
            self.dprint(f'条件1：获取车道信息失败')
            return False

        if distance_to_next_junction(wp) < self.min_distance_to_junction:
            self.dprint(f'条件2：到前方最近路口的距离为{distance_to_next_junction(wp)}，不处于限制区间[{self.min_distance_to_junction}, inf]')
            return False

        return True


class pidFollowingStop(BasicScenario):
    """
    Desc: 场景为跟随前方车辆行驶，前方车辆减速，自车也减速，直至停车
    Special: 用于验证实车是否能够跟随前车减速停车
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180, uniad=False, interfuser=False, assign_distribution=None):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout
        self.starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self.distance_to_junction = distance_to_next_junction(self.starting_wp)
        self.actor_desc = []
        self.traffic_manager = None

        # For: tick autopilot
        self.initialized = False

        # For: Is success
        self.passed_lane_ids = []

        # self.acc_data = []
        self.PID = None

        self.data_tags = {}
        self.following_car_wp = None
        self.following_car = None
        self.drive_plan_func = self.sample_drive_plan()
        self.carla_vars = {
            'world': world,
            'blueprint_library': world.get_blueprint_library(),
            'map': self._map,
        }
        self.ego_last_wp = None
        self.ego_drive_distance = 0
        self.follow_last_wp = None
        self.follow_drive_distance = 0
        self.start_save_sign = False
        starting_wp_lane_info = get_lane_info(self.starting_wp)
        navi_command_candidates = [2]
        if starting_wp_lane_info.r2l <= 1:
            navi_command_candidates.append(0)
        if starting_wp_lane_info.l2r <= 1:
            navi_command_candidates.append(1)
        self.navi_command = random.choice(navi_command_candidates)
        self.start_save_speed_seed = random.random()
        self.more_space = 0


        super().__init__("pidFollowingStop", ego_vehicles, config, world, randomize, debug_mode,
                         criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser, assign_distribution=assign_distribution)

        # PID
        args_lateral_dict = {'K_P': 1.95, 'K_D': 0.2, 'K_I': 0.07, 'dt': 1.0 / 10.0}
        args_long_dict = {'K_P': 0.175, 'K_D': 0.128, 'K_I': 0.18, 'dt': 1.0 / 10.0}
        if self.data_tags['drive_plan'] == 'A':
            self.PID = PIDcontroller(self.ego_vehicles[0], args_lateral=args_lateral_dict,
                                     args_longitudinal=args_long_dict,
                                     acc_lim1=self.default_settings['planA']['lim1'],
                                     acc_lim2=self.default_settings['planA']['lim2'])
        else:
            self.PID = PIDcontroller(self.ego_vehicles[0], args_lateral=args_lateral_dict,
                                     args_longitudinal=args_long_dict,
                                     acc_lim1=self.default_settings['planB']['lim1'],
                                     acc_lim2=self.default_settings['planB']['lim2'])

    def _update_scenario_settings(self):
        planA_ego_init_speed = random.randint(35, 45)
        planA_follow_init_speed = random.randint(planA_ego_init_speed+1, planA_ego_init_speed + 3)
        planA_decrease_start_point_distance = random.randint(50, 100)
        planA_decrease_end_point_distance = random.randint(planA_decrease_start_point_distance + 30, planA_decrease_start_point_distance + 45)

        planB_ego_init_speed = random.randint(15, 25)
        planB_follow_init_speed = random.randint(planB_ego_init_speed, planB_ego_init_speed + 2)
        planB_decrease_start_point_distance = random.randint(40, 50)
        planB_decrease_end_point_distance = random.randint(planB_decrease_start_point_distance + 5, planB_decrease_start_point_distance + 12)

        self.default_settings = {
            'probs': {
                'left_traffic': 0.8,
                'right_traffic': 0.8,
                'opposite_traffic': 0.8,
            },
            'following': {
                'update_light': 0.8,
                'type': {
                    'car': 1.0,
                    'suv': 1.0,
                    'truck': 1.0,
                    'van': 1.0,
                    'bus': 1.0,
                }
            },
            'planA': {
                'ego_init_speed': planA_ego_init_speed,
                'follow_init_speed': planA_follow_init_speed,
                'decrease_start_point': {  # 减速开始点
                    'distance': planA_decrease_start_point_distance,
                    'speed': planA_follow_init_speed
                },
                'decrease_end_point': {  # 减速结束点
                    'distance': planA_decrease_end_point_distance,
                    'speed': 15
                },
                'lim1': planA_ego_init_speed / 10 * 0.75,
                'lim2': planA_ego_init_speed / 10 * 0.75 + 1,
            },
            'planB': {
                'ego_init_speed': planB_ego_init_speed,
                'follow_init_speed': planB_follow_init_speed,
                'decrease_start_point': {  # 减速开始点
                    'distance': planB_decrease_start_point_distance,
                    'speed': planB_follow_init_speed
                },
                'decrease_end_point': {  # 减速结束点
                    'distance': planB_decrease_end_point_distance,
                    'speed': 10
                },
                'lim1': planB_ego_init_speed / 10 * 0.5,
                'lim2': planB_ego_init_speed / 10 * 0.5 + 1,
            },
            'de_count': 0
        }
        self.default_settings = self.update_assign_distribution(self.default_settings)

    def get_data_tags(self):
        return self.data_tags

    def is_success(self):
        # Desc: 非法行为会抛出异常
        return True

    def sample_drive_plan(self):
        distance_to_junc = distance_to_next_junction(self.starting_wp)

        if distance_to_junc > 150:
            move_distance = random.randint(10, 20)
            self.following_car_wp = move_waypoint_forward(self.starting_wp, move_distance)
            self.data_tags['dis2following'] = move_distance
            self.data_tags['drive_plan'] = 'A'
            return self.drive_planA
        else:
            move_distance = random.randint(8, 10)
            self.following_car_wp = move_waypoint_forward(self.starting_wp, move_distance)
            self.data_tags['dis2following'] = move_distance
            self.data_tags['drive_plan'] = 'B'
            return self.drive_planB

    def drive_planA(self):
        # Desc: 驾驶方案A
        # Desc: 加速直到高速行驶，开始保存数据，而后前车开始减速，自车也减速，直至停车
        # Special: 由于不变道回去，导航指令为左转或直行
        ego = self.ego_vehicles[0]
        ego_loc = ego.get_location()
        ego_wp = self._map.get_waypoint(ego_loc)
        follow_loc = self.following_car.get_location()
        follow_wp = self._map.get_waypoint(follow_loc)
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)
        frame_data = {'navi_command': self.navi_command}
        control = ego.get_control()

        acceleration = ego.get_acceleration()

        if self.initialized is False:

            control = self.PID.increase(ego_wp, acceleration, self.default_settings['planA']['ego_init_speed'])
            ego.apply_control(control)
            self.initialized = True

            self.data_tags['init_speed'] = self.default_settings['planA']['ego_init_speed']
            self.data_tags['decrease_length'] = self.default_settings['planA']['decrease_start_point']['distance'] - self.default_settings['planA']['decrease_end_point']['distance']
        # self.acc_data.append([acceleration.x, acceleration.y])
        if not self.start_save_sign:
            if self.start_save_speed_seed < 0.5:
                if ego_speed > self.default_settings['planA']['ego_init_speed'] - random.randint(2, 7):
                    self.start_save_sign = True
                    success(f'开始保存数据，自车速度：{round(ego_speed, 2)}km/h')
            else:
                if ego_speed > self.default_settings['planA']['ego_init_speed'] - 2:
                    self.start_save_sign = True
                    success(f'开始保存数据，自车速度：{round(ego_speed, 2)}km/h')

        # 更新ego_drive_distance
        if self.ego_last_wp is not None:
            self.ego_drive_distance += ego_wp.transform.location.distance(self.ego_last_wp.transform.location)
        else:
            self.ego_last_wp=ego_wp

        # 更新follow_drive_distance
        if self.follow_last_wp is not None:
            self.follow_drive_distance += follow_wp.transform.location.distance(self.follow_last_wp.transform.location)
        else:
            self.follow_last_wp=follow_wp
        if self.ego_drive_distance > 5 and 0.1<abs(ego_speed) < 0.8:
            self._tf_set_actor_speed(0, self.following_car)
            # self._tf_set_ego_speed(0)
            control.brake=1.0
            # print('done')
        else:
            if self.follow_drive_distance < self.default_settings['planA']['decrease_start_point']['distance']:
                self._tf_set_actor_speed(self.default_settings['planA']['follow_init_speed'], self.following_car)
            elif self.default_settings['planA']['decrease_start_point']['distance'] +self.more_space<= self.follow_drive_distance <= \
                    self.default_settings['planA']['decrease_end_point']['distance']+self.more_space:
                # 线性减速
                p1 = (self.default_settings['planA']['decrease_start_point']['distance']+self.more_space,
                      self.default_settings['planA']['decrease_start_point']['speed'])
                p2 = (self.default_settings['planA']['decrease_end_point']['distance']+self.more_space,
                      self.default_settings['planA']['decrease_end_point']['speed'])
                target_speed = calculate_y_for_given_x(self.follow_drive_distance, p1, p2)
                self._tf_set_actor_speed(target_speed, self.following_car)
            else:
                self._tf_set_actor_speed(0, self.following_car)

            if self.follow_drive_distance <= \
                    self.default_settings['planA']['decrease_end_point']['distance']-10 \
                     and distance_between_waypoints(self.ego_last_wp, self.follow_last_wp) > 10:
                # 当起步时，throttle的最大值从0.15逐渐增加
                self.PID.max_throttle_control(self.default_settings['de_count'], self.default_settings['planA']['ego_init_speed'])
                control = self.PID.increase(ego_wp, acceleration, self.default_settings['planA']['ego_init_speed'])

            elif self.follow_drive_distance <= self.default_settings['planA']['decrease_end_point']['distance'] \
                 or 10 < distance_between_waypoints(ego_wp, follow_wp) <= 18:  # random.randint(12,18):
                control = self.PID.decrease(ego_wp, move_waypoint_forward(ego_wp, 4), ego_speed)
                control.brake = 0.0
                # control.throttle = 0.0
            elif self.ego_drive_distance > 5 and distance_between_waypoints(ego_wp, follow_wp) < 8:  # random.randint(8,10):
                if control.brake < 0.008:
                    control = self.PID.decrease(ego_wp, follow_wp, 0)
                else:
                    control.brake = 0.0
                    control.throttle = 0.0
                # message('stop')
        ego.apply_control(control)
            # message(f'{control.throttle}, {control.brake}')

        self.default_settings['de_count'] += 1
        self.ego_last_wp = ego_wp
        self.follow_last_wp = follow_wp
        # print(distance_between_waypoints(self.ego_last_wp, self.follow_last_wp))

        if distance_between_waypoints(self.ego_last_wp, self.follow_last_wp) < 12:
            self.default_settings['de_count'] -= 1

        return self.start_save_sign, frame_data

    def drive_planB(self):
        # Desc: 驾驶方案B
        # Desc: 一开始低速行驶（30-40之间），然后减速到15，距离前方施工区域15米时向左变道，变道后不变回原车道，变完道保持车速30-40
        # Special: 由于不变道回去，导航指令为左转或直行
        ego = self.ego_vehicles[0]
        ego_loc = ego.get_location()
        ego_wp = self._map.get_waypoint(ego_loc)
        follow_loc = self.following_car.get_location()
        follow_wp = self._map.get_waypoint(follow_loc)
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)
        frame_data = {'navi_command': self.navi_command}
        control = ego.get_control()
        acceleration = ego.get_acceleration()

        if self.initialized is False:
            control = self.PID.increase(ego_wp, acceleration,
                                        self.default_settings['planB']['ego_init_speed'], distance=1)
            ego.apply_control(control)

            self.initialized = True

            self.data_tags['init_speed'] = self.default_settings['planB']['ego_init_speed']
            self.data_tags['decrease_length'] = self.default_settings['planB']['decrease_start_point']['distance'] - self.default_settings['planB']['decrease_end_point']['distance']

        # self.acc_data.append([acceleration.x, acceleration.y])
        if not self.start_save_sign:
            if self.start_save_speed_seed < 0.5:
                if ego_speed > self.default_settings['planB']['ego_init_speed'] - random.randint(2, 7):
                    self.start_save_sign = True
                    success(f'开始保存数据，自车速度：{round(ego_speed, 2)}km/h')
            else:
                if ego_speed > self.default_settings['planB']['ego_init_speed'] - 2:
                    self.start_save_sign = True
                    success(f'开始保存数据，自车速度：{round(ego_speed, 2)}km/h')

        # 更新ego_drive_distance
        if self.ego_last_wp is not None:
            self.ego_drive_distance += ego_wp.transform.location.distance(self.ego_last_wp.transform.location)
        else:
            self.ego_last_wp = ego_wp

        # 更新follow_drive_distance
        if self.follow_last_wp is not None:
            self.follow_drive_distance += follow_wp.transform.location.distance(self.follow_last_wp.transform.location)
        else:
            self.follow_last_wp = follow_wp

        if self.ego_drive_distance > 5 and 0.1<abs(ego_speed) < 0.8:
            self._tf_set_actor_speed(0, self.following_car)
            # self._tf_set_ego_speed(0)
            control.brake = 1.0
            # print('done')

        else:
            if self.follow_drive_distance < self.default_settings['planB']['decrease_start_point']['distance']:
                self._tf_set_actor_speed(self.default_settings['planB']['follow_init_speed'], self.following_car)
            elif (self.default_settings['planB']['decrease_start_point']['distance']+self.more_space <= self.follow_drive_distance <=
                  self.default_settings['planB']['decrease_end_point']['distance']+self.more_space):
                # 线性减速
                p1 = (self.default_settings['planB']['decrease_start_point']['distance']+self.more_space, self.default_settings['planB']['decrease_start_point']['speed'])
                p2 = (self.default_settings['planB']['decrease_end_point']['distance']+self.more_space, self.default_settings['planB']['decrease_end_point']['speed'])
                target_speed = calculate_y_for_given_x(self.follow_drive_distance, p1, p2)
                self._tf_set_actor_speed(target_speed, self.following_car)
            else:
                self._tf_set_actor_speed(0, self.following_car)

            if self.follow_drive_distance <= \
                    self.default_settings['planB']['decrease_end_point']['distance']-10 \
                     and distance_between_waypoints(self.ego_last_wp, self.follow_last_wp) > 8:
                # 当起步时，throttle的最大值从0.15逐渐增加
                self.PID.max_throttle_control(self.default_settings['de_count'], self.default_settings['planB']['ego_init_speed'])
                control = self.PID.increase(ego_wp, acceleration, self.default_settings['planB']['ego_init_speed'])

            elif self.follow_drive_distance <= self.default_settings['planB']['decrease_end_point']['distance'] \
                 or 10 < distance_between_waypoints(ego_wp, follow_wp) <= 18:  # random.randint(12,18):
                control = self.PID.decrease(ego_wp, move_waypoint_forward(ego_wp, 4), ego_speed)
                control.brake = 0.0
            elif self.ego_drive_distance > 5 and distance_between_waypoints(ego_wp, follow_wp) < 7:  # random.randint(8,10):
                if control.brake< 0.008:
                    control = self.PID.decrease(ego_wp, follow_wp, 0)
                else:
                    control.brake = 0.0
                    control.throttle = 0.0
                # message('stop')
        ego.apply_control(control)

        self.default_settings['de_count']+=1
        self.ego_last_wp = ego_wp
        self.follow_last_wp = follow_wp

        if distance_between_waypoints(self.ego_last_wp, self.follow_last_wp) < 10:
            self.default_settings['de_count'] -= 1

        return self.start_save_sign, frame_data

    def interfuser_tick_autopilot(self):
        return self.drive_plan_func()

    def _initialize_actors(self, config):
        # Depend: 场景前方跟随车辆
        vehicle_bp_types = self.default_settings['following']['type']
        bp_class = do_sample(vehicle_bp_types)
        if bp_class == 'bus' or bp_class == 'truck':
            self.more_space = 5
        following_car_bp_name = choose_bp_name('+' + bp_class)
        following_car_bp = self.carla_vars['blueprint_library'].find(following_car_bp_name)
        following_car_transform = carla.Transform(self.following_car_wp.transform.location + carla.Location(z=0.1), self.following_car_wp.transform.rotation)
        following_car = self._world.spawn_actor(following_car_bp, following_car_transform)
        self.other_actors.append(following_car)
        self.actor_desc.append('following_car')
        self.data_tags['vehicle_type'] = bp_class
        self.data_tags['front_vehicle_num'] = 1

        # For: 左侧交通流
        if random.random() < self.default_settings['probs']['left_traffic']:
            vehicle_nums = left_traffic_flow_scenario(
                self.starting_wp, self.other_actors, self.actor_desc,
                scene_cfg={'filters': '+hcy1', 'idp': 0.5, 'lanes_num': 99, 'forward_num': random.randint(0, 10), 'backward_num': random.randint(0, 10)},
                gen_cfg={'name_prefix': 'left'},
                assign_distribution=self.assign_distribution
            )
            self.data_tags['left_vehicle_num'] = sum(vehicle_nums)
        else:
            self.data_tags['left_vehicle_num'] = 0

        # For: 右侧交通流
        if random.random() < self.default_settings['probs']['right_traffic']:
            vehicle_nums = right_traffic_flow_scenario(
                self.starting_wp, self.other_actors, self.actor_desc,
                scene_cfg={'filters': '+hcy1', 'idp': 0.5, 'lanes_num': 99, 'forward_num': random.randint(0, 10), 'backward_num': random.randint(0, 10)},
                gen_cfg={'name_prefix': 'right'},
                assign_distribution=self.assign_distribution
            )
            self.data_tags['right_vehicle_num'] = sum(vehicle_nums)
        else:
            self.data_tags['right_vehicle_num'] = 0

        # For: 对向交通流
        if random.random() < self.default_settings['probs']['opposite_traffic']:
            vehicle_nums = opposite_traffic_flow_scenario(
                self.starting_wp, self.other_actors, self.actor_desc,
                scene_cfg={'filters': '+hcy1', 'idp': 0.4, 'forward_num': random.randint(0, 8), 'backward_num': random.randint(3, 20)},
                gen_cfg={'name_prefix': 'opposite'},
                assign_distribution=self.assign_distribution
            )
            self.data_tags['opposite_vehicle_num'] = sum(vehicle_nums)
        else:
            self.data_tags['opposite_vehicle_num'] = 0

        self.traffic_manager = CarlaDataProvider.get_trafficmanager()
        for a_index, (actor, actor_desc) in enumerate(zip(self.other_actors, self.actor_desc)):
            if actor.type_id.startswith('vehicle'):
                actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
                if actor_desc == 'following_car':
                    self.following_car = actor
                    if random.random() < self.default_settings['following']['update_light']:
                        self.traffic_manager.update_vehicle_lights(actor, True)
                        self.data_tags['following_car_light'] = True
                    else:
                        self.traffic_manager.update_vehicle_lights(actor, False)
                        self.data_tags['following_car_light'] = False
                else:
                    self.traffic_manager.update_vehicle_lights(actor, True)
                    if self.data_tags['drive_plan'] == 'A':
                        self.traffic_manager.set_desired_speed(actor, random.randint(30, 50))
                    elif self.data_tags['drive_plan'] == 'B':
                        self.traffic_manager.set_desired_speed(actor, random.randint(15, 25))
                    self.traffic_manager.auto_lane_change(actor, False)

    def _create_behavior(self):
        root = py_trees.composites.Parallel(name="pidFollowingStop", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(TimeoutRaiseException(300))
        root.add_child(CollisionTest(self.ego_vehicles[0], terminate_on_failure=True))
        root.add_child(InTriggerDistanceToNextIntersection(self.ego_vehicles[0], 10))
        c1 = py_trees.composites.Sequence(name="pidFollowingStop_c1")
        c1.add_child(DriveDistance(self.ego_vehicles[0], 10))
        c1.add_child(StandStill(self.ego_vehicles[0], name="StandStill", duration=4))
        root.add_child(c1)
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
