# -*- coding: utf-8 -*-
# @Time    : 2024/05/14
# @Author  : H.Liu
# @File    : obstacledistribution.py
# @Desc    : 前车道分布式障碍物场景（右侧）
# @input   : 输入接口在第270行，可调整障碍物数量/类型/和类别生成概率
#          : 'num_trashcan': [10,20], 障碍物数量将落在（10，20）之间
#          :  'objects' :['trash_can', 'garbage', 'trash_bin', 'barrel', 'box'], 障碍物类型
#          : 'weights' :[0.2, 0.5, 0.1, 0.1, 0.1]，每类别障碍物生成概率                                               
                                                

from __future__ import print_function
import random
import carla
import py_trees
import operator
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance, StandStill,InTriggerDistanceToVehicle, TimeoutRaiseException, InTriggerDistanceToNextIntersection)
from Initialization.Standardimport.basic_scenario import BasicScenario
from Explainablelibrary.explainable_utils import *
from Explainablelibrary.basic_desc import *
from Scenarios.Tinyscenarios import *
from Scenarios.condition_utils import *
from Logger import *
# import pygame

# import ipdb

class SideTurnCondition(AtomicCondition):
    def __init__(self, debug=False):

        self.scenario_name = 'SideTurn'

        self.min_distance_to_junction = 30  
        self.max_distance_to_junction = 50
        super(SideTurnCondition, self).__init__(debug)

    def check_condition_by_waypoint(self, wp):
        lane_info = get_lane_info(wp)
        if lane_info is None:
            self.dprint(f'条件1：获取车道信息失败')
            return False

        if distance_to_next_junction(wp) < self.min_distance_to_junction:
            self.dprint(f'条件2：到前方最近路口的距离为{distance_to_next_junction(wp)}，不处于限制区间[30，50]')
            return False
        
        if distance_to_next_junction(wp) > self.max_distance_to_junction:
            self.dprint(f'条件2：到前方最近路口的距离为{distance_to_next_junction(wp)}，不处于限制区间[30，50]')
            return False

        return True


class SideTurn(BasicScenario):
    """
    Desc: TODO
    Special: 补充TODO的数据
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180, uniad=False, interfuser=False, assign_distribution=None):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        self.starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self.distance_to_junction = distance_to_next_junction(self.starting_wp)
        self.predefined_vehicle_length = 8
        self.actor_desc = []
        self.traffic_manager = None
        self.init_speed = 0

        # For: tick autopilot
        self.side_car_left = None
        self.side_car_right = None
        self.side_care_wp = None
        self.side_car_loc = None
        self.initialized = False
        self.lane_changed = False
        self.avoid_sign = False 
        self.data_tags = {}                        #关键指标日志记录
        self.navi_command = random.choice([0, 2])  #1: left, 2: straight
        self.start_save_sign = False               #是否开始记录
        # self.navigation_cmds = ['Straight']   
        # self.obstacle_index = -1
        self.start_save_speed_seed = random.random()

        # For: 始终保持最右车道
        # while True:
        #     right_lane = self.starting_wp.get_right_lane()
        #     if right_lane is None or right_lane.lane_type != carla.LaneType.Driving:
        #         break
        #     self.starting_wp = right_lane
        # For: Is success
        self.passed_lane_ids = []

        # For: V5 description
        self.carla_desc = CarlaDesc(world, ego_vehicles[0])


        super().__init__("SideTurn", ego_vehicles, config, world, randomize, debug_mode,
                         criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser, assign_distribution=assign_distribution)
        
    
    def get_data_tags(self):
        return self.data_tags

    def is_success(self):
        # Desc: Autopilot是否在当前车道上速度最终变为0且不存在变道行为
        if len(self.passed_lane_ids) == 0:
            return False, 'Autopilot未行驶'

        last_speed = None
        change_num = 0
        first_lane_id = -1
        for item in self.passed_lane_ids:
            if first_lane_id == -1:
                first_lane_id = item

            if last_speed is not None and item != last_speed:
                change_num += 1

            last_speed = item

        if last_speed == 0 and change_num == 0:
            return True, ''
        else:
            return False, 'Autopilot在当前车道上速度未最终变为0或者进行了车道变换'

    def interfuser_tick_autopilot(self):
        # pygame.init()
        # pygame.display.set_mode((400, 300))

        #场景主车辆
        ego = self.ego_vehicles[0]
        frame_data = {'navi_command': self.navi_command}
        if self.side_car_left is None:
            if 'left' in self.actor_desc:
                self.side_car_left = self.other_actors[self.actor_desc.index('left')]

        if self.side_car_right is None:
            if 'right' in self.actor_desc:
                self.side_car_right = self.other_actors[self.actor_desc.index('right')]
        
        
        if self.initialized is False:
            self._tf_set_ego_speed(self.init_speed)        # 初始速度
            # self._tf_disable_ego_auto_lane_change()        # 禁用自动变道
            # self._set_ego_autopilot()               
            self.initialized = True

        return self.start_save_sign, frame_data

        
    def _initialize_actors(self, config):

        lane_info = get_lane_info(self.starting_wp)
        left_car = self.spawn_car_left_lane()
        right_car = self.spawn_car_right_lane()
        # ipdb.set_trace()
        if left_car is not None:
            self.other_actors.append(left_car)
            self.actor_desc.append('left')
        if right_car is not None:
            self.other_actors.append(right_car)
            self.actor_desc.append('right')
        
        
        
        # # For: 左侧交通流
        left_traffic_flow_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.5, 'lanes_num': 4, 'skip_num': 0, 'forward_num': random.randint(2, 5), 'backward_num': random.randint(0, 20)}, gen_cfg={'name_prefix': 'left'}
        )

        # # For: 右侧交通流
        right_traffic_flow_scenario(self.starting_wp, self.other_actors, self.actor_desc,
                                    scene_cfg={'filters': '+hcy1', 'idp': 0.4}, gen_cfg={'name_prefix': 'right'})
        # # For: 对向交通流
        opposite_traffic_flow_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.4, 'forward_num': random.randint(5, 8), 'backward_num': random.randint(0, 20)},
            gen_cfg={'name_prefix': 'opposite'}
        )

        self.traffic_manager = CarlaDataProvider.get_trafficmanager()
        for a_index, (actor, actor_desc) in enumerate(zip(self.other_actors, self.actor_desc)):
            # ipdb.set_trace()
            if actor.type_id.startswith('vehicle'):
                if 'park' in actor_desc:
                    actor.apply_control(carla.VehicleControl(brake=1.0))
                    continue
                actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
                self.traffic_manager.update_vehicle_lights(actor, True)
                self.traffic_manager.set_desired_speed(actor, random.randint(30, 40))
                self.traffic_manager.auto_lane_change(actor, False)

    def _create_behavior(self):
        root = py_trees.composites.Parallel(name="SideTurn",
                                            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        c1 = py_trees.composites.Sequence(name="SideTurn_c1")
        c1.add_child(DriveDistance(self.ego_vehicles[0], 20))
        root.add_child(c1)
        root.add_child(TimeoutRaiseException(30))
        root.add_child(CollisionTest(self.ego_vehicles[0], terminate_on_failure=True))
        root.add_child(InTriggerDistanceToNextIntersection(self.ego_vehicles[0], 10))
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
    
    #右侧向运动障碍
    def spawn_car_right_lane(self):
        if self.starting_wp.get_right_lane() is not None:
            if self.starting_wp.get_right_lane().lane_type == carla.LaneType.Driving:
                side_wp_init = self.starting_wp.get_right_lane()
            else: return None
        else:
            return None

        side_distance = random.choice([0,4])
        side_turn = random.choice(['front','back'])
        if side_turn =='front':
            side_vehicle_wp = move_waypoint_forward(side_wp_init, side_distance)
        else:
            side_vehicle_wp = move_waypoint_backward(side_wp_init,side_distance)
        side_bp_name = choose_bp_name('+wheel4-special')
        side_actor = CarlaDataProvider.request_new_actor(side_bp_name, side_vehicle_wp.transform)
        return side_actor
    
    ## 侧向运动障碍+前后向运动障碍
    # def spawn_car_right_lane(self):
    #     if self.starting_wp.get_right_lane() is not None:
    #         if self.starting_wp.get_right_lane().lane_type == carla.LaneType.Driving:
    #             side_wp_init = self.starting_wp.get_right_lane()
    #         else: side_wp_init = self.starting_wp
    #     else:
    #         side_wp_init = self.starting_wp

    #     side_distance = random.choice([12,20])
    #     side_turn = random.choice(['front','back'])
    #     if side_turn =='front':
    #         side_vehicle_wp = move_waypoint_forward(side_wp_init, side_distance)
    #     else:
    #         side_vehicle_wp = move_waypoint_backward(side_wp_init,side_distance)
    #     side_bp_name = choose_bp_name('+wheel4-special')
    #     side_actor = CarlaDataProvider.request_new_actor(side_bp_name, side_vehicle_wp.transform)
    #     return side_actor
    
    #左侧运动障碍
    def spawn_car_left_lane(self):
        if self.starting_wp.get_left_lane() is not None:
            if self.starting_wp.get_left_lane().lane_type == carla.LaneType.Driving:
                side_wp_init = self.starting_wp.get_left_lane()
            else: 
                return None
        else:
            return None

        side_distance = random.choice([0,4])
        side_turn = random.choice(['front','back'])
        if side_turn =='front':
            side_vehicle_wp = move_waypoint_forward(side_wp_init, side_distance)
        else:
            side_vehicle_wp = move_waypoint_backward(side_wp_init,side_distance)
        side_bp_name = choose_bp_name('+wheel4-special')
        side_actor = CarlaDataProvider.request_new_actor(side_bp_name, side_vehicle_wp.transform)
        return side_actor
    # 左侧运动障碍 + 前后向运动障碍
    # def spawn_car_left_lane(self):
    #     if self.starting_wp.get_left_lane() is not None:
    #         if self.starting_wp.get_left_lane().lane_type == carla.LaneType.Driving:
    #             side_wp_init = self.starting_wp.get_left_lane()
    #         else: side_wp_init = self.starting_wp
    #     else:
    #         side_wp_init = self.starting_wp

    #     side_distance = random.choice([12,20])
    #     side_turn = random.choice(['front','back'])
    #     if side_turn =='front':
    #         side_vehicle_wp = move_waypoint_forward(side_wp_init, side_distance)
    #     else:
    #         side_vehicle_wp = move_waypoint_backward(side_wp_init,side_distance)
    #     side_bp_name = choose_bp_name('+wheel4-special')
    #     side_actor = CarlaDataProvider.request_new_actor(side_bp_name, side_vehicle_wp.transform)
    #     return side_actor
        
    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        pass
    
