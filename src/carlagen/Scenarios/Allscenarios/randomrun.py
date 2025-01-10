# -*- coding: utf-8 -*-
# @Time    : 2024/7/10
# @Author  : H.Liu
# @File    : RandomRun.py
# @Desc    : 随机在各种场景中采集数据
# @input                                            
                                                

from __future__ import print_function
import random
import carla
import py_trees
import operator
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance, StandStill,InTriggerDistanceToVehicle, TimeoutRaiseException, InTriggerDistanceToNextIntersection,WaitEndIntersection)
from Initialization.Standardimport.basic_scenario import BasicScenario
from Explainablelibrary.explainable_utils import *
from Explainablelibrary.basic_desc import *
from Scenarios.Tinyscenarios import *
from Scenarios.condition_utils import *
from Logger import *
# import pygame

# import ipdb

class RandomRunCondition(AtomicCondition):
    def __init__(self, debug=False):

        self.scenario_name = 'RandomRun'
        self.min_distance_to_junction = 50 

        super(RandomRunCondition, self).__init__(debug)

    def check_condition_by_waypoint(self, wp):
        lane_info = get_lane_info(wp)
        if lane_info is None:
            self.dprint(f'条件1：获取车道信息失败')
            return False

        if distance_to_next_junction(wp) < self.min_distance_to_junction:
            self.dprint(f'条件2：到前方最近路口的距离为{distance_to_next_junction(wp)}，不处于限制区间[30，50]')
            return False
        
        if wp.road_id in [65,71,57]:
            return False
        
        #只取左右转路点
        if lane_info.r2l >1 and lane_info.l2r >1:
            return False
        

        return True


class RandomRun(BasicScenario):
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
        self.init_speed = random.randint(40,60)

        # For: tick autopilot
        self.side_car_left = None
        self.side_car_right = None
        self.side_care_wp = None
        self.side_car_loc = None
        self.initialized = False
        self.lane_changed = False
        self.avoid_sign = False 
        self.data_tags = {}                        #关键指标日志记录
        self.start_save_sign = False               #是否开始记录
        self.start_save_speed_seed = random.random()

        starting_wp_lane_info = get_lane_info(self.starting_wp)
        self.in_junction = False
        self.traffic = '222'
        self.last_traffic = '222'
        navi_command_candidates = [2]
        next_navi_command_candidates = [2]

        #可以右转
        if starting_wp_lane_info.r2l <= 1:
            if self.side_turn(self.starting_wp,"right") is not None:
                navi_command_candidates.append(0)
                
        #可以左转
        if starting_wp_lane_info.l2r <= 1:
            if self.side_turn(self.starting_wp,"left") is not None:
                 navi_command_candidates.append(1)
        # self.navi_command = random.choice(navi_command_candidates)
        self.navi_command = navi_command_candidates[-1]

        #左转后的navi_command:
        if self.navi_command == 1:
            next_lane_info = get_lane_info(self.side_turn(self.starting_wp,"left"))
            if next_lane_info .r2l <= 1:
                next_navi_command_candidates.append(0)
            if next_lane_info .l2r <= 1:
                next_navi_command_candidates.append(1)
        
        #右转后的navi_command:
        if self.navi_command == 0:
            next_lane_info = get_lane_info(self.side_turn(self.starting_wp,"right"))
            if next_lane_info .r2l <= 1:
                next_navi_command_candidates.append(0)
            if next_lane_info .l2r <= 1:
                next_navi_command_candidates.append(1)

        self.next_navi_command = random.choice(next_navi_command_candidates)

        
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


        super().__init__("RandomRun", ego_vehicles, config, world, randomize, debug_mode,
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
        ego_loc = ego.get_location()
        ego_rot = ego.get_transform().rotation
        ego_wp = self._map.get_waypoint(ego_loc)
        
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)
        # message(f'ego_speed:{ego_speed}')
        
        if ego_wp.is_junction and not self.in_junction:
           self.in_junction = True
           message(f'进路口:{self.in_junction}')
           frame_data = {'navi_command': self.navi_command}
        #    ipdb.set_trace(context = 10)
        elif not ego_wp.is_junction and self.in_junction:
            frame_data = {'navi_command': self.next_navi_command}
            message(f'出路口:{self.in_junction}')
        else:
            frame_data = {'navi_command': self.navi_command}
        
        message(f'navi_command:{frame_data["navi_command"]}')

        # traffic_light = ego.get_traffic_light()

        # if traffic_light:
        #     state = traffic_light.get_state()
        #     message(f'红绿灯颜色{state}')
        #     if state == 'Green':
        #         frame_data['tf_light_state'] = '111'
        #     else:
        #         frame_data['tf_light_state'] = '000'
        # else:
        #     frame_data['tf_light_state'] = '222'

        # message(f'traffic_light_state{frame_data["tf_light_state"]}')

        traffic_distance, traffic_state, left_turn, right_turn, straight_turn = junction_first_traffic_light(self._world,ego_wp)
        # message(f'左右转分别为{left_turn,right_turn}')
        # message(f'Traffic_state{traffic_state}')
        # ipdb.set_trace()
        #记录路口红绿灯到自车的距离以及红绿灯的状态，返回到frame_data里
        message(f'颜色:{traffic_state}')
        frame_data['tf_light_distance'] = traffic_distance
        if traffic_distance > 60:
            self.traffic = '222'
            self.last_traffic = self.traffic
            frame_data['tf_light_state'] = self.traffic
        else:
            if ego_wp.is_junction:
                frame_data['tf_light_state'] = self.last_traffic
            else:
                if traffic_state in ['red', 'yellow']:
                    if right_turn is not None:
                        self.traffic = '001'
                        self.last_traffic = self.traffic
                        frame_data['tf_light_state'] = self.traffic
                    else:
                        self.traffic = '000'
                        self.last_traffic = self.traffic
                        frame_data['tf_light_state'] = self.traffic
                        
                elif traffic_state =='yellow': 
                    if right_turn is not None:
                        self.traffic = '001'
                        self.last_traffic = self.traffic
                        frame_data['tf_light_state'] = self.traffic

                    else:
                        self.traffic = '000'
                        self.last_traffic = self.traffic
                        frame_data['tf_light_state'] = self.traffic

                elif traffic_state == 'green':
                    if right_turn is None:
                        self.traffic = '112'
                        self.last_traffic = self.traffic
                        frame_data['tf_light_state'] = self.traffic

                    elif left_turn is None:
                        self.traffic = '211'
                        self.last_traffic = self.traffic
                        frame_data['tf_light_state'] = self.traffic

                    elif straight_turn is None:
                        self.traffic = '121'
                        self.last_traffic = self.traffic
                        frame_data['tf_light_state'] = self.traffic

                    else:
                        self.traffic = '111'
                        self.last_traffic = self.traffic
                        frame_data['tf_light_state'] = self.traffic

                else:
                    self.traffic = '222'
                    self.last_traffic = self.traffic
                    frame_data['tf_light_state'] = self.traffic

        
        message(f'交通灯状态:{frame_data["tf_light_state"]}')
        
        if self.initialized is False:
            self._tf_set_ego_route([self.navi_command])
            self._tf_set_ego_speed(self.init_speed)        # 初始速度
            self._tf_disable_ego_auto_lane_change()        # 禁用自动变道
            self._set_ego_autopilot()               
            self.initialized = True

        if not self.start_save_sign:
            if self.start_save_speed_seed < 0.5:
                if ego_speed > 20:
                    self.start_save_sign = True
                    success(f'开始保存数据，自车速度：{round(ego_speed, 2)}km/h')
            else:
                if ego_speed > 15:
                    self.start_save_sign = True
                    success(f'开始保存数据，自车速度：{round(ego_speed, 2)}km/h')
        
        return self.start_save_sign, frame_data


        
    def _initialize_actors(self, config):

        road_id = self.starting_wp.road_id
        
        if road_id in [65]:
            raise ValueError("The starting waypoint lane ID is 65, which is not allowed.")
        
        #前方交通流
        front_traffic_flow_scenario(self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.5, 'lanes_num': 4, 'skip_num': 0, 'forward_num': random.randint(0, 15), 'backward_num': random.randint(0, 15)}, gen_cfg={'name_prefix': 'left'}
        )
        # # For: 左侧交通流
        left_traffic_flow_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.5, 'lanes_num': 4, 'skip_num': 0, 'forward_num': random.randint(0, 15), 'backward_num': random.randint(0, 15)}, gen_cfg={'name_prefix': 'left'}
        )

        # # For: 右侧交通流
        right_traffic_flow_scenario(self.starting_wp, self.other_actors, self.actor_desc,
                                    scene_cfg={'filters': '+hcy1', 'idp': 0.4, 'forward_num': random.randint(0, 15), 'backward_num': random.randint(0, 15)}, gen_cfg={'name_prefix': 'right'})
        # # For: 对向交通流
        opposite_traffic_flow_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.4, 'forward_num': random.randint(0, 15), 'backward_num': random.randint(0, 15)},
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
                self.traffic_manager.set_desired_speed(actor, random.randint(40, 60))
                self.traffic_manager.auto_lane_change(actor, False)

    def _create_behavior(self):

        distance_to_junc = distance_to_next_junction(self.starting_wp)
        if distance_to_junc < 70:
            move_dis = distance_to_junc + random.randint(45,55)
        elif distance_to_junc <90:
            move_dis = distance_to_junc + random.randint(45,55)
        elif distance_to_junc < 110:
            move_dis = distance_to_junc + random.randint(45,55)
        elif distance_to_junc < 150:
            move_dis = distance_to_junc + random.randint(45,55)
        elif distance_to_junc < 200:
            move_dis = distance_to_junc + random.randint(45,55)
        else:
            move_dis = 350
        message(f'move_distance{move_dis}')
        
        root = py_trees.composites.Parallel(name="RandomRun",
                                            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        c1 = py_trees.composites.Sequence(name="RandomRun_c1")
        c1.add_child(DriveDistance(self.ego_vehicles[0], move_dis))
        root.add_child(c1)
        root.add_child(TimeoutRaiseException(300))
        root.add_child(CollisionTest(self.ego_vehicles[0], terminate_on_failure=True))
        # root.add_child(WaitEndIntersection(self.ego_vehicles[0]))
        # root.add_child(InTriggerDistanceToNextIntersection(self.ego_vehicles[0], 10))
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
    

    def side_turn(self,start_wp,direction,visual_manager=None, debug=False):
    
        junction = get_next_junction(start_wp,self._world)
        # ipdb.set_trace(context=10)
        

        #debug
        # return get_junction_turn_after_wp(junction, start_wp, direction,self._world)
        return get_junction_turn_after_wp_new(junction, start_wp, direction,self._world)