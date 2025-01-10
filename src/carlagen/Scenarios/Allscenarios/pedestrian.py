from __future__ import print_function

import math
import random

# import ipdb
import py_trees
import operator
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance, StandStill, InTriggerDistanceToVehicle, TimeoutRaiseException, InTriggerDistanceToNextIntersection)
from Initialization.Standardimport.basic_scenario import BasicScenario

from Explainablelibrary.explainable_utils import *
from Explainablelibrary.basic_desc import *
from Scenarios.Tinyscenarios import *
from Logger import *


class Pedestrian(BasicScenario):
    """
    Desc: TODO
    Special: 补充TODO的数据
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180, uniad=False, interfuser=False):
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        self.starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self.distance_to_junction = distance_to_next_junction(self.starting_wp)
        self.predefined_vehicle_length = 8
        self.actor_desc = []
        self.traffic_manager = None

        # For: tick autopilot
        self.front_obstacle = None
        self.front_obstacle_wp = None
        self.front_obstacle_loc = None
        self.initialized = False
        self.lane_changed = False
        self.stage = 0
        self.init_speed = random.randint(20, 40)
        self.navigation_cmds = ['Straight']
        self.obstacle_index = -1
        self.obstacle_location = None
        self.spawn_npc_point_location =  None
        self.npc = None
        self.random_distance = random.randint(5, 8)
        self.num = 0
        self.npc_all = []
        self.road_ids = set()

        # For: Is success
        self.passed_lane_ids = []

        # For: V5 description
        self.carla_desc = CarlaDesc(world, ego_vehicles[0])

        # For: autopilot stage
        self.stages = {
            '#fix1': (KEEP_L, [], KEEP_S, [], ''),
            '#dynamic1': (KEEP_L, [], STOP, [], ''),
            '#dynamic2': (LEFT_C, [], KEEP_S, [], ''),
            '#fix2': (KEEP_L, [], ACC, [], ''),
            '#dynamic3': (LEFT_C, [], KEEP_S, [], ''),
            '#fix3': (KEEP_L, [], ACC, [], ''),
            '#dynamic4': (RIGHT_C, [], KEEP_S, [], ''),
            '#fix4': (KEEP_L, [], ACC, [], ''),
            '#dynamic5': (RIGHT_C, [], DEC, [], ''),
            '#fix5': (KEEP_L, [], KEEP_S, [], ''),
            '#fix6': (KEEP_L, [], KEEP_S, [], ''),
        }

        super().__init__("Pedestrian", ego_vehicles, config, world, randomize, debug_mode,
                         criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser)

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
        ego = self.ego_vehicles[0]
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)
        cur_ego_loc = ego.get_location()
        cur_wp = self._map.get_waypoint(cur_ego_loc)

        if self.initialized is False:
            self._tf_set_ego_route(self.navigation_cmds)
            self._tf_set_ego_speed(self.init_speed)
            self._tf_disable_ego_auto_lane_change()
            self._tf_set_ego_ignore_signs()
            self._set_ego_autopilot()
            self.initialized = True
            self.road_ids.add(cur_wp.road_id)
            self.first_wp = cur_wp

        explainable_data = {
            'actors': build_actor_data(ego, self.other_actors, eng=True),
            'actors_desc': self.actor_desc,
        }
        ego_location = ego.get_location()
        x_coordinate = ego_location.x
        y_coordinate = ego_location.y
        # ipdb.set_trace()
        # control_pedestrian(self.npc,self.spawn_npc_point)
        npc = self.npc_all[0]
        cur_npc_loc = npc.get_location()

        assert isinstance(cur_wp, carla.Waypoint)
        if not cur_wp.is_junction and not cur_wp.is_intersection:
            self.passed_lane_ids.append(cur_wp.lane_id)
        ego_stage = '#fix1'
        reason = f'because there are no special circumstances, so normal driving.'

        info = f'Speed: {int(ego_speed)} km/h '
        ego_transform = ego.get_transform()
        if self.obstacle_location is not None:
            distance = round(self.obstacle_location.distance(cur_ego_loc), 2)
            #distance = math.sqrt((self.spawn_npc_point.x - cur_ego_loc.x) ** 2 + (self.spawn_npc_point.y - cur_ego_loc.y) ** 2)
            ego_direction = ego_transform.rotation.get_forward_vector()
            delta_x = cur_npc_loc.x - ego_location.x
            delta_y = cur_npc_loc.y - ego_location.y
            # 计算向量与车辆朝向方向的夹角（弧度）
            angle_rad = math.atan2(delta_y, delta_x) - math.atan2(ego_direction.y, ego_direction.x)
            # print("angle_rad    =   ",angle_rad)
            info += f'Distance to obstacle: {distance}m '
            info += f'Distance: {distance}m '

            if distance > 12:
                ego_stage = '#fix1'
                reason = 'because there are no special circumstances, so normal driving.'
                self._tf_set_ego_speed(self.init_speed)
                if distance < 23:
                    control_pedestrian(self.npc, self.spawn_npc_point)
            elif  12 >= distance and math.fabs(angle_rad) < math.atan(cur_wp.lane_width/ 20) :# 观测到前方有静止的障碍物
                # print("math.fabs(angle_rad) =   ",math.fabs(angle_rad))
                # print("math.atan(1.5 / 10) = ", math.atan(1.5 / 10))
                ego_stage = '#dynamic1'
                reason = f'because there is a stationary person , prepare to start decelerating.'
                self._tf_set_ego_speed(0)
                control_pedestrian(self.npc, self.spawn_npc_point)# 设置自动驾驶汽车的速度为 0m/s
            else:
                ego_stage = '#fix1'
                reason = 'because there are no special circumstances, so normal driving.'

        else:
            self._tf_set_ego_speed(self.init_speed)
            ego_stage = '#fix6'
            reason = f'because there are no special circumstances, so normal driving.'

        stage_data = self.stages[ego_stage]

        decision = {'path': (stage_data[0], stage_data[1]), 'speed': (stage_data[2], stage_data[3])}
        env_description = self.carla_desc.get_env_description()

        explainable_data['env_desc'] = env_description
        explainable_data['ego_stage'] = ego_stage
        explainable_data['ego_action'] = decision
        explainable_data['scenario'] = self.name
        explainable_data['nav_command'] = 'follow lane'
        explainable_data['info'] = info

        explainable_data['ego_reason'] = reason

        info += f'{ego_stage} -> 可解释性描述：{env_description}'
        hanzi_num = len(re.findall(r'[\u4e00-\u9fa5]', info))
        info += ' ' * (150 - hanzi_num * 2 - (len(info) - hanzi_num))
        # print(f'\r{info}', end='')

        return explainable_data

    def _initialize_actors(self, config):
        # Depend: 场景保证前方生成一辆静止的车辆
        lane_info = get_lane_info(self.starting_wp)
        # For: 生成NPC
        get_opposite_lane_spawn_transforms(self._world, self.ego_vehicles[0],
                                                                 random.randint(0, lane_info.num * 5))
        pedestrians_bp,self.obstacle_location,self.spawn_npc_point_location,self.spawn_npc_point = spawn_pedestrian_v2(self._world, self.starting_wp)
        self.npc = CarlaDataProvider.request_new_actor(pedestrians_bp, self.spawn_npc_point)
        self.npc_all.append(self.npc)

        # For: 右侧交通流
        right_traffic_flow_scenario(self.starting_wp, self.other_actors, self.actor_desc,
                                    scene_cfg={'filters': '+hcy1', 'idp': 0.4}, gen_cfg={'name_prefix': 'right'})
        # For: 左侧交通流
        left_traffic_flow_scenario(self.starting_wp, self.other_actors, self.actor_desc,
                                   scene_cfg={'filters': '+hcy1', 'idp': 0.4}, gen_cfg={'name_prefix': 'left'})
        # For: 对向交通流
        opposite_traffic_flow_scenario(self.starting_wp, self.other_actors, self.actor_desc,
                                       scene_cfg={'filters': '+hcy1', 'idp': 0.4,
                                                  'backward_num': random.randint(10, 30)},
                                       gen_cfg={'name_prefix': 'opposite'})
        # For: 路边停靠车辆
        right_parking_vehicle_scenario(self.starting_wp, self.other_actors, self.actor_desc,
                                       scene_cfg={'filters': '+wheel4-large', 'idp': 0.4,
                                                  'forward_num': random.randint(8, 15)},
                                       gen_cfg={'name_prefix': 'park'})
        self.traffic_manager = CarlaDataProvider.get_trafficmanager()
        for a_index, (actor, actor_desc) in enumerate(zip(self.other_actors, self.actor_desc)):
            if 'park' in actor_desc:
                continue
            if actor.type_id.startswith('vehicle'):
                actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
                self.traffic_manager.update_vehicle_lights(actor, True)
                self.traffic_manager.auto_lane_change(actor, False)
                self.traffic_manager.set_desired_speed(actor,random.randint(20, 40))

    def _create_behavior(self):
        root = py_trees.composites.Parallel(name="Pedestrian",
                                            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        c1 = py_trees.composites.Sequence(name="Pedestrian_c1")
        c1.add_child(StandStill(self.ego_vehicles[0], name='ego_standstill', duration= 5))
        root.add_child(InTriggerDistanceToNextIntersection(self.ego_vehicles[0],10))
        root.add_child(CollisionTest(self.ego_vehicles[0], terminate_on_failure=True))
        root.add_child(CollisionTest(self.npc_all[0], terminate_on_failure=True))
        root.add_child(TimeoutRaiseException(35))
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