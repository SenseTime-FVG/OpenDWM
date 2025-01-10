from __future__ import print_function

import random

import py_trees

from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance, StandStill)
from Initialization.Standardimport.basic_scenario import BasicScenario

from Explainablelibrary.explainable_utils import *
from Explainablelibrary.basic_desc import *
from Scenarios.Tinyscenarios import *


class ObstacleAhead(BasicScenario):
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
        self.init_speed = 20
        self.navigation_cmds = ['Straight']
        self.obstacle_index = -1
        self.obstacle_location = 0


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

        super().__init__("ObstacleAhead", ego_vehicles, config, world, randomize, debug_mode,
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
        # WalkerManager.check_walker_distance_to_obstacles(self)
        if self.initialized is False:
            self._tf_set_ego_route(self.navigation_cmds)
            self._tf_set_ego_speed(self.init_speed)
            self._tf_disable_ego_auto_lane_change()
            self._set_ego_autopilot()
            self.initialized = True

        explainable_data = {
            'actors': build_actor_data(ego, self.other_actors, eng=True),
            'actors_desc': self.actor_desc,
        }

        cur_ego_loc = ego.get_location()
        cur_wp = self._map.get_waypoint(cur_ego_loc)

        if not cur_wp.is_junction and not cur_wp.is_intersection:
            self.passed_lane_ids.append(cur_wp.lane_id)

        info = f'Speed: {int(ego_speed)} km/h '

        if self.obstacle_location is not None:
            distance = round(self.obstacle_location.distance(cur_ego_loc), 2)
            info += f'Distance to obstacle: {distance}m '
            info += f'Distance: {distance}m '

            if distance > 10:
                ego_stage = '#fix1'
                reason = 'because there are no special circumstances, so normal driving.'
                self._tf_set_ego_speed(20)
            elif 10 >= distance :# 观测到前方有静止的障碍物
                ego_stage = '#dynamic1'
                reason = f'because there is a stationary , prepare to start decelerating.'
                self._tf_set_ego_speed(0)  # 设置自动驾驶汽车的速度为 0m/s
                # else:
                #     reason = f'because there is a {desc.color} {desc.type} moving slowly at the {desc.direction} of ego. It does not affect driving, so it is normal driving.'
                #     self._tf_set_ego_speed(20)

        else:
            vel = random.uniform(20, 40)
            self._tf_set_ego_speed(vel)
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

        self.obstacle_location = ahead_obstacle_scenario(self._world,self.starting_wp, self.other_actors, self.actor_desc,
                                    scene_cfg={ 'num_cones': 10,  # 想要生成的锥筒数量
                                                'cone_interval': 2.5 , # 锥筒之间的间隔距离（米）
                                                'num_garbage': 50 ,   # 想要生成的垃圾数量
                                                'num_workers': 4 },   # 想要生成的行人数量
                                    gen_cfg={'name_prefix': 'construction1'})     # 默认construction 1    construction 1 or construction 2

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
        for a_index, actor in enumerate(self.other_actors):
            if actor.type_id.startswith('vehicle'):
                actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
                self.traffic_manager.update_vehicle_lights(actor, True)
                if a_index == 0:
                    self.traffic_manager.set_desired_speed(actor,random.randint(10, 15))
                else:
                    self.traffic_manager.set_desired_speed(actor, random.randint(10, 15))


    def _create_behavior(self):
        root = py_trees.composites.Parallel(name="ObstacleAhead",
                                            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        c1 = py_trees.composites.Sequence(name="ObstacleAhead_c1")
        # c1.add_child(InTriggerDistanceToVehicle(self.other_actors[self.front_index], self.ego_vehicles[0], 12))
        c1.add_child(StandStill(self.ego_vehicles[0], name='ego_standstill', duration= 5))
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