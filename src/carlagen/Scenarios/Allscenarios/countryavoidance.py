from __future__ import print_function

import math
import random

import carla
import py_trees
import operator
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance, StandStill, InTriggerDistanceToVehicle, TimeoutRaiseException)
from Initialization.Standardimport.basic_scenario import BasicScenario

from Explainablelibrary.explainable_utils import *
from Explainablelibrary.basic_desc import *
from Scenarios.Tinyscenarios import *
from Logger import *


class CountryAvoidance(BasicScenario):
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
        self.front_car = None
        self.front_car_wp = None
        self.front_car_loc = None
        self.initialized = False
        self.lane_changed = False
        self.stage = 0
        self.offset_m = 0
        self.init_speed = random.randint(30, 40)
        self.navigation_cmds = ['Straight', 'Straight', 'Straight', 'Straight']

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

        super().__init__("CountryAvoidance", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser)

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
        ego_stage = '#fix1'
        reason = ''
        ego = self.ego_vehicles[0]
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)
        # WalkerManager.check_walker_distance_to_obstacles(self)
        if self.front_car is None:
            self.front_car = self.other_actors[self.actor_desc.index('front')]
            self.front_car_wp = self._map.get_waypoint(self.front_car.get_location())
            self.front_car_loc = self.front_car.get_location()

        if self.initialized is False:
            self._tf_set_ego_route(self.navigation_cmds)
            self._tf_set_ego_speed(self.init_speed)
            self._tf_disable_ego_auto_lane_change()
            self._set_ego_autopilot()
            self._tf_set_ego_ignore_signs()
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
        if self.front_car.is_alive:
            desc = Edict(explainable_data['actors'][self.front_car.id]['description'])
            data = Edict(explainable_data['actors'][self.front_car.id])
            distance = round(self.front_car.get_location().distance(cur_ego_loc), 2)
            info += f'Distance: {distance}m '

            if actor_front_of_ego(data):
                ego_stage = '#fix1'
                reason = '前方车辆占道，准备避让'

                if self.stage == 0:
                    if distance < 30:
                        self._tf_set_ego_speed(20)
                    if distance < 20:
                        ego_stage = '#fix2'
                        reason = '前方有车，减速'
                        self._tf_set_ego_speed(15)
                        self.offset_m = cur_wp.lane_width / 12.0 * 5
                        self._tf_set_ego_offset(self.offset_m)
                        message(f'向右避让{self.offset_m}米')
                        self.stage = 1
                elif self.stage == 1:
                    dis = math.sqrt((cur_ego_loc.x - cur_wp.transform.location.x) ** 2 + (cur_ego_loc.y - cur_wp.transform.location.y) ** 2)
                    print(f'偏移距离：{round(dis, 3)}m')
                    if abs(dis - self.offset_m) < 0.3:
                        self._tf_set_ego_speed(0)
                        message(f'自车静止让行')
                        self.stage = 2
            else:
                self._tf_set_ego_offset(self.offset_m)
                self.offset_m = 0
                self._tf_set_ego_speed(self.init_speed)
        else:
            ego_stage = '#fix3'
            reason = 'because there are no special circumstances, so normal driving.'
            self._tf_set_ego_speed(self.init_speed)

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
        get_opposite_lane_spawn_transforms(self._world, self.ego_vehicles[0], random.randint(0, lane_info.num * 5))

        # ipdb.set_trace()
        change_wp_transform_v1 = opposite_wp_through_junction_v1(self.starting_wp)
        veh_bp = choose_bp_name('+car')
        # 请求新的 actor
        front_actor = CarlaDataProvider.request_new_actor(veh_bp, change_wp_transform_v1)
        self.other_actors.append(front_actor)
        self.actor_desc.append('front')
        self.front_index = len(self.other_actors) - 1
        # For: 右侧交通流
        right_traffic_flow_scenario(self.starting_wp, self.other_actors, self.actor_desc, scene_cfg={'filters': '+hcy1', 'idp': 0.4}, gen_cfg={'name_prefix': 'right'})
        # For: 左侧交通流
        left_traffic_flow_scenario(self.starting_wp, self.other_actors, self.actor_desc, scene_cfg={'filters': '+hcy1', 'idp': 0.4}, gen_cfg={'name_prefix': 'left'})
        # # For: 对向交通流
        opposite_traffic_flow_scenario(self.starting_wp, self.other_actors, self.actor_desc, scene_cfg={'filters': '+hcy1', 'idp': 0.4, 'backward_num': random.randint(2, 6)}, gen_cfg={'name_prefix': 'opposite'})
        # For: 路边停靠车辆
        right_parking_vehicle_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={
                'filters': '+wheel4-large', 'idp': 0.4,
                'forward_num': random.randint(8, 15)
            },
            gen_cfg={'name_prefix': 'park'}
        )
        self.traffic_manager = CarlaDataProvider.get_trafficmanager()
        for a_index, actor in enumerate(self.other_actors):
            if actor.type_id.startswith('vehicle'):
                actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
                self.traffic_manager.update_vehicle_lights(actor, True)
                if a_index == 0:
                    half_lane_with = self.starting_wp.lane_width / 12.0 * 5
                    self.traffic_manager.vehicle_lane_offset(actor, -half_lane_with)
                    self.traffic_manager.set_desired_speed(actor, random.randint(30, 35))
                    self.traffic_manager.set_route(actor, ['Straight', 'Straight', 'Straight', 'Straight'])
                    self.traffic_manager.ignore_signs_percentage(actor, 100)
                else:
                    self.traffic_manager.set_desired_speed(actor, random.randint(10, 15))

    def _create_behavior(self):
        root = py_trees.composites.Parallel(name="CountryAvoidance", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        c1 = py_trees.composites.Sequence(name="CountryAvoidance_c1")
        c1.add_child(InTriggerDistanceToVehicle(self.other_actors[self.front_index], self.ego_vehicles[0], 8))
        c1.add_child(InTriggerDistanceToVehicle(self.other_actors[self.front_index], self.ego_vehicles[0], 35, comparison_operator=operator.gt))
        root.add_child(c1)
        root.add_child(TimeoutRaiseException(20))
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