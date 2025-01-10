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
from Logger import *


class TrafficOccupation(BasicScenario):
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
        self.avoid_sign = False
        self.stage = 0
        self.init_speed = random.randint(20, 30)
        self.navigation_cmds = ['Straight']
        self.obstacle_index = -1

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
            '#dynamic3': (LEFT_C, [], DEC, [], ''),
            '#fix3': (KEEP_L, [], ACC, [], ''),
            '#dynamic4': (RIGHT_C, [], KEEP_S, [], ''),
            '#fix4': (KEEP_L, [], ACC, [], ''),
            '#dynamic5': (RIGHT_C, [], DEC, [], ''),
            '#fix5': (KEEP_L, [], KEEP_S, [], ''),
            '#fix6': (KEEP_L, [], KEEP_S, [], ''),
        }

        self.front_car_category = 'bus'

        super().__init__("TrafficOccupation", ego_vehicles, config, world, randomize, debug_mode,
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

        if self.front_car is None:
            self.front_car = self.other_actors[self.actor_desc.index('front')]
            self.front_car_wp = self._map.get_waypoint(self.front_car.get_location())
            self.front_car_loc = self.front_car.get_location()

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
        # front_car_wp = self._map.get_waypoint(self.front_car.get_location())
        # front_car_loc = self.front_car.get_location()

        if not cur_wp.is_junction and not cur_wp.is_intersection:
            self.passed_lane_ids.append(cur_wp.lane_id)

        info = f'Speed: {int(ego_speed)} km/h '
        if self.front_car.is_alive:
            desc = Edict(explainable_data['actors'][self.front_car.id]['description'])
            data = Edict(explainable_data['actors'][self.front_car.id])
            distance = round(self.front_car.get_location().distance(cur_ego_loc), 2)
            info += f'Distance: {distance}m '

            if self.front_car_category == 'car':
                decrease_p1 = (20, self.init_speed)
                decrease_p2 = (8, 15)
                increase_p1 = (3, 15)
                increase_p2 = (15, self.init_speed)
            elif self.front_car_category == 'bus':
                decrease_p1 = (25, self.init_speed)
                decrease_p2 = (12, 15)
                increase_p1 = (5, 15)
                increase_p2 = (17, self.init_speed)
            else:
                raise ValueError('Unknown front car category')

            if actor_front_of_ego(data):
                ego_stage = '#fix1'
                reason = '前方障碍车辆，准备绕行'
                if distance > decrease_p1[0]:
                    self._tf_set_ego_speed(self.init_speed)
                elif distance > decrease_p2[0]:
                    ego_stage = '#fix2'
                    reason = '前方有车，减速'
                    target_speed = (self.init_speed - decrease_p2[1]) / (decrease_p1[0] - decrease_p2[0]) * (distance - decrease_p2[0]) + decrease_p2[1]
                    # print(f'目标速度：{target_speed}')
                    self._tf_set_ego_speed(target_speed)
                else:
                    if self.avoid_sign is False:
                        self._tf_set_ego_speed(decrease_p2[1])
                        offset_m = cur_wp.lane_width / 4.0
                        self._tf_set_ego_offset(-offset_m)
                        self.traffic_manager.ignore_vehicles_percentage(ego, 100)
                        self.avoid_sign = True
                        message(f'向左避让{offset_m}米')
                    self._tf_set_ego_speed(decrease_p2[1])
            else:
                ego_stage = '#fix2'
                reason = '完成绕行'
                if distance > increase_p1[0] and self.avoid_sign is True:
                    self._tf_set_ego_offset(0)
                    self.traffic_manager.ignore_vehicles_percentage(ego, 0)
                    self.avoid_sign = False
                    message('恢复车道')

                if distance < increase_p2[0]:
                    target_speed = (self.init_speed - increase_p1[1]) / (increase_p2[0] - increase_p1[0]) * (distance - increase_p1[0]) + increase_p1[1]
                    # print(f'目标速度：{target_speed}')
                    self._tf_set_ego_speed(target_speed)
                else:
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
        # For: 在自车前方50米生成一辆车
        change_wp_transform_v1, _ = trans_wp_v2(self.starting_wp, p1_range=(25, 50))

        veh_bp = choose_bp_name(f'+{self.front_car_category}')
        # 请求新的 actor
        front_actor = CarlaDataProvider.request_new_actor(veh_bp, change_wp_transform_v1)
        self.other_actors.append(front_actor)
        self.actor_desc.append('front')
        self.front_index = len(self.other_actors) - 1
        # ipdb.set_trace()
        # next_actor = CarlaDataProvider.request_new_actor(veh_bp, change_wp_transform_v2)
        # self.other_actors.append(next_actor)
        # self.actor_desc.append('next')
        # self.next_index = len(self.other_actors) - 1

        # world = self._world
        # ego_location = self.starting_wp
        # vehicle_breakdown(world, ego_location)
        # road_construction(world, ego_location)

        # For: 后方车流
        behind_traffic_flow_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.5, 'backward_num': random.randint(1, 5)}, gen_cfg={'name_prefix': 'behind'}
        )
        # For: 前方车流
        front_traffic_flow_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.0, 'forward_num': random.randint(1, 5)}, gen_cfg={'name_prefix': 'front'}
        )
        # For: 右侧交通流
        right_traffic_flow_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.4, 'lanes_num': 2}, gen_cfg={'name_prefix': 'right'}
        )
        # For: 左侧交通流
        left_traffic_flow_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.4, 'lanes_num': 2}, gen_cfg={'name_prefix': 'left'}
        )
        # For: 对向交通流
        opposite_traffic_flow_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.4, 'backward_num': random.randint(0, 10)},
            gen_cfg={'name_prefix': 'opposite'}
        )
        # For: 路边停靠车辆
        right_parking_vehicle_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+wheel4-large', 'idp': 0.4, 'forward_num': 10},
            gen_cfg={'name_prefix': 'park'}
        )
        self.traffic_manager = CarlaDataProvider.get_trafficmanager()
        for a_index, (actor, actor_desc) in enumerate(zip(self.other_actors, self.actor_desc)):
            if actor.type_id.startswith('vehicle'):
                if a_index == 0 or 'park' in actor_desc:
                    actor.apply_control(carla.VehicleControl(brake=1.0))
                    continue
                actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
                self.traffic_manager.update_vehicle_lights(actor, True)
                self.traffic_manager.set_desired_speed(actor, random.randint(15, 30))

    def _create_behavior(self):
        root = py_trees.composites.Parallel(name="TrafficOccupation",
                                            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        c1 = py_trees.composites.Sequence(name="TrafficOccupation_c1")
        # c1.add_child(InTriggerDistanceToVehicle(self.other_actors[self.front_index], self.ego_vehicles[0], 15))
        # c1.add_child(StandStill(self.ego_vehicles[0], name='ego_standstill', duration=2))
        c1.add_child(InTriggerDistanceToVehicle(self.other_actors[self.front_index], self.ego_vehicles[0], 10))
        c1.add_child(InTriggerDistanceToVehicle(self.other_actors[self.front_index], self.ego_vehicles[0], 30,
                                                comparison_operator=operator.gt))
        root.add_child(c1)
        root.add_child(TimeoutRaiseException(300))
        root.add_child(CollisionTest(self.ego_vehicles[0], terminate_on_failure=True))
        root.add_child(InTriggerDistanceToNextIntersection(self.ego_vehicles[0], 8))
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