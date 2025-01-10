from __future__ import print_function

import random

import py_trees

from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance, StandStill)
from Initialization.Standardimport.basic_scenario import BasicScenario

from Explainablelibrary.explainable_utils import *
from Explainablelibrary.basic_desc import *
from Scenarios.Tinyscenarios import *


class FrontBrake(BasicScenario):
    """
    Desc: 前车运动刹车，自车也刹车
    Special: 补充刹车的数据
    """
    
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=180, uniad=False, interfuser=False):
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
        self.initialized = False
        self.lane_changed = False
        self.init_speed = 40
        self.navigation_cmds = ['straight']
        
        # For: Is success
        self.passed_lane_ids = []
        
        # For: V5 description
        self.carla_desc = CarlaDesc(world, ego_vehicles[0])

        # For: autopilot stage
        self.stages = {
            '#unknown': (UNKNOWN, [], UNKNOWN, [], ''),
        }

        # For: Front car control
        self.front_car_last_loc = None
        self.front_car_drive_distance = 0
        self.front_car_brake_sign = False

        super().__init__("FrontBrake", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser)

    def is_success(self):
        # Desc: Autopilot是否按照预期行驶
        return True, ''

    def interfuser_tick_autopilot(self):
        ego = self.ego_vehicles[0]
        ego_speed = round(math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6, 2)

        if self.front_car is None:
            self.front_car = self.other_actors[self.actor_desc.index('front')]

        if self.initialized is False:
            self._tf_set_ego_route(self.navigation_cmds)
            self._tf_set_ego_speed(self.init_speed)
            self._set_ego_autopilot()
            self.initialized = True

        explainable_data = {
            'actors': build_actor_data(ego, self.other_actors),
            'actors_desc': self.actor_desc,
        }
        
        cur_ego_loc = ego.get_location()
        cur_wp = self._map.get_waypoint(cur_ego_loc)
        front_car_wp = self._map.get_waypoint(self.front_car.get_location())
        front_car_loc = self.front_car.get_location()
        if self.front_car_last_loc is not None:
            self.front_car_drive_distance += front_car_loc.distance(self.front_car_last_loc)
        self.front_car_last_loc = front_car_loc
        
        info = f'Speed: {int(ego_speed)} km/h '

        ego_stage = '#unknown'
        reason = ''
        if self.front_car.is_alive:
            # desc = Edict(explainable_data['actors'][self.front_car.id]['description'])
            distance = round(self.front_car.get_location().distance(cur_ego_loc), 2)
            info += f'Distance: {distance}m '

            if self.front_car_brake_sign is False and self.front_car_drive_distance > 20:
                self.front_car_brake_sign = True
                self.traffic_manager.set_desired_speed(self.front_car, 0)
                print(f'\n++++++++++++++++\n{self.front_car} brake.')

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

        # info += f'{ego_stage} -> 可解释性描述：{reason}'
        # hanzi_num = len(re.findall(r'[\u4e00-\u9fa5]', info))
        # info += ' ' * (150 - hanzi_num * 2 - (len(info) - hanzi_num))
        # print(f'\r{info}', end='')

        return explainable_data
        
    def _initialize_actors(self, config):
        # For: 在自车前方50米生成一辆车
        _first_vehicle_wp = move_waypoint_forward(self.starting_wp, random.randint(15, 25))
        front_bp_name = choose_bp_name('+wheel4-special')
        front_actor = CarlaDataProvider.request_new_actor(front_bp_name, _first_vehicle_wp.transform)
        self.other_actors.append(front_actor)
        self.actor_desc.append('front')
        self.front_index = len(self.other_actors) - 1

        # For: 右侧交通流
        right_traffic_flow_scenario(self.starting_wp, self.other_actors, self.actor_desc, scene_cfg={'filters': '+hcy1', 'idp': 0.4}, gen_cfg={'name_prefix': 'right'})
        # For: 左侧交通流
        left_traffic_flow_scenario(self.starting_wp, self.other_actors, self.actor_desc, scene_cfg={'filters': '+hcy1', 'idp': 0.4}, gen_cfg={'name_prefix': 'left'})
        # For: 对向交通流
        opposite_traffic_flow_scenario(self.starting_wp, self.other_actors, self.actor_desc, scene_cfg={'filters': '+hcy1', 'idp': 0.4, 'backward_num': random.randint(10, 30)}, gen_cfg={'name_prefix': 'opposite'})
        # For: 路边停靠车辆
        right_parking_vehicle_scenario(self.starting_wp, self.other_actors, self.actor_desc, scene_cfg={'filters': '+wheel4-large', 'idp': 0.4, 'forward_num': random.randint(8, 15)}, gen_cfg={'name_prefix': 'park'})

        self.traffic_manager = CarlaDataProvider.get_trafficmanager()
        for a_index, (actor, actor_desc) in enumerate(zip(self.other_actors, self.actor_desc)):
            if 'park' in actor_desc:
                continue
            if actor.type_id.startswith('vehicle'):
                actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
                self.traffic_manager.update_vehicle_lights(actor, True)
                if a_index == 0:
                    self.traffic_manager.set_desired_speed(actor, random.randint(35, 45))
                else:
                    self.traffic_manager.set_desired_speed(actor, random.randint(30, 45))

    def _create_behavior(self):
        root = py_trees.composites.Parallel(name="FrontBrake", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        c1 = py_trees.composites.Sequence(name="FrontBrake_c1")
        c1.add_child(DriveDistance(self.other_actors[self.front_index], 15))
        c1.add_child(DriveDistance(self.ego_vehicles[0], 3))
        c1.add_child(StandStill(self.other_actors[self.front_index], name="StandStill", duration= 5))
        c1.add_child(StandStill(self.ego_vehicles[0], name="StandStill", duration=5))
        root.add_child(c1)
        root.add_child(StandStill(self.ego_vehicles[0], name="StandStill", duration=6))
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

    