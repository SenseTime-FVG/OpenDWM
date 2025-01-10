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


class Construction01Condition(AtomicCondition):
    def __init__(self, debug=False):

        self.scenario_name = 'Construction01'

        self.min_distance_to_junction = 60  #

        super(Construction01Condition, self).__init__(debug)

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
            self.dprint(f'条件4：到前方最近路口的距离为{distance_to_next_junction(wp)}，不处于限制区间[{self.min_distance_to_junction}, inf]')
            return False

        return True


class Construction01(BasicScenario):
    """
    Desc: 场景为前方道路施工，道路为同向双车道以上，前方切角锥桶构成施工区域，自车向左变道绕行，变道后不变回原车道
    Special: 补充避让施工场景的数据
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

        self.data_tags = {}
        self.cone_start_wp = None
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
        self.navi_command = random.choice([1, 2])  # 1: left, 2: straight
        self.start_save_speed_seed = random.random()

        super().__init__("Construction01", ego_vehicles, config, world, randomize, debug_mode,
                         criteria_enable=criteria_enable, uniad=uniad, interfuser=interfuser, assign_distribution=assign_distribution)

    def get_data_tags(self):
        return self.data_tags

    def is_success(self):
        # Desc: 非法行为会抛出异常
        return True

    def sample_drive_plan(self):
        distance_to_junc = distance_to_next_junction(self.starting_wp)
        if distance_to_junc > 150:
            move_distance = random.randint(60, 80)
            self.cone_start_wp = move_waypoint_forward(self.starting_wp, move_distance)
            self.data_tags['scenario_sep_dis'] = move_distance
            self.data_tags['scenario_length'] = move_distance
            return self.drive_planA1
        else:
            move_distance = random.randint(30, 40)
            self.cone_start_wp = move_waypoint_forward(self.starting_wp, move_distance)
            self.data_tags['scenario_sep_dis'] = move_distance
            self.data_tags['scenario_length'] = move_distance
            return self.drive_planB

    def drive_planB(self):
        # Desc: 驾驶方案A
        # Desc: 一开始低速行驶（30-40之间），然后减速到15，距离前方施工区域15米时向左变道，变道后不变回原车道，变完道保持车速30-40
        # Special: 由于不变道回去，导航指令为左转或直行
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
                    'distance': random.randint(23, 28),
                    'speed': init_speed
                },
                'decrease_end_point': {
                    'distance': random.randint(16, 19),
                    'speed': 15
                },
                'lane_change_sign': False,
                'left_lane_cone_end_wp': self.cone_end_wp.get_left_lane(),
                'middle_area_speed': random.randint(15, 20),
                # 使用sigmoid函数变道
                'lane_change_start_point': {
                    'distance': random.randint(14, 16),
                },
                'lane_change_end_point': {
                    'distance': random.randint(5, 8),
                },
                'lane_change_curvature': 0.05,
                'lane_change_sigmoid': {
                    'a': None,
                    'b': float(ego_wp.lane_width),
                }
            })
            self.stage_values = self.update_assign_distribution(self.stage_values)
            self.stage_values['lane_change_sigmoid']['a'] = (-1 * math.log((self.stage_values['lane_change_sigmoid']['b'] / self.stage_values['lane_change_curvature']) - 1)) / ((self.stage_values['lane_change_start_point']['distance'] - self.stage_values['lane_change_end_point']['distance']) / 2.0 * -1)

            self._tf_set_ego_route([self.navi_command])
            self._tf_set_ego_speed(self.stage_values['init_speed'])
            self._tf_disable_ego_auto_lane_change()
            self._set_ego_autopilot()
            self.initialized = True

            self.data_tags['init_speed'] = init_speed
            self.data_tags['decrease_length'] = self.stage_values['decrease_start_point']['distance'] - self.stage_values['decrease_end_point']['distance']
            self.data_tags['lane_change_length'] = self.stage_values['lane_change_start_point']['distance'] - self.stage_values['lane_change_end_point']['distance']

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
            dis2cone_start = distance_between_waypoints(ego_wp, self.cone_start_wp)
            if dis2cone_start > self.stage_values['decrease_start_point']['distance']:
                self._tf_set_ego_speed(self.stage_values['init_speed'])
            elif self.stage_values['decrease_start_point']['distance'] >= dis2cone_start > \
                    self.stage_values['decrease_end_point']['distance']:
                # 线性减速
                p1 = (self.stage_values['decrease_start_point']['distance'], self.stage_values['decrease_start_point']['speed'])
                p2 = (self.stage_values['decrease_end_point']['distance'], self.stage_values['decrease_end_point']['speed'])
                target_speed = calculate_y_for_given_x(dis2cone_start, p1, p2)
                self._tf_set_ego_speed(target_speed)

            if self.stage_values['lane_change_start_point']['distance'] >= dis2cone_start > self.stage_values['lane_change_end_point']['distance']:
                # sigmoid变道
                sigmoid_x = self.stage_values['lane_change_start_point']['distance'] - dis2cone_start - (self.stage_values['lane_change_start_point']['distance'] - self.stage_values['lane_change_end_point']['distance']) / 2.0
                ego_offset = self.stage_values['lane_change_sigmoid']['b'] / (1 + math.exp(-self.stage_values['lane_change_sigmoid']['a'] * sigmoid_x))
                self._tf_set_ego_offset(-ego_offset)
                self._tf_set_ego_force_go(100)
            elif self.stage_values['lane_change_end_point']['distance'] >= dis2cone_start:
                self.stage_values['stage'] = 2
                self.ego_drive_distance = 0
                self._tf_set_ego_offset(-self.stage_values['lane_change_sigmoid']['b'])

        elif self.stage_values['stage'] == 2:
            if self.ego_drive_distance < self.cone_start_wp.transform.location.distance(self.stage_values['left_lane_cone_end_wp'].transform.location) - 5:
                self._tf_set_ego_speed(self.stage_values['middle_area_speed'])
            else:
                self._tf_set_ego_speed(self.stage_values['init_speed'])

        return self.start_save_sign, frame_data

    def drive_planA1(self):
        # Desc: 驾驶方案A
        # Desc: 一开始高速行驶，距离前方施工区域30米时减速，减速到15之间，距离到10m左右时向左变道，变道后不变回原车道，变完道加速到30-40之间，而后恢复到40-50
        # Special: 由于不变道回去，导航指令为左转或直行
        ego = self.ego_vehicles[0]
        ego_loc = ego.get_location()
        ego_wp = self._map.get_waypoint(ego_loc)
        frame_data = {'navi_command': self.navi_command}
        
        if self.initialized is False:
            init_speed = random.randint(35, 45)
            self.stage_values.update({
                'init_speed': init_speed,  # 初始速度
                'stage': 1,  # 阶段
                'decrease_start_point': {  # 减速开始点
                    'distance': random.randint(38, 45),
                    'speed': init_speed
                },
                'decrease_end_point': {  # 减速结束点
                    'distance': random.randint(25, 30),
                    'speed': 15
                },
                'lane_change_sign': False,  # 变道标志
                'left_lane_cone_end_wp': self.cone_end_wp.get_left_lane(),  # 左车道施工区域结束点
                'middle_area_speed': random.randint(15, 20),  # 中间区域速度
                # 使用sigmoid函数变道
                'lane_change_start_point': {
                    'distance': random.randint(20, 25),
                },
                'lane_change_end_point': {
                    'distance': random.randint(6, 10),
                },
                'lane_change_curvature': 0.01,
                'lane_change_sigmoid': {
                    'a': None,
                    'b': float(ego_wp.lane_width),
                }
            })
            self.stage_values = self.update_assign_distribution(self.stage_values)
            self.stage_values['lane_change_sigmoid']['a'] = (-1 * math.log((self.stage_values['lane_change_sigmoid']['b'] / self.stage_values['lane_change_curvature']) - 1)) / ((self.stage_values['lane_change_start_point']['distance'] - self.stage_values['lane_change_end_point']['distance']) / 2.0 * -1)

            self._tf_set_ego_route([self.navi_command])
            self._tf_set_ego_speed(self.stage_values['init_speed'])
            self._tf_disable_ego_auto_lane_change()
            self._set_ego_autopilot()
            self.initialized = True

            self.data_tags['init_speed'] = init_speed
            self.data_tags['decrease_length'] = self.stage_values['decrease_start_point']['distance'] - self.stage_values['decrease_end_point']['distance']
            self.data_tags['lane_change_length'] = self.stage_values['lane_change_start_point']['distance'] - self.stage_values['lane_change_end_point']['distance']

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

        # version: 2.0，采用ego-offset + sigmoid函数变道
        if self.stage_values['stage'] == 1:
            # 还未变道
            dis2cone_start = distance_between_waypoints(ego_wp, self.cone_start_wp)
            if dis2cone_start > self.stage_values['decrease_start_point']['distance']:
                self._tf_set_ego_speed(self.stage_values['init_speed'])
            elif self.stage_values['decrease_start_point']['distance'] >= dis2cone_start > self.stage_values['decrease_end_point']['distance']:
                # 线性减速
                p1 = (self.stage_values['decrease_start_point']['distance'], self.stage_values['decrease_start_point']['speed'])
                p2 = (self.stage_values['decrease_end_point']['distance'], self.stage_values['decrease_end_point']['speed'])
                target_speed = calculate_y_for_given_x(dis2cone_start, p1, p2)
                self._tf_set_ego_speed(target_speed)

            if self.stage_values['lane_change_start_point']['distance'] >= dis2cone_start > self.stage_values['lane_change_end_point']['distance']:
                # sigmoid变道
                sigmoid_x = self.stage_values['lane_change_start_point']['distance'] - dis2cone_start - (self.stage_values['lane_change_start_point']['distance'] - self.stage_values['lane_change_end_point']['distance']) / 2.0
                ego_offset = self.stage_values['lane_change_sigmoid']['b'] / (1 + math.exp(-self.stage_values['lane_change_sigmoid']['a'] * sigmoid_x))
                self._tf_set_ego_offset(-ego_offset)
                self._tf_set_ego_force_go(100)
            elif self.stage_values['lane_change_end_point']['distance'] >= dis2cone_start:
                self.stage_values['stage'] = 2
                self.ego_drive_distance = 0
                self._tf_set_ego_offset(-self.stage_values['lane_change_sigmoid']['b'])

        elif self.stage_values['stage'] == 2:
            if self.ego_drive_distance < self.cone_start_wp.transform.location.distance(self.stage_values['left_lane_cone_end_wp'].transform.location) - 5:
                self._tf_set_ego_speed(self.stage_values['middle_area_speed'])
            else:
                self._tf_set_ego_speed(self.stage_values['init_speed'])

        return self.start_save_sign, frame_data

    def interfuser_tick_autopilot(self):
        return self.drive_plan_func()

    def _initialize_actors(self, config):
        # Depend: 场景前方生成施工区域
        self.cone_end_wp, self.data_tags = _construction01_cones(self.carla_vars, self.cone_start_wp, assign_distribution=self.assign_distribution, input_data_tags=self.data_tags)

        # For: 左侧交通流
        left_traffic_flow_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.5, 'lanes_num': 4, 'skip_num': 1, 'forward_num': random.randint(5, 8), 'backward_num': random.randint(0, 20)},
            gen_cfg={'name_prefix': 'left'},
            assign_distribution=self.assign_distribution
        )
        # For: 对向交通流
        opposite_traffic_flow_scenario(
            self.starting_wp, self.other_actors, self.actor_desc,
            scene_cfg={'filters': '+hcy1', 'idp': 0.4, 'forward_num': random.randint(5, 8), 'backward_num': random.randint(0, 20)},
            gen_cfg={'name_prefix': 'opposite'},
            assign_distribution=self.assign_distribution
        )
        self.traffic_manager = CarlaDataProvider.get_trafficmanager()
        for a_index, (actor, actor_desc) in enumerate(zip(self.other_actors, self.actor_desc)):
            if actor.type_id.startswith('vehicle'):
                actor.set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())
                self.traffic_manager.update_vehicle_lights(actor, True)
                self.traffic_manager.set_desired_speed(actor, random.randint(15, 30))
                if 'left' in actor_desc:
                    self.traffic_manager.auto_lane_change(actor, False)

    def _create_behavior(self):
        root = py_trees.composites.Parallel(name="Construction01", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(DriveDistance(self.ego_vehicles[0], self.data_tags['scenario_length'] + 25))
        root.add_child(TimeoutRaiseException(300))
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

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        pass


def _construction01_cones(carla_vars, wp, assign_distribution=None, input_data_tags=None):
    # Desc: 在指定位置开始生成向左切角施工锥桶
    # For: Construction01
    """
    参数说明
    :param wp: 起始waypoint（位于车道中心）
    :return:

    示意图：
    │    .│---------------------
    │   .θ│         │
    │  .xx│    cut_in_length
    │ .xxx│         │
    │.xxxx│---------------------
    │.xxxx│         │
    │.xxxx│  middle_area_length
    │.xxxx│         │
    │.xxxx│---------------------
    │ .xxx│         │
    │  .xx│    cut_in_length
    │   .θ│         │
    │    .│---------------------
    """

    world = carla_vars['world']
    bp_lib = carla_vars['blueprint_library']
    _map = carla_vars['map']
    # spectator_focus_wp(world.get_spectator(), wp)
    data_tags = {} if input_data_tags is None else input_data_tags
    default_settings = {
        'params': {
            'cut_in_min_angle': 5,  # 最小切角角度
            'cut_in_max_angle': 60,  # 最大切角角度
            'cut_in_cone_distance': 1.5,  # 切角锥桶间距
            'middle_area_cone_distance': 2.0,  # 施工中间区域锥桶间距
            'max_construction_num': 999,  # 最多施工材料区域数量
        },
        'distributions': {
            'theta': 'default',  # 切角角度
            'middle_area_length': 'default',  # 施工中间区域长度
            'cone_type': 'default',  # 锥桶类型
            'construction_material': 'default',  # 建筑材料类型
            'worker_num': 'default',  # 每个建筑材料旁的工人数量
        },
        'probs': {
            'daylight_police': 0.2,  # 白天警车生成概率
            'night_police': 1.0,  # 夜晚警车生成概率
            'traffic_warning': 0.8,  # 交通警示生成概率
            'construction_material': 0.5,  # 建筑材料生成概率
            'worker': 0.3,  # 工人生成概率
        }
    }
    if isinstance(assign_distribution, dict):
        default_settings['params'].update(assign_distribution.get('params', {}))
        default_settings['distributions'].update(assign_distribution.get('distributions', {}))
        default_settings['probs'].update(assign_distribution.get('probs', {}))

    # 施工中间区域最小长度（固定）
    fixed_min_middle_area_length = 15
    # 距离下一个路口的距离（留出14m余量，固定）
    distance_to_junc = distance_to_next_junction(wp) - 14
    # 道路宽度
    road_width = wp.lane_width

    if default_settings['distributions']['theta'] == 'default':
        if distance_to_junc < 0:
            print(f'当前位置已经距离路口小于15m，无法生成施工区域')
            return None
        if distance_to_junc <= fixed_min_middle_area_length:
            print(f'距离下一个路口的距离小于最小施工区域长度{fixed_min_middle_area_length}米，无法生成施工区域')
            return None
        # 最小角度（最小5度，最大90度）
        min_theta = max(min(math.degrees(math.atan((2 * road_width) / (distance_to_junc - fixed_min_middle_area_length))), default_settings['params']['cut_in_max_angle']), default_settings['params']['cut_in_min_angle'])
        # Special: 切角等概率采样
        theta_distribution = {i: 1.0 for i in range(math.ceil(min_theta), default_settings['params']['cut_in_max_angle'] + 1)}
        theta = do_sample(theta_distribution)
    else:
        theta = do_sample(default_settings['distributions']['theta'])
    data_tags['theta'] = theta

    # 切角长度
    cut_in_length = math.floor(road_width / math.tan(math.radians(theta)))
    if default_settings['distributions']['middle_area_length'] == 'default':
        # Special: 施工区域宽度等概率采样
        min_middle_area_length = fixed_min_middle_area_length
        max_middle_area_length = min(distance_to_junc - 2 * cut_in_length, 50)
        middle_area_length_distribution = {i: 1.0 for i in range(min_middle_area_length, max_middle_area_length + 1)}
        middle_area_length = do_sample(middle_area_length_distribution)
    else:
        middle_area_length = do_sample(default_settings['distributions']['middle_area_length'])
    suitable_middle_area_length = middle_area_length - 5
    data_tags['middle_area_length'] = middle_area_length

    if default_settings['distributions']['cone_type'] == 'default':
        # 锥桶类型
        cone_type = do_sample({'static.prop.constructioncone': 0.5, 'static.prop.trafficcone01': 0.5})
    else:
        cone_type = default_settings['distributions']['cone_type']
    # 锥桶蓝图
    cone_bp = bp_lib.find(cone_type)

    # Effect: 生成进入切角施工锥桶
    # Parameter1: 两个锥桶之间的距离
    cut_in_cone_distance = default_settings['params']['cut_in_cone_distance']
    accumulate_distance = 0
    cur_cut_in_wp = wp
    wp_move_forward_distance = cut_in_cone_distance * math.cos(math.radians(theta))
    while accumulate_distance * math.sin(math.radians(theta)) < road_width:
        right_vector = cur_cut_in_wp.transform.rotation.get_right_vector()
        left_vector = right_vector * -1

        road_right_loc = cur_cut_in_wp.transform.location + road_width / 2 * right_vector
        left_vec_coefficient = accumulate_distance * math.sin(math.radians(theta))
        left_multiply_vector = left_vec_coefficient * left_vector
        cur_cone_loc = road_right_loc + left_multiply_vector

        world.spawn_actor(cone_bp, carla.Transform(cur_cone_loc, carla.Rotation()))
        accumulate_distance += cut_in_cone_distance
        cur_cut_in_wp = cur_cut_in_wp.next(wp_move_forward_distance)[0]

    construction_obj_wp = cur_cut_in_wp
    # Effect: 生成施工中间区域锥桶
    # Parameter2: 两个锥桶之间的距离
    middle_area_cone_distance = default_settings['params']['middle_area_cone_distance']
    accumulate_distance = 0
    cur_middle_wp = cur_cut_in_wp
    wp_move_forward_distance = middle_area_cone_distance
    while accumulate_distance < middle_area_length:
        left_vector = cur_middle_wp.transform.rotation.get_right_vector() * -1
        cur_cone_loc = cur_middle_wp.transform.location + (road_width * 9.0 / 20.0) * left_vector
        world.spawn_actor(cone_bp, carla.Transform(cur_cone_loc, carla.Rotation()))
        accumulate_distance += middle_area_cone_distance
        cur_middle_wp = cur_middle_wp.next(wp_move_forward_distance)[0]

    # Effect: 生成离开切角施工锥桶
    last_cone_wp = None
    step3_distance = cut_in_cone_distance
    accumulate_distance = 0
    cur_cut_out_wp = cur_middle_wp
    wp_move_forward_distance = step3_distance * math.cos(math.radians(theta))
    while accumulate_distance * math.sin(math.radians(theta)) < road_width + step3_distance / 2.0:
        right_vector = cur_cut_out_wp.transform.rotation.get_right_vector()

        road_left_loc = cur_cut_out_wp.transform.location - road_width * 9.0 / 20.0 * right_vector
        right_vec_coefficient = accumulate_distance * math.sin(math.radians(theta))
        right_multiply_vector = right_vec_coefficient * right_vector
        cur_cone_loc = road_left_loc + right_multiply_vector

        world.spawn_actor(cone_bp, carla.Transform(cur_cone_loc, carla.Rotation()))
        accumulate_distance += step3_distance
        cur_cut_out_wp = cur_cut_out_wp.next(wp_move_forward_distance)[0]
        last_cone_wp = cur_cut_out_wp

    obj_accumulate_distance = 0

    # Effect: 生成police car
    # 获取当前天气中的太阳高度角
    weather_parameters = world.get_weather()
    sun_altitude_angle = weather_parameters.sun_altitude_angle
    # Parameter3: 生成概率
    if sun_altitude_angle < 10:
        police_prob = default_settings['probs']['night_police']
        data_tags['light'] = 'night'
    else:
        police_prob = default_settings['probs']['daylight_police']
        data_tags['light'] = 'day'
    if random.random() < police_prob and obj_accumulate_distance < suitable_middle_area_length:
        police_bp = random.choice(list(bp_lib.filter('*police*')))
        police_wp = construction_obj_wp.next(3)[0]
        right_vector = police_wp.transform.rotation.get_right_vector() * 0.125
        police_loc = police_wp.transform.location + carla.Location(z=0.5) + right_vector
        police_rot = carla.Rotation(yaw=police_wp.transform.rotation.yaw+180, roll=police_wp.transform.rotation.roll, pitch=police_wp.transform.rotation.pitch)
        police_actor = world.spawn_actor(police_bp, carla.Transform(police_loc, police_rot))
        police_actor.set_light_state(carla.VehicleLightState(carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.RightBlinker | carla.VehicleLightState.Reverse | carla.VehicleLightState.Interior | carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2 | carla.VehicleLightState.LowBeam))
        construction_obj_wp = police_wp
        obj_accumulate_distance += 3
        data_tags['police'] = 1
    else:
        data_tags['police'] = 0

    # Effect: 生成traffic warning
    # Parameter4: 生成概率
    tw_prob = default_settings['probs']['traffic_warning']
    if random.random() < tw_prob and obj_accumulate_distance < suitable_middle_area_length:
        tw_bp = bp_lib.find('static.prop.trafficwarning')
        tw_wp = construction_obj_wp.next(4)[0]
        tw_actor = world.spawn_actor(tw_bp, carla.Transform(tw_wp.transform.location, carla.Rotation(yaw=tw_wp.transform.rotation.yaw-90, roll=tw_wp.transform.rotation.roll, pitch=tw_wp.transform.rotation.pitch)))
        construction_obj_wp = tw_wp
        obj_accumulate_distance += 4

    construction_area_count = 0
    worker_count = 0
    while obj_accumulate_distance < suitable_middle_area_length and construction_area_count < default_settings['params']['max_construction_num']:
        # Effect: 生成建筑材料
        # Parameter5: 生成概率
        construction_material_prob = default_settings['probs']['construction_material']
        construction_wp = construction_obj_wp.next(3)[0]
        if random.random() < construction_material_prob and obj_accumulate_distance < middle_area_length:
            if default_settings['distributions']['construction_material'] == 'default':
                construction_material_type = do_sample({'steel': 1.0, 'barrier': 1.0, 'haybale': 1.0})
            else:
                construction_material_type = default_settings['distributions']['construction_material']
            if construction_material_type == 'steel':
                steel_num = do_sample({i: 1.0 for i in range(3, 16)})
                steel_intrinsic_params = {
                    'length': 0.725029,
                    'width': 0.588164,
                    'height': 0.016910,
                }
                steel_bp = bp_lib.find('static.prop.ironplank')
                random_steel_yaw = random.randint(0, 360)
                for i in range(steel_num):
                    steel_loc = construction_wp.transform.location + carla.Location(z=0.1*i)
                    steel_actor = world.spawn_actor(steel_bp, carla.Transform(steel_loc, carla.Rotation(yaw=random_steel_yaw, roll=construction_wp.transform.rotation.roll, pitch=construction_wp.transform.rotation.pitch)))
                    sleep(0.1)
            elif construction_material_type == 'barrier':
                barrier_num = do_sample({i: 1.0 for i in range(1, 2)})
                barrier_bp = bp_lib.find('static.prop.streetbarrier')
                for i in range(barrier_num):
                    random_barrier_yaw = random.randint(0, 360)
                    barrier_loc = construction_wp.transform.location + carla.Location(z=0.1 * i)
                    barrier_actor = world.spawn_actor(barrier_bp, carla.Transform(barrier_loc, carla.Rotation(yaw=random_barrier_yaw, roll=construction_wp.transform.rotation.roll, pitch=construction_wp.transform.rotation.pitch)))
                    barrier_actor.set_simulate_physics(True)
                    sleep(0.2)
            elif construction_material_type == 'haybale':
                haybale_bp = random.choice(list(bp_lib.filter('haybale')))
                random_barrier_yaw = random.randint(0, 360)
                haybale_loc = construction_wp.transform.location + carla.Location(z=1)
                haybale_actor = world.spawn_actor(haybale_bp, carla.Transform(haybale_loc, carla.Rotation(yaw=random_barrier_yaw, roll=construction_wp.transform.rotation.roll, pitch=construction_wp.transform.rotation.pitch)))
                haybale_actor.set_simulate_physics(True)
            construction_area_count += 1
        construction_obj_wp = construction_wp

        # Effect: 生成工人
        # Parameter6: 生成概率
        worker_prob = default_settings['probs']['worker']
        if random.random() < worker_prob:
            worker_wp = construction_obj_wp
            if default_settings['distributions']['worker_num'] == 'default':
                worker_num = do_sample({i: 1.0 for i in range(1, 3)})
            else:
                worker_num = default_settings['distributions']['worker_num']

            offset_unit = road_width / 2.0 * 3 / 4
            right_vector = worker_wp.transform.rotation.get_right_vector()
            forward_vector = worker_wp.transform.rotation.get_forward_vector()
            for worker_index in range(worker_num):
                horizontal_offset_coefficient = random.uniform(-offset_unit, offset_unit)
                vertical_offset_coefficient = random.uniform(-1.5, 1.5)
                blueprint = bp_lib.find(choose_pedestrian_bp_name('+adult+man+middle-women-police-fat-old'))
                worker_loc = worker_wp.transform.location + horizontal_offset_coefficient * right_vector + vertical_offset_coefficient * forward_vector + carla.Location(z=0.5)
                worker_rot = carla.Rotation(yaw=random.randint(0, 360), roll=0, pitch=0)
                worker = world.try_spawn_actor(blueprint, carla.Transform(worker_loc, worker_rot))
                sleep(0.1)
                worker_count += 1
                if worker is not None:
                    worker_real_loc = worker.get_location()
                    if _map.get_waypoint(worker_real_loc).lane_id != worker_wp.lane_id:
                        print(f'destroy worker {worker.id}')
                        worker.destroy()
                        worker_count -= 1
                        continue

                    worker.set_simulate_physics(True)
                # if worker is not None:
                #     worker_real_wp = _map.get_waypoint(worker.get_location())
                #     if worker_real_wp is not None and worker_real_wp.lane_id != worker_wp.lane_id:
                #         worker.destroy()
                #         sleep(0.1)
                #         worker_count -= 1
                #         continue

        obj_accumulate_distance += 3
    data_tags['worker_num'] = worker_count
    data_tags['scenario_length'] += 2 * cut_in_length + middle_area_length

    return last_cone_wp, data_tags


