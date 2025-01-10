# -*- coding: utf-8 -*-
# @Time    : 2023/11/4 下午8:35
# @Author  : Hcyang
# @File    : tiny_scenarios.py
# @Desc    : xxx

import re
import random
import carla
from Initialization.Standardimport.scenariomanager.carla_data_provider import CarlaDataProvider
from functools import reduce
import math
# from functions import *
from .walker_manager import *


def do_sample(data_dict):
    return random.choices(list(data_dict.keys()), weights=list(data_dict.values()), k=1)[0]


# # import ipdb


# Special: 参数说明
# - wp: 参考waypoint
# - actor_list: 生成的actor列表
# - actor_desc: 生成的actor描述
# - scene_cfg: 生成车辆的配置
#   - filters: actor蓝图的过滤条件           | '+common'
#   - idp: 生成车辆的概率                    | 0.5
#   - lane_num: 生成车辆的车道数             | 999
#   - self_lane: 是否生成在当前车道上        | False
#   - forward_num: 向前生成车辆的数量         | 6
#   - backward_num: 向后生成车辆的数量        | 4
# - gen_cfg: 生成车辆的配置
#   - name_prefix: 生成车辆的前缀            | 'vehicle'


VEHICLE_TYPE_DICT = {
    'vehicle.audi.a2': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.audi.etron': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.audi.tt': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.bmw.grandtourer': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.chevrolet.impala': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.citroen.c3': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.dodge.charger_2020': ['car', 'wheel4', 'common', 'hcy1','door'],
    'vehicle.dodge.charger_police': ['car', 'special', 'police', 'wheel4'],
    'vehicle.dodge.charger_police_2020': ['car', 'special', 'police', 'wheel4'],
    'vehicle.ford.crown': ['car', 'wheel4', 'common', 'hcy1','door'],
    'vehicle.ford.mustang': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.jeep.wrangler_rubicon': ['car', 'suv', 'wheel4', 'common', 'hcy1'],
    'vehicle.lincoln.mkz_2017': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.lincoln.mkz_2020': ['car', 'wheel4', 'common', 'hcy1','door'],
    'vehicle.mercedes.coupe': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.mercedes.coupe_2020': ['car', 'wheel4', 'common', 'hcy1','door'],
    'vehicle.micro.microlino': ['car', 'small', 'wheel4'],
    'vehicle.mini.cooper_s': ['wheel4', 'common'],
    'vehicle.mini.cooper_s_2021': ['wheel4', 'common'],
    'vehicle.nissan.micra': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.nissan.patrol': ['car', 'suv', 'wheel4', 'common', 'hcy1'],
    'vehicle.nissan.patrol_2021': ['car', 'suv', 'wheel4', 'common', 'hcy1','door'],
    'vehicle.seat.leon': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.tesla.model3': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.toyota.prius': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.carlamotors.carlacola': ['truck', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.carlamotors.firetruck': ['truck', 'special', 'fire', 'large', 'wheel4'],
    'vehicle.tesla.cybertruck': ['truck', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.ford.ambulance': ['van', 'special', 'ambulance', 'large', 'wheel4'],
    'vehicle.mercedes.sprinter': ['van', 'large', 'wheel4', 'common', 'hcy1','door'],
    'vehicle.volkswagen.t2': ['bus', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.volkswagen.t2_2021': ['bus', 'large', 'wheel4', 'common', 'hcy1','door'],
    'vehicle.mitsubishi.fusorosa': ['bus', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.harley-davidson.low_rider': ['moto', 'wheel2', 'common'],
    'vehicle.kawasaki.ninja': ['moto', 'wheel2', 'common'],
    'vehicle.vespa.zx125': ['electric', 'wheel2'],
    'vehicle.yamaha.yzf': ['moto', 'wheel2', 'common'],
    'vehicle.bh.crossbike': ['bicycle', 'wheel2'],
    'vehicle.diamondback.century': ['bicycle', 'wheel2'],
    'vehicle.gazelle.omafiets': ['bicycle', 'wheel2'],
}
TYPE_VEHICLE_DICT = {}
for bp_name_outside, bp_filters_outside in VEHICLE_TYPE_DICT.items():
    for bp_filter_outside in bp_filters_outside:
        if bp_filter_outside not in TYPE_VEHICLE_DICT:
            TYPE_VEHICLE_DICT[bp_filter_outside] = []
        TYPE_VEHICLE_DICT[bp_filter_outside].append(bp_name_outside)

PEDESTRIAN_TYPE_DICT = {
    'walker.pedestrian.0002': ['adult', 'man', 'middle'],
    'walker.pedestrian.0003': ['adult', 'man', 'middle'],
    'walker.pedestrian.0004': ['adult', 'man', 'middle'],
    'walker.pedestrian.0001': ['adult', 'women', 'middle'],
    'walker.pedestrian.0005': ['adult', 'women', 'middle'],
    'walker.pedestrian.0006': ['adult', 'women', 'middle'],
    'walker.pedestrian.0007': ['adult', 'women', 'middle'],
    'walker.pedestrian.0008': ['adult', 'women', 'middle'],
    'walker.pedestrian.0015': ['adult', 'women', 'old'],
    'walker.pedestrian.0019': ['adult', 'women', 'old'],
    'walker.pedestrian.0016': ['adult', 'man', 'old'],
    'walker.pedestrian.0017': ['adult', 'man', 'old'],
    'walker.pedestrian.0026': ['adult', 'man', 'middle'],
    'walker.pedestrian.0018': ['adult', 'man', 'middle'],
    'walker.pedestrian.0021': ['adult', 'women', 'middle'],
    'walker.pedestrian.0020': ['adult', 'women', 'middle'],
    'walker.pedestrian.0023': ['adult', 'women', 'middle'],
    'walker.pedestrian.0022': ['adult', 'women', 'middle'],
    'walker.pedestrian.0024': ['adult', 'man', 'middle'],
    'walker.pedestrian.0025': ['adult', 'man', 'middle'],
    'walker.pedestrian.0027': ['adult', 'man', 'middle'],
    'walker.pedestrian.0029': ['adult', 'man', 'middle'],
    'walker.pedestrian.0028': ['adult', 'man', 'middle'],
    'walker.pedestrian.0041': ['adult', 'women', 'middle'],
    'walker.pedestrian.0040': ['adult', 'women', 'middle'],
    'walker.pedestrian.0033': ['adult', 'women', 'middle'],
    'walker.pedestrian.0031': ['adult', 'women', 'middle'],
    'walker.pedestrian.0034': ['adult', 'man', 'middle', 'fat'],
    'walker.pedestrian.0038': ['adult', 'man', 'middle', 'fat'],
    'walker.pedestrian.0035': ['adult', 'women', 'middle', 'fat'],
    'walker.pedestrian.0036': ['adult', 'women', 'middle', 'fat'],
    'walker.pedestrian.0037': ['adult', 'women', 'middle', 'fat'],
    'walker.pedestrian.0039': ['adult', 'man', 'middle'],
    'walker.pedestrian.0042': ['adult', 'women', 'middle', 'fat'],
    'walker.pedestrian.0043': ['adult', 'women', 'middle', 'fat'],
    'walker.pedestrian.0044': ['adult', 'women', 'middle', 'fat'],
    'walker.pedestrian.0047': ['adult', 'women', 'old'],
    'walker.pedestrian.0046': ['adult', 'women', 'old'],
    'walker.pedestrian.0030': ['adult', 'man', 'middle', 'police'],
    'walker.pedestrian.0032': ['adult', 'women', 'middle', 'police'],
}
TYPE_PEDESTRIAN_DICT = {}
for bp_name_outside, bp_filters_outside in PEDESTRIAN_TYPE_DICT.items():
    for bp_filter_outside in bp_filters_outside:
        if bp_filter_outside not in TYPE_PEDESTRIAN_DICT:
            TYPE_PEDESTRIAN_DICT[bp_filter_outside] = []
        TYPE_PEDESTRIAN_DICT[bp_filter_outside].append(bp_name_outside)


def choose_bp_name(filters):
    """
    Desc: 根据车辆类型和车轮数选择对应的blueprint
    @param filters: +x: 添加类型 -x: 排除类型，按顺序计算
    """
    # Special: 类型说明
    # car: 轿车
    # suv: SUV
    # truck: 卡车
    # van: 箱型车
    # bus: 巴士
    # moto: 摩托车
    # electric: 电瓶车
    # bicycle: 自行车
    # special: 特种车辆
    # police: 警车
    # fire: 消防车
    # wheel2: 两轮车辆
    # wheel4: 四轮车辆
    # large: 大型车辆
    # small: 小型车辆
    # common: 常见车辆：排除了特种车辆和自行车和小型车辆
    # hcy1: huchuanyang自定义的车辆集合

    # e.g. +wheel4-special
    filters = [item.strip() for item in re.split(r'([+\-])', filters.strip()) if item.strip()]

    # 不能为单数
    if len(filters) % 2 != 0:
        return ""

    candidate_bp_names = []
    for index in range(0, len(filters), 2):
        op = filters[index]
        filter_type = filters[index + 1]
        if op == '+':
            candidate_bp_names.extend(TYPE_VEHICLE_DICT[filter_type])
        elif op == '-':
            candidate_bp_names = list(set(candidate_bp_names) - set(TYPE_VEHICLE_DICT[filter_type]))
        else:
            print(f'Error: {op} is not supported in blueprint choosing.')
            return ""

    if len(candidate_bp_names) == 0:
        print(f'Error: candidate_bp_names is empty.')
        return ""

    return random.choice(candidate_bp_names)


def choose_pedestrian_bp_name(filters):
    """
    Desc: 根据行人类型选择对应的blueprint
    @param filters: +x: 添加类型 -x: 排除类型，按顺序计算
    """
    # Special: 类型说明
    # adult: 成年人
    # man: 男性
    # women: 女性
    # old: 老年人
    # fat: 胖人
    # middle: 中年人
    # police: 警察

    # e.g. +adult
    filters = [item.strip() for item in re.split(r'([+\-])', filters.strip()) if item.strip()]

    # 不能为单数
    if len(filters) % 2 != 0:
        return ""

    candidate_bp_names = []
    for index in range(0, len(filters), 2):
        op = filters[index]
        filter_type = filters[index + 1]
        if op == '+':
            candidate_bp_names.extend(TYPE_PEDESTRIAN_DICT[filter_type])
        elif op == '-':
            candidate_bp_names = list(set(candidate_bp_names) - set(TYPE_PEDESTRIAN_DICT[filter_type]))
        else:
            print(f'Error: {op} is not supported in blueprint choosing.')
            return ""

    if len(candidate_bp_names) == 0:
        print(f'Error: candidate_bp_names is empty.')
        return ""

    return random.choice(candidate_bp_names)


def _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, name_prefix='vehicle'):
    offset_index = 0
    for v_index, (v_bp, v_transform) in enumerate(bp_and_transforms):
        right_actor = CarlaDataProvider.request_new_actor(v_bp, v_transform, retry=1)
        if right_actor is not None:
            actor_list.append(right_actor)
            actor_desc.append('_'.join([name_prefix, str(v_index - offset_index)]))
        else:
            offset_index += 1


def _warning_unused_keys(kwargs):
    for k in kwargs:
        print(f'Warning: Unused key {k} in kwargs')


def _traffic_flow_scenario(wp, filters='+common', idp=0.5, forward_num=6, backward_num=4, skip_cur_loc=False, **kwargs):
    # Desc: 在当前waypoint的左侧车道或者右侧车道生成车流
    results = []

    # Desc: 先向前生成车流
    _vehicle_wp = wp
    right_forward_index = 1
    while right_forward_index <= forward_num:
        bp_name = choose_bp_name(filters)
        if right_forward_index == 1:
            if skip_cur_loc:
                pass
            else:
                if random.random() < idp:
                    results.append((bp_name, _vehicle_wp.transform))
        else:
            if random.random() < idp:
                results.append((bp_name, _vehicle_wp.transform))
        _vehicle_wps = _vehicle_wp.next(random.randint(8, 15))
        if len(_vehicle_wps) == 0:
            break
        _vehicle_wp = _vehicle_wps[0]
        right_forward_index += 1

    # Desc: 再向后生成车流
    _vehicle_wp = wp
    right_backward_index = 1
    while right_backward_index <= backward_num:
        _vehicle_wps = _vehicle_wp.previous(8)
        if len(_vehicle_wps) == 0:
            break
        _vehicle_wp = _vehicle_wps[0]
        bp_name = choose_bp_name(filters)
        if random.random() < idp:
            results.append((bp_name, _vehicle_wp.transform))
        right_backward_index += 1

    return results


def behind_traffic_flow_scenario(wp, actor_list, actor_desc, scene_cfg={}, gen_cfg={}, assign_distribution=None):
    # Desc: 在当前waypoint的后方生成车流
    if assign_distribution is not None:
        scene_cfg.update(assign_distribution.get('behind_traffic', {}))

    scene_cfg['forward_num'] = 0
    bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
    _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, **gen_cfg)
    return [len(bp_and_transforms)]


def front_traffic_flow_scenario(wp, actor_list, actor_desc, scene_cfg={}, gen_cfg={}, assign_distribution=None):
    # Desc: 在当前waypoint的前方生成车流
    if assign_distribution is not None:
        scene_cfg.update(assign_distribution.get('front_traffic', {}))

    scene_cfg['backward_num'] = 0
    scene_cfg['skip_cur_loc'] = True
    bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
    _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, **gen_cfg)
    return [len(bp_and_transforms)]


def right_traffic_flow_scenario(wp, actor_list, actor_desc, scene_cfg={}, gen_cfg={}, assign_distribution=None):
    # Desc: 在当前车道的右侧车道生成交通流，如果右侧为行驶车道
    if assign_distribution is not None:
        scene_cfg.update(assign_distribution.get('right_traffic', {}))

    processed_lanes = []
    if scene_cfg.get('self_lane', False):
        bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
        _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, **gen_cfg)
    processed_lanes.append(wp.lane_id)

    driving_lane_count = 0
    vehicle_nums = []
    while wp is not None:
        wp = wp.get_right_lane()
        if reduce(lambda x, y: x * y, [wp.lane_id, processed_lanes[0]]) < 0:
            break
        if wp.lane_type != carla.LaneType.Driving or wp.lane_id in processed_lanes or driving_lane_count >= scene_cfg.get(
                'lane_num', 999):
            break
        bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
        _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, **gen_cfg)
        processed_lanes.append(wp.lane_id)
        driving_lane_count += 1
        vehicle_nums.append(len(bp_and_transforms))
    return vehicle_nums


def left_traffic_flow_scenario(wp, actor_list, actor_desc, scene_cfg={}, gen_cfg={}, assign_distribution=None):
    # Desc: 在当前车道的左侧车道生成交通流，如果左侧为行驶车道
    if assign_distribution is not None:
        scene_cfg.update(assign_distribution.get('left_traffic', {}))

    processed_lanes = []
    if scene_cfg.get('self_lane', False):
        bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
        _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, **gen_cfg)
    processed_lanes.append(wp.lane_id)

    driving_lane_count = 0
    vehicle_nums = []
    while wp is not None:
        wp = wp.get_left_lane()
        if reduce(lambda x, y: x * y, [wp.lane_id, processed_lanes[0]]) < 0:
            break
        if scene_cfg.get('skip_num', 0) > driving_lane_count:
            driving_lane_count += 1
            continue
        if wp.lane_type != carla.LaneType.Driving or wp.lane_id in processed_lanes or driving_lane_count >= scene_cfg.get(
                'lane_num', 999):
            break
        bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
        _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, **gen_cfg)
        processed_lanes.append(wp.lane_id)
        driving_lane_count += 1
        vehicle_nums.append(len(bp_and_transforms))
    return vehicle_nums


def opposite_traffic_flow_scenario(wp, actor_list, actor_desc, scene_cfg={}, gen_cfg={}, assign_distribution=None):
    # Desc: 在当前道路的对向车道生成交通流
    if assign_distribution is not None:
        scene_cfg.update(assign_distribution.get('opposite_traffic', {}))

    # Special: 获取当前车道的对向车道的最左侧的waypoint
    added_lanes = []
    last_wp = None
    while True:
        if wp is None:
            return
        if wp.lane_id in added_lanes:
            break
        added_lanes.append(wp.lane_id)
        last_wp = wp
        wp = wp.get_left_lane()

    if last_wp is None:
        return

    while last_wp.lane_type != carla.LaneType.Driving:
        if last_wp is None:
            return
        last_wp = last_wp.get_right_lane()

    scene_cfg.update({'self_lane': True})
    return right_traffic_flow_scenario(last_wp, actor_list, actor_desc, scene_cfg, gen_cfg)


def right_parking_vehicle_scenario(wp, actor_list, actor_desc, scene_cfg={}, gen_cfg={}, assign_distribution=None):
    # Desc: 在当前车道的右侧车道生成停车车辆
    if assign_distribution is not None:
        scene_cfg.update(assign_distribution.get('right_parking', {}))

    processed_lanes = set()
    if scene_cfg.get('self_lane', False):
        if wp.lane_type == carla.LaneType.Stop or (wp.lane_type == carla.LaneType.Shoulder and wp.lane_width >= 2):
            bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
            _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, **gen_cfg)
    processed_lanes.add(wp.lane_id)

    stop_lane_count = 0
    while True:
        wp = wp.get_right_lane()
        if wp is None:
            return
        if wp.lane_type != carla.LaneType.Stop and (wp.lane_type != carla.LaneType.Shoulder or wp.lane_width < 2):
            continue
        if wp.lane_id in processed_lanes or stop_lane_count >= scene_cfg.get('lane_num', 999):
            return
        bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)
        _apply_bp_generation(actor_list, actor_desc, bp_and_transforms, **gen_cfg)
        processed_lanes.add(wp.lane_id)

def right_parking_open_door_scenario(wp, actor_list, actor_desc, data_tags, scene_cfg={},
                                        gen_cfg={}):
    # Desc: 在当前车道的右侧车道生成沿路的停车车辆并随机一辆车门打开（前左FL车门或后左RL车门或FLRL同时）
    wp = wp.get_right_lane()
    # if wp.lane_width < 2:
    half_lane_with = 0.9 - wp.lane_width / 2.0

    new_bp_and_transforms = []
    right_forward_index = 0
    nex_wp = wp
    dis = 0

    while right_forward_index < scene_cfg['length']:
        if dis == 0:
            dis = random.randint(12, 14)
        else:
            dis = random.randint(max(9, data_tags['tidy_dis_min']),
                                 data_tags['tidy_dis_max'] if data_tags['tidy_dis_max'] is not None else 10)
        nex_wp = move_waypoint_forward(nex_wp, dis)
        if distance_to_next_junction(nex_wp) - 20 <= 0 and scene_cfg['length']>1:
            break
        nxt_transform = nex_wp.transform

        x_offset_v1 = -half_lane_with * math.sin(math.radians(nxt_transform.rotation.yaw))
        y_offset_v1 = half_lane_with * math.cos(math.radians(nxt_transform.rotation.yaw))

        bp_name = choose_bp_name(scene_cfg['filters'])
        if random.random() < scene_cfg['idp']:
            new_bp_and_transforms.append((bp_name, carla.Transform(location=carla.Location(
                x=nxt_transform.location.x + x_offset_v1,
                y=nxt_transform.location.y + y_offset_v1,
                z=nxt_transform.location.z + 0.5
            ),
                rotation=nxt_transform.rotation
            )
                                          ))
        right_forward_index += dis

    # bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)

    _apply_bp_generation(actor_list, actor_desc, new_bp_and_transforms, **gen_cfg)



def right_parking_tidy_vehicle_scenario(wp, actor_list, actor_desc, data_tags, default_settings, scene_cfg={},
                                        gen_cfg={}):
    # Desc: 在当前车道的右侧车道生成沿路的停车车辆
    # wp = wp.get_right_lane()
    half_lane_with = wp.lane_width / 5.0 * 2

    new_bp_and_transforms = []
    right_forward_index = 0
    nex_wp = wp
    dis = 0
    while right_forward_index < scene_cfg['length']:
        if dis == 0:
            dis = random.randint(7, 8)
        else:
            dis = random.randint(max(9, data_tags['tidy_dis_min']),
                                 data_tags['tidy_dis_max'] if data_tags['tidy_dis_max'] is not None else 10)
        nex_wp = move_waypoint_forward(nex_wp, dis)
        if distance_to_next_junction(nex_wp) - 15 <= 0:
            break
        nxt_transform = nex_wp.transform

        x_offset_v1 = -half_lane_with * math.sin(math.radians(nxt_transform.rotation.yaw))
        y_offset_v1 = half_lane_with * math.cos(math.radians(nxt_transform.rotation.yaw))

        bp_name = choose_bp_name(scene_cfg['filters'])
        if random.random() < scene_cfg['idp']:
            new_bp_and_transforms.append((bp_name, carla.Transform(location=carla.Location(
                x=nxt_transform.location.x + x_offset_v1,
                y=nxt_transform.location.y + y_offset_v1,
                z=nxt_transform.location.z + 0.5
            ),
                rotation=nxt_transform.rotation
            )
                                          ))
        right_forward_index += dis

    # bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)

    _apply_bp_generation(actor_list, actor_desc, new_bp_and_transforms, **gen_cfg)


def right_parking_many_vehicle_scenario(wp, actor_list, actor_desc, data_tags, default_settings, scene_cfg={},
                                        gen_cfg={}):
    # Desc: 在当前车道的右侧生成大量停车车辆

    half_lane_with = wp.lane_width / 3.0  #* 2

    new_bp_and_transforms = []
    right_forward_index = 0
    nex_wp = wp
    dis = 0
    direction = [0, 0]
    if default_settings['distributions']['direct_flip'] == 'default':
        sign = random.randint(-1, 1)
    else:
        sign = do_sample(default_settings['distributions']['direct_flip'])
    if default_settings['distributions']['theta'] == 'default':
        theta_distribution = {i: 1.0 for i in range(math.ceil(default_settings['params']['yaw_min']),
                                                    math.ceil(default_settings['params']['yaw_max']) + 1)}
    else:
        theta_distribution = default_settings['distributions']['theta']

    while right_forward_index < scene_cfg['length']:
        if dis == 0:
            dis = random.randint(7, 8)
        else:
            dis = random.randint(max(4, data_tags['mass_dis_min']),
                                 data_tags['mass_dis_max'] if data_tags['mass_dis_max'] is not None else 6)
        nex_wp = move_waypoint_forward(nex_wp, dis)
        if distance_to_next_junction(nex_wp) - 15 <= 0:
            break
        nxt_transform = nex_wp.transform
        new_yaw = do_sample(theta_distribution) * sign

        if default_settings['distributions']['direct_out'] == 'default':
            out_flag = random.randint(0, 1)
        else:
            out_flag = do_sample(default_settings['distributions']['direct_out'])

        if out_flag:
            new_yaw += 180
            direction[0] += 1  # 车头朝外
        else:
            direction[1] += 1  # 车头朝内
        new_yaw += nxt_transform.rotation.yaw

        x_offset_v1 = -half_lane_with * math.sin(math.radians(nxt_transform.rotation.yaw))
        y_offset_v1 = half_lane_with * math.cos(math.radians(nxt_transform.rotation.yaw))

        bp_name = choose_bp_name(scene_cfg['filters'])
        if random.random() < scene_cfg['idp']:
            new_bp_and_transforms.append((bp_name, carla.Transform(location=carla.Location(
                x=nxt_transform.location.x + x_offset_v1,
                y=nxt_transform.location.y + y_offset_v1,
                z=nxt_transform.location.z + 0.5
            ),
                rotation=carla.Rotation(
                    pitch=nxt_transform.rotation.pitch,
                    yaw=new_yaw,
                    roll=nxt_transform.rotation.roll)
            )
                                          ))
        right_forward_index += dis
    data_tags['direct_inside'] += direction[1]
    data_tags['direct_outside'] += direction[0]
    # bp_and_transforms = _traffic_flow_scenario(wp, **scene_cfg)

    _apply_bp_generation(actor_list, actor_desc, new_bp_and_transforms, **gen_cfg)
    # ipdb.set_trace()


def trans_wp_v2(wp, p1_range=None, p2_range=None):
    half_lane_with = wp.lane_width / 3.0 * 2
    # soffset_v1 = 1.0  # 偏移量为1m
    soffset_v1 = half_lane_with

    # 移动起始路径点
    if p1_range is not None:
        first_vehicle_wp = move_waypoint_forward(wp, random.randint(p1_range[0], p1_range[1]))
    else:
        first_vehicle_wp = move_waypoint_forward(wp, random.randint(25, 35))
    angle_v1 = first_vehicle_wp.transform.rotation.yaw

    # 计算新的 x 和 y 坐标
    x_offset_v1 = -soffset_v1 * math.sin(math.radians(angle_v1))
    y_offset_v1 = soffset_v1 * math.cos(math.radians(angle_v1))

    # 创建新的 carla.Location 对象
    location_v1 = carla.Location(
        x=first_vehicle_wp.transform.location.x + x_offset_v1,
        y=first_vehicle_wp.transform.location.y + y_offset_v1,
        z=first_vehicle_wp.transform.location.z + 0.5
    )

    # 创建新的 carla.Rotation 对象
    rotation_v1 = carla.Rotation(
        pitch=first_vehicle_wp.transform.rotation.pitch,
        yaw=first_vehicle_wp.transform.rotation.yaw,
        roll=first_vehicle_wp.transform.rotation.roll
    )

    # 创建新的 carla.Transform 对象
    change_wp_transform_v1 = carla.Transform(location=location_v1, rotation=rotation_v1)

    if p2_range is not None:
        second_vehicle_wp = move_waypoint_forward(wp, random.randint(p2_range[0], p2_range[1]))
    else:
        second_vehicle_wp = move_waypoint_forward(wp, random.randint(48, 55))
    angle_v2 = second_vehicle_wp.transform.rotation.yaw
    assert isinstance(second_vehicle_wp, carla.Waypoint)
    lane_width = second_vehicle_wp.lane_width
    soffset_v2 = lane_width + 0.5

    # 计算新的 x 和 y 坐标
    x_offset_v2 = soffset_v2 * math.sin(math.radians(angle_v2))
    y_offset_v2 = -soffset_v2 * math.cos(math.radians(angle_v2))  # 注意这里的负号，因为 y 轴在地图上通常是向下的

    # 创建新的 carla.Location 对象
    location_v2 = carla.Location(x=second_vehicle_wp.transform.location.x + x_offset_v2,
                                 y=second_vehicle_wp.transform.location.y + y_offset_v2,
                                 z=second_vehicle_wp.transform.location.z + 0.5)

    # 创建新的 carla.Rotation 对象
    rotation_v2 = carla.Rotation(pitch=second_vehicle_wp.transform.rotation.pitch,
                                 yaw=second_vehicle_wp.transform.rotation.yaw,
                                 roll=second_vehicle_wp.transform.rotation.roll)

    # 创建新的 carla.Transform 对象
    change_wp_transform_v2 = carla.Transform(location=location_v2, rotation=rotation_v2)
    return change_wp_transform_v1, change_wp_transform_v2


def trans_wp_v1(wp, dis=None, yaw=None):
    half_lane_with = wp.lane_width / 3.0 * 2
    # soffset_v1 = 1.0  # 偏移量为1m
    soffset_v1 = random.randint(1, 10) * 0.1 * half_lane_with

    # 移动起始路径点
    first_vehicle_wp = move_waypoint_forward(wp, dis)
    angle_v1 = first_vehicle_wp.transform.rotation.yaw

    # 计算新的 x 和 y 坐标
    x_offset_v1 = -soffset_v1 * math.sin(math.radians(angle_v1))
    y_offset_v1 = soffset_v1 * math.cos(math.radians(angle_v1))

    # 创建新的 carla.Location 对象
    location_v1 = carla.Location(
        x=first_vehicle_wp.transform.location.x + x_offset_v1,
        y=first_vehicle_wp.transform.location.y + y_offset_v1,
        z=first_vehicle_wp.transform.location.z + 0.5
    )

    # 创建新的 carla.Rotation 对象
    rotation_v1 = carla.Rotation(
        pitch=first_vehicle_wp.transform.rotation.pitch,
        yaw=yaw,
        roll=first_vehicle_wp.transform.rotation.roll
    )

    # 创建新的 carla.Transform 对象
    change_wp_transform_v1 = carla.Transform(location=location_v1, rotation=rotation_v1)
    return change_wp_transform_v1


def opposite_wp_through_junction_v1(wp):
    half_lane_with = wp.lane_width * 4 / 3
    # soffset_v1 = 1.0  # 偏移量为1m
    soffset_v1 = half_lane_with

    # 移动起始路径点
    forward_meters = random.randint(50, 60)
    first_vehicle_wp = move_waypoint_forward_through_junction(wp, forward_meters)
    print(f'forward_meters: {forward_meters}')
    change_wp_transform_v1 = first_vehicle_wp.get_left_lane().transform

    return change_wp_transform_v1


def trans_wp_right(wp):
    spawn_point = move_waypoint_forward(wp, random.randint(10, 15))
    random_yaw = random.uniform(0, 180)
    assert isinstance(spawn_point, carla.Waypoint)
    spawn_npc_point = carla.Transform(
        location=carla.Location(x=spawn_point.transform.location.x + -1 * (spawn_point.lane_width - 1) / 2 * math.sin(
            math.radians(spawn_point.transform.rotation.yaw)),
                                y=spawn_point.transform.location.y + ((spawn_point.lane_width - 1) / 2) * math.cos(
                                    math.radians(spawn_point.transform.rotation.yaw)),
                                z=spawn_point.transform.location.z + 0.5),
        rotation=carla.Rotation(pitch=spawn_point.transform.rotation.pitch,
                                yaw=spawn_point.transform.rotation.yaw,
                                roll=spawn_point.transform.rotation.roll))

    return spawn_npc_point
