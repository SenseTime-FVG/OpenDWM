# -*- coding: utf-8 -*-
# @Time    : 2024/3/12
# @Author  : ZHIWEN
# @File    : tiny_scenarios_obstacle.py
# @Desc    : xxx
import math
import re
import random
import carla
# import ipdb

import numpy as np
from random import choice
import os
import argparse
import pickle
import json
from Initialization.Standardimport.scenariomanager.carla_data_provider import CarlaDataProvider
# from functions import *
from .walker_manager import *
from Scenarios.Tinyscenarios.tiny_scenarios_v2 import choose_bp_name

from functools import reduce
# # import ipdb
import time


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


OBSTACLE_TYPE_DICT = {
    # traffic obstacles
    'static.prop.garbage01': ['garbage'],  # 建筑垃圾
    'static.prop.garbage02': ['garbage'],
    'static.prop.garbage03': ['garbage'],
    'static.prop.garbage04': ['garbage'],
    'static.prop.busstop': ['bus_stop'],   # 公交车站
    'static.prop.constructioncone': ['construction'],   # 施工锥，用于标记施工区域或指引行人和车辆
    'static.prop.streetbarrier': ['street_barrier'],   # 用于限制车辆通行或指引行人。
    'static.prop.warningconstruction': ['street_barrier'],   # 用于限制车辆通行或指引行人。
    'static.prop.trafficcone01': ['traffic_barrier'],  # 交通锥，用于标记道路施工区域或指引交通
    'static.prop.trafficcone02': ['traffic_barrier'],  # 交通锥，用于标记道路施工区域或指引交通
    'static.prop.warningaccident': ['accident'],
    'walker.pedestrian.0004': ['workers'],
    'walker.pedestrian.0003': ['workers'],
    'walker.pedestrian.0015': ['workers'],
    'walker.pedestrian.0019': ['workers'],
    'walker.pedestrian.0016': ['workers'],
    'walker.pedestrian.0023': ['workers'],
    'static.prop.creasedbox02': ['creasedbox'],
    'static.prop.ironplank': ['plank']
}
TYPE_OBSTACLE_DICT = {}
for bp_obstacle_name, bp_obstacle_filters in OBSTACLE_TYPE_DICT.items():
    for bp_obstacle_filters in bp_obstacle_filters:
        if bp_obstacle_filters not in TYPE_OBSTACLE_DICT:
            TYPE_OBSTACLE_DICT[bp_obstacle_filters] = []
        TYPE_OBSTACLE_DICT[bp_obstacle_filters].append(bp_obstacle_name)

def choose_obsbp_name(filters):
    """
    Desc: 根据障碍物类型选择对应的blueprint
    @param filters: +x: 添加类型 -x: 排除类型，按顺序计算
    """
    # garbage: 道路垃圾，废弃物
    # bus_stop: 公交车站
    # construction： 施工
    # street_barrier: 道路指引
    # traffic_barrier: 交通障碍物

    filters = [item.strip() for item in re.split(r'([+\-])', filters.strip()) if item.strip()]

    # 不能为单数
    if len(filters) % 2 != 0:
        return ""

    candidate_obsbp_names = []
    for index in range(0, len(filters), 2):
        op = filters[index]
        filter_type = filters[index + 1]
        if op == '+':
            candidate_obsbp_names.extend(TYPE_OBSTACLE_DICT[filter_type])
        elif op == '-':
            candidate_obsbp_names = list(set(candidate_obsbp_names) - set(TYPE_OBSTACLE_DICT[filter_type]))
        else:
            print(f'Error: {op} is not supported in blueprint choosing.')
            return ""

    if len(candidate_obsbp_names) == 0:
        print(f'Error: candidate_bp_names is empty.')
        return ""

    return random.choice(candidate_obsbp_names)

def vehicle_breakdown(world,ego_location):
    """
       Simulates a vehicle breakdown scenario by spawning a stationary vehicle and a barrier in front of the ego vehicle.

       :param world: The CARLA world object.
       :param ego_location: The location of the ego vehicle.
       """

    # For: 在自车前方20-25米随机生成一辆车
    first_vehicle_wp = move_waypoint_forward(ego_location, random.randint(20, 25))
    front_actor = CarlaDataProvider.request_new_actor('+wheel4', first_vehicle_wp.transform)
    traffic_manager = CarlaDataProvider.get_trafficmanager()
    traffic_manager.set_desired_speed(front_actor, 0)

    blueprint_library = world.get_blueprint_library()
    # 根据过滤条件选择障碍物蓝图
    barrier_bp = choose_obsbp_name('+traffic_barrier')
    print(f"barrier_bp : {barrier_bp}\n")
    # 计算障碍物的位置
    barrier_transform = move_waypoint_backward(first_vehicle_wp, random.randint(2, 3))
    # 从蓝图库中获取障碍物的ActorBlueprint对象
    barrier_blueprint = blueprint_library.find(barrier_bp)
    if barrier_blueprint is not None:
        barrier = world.spawn_actor(barrier_blueprint, barrier_transform.transform)
        barrier.set_simulate_physics(True)  # Ensure the barrier has physics simulation
        return barrier_transform.transform.location
    else:
        print("Warning: Barrier blueprint not found.")


def gen_barrier(world,wp):
    blueprint_library = world.get_blueprint_library()
    # 根据过滤条件选择障碍物蓝图
    barrier_bp = choose_obsbp_name('+street_barrier')
    print(f"barrier_bp : {barrier_bp}\n")
    # 计算障碍物的位置
    barrier_spawn = move_waypoint_forward(wp, random.randint(20, 25))
    barrier_transform = barrier_spawn.transform
    # 从蓝图库中获取障碍物的ActorBlueprint对象
    barrier_blueprint = blueprint_library.find(barrier_bp)
    if barrier_blueprint is not None:
        new_yaw = barrier_transform.rotation.yaw + 90
        # 创建一个新的Transform对象，使用新的yaw值
        new_transform = carla.Transform(
            location=barrier_transform.location,  # 保持位置不变
            rotation=carla.Rotation(pitch=barrier_transform.rotation.pitch, yaw=new_yaw,
                                    roll=barrier_transform.rotation.roll))
        barrier = world.spawn_actor(barrier_blueprint, new_transform)
        barrier.set_simulate_physics(False)  # Ensure the barrier has physics simulation
    return barrier_spawn


def gen_two_planks(world,barrier_spawn):
    blueprint_library = world.get_blueprint_library()
    plank_bp = choose_obsbp_name('+plank')
    print(f"plank : {plank_bp}\n")
    # 计算障碍物的位置
    plank_spawn_first = move_waypoint_forward(barrier_spawn, random.randint(2, 3))
    plank_transform_first = plank_spawn_first.transform
    # 从蓝图库中获取障碍物的ActorBlueprint对象
    plank_blueprint_first = blueprint_library.find(plank_bp)
    if plank_blueprint_first is not None:
        new_yaw = plank_transform_first.rotation.yaw - 20
        # 创建一个新的Transform对象，使用新的yaw值
        plank_transform_first = carla.Transform(
            location=carla.Location(x=plank_transform_first.location.x + 0.5, y=plank_transform_first.location.y - 0.5,
                                    z=plank_transform_first.location.z + 0.5),
            rotation=carla.Rotation(pitch=plank_transform_first.rotation.pitch, yaw=new_yaw,
                                    roll=plank_transform_first.rotation.roll))
        plank_first = world.spawn_actor(plank_blueprint_first, plank_transform_first)
        plank_first.set_simulate_physics(False)  # Ensure the barrier has physics simulation

    plank_spawn_second = move_waypoint_forward(barrier_spawn, random.randint(8, 9))
    plank_transform_second = plank_spawn_second.transform
    # 从蓝图库中获取障碍物的ActorBlueprint对象
    plank_blueprint_second = blueprint_library.find(plank_bp)
    if plank_blueprint_second is not None:
        new_yaw_second = plank_transform_second.rotation.yaw - 70
        # 创建一个新的Transform对象，使用新的yaw值
        plank_transform_second = carla.Transform(
            location=carla.Location(x=plank_transform_second.location.x - 0.3,
                                    y=plank_transform_second.location.y + 0.3,
                                    z=plank_transform_second.location.z + 0.5),
            rotation=carla.Rotation(pitch=plank_transform_second.rotation.pitch, yaw=new_yaw_second,
                                    roll=plank_transform_second.rotation.roll))
        plank_second = world.spawn_actor(plank_blueprint_second, plank_transform_second)
        plank_second.set_simulate_physics(False)  # Ensure the barrier has physics simulation
    return plank_spawn_first


def gen_creasedbox(world,barrier_spawn):
    blueprint_library = world.get_blueprint_library()
    creasedbox_bp = choose_obsbp_name('+creasedbox')
    print(f"creasedbox : {creasedbox_bp}\n")
    # 计算障碍物的位置
    creasedbox_spawn = move_waypoint_forward(barrier_spawn, random.randint(5, 8))
    creasedbox_transform = creasedbox_spawn.transform
    # 从蓝图库中获取障碍物的ActorBlueprint对象
    creasedbox_blueprint = blueprint_library.find(creasedbox_bp)
    if creasedbox_blueprint is not None:
        new_yaw = creasedbox_transform.rotation.yaw + 45
        # 创建一个新的Transform对象，使用新的yaw值
        new_creasedbox_transform = carla.Transform(
            location=carla.Location(x=creasedbox_transform.location.x, y=creasedbox_transform.location.y,
                                    z=creasedbox_transform.location.z + 0.5),
            rotation=carla.Rotation(pitch=creasedbox_transform.rotation.pitch, yaw=new_yaw,
                                    roll=creasedbox_transform.rotation.roll))
        creasedbox = world.spawn_actor(creasedbox_blueprint, new_creasedbox_transform)
        creasedbox.set_simulate_physics(False)  # Ensure the barrier has physics simulation
    return creasedbox_spawn

def gen_garbage(world,barrier_spawn,num_garbage):
    blueprint_library = world.get_blueprint_library()
    gar_spawn = move_waypoint_forward(barrier_spawn, random.randint(2, 3))
    for i in range(num_garbage):
        garbage_bp = choose_obsbp_name('+garbage')
        garbage_blueprint = blueprint_library.find(garbage_bp)
        k_gar_soffset = random.choice([2, 3])
        x_gar_soffset = random.uniform(0, 0.3)
        y_gar_soffset = random.uniform(0, 0.5)
        yaw_gar_soffset = random.uniform(0, 360)
        spawn_garbage_point = carla.Transform(
            location=carla.Location(
                x=gar_spawn.transform.location.x + 1.0 + (-1) ** (-1 * k_gar_soffset) * x_gar_soffset * 0.4,
                y=gar_spawn.transform.location.y + (-1) ** (k_gar_soffset) * y_gar_soffset * 0.3,
                z=gar_spawn.transform.location.z + 0.5),
            rotation=carla.Rotation(pitch=gar_spawn.transform.rotation.pitch,
                                    yaw=gar_spawn.transform.rotation.yaw + (-1) ** (k_gar_soffset) * yaw_gar_soffset,
                                    roll=gar_spawn.transform.rotation.roll)
        )
        while True:
            garbage = world.spawn_actor(garbage_blueprint, spawn_garbage_point)
            garbage.set_simulate_physics(False)
            break
    return gar_spawn


def gen_cones(world,barrier_spawn):
    blueprint_library = world.get_blueprint_library()
    cone_bp = choose_obsbp_name('+traffic_barrier')
    cone_blueprint = blueprint_library.find(cone_bp)
    if cone_bp is None:
        raise ValueError("Traffic cone blueprint not found in the library.")

    # Get the waypoint just ahead of the barrier
    _map = CarlaDataProvider.get_map()
    barrier_waypoint = _map.get_waypoint(barrier_spawn.transform.location)
    first_cone_waypoint = barrier_waypoint.next(0.3)[0]  # Get the next waypoint after the barrier
    num_cones = random.choice([4,5,6,7,8])
    cone_interval = random.choice([0.5,1,1.5])
    # Spawn the traffic cones
    for i in range(num_cones):
        try:
            # 尝试获取锥筒的目标waypoint
            target_waypoint = first_cone_waypoint.next((i + 1) * int(cone_interval))[0]
            if target_waypoint is not None:  # 确保waypoint是有效的
                assert isinstance(target_waypoint, carla.Waypoint)
                # 计算锥筒的位置
                cone_left_location = carla.Location(
                    x=target_waypoint.transform.location.x + (
                            ((target_waypoint.lane_width - 0.8) / 2) * math.sin(
                        math.radians(target_waypoint.transform.rotation.yaw))),
                    y=target_waypoint.transform.location.y - (
                            ((target_waypoint.lane_width - 0.8) / 2) * math.cos(
                        math.radians(target_waypoint.transform.rotation.yaw))),
                    z=target_waypoint.transform.location.z)

                # 创建锥筒的变换对象
                cone_left_transform = carla.Transform(
                    location=cone_left_location,
                    rotation=carla.Rotation(pitch=target_waypoint.transform.rotation.pitch,
                                            yaw=target_waypoint.transform.rotation.yaw,
                                            roll=target_waypoint.transform.rotation.roll))
                # 在计算出的位置和方向上生成锥筒
                cone = world.spawn_actor(cone_blueprint, cone_left_transform)
                cone.set_simulate_physics(False)

                cone_right_location = carla.Location(
                    x=target_waypoint.transform.location.x - (
                            ((target_waypoint.lane_width - 0.8) / 2) * math.sin(
                        math.radians(target_waypoint.transform.rotation.yaw))),
                    y=target_waypoint.transform.location.y + (
                            ((target_waypoint.lane_width - 0.8) / 2) * math.cos(
                        math.radians(target_waypoint.transform.rotation.yaw))),
                    z=target_waypoint.transform.location.z)

                # 创建锥筒的变换对象
                cone_right_transform = carla.Transform(
                    location=cone_right_location,
                    rotation=carla.Rotation(pitch=target_waypoint.transform.rotation.pitch,
                                            yaw=target_waypoint.transform.rotation.yaw,
                                            roll=target_waypoint.transform.rotation.roll))
                # 在计算出的位置和方向上生成锥筒
                cone = world.spawn_actor(cone_blueprint, cone_right_transform)
                cone.set_simulate_physics(False)

            else:
                print(f"Invalid waypoint for cone placement at i = {i}")
        except RuntimeError as e:
            print(f"Error placing cones at i = {i}: {e}")

        for i in range(num_cones):
            try:
                # 尝试获取锥筒的目标waypoint
                target_waypoint = first_cone_waypoint.next((i + 1 ) * int(cone_interval))[0]
                if target_waypoint is not None:  # 确保waypoint是有效的
                    assert isinstance(target_waypoint, carla.Waypoint)
                    # 计算锥筒的位置
                    cone_left_location = carla.Location(
                        x=target_waypoint.transform.location.x + (
                                ((target_waypoint.lane_width - 0.8) / 2) * math.sin(
                            math.radians(target_waypoint.transform.rotation.yaw))),
                        y=target_waypoint.transform.location.y - (
                                ((target_waypoint.lane_width - 0.8) / 2) * math.cos(
                            math.radians(target_waypoint.transform.rotation.yaw))),
                        z=target_waypoint.transform.location.z)

                    # 创建锥筒的变换对象
                    cone_left_transform = carla.Transform(
                        location=cone_left_location,
                        rotation=carla.Rotation(pitch=target_waypoint.transform.rotation.pitch,
                                                yaw=target_waypoint.transform.rotation.yaw,
                                                roll=target_waypoint.transform.rotation.roll))
                    # 在计算出的位置和方向上生成锥筒
                    cone = world.spawn_actor(cone_blueprint, cone_left_transform)
                    cone.set_simulate_physics(False)

                    cone_right_location = carla.Location(
                        x=target_waypoint.transform.location.x - (
                                ((target_waypoint.lane_width - 1) / 2) * math.sin(
                            math.radians(target_waypoint.transform.rotation.yaw))),
                        y=target_waypoint.transform.location.y + (
                                ((target_waypoint.lane_width - 1) / 2) * math.cos(
                            math.radians(target_waypoint.transform.rotation.yaw))),
                        z=target_waypoint.transform.location.z)

                    # 创建锥筒的变换对象
                    cone_right_transform = carla.Transform(
                        location=cone_right_location,
                        rotation=carla.Rotation(pitch=target_waypoint.transform.rotation.pitch,
                                                yaw=target_waypoint.transform.rotation.yaw,
                                                roll=target_waypoint.transform.rotation.roll))
                    # 在计算出的位置和方向上生成锥筒
                    cone = world.spawn_actor(cone_blueprint, cone_right_transform)
                    cone.set_simulate_physics(False)

                    # 如果是最后一个锥筒，存储其位置
                    if i == num_cones - 1:
                        last_cone_transform = cone_left_transform
                        print('-------last_cone_transform', last_cone_transform)
                        # Spawn the last barrier
                        last_barrier_bp = choose_obsbp_name('+street_barrier')
                        print(f"last_barrier_bp : {last_barrier_bp}\n")
                        # 从蓝图库中获取障碍物的ActorBlueprint对象
                        last_barrier_blueprint = blueprint_library.find(last_barrier_bp)
                        if last_barrier_blueprint is not None:
                            new_transform_last = carla.Transform(
                                location=carla.Location(x=last_cone_transform.location.x - (
                                            target_waypoint.lane_width - 0.8) / 2 * math.sin(
                                    math.radians(last_cone_transform.rotation.yaw)),
                                                        y=last_cone_transform.location.y + (
                                                                    target_waypoint.lane_width - 0.8) / 2 * math.cos(
                                                            math.radians(last_cone_transform.rotation.yaw)),
                                                        z=last_cone_transform.location.z + 0.1),  # 保持位置不变
                                rotation=carla.Rotation(pitch=last_cone_transform.rotation.pitch,
                                                        yaw=last_cone_transform.rotation.yaw - 90,
                                                        roll=last_cone_transform.rotation.roll))
                            last_barrier = world.spawn_actor(last_barrier_blueprint, new_transform_last)
                            print("-------", last_barrier)
                            last_barrier.set_simulate_physics(False)  # Ensure the barrier has physics simulation
                            return new_transform_last
                else:
                    print(f"Invalid waypoint for cone placement at i = {i}")
            except RuntimeError as e:
                print(f"Error placing cones at i = {i}: {e}")


def gen_Walker(world,num_workers, ref_spawn):
    blueprint_library = world.get_blueprint_library()
    pedestrians = []
    pedestrians_bp = WalkerManager.choose_walker_name('+workers')
    pedestrians_blueprint = blueprint_library.find(pedestrians_bp)
    for i in range(num_workers):
        spawn_point = move_waypoint_forward(ref_spawn, random.randint(0, 5))
        random_yaw = random.uniform(0, 180)
        spawn_npc_point = carla.Transform(
            location=carla.Location(x=spawn_point.transform.location.x, y=spawn_point.transform.location.y,
                                    z=spawn_point.transform.location.z + 0.5),
            rotation=carla.Rotation(pitch=spawn_point.transform.rotation.pitch,
                                    yaw=spawn_point.transform.rotation.yaw + random_yaw,
                                    roll=spawn_point.transform.rotation.roll))
        while True:
            try:
                # 如果位置安全，尝试生成行人
                npc = world.spawn_actor(pedestrians_blueprint, spawn_npc_point)
                pedestrians.append(npc)
                break  # 成功生成行人，退出循环
            except RuntimeError as e:
                # 如果生成失败，打印错误信息并尝试新的位置
                print(f"Spawn failed at {spawn_npc_point}: {e}")
                spawn_point = move_waypoint_forward(ref_spawn, random.randint(3, 6))
                # 重新计算生成点
                random_yaw = random.uniform(0, 180)
                spawn_npc_point = carla.Transform(
                    location=carla.Location(x=spawn_point.transform.location.x,
                                            y=spawn_point.transform.location.y,
                                            z=spawn_point.transform.location.z + 0.5),
                    rotation=carla.Rotation(pitch=spawn_point.transform.rotation.pitch,
                                            yaw=spawn_point.transform.rotation.yaw + random_yaw,
                                            roll=spawn_point.transform.rotation.roll))

def ahead_obstacle_scenario_first(world,wp, actor_list, actor_desc, scene_cfg={}, gen_cfg={}):
    # Desc: 在当前车道的前方生成施工现场
    # 从场景配置字典中获取锥筒数量和间隔距离

    num_garbage = scene_cfg.get('num_garbage', 50)
    num_workers = scene_cfg.get('num_workers', 4)
    # 1.生成施工牌/水马
    barrier_spawn = gen_barrier(world, wp)
    # 2.生成纸板
    ref_spawn = gen_creasedbox(world, barrier_spawn)
    # 3.生成锥筒
    gen_cones(world, barrier_spawn)
    # 4.生成垃圾
    gen_garbage(world, barrier_spawn, num_garbage)
    # 5.生成行人
    # walker_manager = WalkerManager(world, num_workers, ref_spawn)
    # walker_manager.gen_walkers(num_workers, ref_spawn)
    gen_Walker(world,num_workers, ref_spawn)

    return barrier_spawn.transform.location

def ahead_obstacle_scenario_second(world,wp, actor_list, actor_desc, scene_cfg={}, gen_cfg={}):
    # Desc: 在当前车道的前方生成施工现场
    # 从场景配置字典中获取锥筒数量和间隔距离

    num_workers = scene_cfg.get('num_workers', 4)
    # 1.生成施工牌/水马
    barrier_spawn = gen_barrier(world, wp)
    # 2.生成两块钢板
    ref_spawn = gen_two_planks(world, barrier_spawn)
    # 3.生成锥筒
    gen_cones(world, barrier_spawn)
    # 4.生成行人
    # walker_manager = WalkerManager(world, num_workers, ref_spawn)
    # walker_manager.gen_walkers(num_workers, ref_spawn)
    gen_Walker(world,num_workers, ref_spawn)
    return barrier_spawn.transform.location

def ahead_obstacle_scenario(world,wp, actor_list, actor_desc, scene_cfg={}, gen_cfg={}) :    # construction 1 or construction 2
    # 检查gen_cfg字典中是否有name_prefix键，根据其值决定调用哪个函数
    name_prefix = gen_cfg.get('gen_cfg', 'construction1')  # 默认值为construction1
    if name_prefix == 'construction1':
        return ahead_obstacle_scenario_first(world, wp, actor_list, actor_desc, scene_cfg, gen_cfg)
    elif name_prefix == 'construction2':
        return ahead_obstacle_scenario_second(world, wp, actor_list, actor_desc, scene_cfg, gen_cfg)
    else:
        raise ValueError("Invalid generation configuration. 'name_prefix' must be 'construction1' or 'construction2'.")

def spawn_pedestrian(world,wp):
    blueprint_library = world.get_blueprint_library()
    pedestrians_bp = WalkerManager.choose_walker_name('+workers')
    pedestrians_blueprint = blueprint_library.find(pedestrians_bp)
    spawn_point = move_waypoint_forward(wp, random.randint(30, 36))
    random_k = random.choice([1,2])
    random_yaw = random.uniform(0, 180)
    random_x = random.uniform(0.3, 0.6)
    random_y = random.uniform(0.2,0.6)
    spawn_npc_point = carla.Transform(
        location=carla.Location(x=spawn_point.transform.location.x + (-1) ** random_k * random_x, y=spawn_point.transform.location.y + (-1)** random_k * random_y,
                                z=spawn_point.transform.location.z + 0.5),
        rotation=carla.Rotation(pitch=spawn_point.transform.rotation.pitch,
                                yaw=spawn_point.transform.rotation.yaw + random_yaw,
                                roll=spawn_point.transform.rotation.roll))
    while True:
        try:
            # 如果位置安全，尝试生成行人
            npc = world.spawn_actor(pedestrians_blueprint, spawn_npc_point)
            break  # 成功生成行人，退出循环
        except RuntimeError as e:
            # 如果生成失败，打印错误信息并尝试新的位置
            print(f"Spawn failed at {spawn_npc_point}: {e}")
            spawn_point = move_waypoint_forward(wp, random.randint(30, 35))
            # 重新计算生成点
            random_yaw = random.uniform(0, 180)
            spawn_npc_point = carla.Transform(
                location=carla.Location(x=spawn_point.transform.location.x,
                                        y=spawn_point.transform.location.y,
                                        z=spawn_point.transform.location.z + 0.5),
                rotation=carla.Rotation(pitch=spawn_point.transform.rotation.pitch,
                                        yaw=spawn_point.transform.rotation.yaw + random_yaw,
                                        roll=spawn_point.transform.rotation.roll))

    return spawn_npc_point.location ,npc,spawn_npc_point.location

def spawn_pedestrian_v2(world,wp):
    blueprint_library = world.get_blueprint_library()
    pedestrians_bp = WalkerManager.choose_walker_name('+workers')
    pedestrians_blueprint = blueprint_library.find(pedestrians_bp)
    spawn_point = move_waypoint_forward(wp, random.randint(35, 60))
    random_k = random.choice([1,2])
    random_yaw = random.uniform(0, 180)
    random_x = random.uniform(0, 0)
    random_y = random.uniform(0,0)
    spawn_npc_point = carla.Transform(
        location=carla.Location(x=spawn_point.transform.location.x + (
            ((spawn_point.lane_width ) / 2) * math.sin(math.radians(spawn_point.transform.rotation.yaw))),
                                y=spawn_point.transform.location.y - (
            ((spawn_point.lane_width) / 2) * math.cos(
                math.radians(spawn_point.transform.rotation.yaw))),
                                z=spawn_point.transform.location.z + 0.5),
        rotation=carla.Rotation(pitch=spawn_point.transform.rotation.pitch,
                                yaw=spawn_point.transform.rotation.yaw + 90 ,
                                roll=spawn_point.transform.rotation.roll))
    return pedestrians_bp,spawn_npc_point.location , spawn_npc_point.location, spawn_npc_point

def spawn_pedestrian_right(world,wp):
    blueprint_library = world.get_blueprint_library()
    pedestrians_bp = WalkerManager.choose_walker_name('+workers')
    pedestrians_blueprint = blueprint_library.find(pedestrians_bp)
    spawn_point = move_waypoint_forward(wp, random.randint(30, 35))
    random_yaw = random.uniform(0, 180)
    assert isinstance(spawn_point, carla.Waypoint)
    spawn_npc_point = carla.Transform(
        location=carla.Location(x=spawn_point.transform.location.x + -1 * (spawn_point.lane_width + 0.4)/ 2 * math.sin(
        math.radians(spawn_point.transform.rotation.yaw)),
                                y=spawn_point.transform.location.y + ((spawn_point.lane_width + 0.4 ) / 2) * math.cos(
        math.radians(spawn_point.transform.rotation.yaw)),
                                z=spawn_point.transform.location.z + 0.5),
        rotation=carla.Rotation(pitch=spawn_point.transform.rotation.pitch,
                                yaw=spawn_point.transform.rotation.yaw ,
                                roll=spawn_point.transform.rotation.roll))

    while True:
        try:
            # 如果位置安全，尝试生成行人
            npc = world.spawn_actor(pedestrians_blueprint, spawn_npc_point)
            break  # 成功生成行人，退出循环
        except RuntimeError as e:
            # 如果生成失败，打印错误信息并尝试新的位置
            print(f"Spawn failed at {spawn_npc_point}: {e}")
            spawn_point = move_waypoint_forward(wp, random.randint(30, 35))
            # 重新计算生成点
            random_yaw = random.uniform(0, 180)
            assert isinstance(spawn_point, carla.Waypoint)
            spawn_npc_point = carla.Transform(
                location=carla.Location(
                    x=spawn_point.transform.location.x + -1 * (spawn_point.lane_width ) / 2 * math.sin(
                        math.radians(spawn_point.transform.rotation.yaw)),
                    y=spawn_point.transform.location.y + ((spawn_point.lane_width ) / 2) * math.cos(
                        math.radians(spawn_point.transform.rotation.yaw)),
                    z=spawn_point.transform.location.z + 0.5),
                rotation=carla.Rotation(pitch=spawn_point.transform.rotation.pitch,
                                        yaw=spawn_point.transform.rotation.yaw,
                                        roll=spawn_point.transform.rotation.roll))

    return spawn_npc_point.location ,npc,spawn_npc_point

def control_pedestrian(npc,spawn_npc_point):
    revert_flag = False
    control = carla.WalkerControl()
    control.direction.y = 0
    control.direction.z = 0
    point = spawn_npc_point
    ped_location = npc.get_location()
    l = math.sqrt((ped_location.x - spawn_npc_point.location.x) ** 2 +(ped_location.y- spawn_npc_point.location.y) ** 2 )
    yaw = point.rotation.yaw
    if (math.fabs(ped_location.x - spawn_npc_point.location.x)) > 1:
        revert_flag = True
        control.speed = 0
    if (math.fabs(ped_location.x - spawn_npc_point.location.x)) <= 1:
        revert_flag = False
        control.speed = 0.3
    if (revert_flag):
        control.direction.x = -1 * math.sin((ped_location.x - spawn_npc_point.location.x)/l)
        control.direction.y = -1 * math.cos((ped_location.y - spawn_npc_point.location.y)/l)
    else:
        control.direction.x = math.cos(math.radians(yaw))
        control.direction.y = math.sin(math.radians(yaw))
    npc.apply_control(control)

def control_pedestrian_right(npc,spawn_npc_point):
    control = carla.WalkerControl()
    point = spawn_npc_point
    yaw = point.rotation.yaw
    control.direction.x = math.cos(math.radians(yaw))
    control.direction.y = math.sin(math.radians(yaw))
    control.direction.z = 0
    control.speed = 0.2
    npc.apply_control(control)

def spawn_motor_right(world,wp):
    blueprint_library = world.get_blueprint_library()
    moto_bp = choose_bp_name('+moto')
    motocycle_blueprint = blueprint_library.find(moto_bp)
    spawn_point = move_waypoint_forward_through_junction(wp, 28)
    # spawn_point = move_waypoint_forward(wp, random.randint(30, 35))
    random_yaw = random.uniform(0, 180)
    assert isinstance(spawn_point, carla.Waypoint)

    spawn_npc_point = carla.Transform(
        location=carla.Location(x=spawn_point.transform.location.x + -1 * (spawn_point.lane_width + 1) / 2 * math.sin(
            math.radians(spawn_point.transform.rotation.yaw)),
                                y=spawn_point.transform.location.y + ((spawn_point.lane_width + 1) / 2) * math.cos(
                                    math.radians(spawn_point.transform.rotation.yaw)),
                                z=spawn_point.transform.location.z + 0.5),
        rotation=carla.Rotation(pitch=spawn_point.transform.rotation.pitch,
                                yaw=spawn_point.transform.rotation.yaw,
                                roll=spawn_point.transform.rotation.roll))

    # while True:
    #     try:
    #         # 如果位置安全，尝试生成行人
    #         moto = world.spawn_actor(motocycle_blueprint, spawn_npc_point)
    #         break  # 成功生成行人，退出循环
    #     except RuntimeError as e:
    #         # 如果生成失败，打印错误信息并尝试新的位置
    #         print(f"Spawn failed at {spawn_point}: {e}")
    #         spawn_point = move_waypoint_forward_through_junction(wp, 28)
    #         # 重新计算生成点
    #         random_yaw = random.uniform(0, 180)
    #         spawn_npc_point = carla.Transform(
    #             location=carla.Location(
    #                 x=spawn_point.transform.location.x + -1 * (spawn_point.lane_width+ 1) / 2 * math.sin(
    #                     math.radians(spawn_point.transform.rotation.yaw)),
    #                 y=spawn_point.transform.location.y + ((spawn_point.lane_width+ 1) / 2) * math.cos(
    #                     math.radians(spawn_point.transform.rotation.yaw)),
    #                 z=spawn_point.transform.location.z + 0.5),
    #             rotation=carla.Rotation(pitch=spawn_point.transform.rotation.pitch,
    #                                     yaw=spawn_point.transform.rotation.yaw,
    #                                     roll=spawn_point.transform.rotation.roll))
    return moto_bp,spawn_npc_point