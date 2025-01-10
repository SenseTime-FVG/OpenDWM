import math
import re
import random
import carla
# import ipdb
import numpy as np
from random import choice
from random import uniform
import os
import argparse
import pickle
import json

from Initialization.Standardimport.scenariomanager.carla_data_provider import CarlaDataProvider
from . functions import *
from functools import reduce
# # import ipdb
import time

WALKER_TYPE_DICT = {
        'walker.pedestrian.0004': ['workers'],
        'walker.pedestrian.0003': ['workers'],
        'walker.pedestrian.0015': ['workers'],
        'walker.pedestrian.0019': ['workers'],
        'walker.pedestrian.0016': ['workers'],
        'walker.pedestrian.0023': ['workers'],
    }
TYPE_WALKER_DICT = {}
for bp_WALKER_name, bp_WALKER_filters in WALKER_TYPE_DICT.items():
    for bp_WALKER_filters in bp_WALKER_filters:
        if bp_WALKER_filters not in TYPE_WALKER_DICT:
            TYPE_WALKER_DICT[bp_WALKER_filters] = []
        TYPE_WALKER_DICT[bp_WALKER_filters].append(bp_WALKER_name)
class WalkerManager:

    def choose_walker_name(filters):
        """
        Desc: 根据障碍物类型选择对应的blueprint
        @param filters: +x: 添加类型 -x: 排除类型，按顺序计算
        """
        filters = [item.strip() for item in re.split(r'([+\-])', filters.strip()) if item.strip()]

        # 不能为单数
        if len(filters) % 2 != 0:
            return ""

        candidate_WALKERbp_names = []
        for index in range(0, len(filters), 2):
            op = filters[index]
            filter_type = filters[index + 1]
            if op == '+':
                candidate_WALKERbp_names.extend(TYPE_WALKER_DICT[filter_type])
            elif op == '-':
                candidate_WALKERbp_names = list(set(candidate_WALKERbp_names) - set(TYPE_WALKER_DICT[filter_type]))
            else:
                print(f'Error: {op} is not supported in blueprint choosing.')
                return ""

        if len(candidate_WALKERbp_names) == 0:
            print(f'Error: candidate_bp_names is empty.')
            return ""

        return random.choice(candidate_WALKERbp_names)

    def gen_walkers(self,num_workers, ref_spawn):
        blueprint_library = self._world.get_blueprint_library()
        num_walkers = 0
        pedestrians = []
        pedestrians_bp = WalkerManager.choose_walker_name('+workers')
        pedestrians_blueprint = blueprint_library.find(pedestrians_bp)
        for i in range(num_workers):
            if i % 2 != 0 or i == 0:
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
                        npc = self._world.spawn_actor(pedestrians_blueprint, spawn_npc_point)
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
            # else:
            #     num_walkers = num_walkers + 1
            #     print(f"num_walkers : {num_walkers}")

        # walker_batch = []
        # walker_speed = []
        # walker_ai_batch = []
        # walker_list = []
        # ped_spawn_points = []
        # for i in range(num_walkers):
        #     random_distance = random.uniform(0, 0.5)
        #     # 计算前方生成点的相对位置
        #     relative_location = carla.Location(x=random_distance, y=-1 * random_distance, z=0.5)
        #     # 从原始的spawn_point位置添加相对位置得到新的行人生成点
        #     new_spawn_point = carla.Transform(ref_spawn.transform.location + relative_location)
        #     print('---------new_spawn_point----------', new_spawn_point)
        #     # 将新的行人生成点添加到列表中
        #     ped_spawn_points.append(new_spawn_point)
        #     # ipdb.set_trace()
        #     print("-----", new_spawn_point)
        #     if i <= len(ped_spawn_points):
        #         walker_bp = pedestrians_blueprint
        #         # 取消行人无敌状态
        #         if walker_bp.has_attribute('is_invincible'):
        #             walker_bp.set_attribute('is_invincible', 'false')
        #         # 设置行人的移动速度
        #         if walker_bp.has_attribute('speed'):
        #             walker_speed.append(0.2)
        #         # 从可生成行人的生成点中随机选择并生成随机行人，之后将生成的行人添加到同一批中
        #         walker = self._world.try_spawn_actor(pedestrians_blueprint, ped_spawn_points[i])
        #         walker_ai_blueprint = self._world.get_blueprint_library().find('controller.ai.walker')
        #         walker_ai_batch.append(self._world.spawn_actor(walker_ai_blueprint, carla.Transform(), walker))
        #         walker_list.append(walker)
        #         print('--------', walker_list)
        #         walker_batch.append(walker)
        #     else:
        #         print(f"Warning: Not enoug"
        #               f"h spawn points for walker index {i}.")
        # # 从蓝图库中寻找控制行人行动逻辑的控制器
        # # 批量启动行人控制器，并设置控制器参数
        # print('------------walker_ai_batch----------',walker_ai_batch)
        # if walker_ai_batch is not None:
        #     for i in range(len(walker_ai_batch)):
        #         # 启动控制器
        #         if walker_list[i] is not None:
        #             walker_ai_batch[i].start()
        #             # 通过控制器设置行人的目标点
        #             # ipdb.set_trace()
        #             # 为每个行人计算独特的目标位置
        #             random_distance = random.uniform(0, 1)
        #             des_location = carla.Location(x=ref_spawn.transform.location.x + random_distance,
        #                                           y=ref_spawn.transform.location.y - random_distance,
        #                                           z=ref_spawn.transform.location.z)
        #             walker_ai_batch[i].go_to_location(des_location)
        #             # 通过控制器设置行人的行走速度
        #             walker_ai_batch[i].set_max_speed(0.2)
        #             walker = walker_list[i]
        #             # 检查行人是否到达目标位置
        #             walker_location = walker.get_location()
        #             distance_to_des = math.sqrt((walker_location.x - des_location.x) ** 2 +
        #                                         (walker_location.y - des_location.y) ** 2)
        #             print('walker_location:', walker_location)
        #             print('distance_to_des:', distance_to_des)
        #             # 如果行人到达目标位置或超出范围，停止行人并退出循环
        #             if distance_to_des >= 6.0:
        #                 walker_ai_batch[i].set_max_speed(0)
        #                 print("行人停止")
        #
        #             print(walker_ai_batch, walker_list, des_location)

                    # return walker_ai_batch, walker_list, des_location

    def __init__(self, world, num_workers, ref_spawn):
        self._world = world
        self.walker_ai_batch = []
        self.walker_list = []
        self.ped_spawn_points = []
        # 调用 gen_walkers 并存储返回值
        # self.walker_ai_batch, self.walker_list, self.des_location = self.gen_walkers(num_workers, ref_spawn)


    # def check_walker_distance_to_obstacles(self):
    #     for i, walker in enumerate(self.walker_list):
    #         # 为目标点计算一个随机的偏航角
    #         random_yaw = random.uniform(0, 180)
    #         target_location = carla.Location(x=self.des_location.x + random.uniform(0, 1),
    #                                          y=self.des_location.y - random.uniform(0, 1),
    #                                          z=self.des_location.z)
    #         # 获取行人当前位置
    #         walker_location = walker.get_location()
    #         # 计算行人与目标点之间的距离
    #         distance_to_target = math.sqrt((walker_location.x - target_location.x) ** 2 +
    #                                        (walker_location.y - target_location.y) ** 2)
    #         print(f"行人 {i} 的位置: {walker_location}, 目标位置: {target_location}, 距离: {distance_to_target}m")
    #         # 如果行人到达目标位置或超出范围，停止行人并退出循环
    #         if distance_to_target <= 0.5 or distance_to_target >= 4 :  # 假设的停止距离阈值
    #             self.walker_ai_batch[i].set_max_speed(0)
    #             print(f"行人 {i} 已到达目标位置或超出范围，停止行走。")
