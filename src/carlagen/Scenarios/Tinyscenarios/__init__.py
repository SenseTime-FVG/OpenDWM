# -*- coding: utf-8 -*-
# @Time    : 2023/11/3 下午10:50
# @Author  : Hcyang
# @File    : __init__.py.py
# @Desc    : xxx

import random

import numpy as np
import carla
import math
import traceback
from random import choice
from .tiny_scenarios_obstacle import *
from .tiny_scenarios_v2 import *
from .functions import *
from typing import List, Tuple
from .walker_manager import *


SUV = [
    'vehicle.audi.etron',
    'vehicle.nissan.patrol',
    'vehicle.nissan.patrol_2021'
]
TRUCK = [
    'vehicle.carlamotors.carlacola',
    'vehicle.tesla.cybertruck'
]
LARGE_VEHICLE = SUV + TRUCK


def get_value_parameter(config, name, p_type, default):
    # Desc: 获取配置文件中的参数
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    else:
        return default

def get_interval_parameter(config, name, p_type, default):
    # Desc: 获取配置文件中的参数
    if name in config.other_parameters:
        return [
            p_type(config.other_parameters[name]['from']),
            p_type(config.other_parameters[name]['to'])
        ]
    else:
        return default


def get_nearby_obstacles_by_azimuth(world: carla.World, ego: carla.Vehicle, normal_radius=20.0, fb_radius=40.0):
    # Desc: 获取指定半径内的所有其他车辆位置，去遮挡后按照方位角分类
    # Special: 条件1: 非正常行驶的车辆 (车祸：roll > 45 or roll < -45)
    #          条件2: 障碍物（static.prop.box*）
    ego_transform = ego.get_transform()
    ego_loc = ego_transform.location

    obstacles = []
    # Desc: 车祸
    obstacles.extend([item for item in world.get_actors().filter('vehicle.*') if item.get_transform().rotation.roll > 45 or item.get_transform().rotation.roll < -45])
    # Desc: 道路上的障碍物
    obstacles.extend([item for item in world.get_actors().filter('static.prop.box*')])

    azimuth_obstacles = []
    radius = max(normal_radius, fb_radius)
    for obstacle in obstacles:
        obstacle_transform = obstacle.get_transform()
        obstacle_loc = obstacle_transform.location
        distance = obstacle_loc.distance(ego_loc)
        if abs(ego_loc.z - obstacle_loc.z) > 5 or distance > radius:
            continue
        azimuth_angle = calc_relative_position(ego_transform, obstacle_transform, only_azimuth=True)
        azimuth_obstacles.append((obstacle, azimuth_angle, distance))
    azimuth_obstacles = process_obstacles(azimuth_obstacles)

    nearby_actors = {key: [] for key in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']}
    for obstacle, azimuth_angle, distance in azimuth_obstacles:
        azimuth_key = angle_to_azimuth_of_obstacle(azimuth_angle)
        if azimuth_key not in ['N', 'S', 'NE', 'NW', 'SE', 'SW'] and distance > normal_radius:
            continue
        data = {
            'instance': obstacle,
            'azimuth': azimuth_angle,
            'distance': distance,
        }
        nearby_actors[azimuth_key].append(data)

    return nearby_actors


def get_nearby_pedestrians_by_azimuth(world: carla.World, ego: carla.Vehicle, normal_radius=20.0, fb_radius=40.0):
    # Desc: 获取指定半径内的所有行人的位置，按照方位角分类
    # Special: 条件1: 排除高度差大于5米的行人（立交桥的情况）
    ego_transform = ego.get_transform()
    ego_loc = ego_transform.location

    pedestrians = [item for item in world.get_actors().filter('walker.*')]

    azimuth_pedestrians = []
    radius = max(normal_radius, fb_radius)
    for pedestrian in pedestrians:
        pedestrian_transform = pedestrian.get_transform()
        pedestrian_loc = pedestrian_transform.location
        distance = pedestrian_loc.distance(ego_loc)
        if abs(ego_loc.z - pedestrian_loc.z) > 5 or distance > radius:
            continue
        azimuth_angle = calc_relative_position(ego_transform, pedestrian_transform, only_azimuth=True)
        azimuth_pedestrians.append((pedestrian, azimuth_angle, distance))

    nearby_actors = {key: [] for key in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']}
    for pedestrian, azimuth_angle, distance in azimuth_pedestrians:
        azimuth_key = angle_to_azimuth_of_pedestrian(azimuth_angle)
        if azimuth_key not in ['N', 'S', 'NE', 'NW', 'SE', 'SW'] and distance > normal_radius:
            continue
        data = {
            'instance': pedestrian,
            'azimuth': azimuth_angle,
            'distance': distance,
        }
        nearby_actors[azimuth_key].append(data)

    return nearby_actors


def get_nearby_spawn_transforms(world: carla.World, ego, spawn_points_num, inside_radius=2, outside_radius=50) -> List[Tuple[str, carla.Transform]]:
    # Desc: 获取指定半径内的所有其他车辆允许的生成位置，去除在路口里的生成点
    wmap = world.get_map()
    spawn_points = wmap.get_spawn_points()
    ego_loc = ego.get_location()

    results = []
    for spawn_point in spawn_points:
        spawn_loc = spawn_point.location
        spawn_wp = wmap.get_waypoint(spawn_loc)
        if spawn_wp.is_junction or spawn_wp.is_intersection:
            continue
        if len(results) >= spawn_points_num:
            break
        if outside_radius > spawn_point.location.distance(ego_loc) > inside_radius:
            results.append(('*vehicle*', spawn_point))

    return results


def get_different_lane_spawn_transforms(world: carla.World, ego, spawn_points_num, radius=50, allow_same_side=True, allow_behind=False) -> List[Tuple[str, carla.Transform]]:
    """
    Desc: 获取指定半径内的所有其他车辆允许的生成位置，去除在路口里的生成点，且分车道判断
    :param world: carla.World
    :param ego: carla.Vehicle
    :param spawn_points_num: int
    :param radius: float
    :param allow_same_side: 允许在同一侧的其他车道生成
    :param allow_behind: 允许在自车后方生成
    """
    wmap = world.get_map()
    spawn_points = wmap.get_spawn_points()
    ego_loc = ego.get_location()
    ego_wp = wmap.get_waypoint(ego_loc)
    ego_road_id = ego_wp.road_id
    ego_lane_id = ego_wp.lane_id

    results = []
    for spawn_point in spawn_points:
        spawn_loc = spawn_point.location
        spawn_wp = wmap.get_waypoint(spawn_loc)
        if spawn_wp.is_junction or spawn_wp.is_intersection:
            continue
        if len(results) >= spawn_points_num:
            break
        if spawn_point.location.distance(ego_loc) > radius:
            continue

        add_sign = False
        if spawn_wp.road_id != ego_road_id:
            add_sign = True
        else:
            if spawn_wp.lane_id * ego_lane_id < 0:
                add_sign = True
            else:
                if spawn_wp.lane_id != ego_lane_id:
                    if allow_same_side:
                        add_sign = True
                else:
                    if allow_behind:
                        v1 = ego_wp.transform.get_forward_vector()
                        v2 = spawn_loc - ego_loc
                        if calc_cos_between_vector(v1, v2) < 0:
                            add_sign = True

        if add_sign:
            results.append(('*vehicle*', spawn_point))

    return results


def get_opposite_lane_spawn_transforms(world: carla.World, ego, spawn_points_num) -> List[Tuple[str, carla.Transform]]:
    """
    Desc: 获取对向车道的生成点
    :param world: carla.World
    :param ego: carla.Vehicle
    :param spawn_points_num: int
    """
    wmap = world.get_map()
    spawn_points = wmap.get_spawn_points()
    ego_loc = ego.get_location()
    ego_wp = wmap.get_waypoint(ego_loc)
    ego_road_id = ego_wp.road_id
    ego_lane_id = ego_wp.lane_id

    results = []
    same_road_spawn_wps = []
    for spawn_point in spawn_points:
        spawn_loc = spawn_point.location
        spawn_wp = wmap.get_waypoint(spawn_loc)
        if spawn_wp.is_junction or spawn_wp.is_intersection:
            continue
        if len(results) >= spawn_points_num:
            break
        add_sign = False
        if spawn_wp.road_id != ego_road_id:
            continue
        else:
            if spawn_wp.lane_id * ego_lane_id < 0:
                add_sign = True

        if add_sign:
            results.append(('*vehicle*', spawn_point))
            same_road_spawn_wps.append(spawn_wp)

    if len(same_road_spawn_wps) == 0:
        tmp = ego_wp.get_left_lane()
        direction_changed = False
        while tmp is not None and tmp.lane_type == carla.LaneType.Driving:
            if tmp.lane_id * ego_lane_id < 0:
                direction_changed = True
                same_road_spawn_wps.append(tmp)
            if direction_changed:
                tmp = tmp.get_right_lane()
            else:
                tmp = tmp.get_left_lane()

    retry = int(1.5 * (spawn_points_num - len(results)))
    if len(results) < spawn_points_num and len(same_road_spawn_wps) > 0:
        while retry > 0 and len(results) < spawn_points_num:
            retry -= 1
            spawn_wp = choice(same_road_spawn_wps)

            candidate_wps = []
            forward_max = distance_to_next_junction(spawn_wp) - 6
            if forward_max > 10:
                forward_spawn_wp = spawn_wp.next(random.randint(10, forward_max))[0]
                candidate_wps.append(forward_spawn_wp)

            backward_max = distance_to_previous_junction(spawn_wp) - 6
            if backward_max > 10:
                backward_spawn_wp = spawn_wp.previous(random.randint(10, backward_max))[0]
                candidate_wps.append(backward_spawn_wp)

            for wp in candidate_wps:
                add_sign = True
                for selected_wp in results:
                    if wp.lane_id == selected_wp and wp.transform.location.distance(selected_wp[1].location) < 8:
                        add_sign = False
                        break
                if add_sign:
                    results.append(('*vehicle*', wp.transform))

    return results


def get_random_pedestrian_transforms(world: carla.World, ego, spawn_points_num, debug=False) -> List[Tuple[str, carla.Transform, carla.Location]]:
    ego_wp = world.get_map().get_waypoint(ego.get_location())
    right_ref_wps = ego_wp.previous_until_lane_start(1)
    if len(right_ref_wps) > 2:
        _, right_sidewalk_wp = get_sidewalk_wps(right_ref_wps[-2])
    else:
        _, right_sidewalk_wp = get_sidewalk_wps(ego_wp)
    left_ref_wps = ego_wp.next_until_lane_end(1)
    if len(left_ref_wps) > 2:
        left_sidewalk_wp, _ = get_sidewalk_wps(left_ref_wps[-2])
    else:
        left_sidewalk_wp, _ = get_sidewalk_wps(ego_wp)
    candidate_wps = []
    if left_sidewalk_wp is not None:
        candidate_wps.extend(left_sidewalk_wp.next_until_lane_end(1))
    if right_sidewalk_wp is not None:
        candidate_wps.extend(right_sidewalk_wp.next_until_lane_end(1))
    random.shuffle(candidate_wps)

    if debug:
        debug = world.debug
        for wp in ego_wp.previous_until_lane_start(1):
            debug.draw_point(wp.transform.location, size=0.2, color=carla.Color(0, 0, 255))
        for wp in ego_wp.next_until_lane_end(1):
            debug.draw_point(wp.transform.location, size=0.2, color=carla.Color(0, 255, 255))
        debug.draw_point(right_sidewalk_wp.transform.location, size=0.2, color=carla.Color(0, 255, 0))
        debug.draw_point(left_sidewalk_wp.transform.location, size=0.2, color=carla.Color(0, 255, 0))
        for wp in candidate_wps:
            debug.draw_point(wp.transform.location, size=0.2, color=carla.Color(255, 0, 0))
    if len(candidate_wps) > 0:
        if len(candidate_wps) > spawn_points_num:
            candidate_wps = random.sample(candidate_wps, spawn_points_num)
        else:
            candidate_wps = random.sample(candidate_wps, len(candidate_wps))
    else:
        return []

    return [('*walker*', wp.transform, choice(candidate_wps).transform.location) for wp in candidate_wps]


def apply_brake(vehicles):
    for vehicle in vehicles:
        vehicle.apply_control(carla.VehicleControl(brake=1.0))


def get_nearby_vehicles_by_azimuth(world: carla.World, ego: carla.Vehicle, normal_radius=20.0, fb_radius=40.0):
    # Desc: 获取指定半径内的所有其他车辆位置，去遮挡后按照方位角分类
    # Special: 条件1: 排除高度差大于5米的车辆（立交桥的情况）
    #          条件2: 排除非正常行驶的车辆 (车祸：roll > 45 or roll < -45)
    # Desc: 8个方位角，N和S感知距离为40米，其他方位角感知距离为20米，8个方位角具体分类参见angle_to_azimuth函数
    ego_transform = ego.get_transform()
    ego_loc = ego_transform.location
    # ego_forward_vec = ego_transform.get_forward_vector()
    ego_speed = math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6

    actors = world.get_actors().filter('vehicle.*')
    azimuth_actors = []
    radius = max(normal_radius, fb_radius)
    for actor in actors:
        actor_transform = actor.get_transform()
        actor_loc = actor_transform.location
        actor_rot = actor_transform.rotation
        actor_speed = math.sqrt(actor.get_velocity().x ** 2 + actor.get_velocity().y ** 2) * 3.6
        distance = actor_loc.distance(ego_loc)
        if abs(ego_loc.z - actor_loc.z) > 5 or not radius > distance >= 2 or actor_rot.roll > 45 or actor_rot.roll < -45:
            continue
        else:
            azimuth_angle, relative_direction = calc_relative_position(ego_transform, actor_transform, only_azimuth=False, ego_speed=ego_speed, actor_speed=actor_speed)
            azimuth_actors.append((actor, azimuth_angle, relative_direction, actor_speed, distance))
    azimuth_actors = process_obscured_vehicles(azimuth_actors)

    nearby_actors = {key: [] for key in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']}
    for actor, azimuth_angle, relative_direction, speed, distance in azimuth_actors:
        if relative_direction == '':
            continue
        azimuth_key = angle_to_azimuth(azimuth_angle)
        if azimuth_key not in ['N', 'S', 'NE', 'NW', 'SE', 'SW'] and distance > normal_radius:
            continue
        data = {
            'instance': actor,
            'azimuth': azimuth_angle,
            'distance': distance,
            'direction': relative_direction,
            'speed': speed
        }
        nearby_actors[azimuth_key].append(data)

    return nearby_actors


def process_obscured_vehicles(azimuth_actors):
    # Desc: 去除被遮挡的车辆
    # Special: 条件1: 存在方位角差值小于7度的车辆
    #          条件2: 他车距离自车更小
    azimuth_actors = sorted(azimuth_actors, key=lambda x: x[-1])
    clear_azimuth_actors = []
    for i in range(len(azimuth_actors)):
        if i == 0:
            clear_azimuth_actors.append(azimuth_actors[i])
        else:
            is_clear = True
            for j in range(len(clear_azimuth_actors)):
                if azimuth_diff(azimuth_actors[i][1], clear_azimuth_actors[j][1]) < 5:
                    is_clear = False
                    break
            if is_clear:
                clear_azimuth_actors.append(azimuth_actors[i])
    return clear_azimuth_actors


def process_obstacles(azimuth_actors):
    # Desc: 处理障碍物
    # Special: 条件1: 每个方位角只保留最近的障碍物
    azimuth_actors = sorted(azimuth_actors, key=lambda x: x[-1])
    azimuth_cache = {key: False for key in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']}
    clear_azimuth_actors = []
    for i in range(len(azimuth_actors)):
        azimuth_key = angle_to_azimuth_of_obstacle(azimuth_actors[i][1])
        if azimuth_cache[azimuth_key]:
            continue
        else:
            azimuth_cache[azimuth_key] = True
            clear_azimuth_actors.append(azimuth_actors[i])
    return clear_azimuth_actors


def angle_to_azimuth(angle):
    # Desc: 将角度转换为方位角
    if 0 <= angle < 2.5:
        return 'N'
    elif 2.5 <= angle < 67.5:
        return 'NE'
    elif 67.5 <= angle < 112.5:
        return 'E'
    elif 112.5 <= angle < 175.0:
        return 'SE'
    elif 175.0 <= angle < 185.0:
        return 'S'
    elif 185.0 <= angle < 247.5:
        return 'SW'
    elif 247.5 <= angle < 292.5:
        return 'W'
    elif 292.5 <= angle < 357.5:
        return 'NW'
    elif 357.5 <= angle <= 360.0:
        return 'N'
    else:
        raise ValueError(f'angle {angle} is invalid')


def angle_to_azimuth_of_obstacle(angle):
    # Desc: 将角度转换为方位角
    if 0 <= angle < 10.0:
        return 'N'
    elif 10.0 <= angle < 67.5:
        return 'NE'
    elif 67.5 <= angle < 112.5:
        return 'E'
    elif 112.5 <= angle < 175.0:
        return 'SE'
    elif 175.0 <= angle < 185.0:
        return 'S'
    elif 185.0 <= angle < 247.5:
        return 'SW'
    elif 247.5 <= angle < 292.5:
        return 'W'
    elif 292.5 <= angle < 350.0:
        return 'NW'
    elif 350.0 <= angle <= 360.0:
        return 'N'
    else:
        raise ValueError(f'angle {angle} is invalid')


def angle_to_azimuth_of_pedestrian(angle):
    return angle_to_azimuth_of_obstacle(angle)


def azimuth_diff(a1, a2):
    # Desc: 计算两个方位角的差值
    diff1 = abs(a1 - a2)
    if a1 < a2:
        diff2 = abs(a1 + 360 - a2)
    else:
        diff2 = abs(a2 + 360 - a1)
    return min(diff1, diff2)


def calc_relative_position(ego, actor, only_azimuth=False, ego_speed=-1.0, actor_speed=-1.0):
    # Desc: 计算相对位置
    if isinstance(ego, carla.Transform):
        ego_transform = ego
    elif isinstance(ego, carla.Actor):
        ego_transform = ego.get_transform()
    else:
        raise NotImplementedError

    if isinstance(actor, carla.Transform):
        actor_transform = actor
    elif isinstance(actor, carla.Actor):
        actor_transform = actor.get_transform()
    else:
        raise NotImplementedError

    # Desc: 计算他车相对于自车的方位角
    # Special: 自车前进向量顺时针旋转到自车指向他车的向量的角度（360度制）
    v1 = ego_transform.get_forward_vector()
    v2 = actor_transform.location - ego_transform.location
    v1 = np.array([-v1.x, v1.y])
    v2 = np.array([-v2.x, v2.y])
    v2 = normalize(v2)
    v12_cos_value = np.dot(v1, v2)
    v12_cos_value = np.clip(v12_cos_value, -1, 1)
    v12_sin_value = np.cross(v1, v2)
    v12_sin_value = np.clip(v12_sin_value, -1, 1)
    v12_180_angle = math.degrees(math.acos(v12_cos_value))
    v12_360_angle = 360 - v12_180_angle if v12_sin_value > 0 else v12_180_angle

    if only_azimuth:
        return v12_360_angle

    # Desc: 计算他车相对于自车是驶来还是驶去还是静止
    # Special: 驶来条件1：自车前进向量与自车指向他车向量的180度制角度小于90度
    #          驶来条件2：自车前进向量与他车前进向量180度制角度大于90度
    #          驶来条件3：自车速度大于他车速度
    if ego_speed == -1.0:
        ego_speed = math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6
    if actor_speed == -1.0:
        actor_speed = math.sqrt(actor.get_velocity().x ** 2 + actor.get_velocity().y ** 2) * 3.6

    if actor_speed > 0.1:
        v3 = actor_transform.get_forward_vector()
        v3 = np.array([-v3.x, v3.y])

        ego_loc = np.array([-ego_transform.location.x, ego_transform.location.y])
        actor_loc = np.array([-actor_transform.location.x, actor_transform.location.y])

        old_distance = math.sqrt((ego_loc[0] - actor_loc[0]) ** 2 + (ego_loc[1] - actor_loc[1]) ** 2)
        actor_next_loc = actor_loc + v3 * actor_speed / 3.6 / 20
        ego_next_loc = ego_loc + v1 * ego_speed / 3.6 / 20
        next_distance = math.sqrt((ego_next_loc[0] - actor_next_loc[0]) ** 2 + (ego_next_loc[1] - actor_next_loc[1]) ** 2)

        if abs(next_distance - old_distance) > 0.1 or True:
            if next_distance < old_distance:
                relative_direction = 'approaching'
            else:
                relative_direction = 'leaving'
        else:
            relative_direction = ''
    else:
        relative_direction = 'stilling'

    return v12_360_angle, relative_direction


def move_waypoint_forward(wp, distance):
    # Desc: 将waypoint沿着前进方向移动一定距离
    dist = 0
    next_wp = wp
    while dist < distance:
        next_wps = next_wp.next(1)
        if not next_wps or next_wps[0].is_junction:
            break
        next_wp = next_wps[0]
        dist += 1
    return next_wp


def move_waypoint_backward(wp, distance):
    # Desc: 将waypoint沿着反方向移动一定距离
    dist = 0
    next_wp = wp
    while dist < distance:
        next_wps = next_wp.previous(1)
        if not next_wps or next_wps[0].is_junction:
            break
        next_wp = next_wps[0]
        dist += 1
    return next_wp


def get_description(explainable_data, actor):
    return Edict(explainable_data['actors'][actor.id]['description'])


def get_actor_data(explainable_data, actor):
    return Edict(explainable_data['actors'][actor.id])


def get_next_junction(wp, debug=False):
    # Desc: 获取下一个路口
    while wp.is_junction or wp.is_intersection:
        wp = wp.next(1)[0]
    try:
        wp = wp.next_until_lane_end(1)[-1]
    except Exception as e:
        return None
        # traceback.print_exc()
        # print(e)
        # fast_debug(wp)
    wp = wp.next(2)[0]
    if not wp.is_junction:
        if debug:
            print(f'get_next_junction: wp {wp} is not a junction')
        return None
    return wp.get_junction()


def get_junction_sidewalk_wps(junction):
    side_walk_wps = []
    for item in junction.get_waypoints(carla.LaneType.Shoulder):
        side_walk_wps.append(item[0])
        side_walk_wps.append(item[-1])
    return side_walk_wps


def sort_wp_by_ref(wps, wp) -> List[carla.Waypoint]:
    # Desc: 将waypoint按照与参考点的距离排序
    wps = sorted(wps, key=lambda x: x.transform.location.distance(wp.transform.location))
    return wps


def get_sidewalk_wps(ori_wp):
    wp = ori_wp.get_right_lane()
    while wp is not None and wp.lane_type != carla.LaneType.Sidewalk:
        wp = wp.get_right_lane()
    if wp is None:
        right_sidewalk_wp = None
    else:
        right_sidewalk_wp = wp

    wp = ori_wp.get_left_lane()
    direction_change = False
    while wp is not None and wp.lane_type != carla.LaneType.Sidewalk:
        if wp.lane_id * ori_wp.lane_id < 0:
            direction_change = True
        if direction_change:
            wp = wp.get_right_lane()
        else:
            wp = wp.get_left_lane()
    if wp is None:
        left_sidewalk_wp = None
    else:
        left_sidewalk_wp = wp
    return left_sidewalk_wp, right_sidewalk_wp


def gen_ai_walker(world, transform, CarlaDataProvider):
    pedestrian = CarlaDataProvider.request_new_actor('walker.*', transform)
    if pedestrian is not None:
        controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        controller = world.spawn_actor(controller_bp, pedestrian.get_transform(), pedestrian)
        return pedestrian, controller
    else:
        return None, None


def junction_has_traffic_light(world, wp):
    # Desc: 判断路口是否有红绿灯
    wp = wp.next_until_lane_end(1)[-1]
    wp = wp.next(2)[0]
    if not wp.is_junction:
        print(f'junction_has_traffic_light: wp {wp} is not a junction')
        return False
    front_junction = wp.get_junction()
    traffic_lights_list = world.get_traffic_lights_in_junction(front_junction.id)
    return len(traffic_lights_list) > 0

# def junction_first_traffic_light_bbox(world, wp):
#     """
#     判断路口是否有红绿灯，并返回找到的第一个红绿灯的第一个bbox。
#     """
#     wp = wp.next_until_lane_end(1)[-1]
#     wp = wp.next(2)[0]
#     if not wp.is_junction:
#         print(f'junction_has_traffic_light: wp {wp} is not a junction')
#         return None
#     front_junction = wp.get_junction()
#     traffic_lights_list = world.get_traffic_lights_in_junction(front_junction.id)
#     if len(traffic_lights_list) > 0:
#         first_traffic_light = traffic_lights_list[0]
#         light_boxes = first_traffic_light.get_light_boxes()
#         if light_boxes:
#             return light_boxes[0]  # 返回路灯的第一个bbox
#     return None

# 获取当前车道最左边车道的对应点
def left_wp(wp):
    leftmost_waypoint = wp
    while leftmost_waypoint.lane_change == carla.LaneChange.Left:
        next_waypoint = leftmost_waypoint.get_left_lane()
        if next_waypoint is None or next_waypoint.lane_type != carla.LaneType.Driving:
            break
        leftmost_waypoint = next_waypoint
    return leftmost_waypoint

#获取当前车道最右边车道的对应点
def right_wp(wp):
    rightmost_waypoint = wp
    while rightmost_waypoint.lane_change == carla.LaneChange.Right:
        next_waypoint = rightmost_waypoint.get_right_lane()
        if next_waypoint is None or next_waypoint.lane_type != carla.LaneType.Driving:
            break
        rightmost_waypoint = next_waypoint
    return rightmost_waypoint

def junction_first_traffic_light(world, wp):
    """
    判断路口是否有红绿灯，并返回自车 Waypoint 到下一个红绿灯的距离和当前状态green/yellow/red.
    """
    debug = world.debug
    lane_info = get_lane_info(wp)
    # print(f'车道信息{lane_info}')
    ego_point = wp # 每一个tick的自车坐标
    ego_point_left = ego_point
    ego_point_right = ego_point
    if lane_info.r2l > 1 :
        ego_point_right = right_wp(ego_point)
    if lane_info.l2r > 1 :
        ego_point_left = left_wp(ego_point)

    # debug.draw_point(ego_point_left.transform.location + carla.Location(z=4.3), size=0.5, color=carla.Color(255, 255, 0), life_time=999)
    
    # 获取道路终点的下一个 Waypoint
    wp = wp.next_until_lane_end(1)[-1]
    # debug.draw_point(wp.transform.location + carla.Location(z=4.3), size=0.15, color=carla.Color(0, 255, 0), life_time=999)
    wp = wp.next(2)[0]
    # debug.draw_point(wp.transform.location + carla.Location(z=4.3), size=0.15, color=carla.Color(255, 255, 0), life_time=999)

    
    

    # 检查是否是路口
    if not wp.is_junction:
        # print(f'junction_has_traffic_light: wp {wp} is not a junction')
        return 0,'off',None,None,None
    
    # 获取前方路口
    front_junction = wp.get_junction()
    
    #判断能否左右转
    left_turn = get_junction_turn_after_wp(front_junction, ego_point_left, 'left', visual_manager=None, debug=False)
    
    right_turn = get_junction_turn_after_wp(front_junction, ego_point_right, 'right', visual_manager=None, debug=False)
    # debug.draw_point(right_turn.transform.location + carla.Location(z=4.3), size=0.5, color=carla.Color(255, 255, 0), life_time=999)
    straight_turn = get_junction_turn_after_wp(front_junction, ego_point, 'straight', visual_manager=None, debug=False)
    
    # 获取自车位置
    ego_location = ego_point.transform.location
    # ego_transform = ego_vehicle.get_transform()
    # ego_rotation = ego_transform.rotation
    # 获取路口内的所有交通灯

    traffic_lights_list = world.get_traffic_lights_in_junction(front_junction.id)
    # print(f'所有信号灯{traffic_lights_list}')
    

    # # 算角度
    # angle_threshold = 8.0
    # if len(traffic_lights_list) > 0:
    #     min_distance = float('inf')
    #     first_traffic_light = None
    #     for traffic_light in traffic_lights_list:
    #         # 获取交通灯位置
    #         traffic_light_transform = traffic_light.get_transform()
    #         traffic_light_location = traffic_light.get_transform().location
            
    #         # 计算交通灯相对于自车的方向
    #         vector_to_traffic_light = traffic_light_location - ego_location
    #         angle_to_traffic_light = math.degrees(math.atan2(vector_to_traffic_light.y, vector_to_traffic_light.x)) - ego_rotation.yaw
    #         angle_to_traffic_light = (angle_to_traffic_light + 360) % 360  # 规范化到0-360度
            
    #         # 判断交通灯是否在车辆前方一定角度范围内
    #         if angle_to_traffic_light < angle_threshold or angle_to_traffic_light > 360 - angle_threshold:
    #             # 计算与自车位置的距离
    #             distance = ego_location.distance(traffic_light_location)
                
    #             # 更新最小距离和对应的交通灯
    #             if distance < min_distance:
    #                 min_distance = distance
    #                 first_traffic_light = traffic_light
    #找最小距离
    if len(traffic_lights_list) > 0:

        if straight_turn is not None:
            # 初始化最小距离和相应的交通灯
            min_distance = 100000
            first_traffic_light = None
            for traffic_light in traffic_lights_list:
                # 获取交通灯位置
                traffic_light_location = traffic_light.get_transform().location
                straight_turn_location = straight_turn.transform.location
                # debug.draw_point(straight_turn_location + carla.Location(z=4.3), size=0.5, color=carla.Color(0, 255, 0), life_time=999)
                # 计算与前方路点位置的距离
                distance = straight_turn_location.distance(traffic_light_location)
                
                # 更新最小距离和对应的交通灯
                if distance < min_distance:
                    min_distance = distance
                    first_traffic_light = traffic_light
        
            #初始化最大距离和相应的交通灯
            # max_distance = 200
            # first_traffic_light = None

            # for traffic_light in traffic_lights_list:
            #     # 获取交通灯位置
            #     traffic_light_location = traffic_light.get_transform().location
                
            #     # 计算与自车位置的距离
            #     distance = straight_turn.transform.location.distance(traffic_light_location)
                
            #     # 更新最大距离和对应的交通灯
            #     if distance < max_distance:
            #         max_distance = distance
            #         first_traffic_light = traffic_light

    #原版            
    # if len(traffic_lights_list) > 0:
    #     first_traffic_light = traffic_lights_list[0]
        # if len(traffic_lights_list) > 2:
        #     first_traffic_light = traffic_lights_list[1]
        # elif len(traffic_lights_list) == 2:
        #     first_traffic_light = traffic_lights_list[1]
        # elif len(traffic_lights_list) == 1:
        #     first_traffic_light = traffic_lights_list[0]
        

        else:
        #原版            
        
            first_traffic_light = traffic_lights_list[0]
            if len(traffic_lights_list) > 2:
                first_traffic_light = traffic_lights_list[1]
            elif len(traffic_lights_list) == 2:
                first_traffic_light = traffic_lights_list[1]
            elif len(traffic_lights_list) == 1:
                first_traffic_light = traffic_lights_list[0]
            

        # print(f'自车坐标{ego_location}')
        # 获取交通灯位置
        traffic_light_location = first_traffic_light.get_transform().location

        # debug.draw_point(traffic_light_location + carla.Location(z=4.3), size=0.5, color=carla.Color(0, 255, 0), life_time=999)
        # 计算距离
        distance = ego_location.distance(traffic_light_location)
        
        # 获取交通灯状态
        traffic_light_state = first_traffic_light.get_state()
        # 将交通灯状态转换为字符串
        state_str = 'off'
        if traffic_light_state == carla.TrafficLightState.Red:
            state_str = 'red'
        elif traffic_light_state == carla.TrafficLightState.Yellow:
            state_str = 'yellow'
        elif traffic_light_state == carla.TrafficLightState.Green:
            state_str = 'green'
        return distance, state_str,left_turn,right_turn,straight_turn
    
    return 0,'off',None,None,None

def distance_to_next_junction(wp, default=0):
    # Desc: 获取到下一个路口的距离
    try:
        distance = 0
        while True:
            if wp.is_junction or wp.is_intersection:
                break
            next_wps = wp.next(1.0)
            if len(next_wps) >= 1:
                wp = next_wps[0]
            else:
                break
            distance += 1
        return distance
    except Exception:
        return default

def distance_to_previous_junction(wp, default=0):
    # Desc: 获取到上一个路口的距离
    try:
        distance = 0
        while True:
            if wp.is_junction or wp.is_intersection:
                break
            next_wps = wp.previous(1.0)
            if len(next_wps) >= 1:
                wp = next_wps[0]
            else:
                break
            distance += 1
        return distance
    except Exception:
        return default


def get_road_end_wp(wp):
    # Desc: 获取路的终点
    while wp.is_junction or wp.is_intersection:
        wp = wp.next(1)[0]
    wp = wp.next_until_lane_end(2)[-1]
    return wp


def dot_product(v1, v2, norm=True):
    # Desc: 计算两个向量的点积
    if isinstance(v1, carla.Vector3D):
        v1 = vector3d_to_2darray(v1)
        if norm:
            v1 = normalize(v1)
    if isinstance(v2, carla.Vector3D):
        v2 = vector3d_to_2darray(v2)
        if norm:
            v2 = normalize(v2)
    return np.cross(v1, v2)


def vector3d_to_2darray(vector3d):
    # Desc: 将carla.Vector3D转换为np.array
    return np.array([vector3d.x, vector3d.y])


def normalize(v):
    # Desc: 将向量归一化
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def get_road_middle_wp(wp):
    # Desc: 获取路的中点
    while wp.is_junction:
        wp = wp.next(1)[0]
    tmp = wp.next_until_lane_end(1)
    if len(tmp) == 1:
        return tmp[0]
    else:
        return tmp[len(tmp) // 2]


def get_vehicle_front_trafficlight(all_trafficlights, vehicle, wmap, distance=30):
    # Desc: 获取车辆前方30米内的红绿灯
    if vehicle.is_at_traffic_light():
        traffic_light = vehicle.get_traffic_light()
        return traffic_light, 0
    else:
        vehicle_wp = wmap.get_waypoint(vehicle.get_location())
        true_distance = distance_to_next_junction(vehicle_wp)
        if true_distance > distance:
            return None, -1

        vehicle_road = vehicle_wp.road_id
        vehicle_lane = vehicle_wp.lane_id
        for traffic_light in all_trafficlights:
            stop_wps = traffic_light.get_stop_waypoints()
            for stop_wp in stop_wps:
                if stop_wp.road_id == vehicle_road and stop_wp.lane_id == vehicle_lane:
                    return traffic_light, true_distance
        return None, -1


def get_junction_turn_after_wp(junction, start_wp, direction, visual_manager=None, debug=False):
    # Desc: 获取在路口某个方向转向后的第一个waypoint
    junc_wps = junction.get_waypoints(carla.LaneType.Driving)
    start_wp = get_road_end_wp(start_wp)
    threshold_distance = start_wp.lane_width / 2.0

    connections = []
    for wps in junc_wps:
        cur_distance = wps[0].transform.location.distance(start_wp.transform.location)
        if cur_distance > threshold_distance:
            continue
        else:
            connections.append(wps)

    if direction == 'any':
        return choice(connections)[1]

    if visual_manager is not None:
        for connection in connections:
            visual_manager.show_waypoints(connection, color=carla.Color(255, 0, 0))

    forward_vector = start_wp.transform.get_forward_vector()
    for connection in connections:
        connection_vector = connection[1].transform.location - connection[0].transform.location
        dot_value = dot_product(connection_vector, forward_vector, norm=True)

        if direction == 'left' and dot_value > 0.1:
            return connection[1]
        elif direction == 'right' and dot_value < -0.1:
            return connection[1]
        elif direction == 'straight' and abs(dot_value) < 0.1:
            return connection[1]

    if debug:
        print(f'get_junction_turn_after_wp: can not find a waypoint in direction {direction} after wp {start_wp}')
    return None

#liuhao version
def get_junction_turn_after_wp_new(junction, start_wp, direction, world,visual_manager=None, debug=False):
    # Desc: 获取在路口某个方向转向后的第一个waypoint

    if junction is None:
        return None
    else:
        junc_wps = junction.get_waypoints(carla.LaneType.Driving)
        start_wp = get_road_end_wp(start_wp)
        threshold_distance = start_wp.lane_width / 2.0

        connections = []
        for wps in junc_wps:
            cur_distance = wps[0].transform.location.distance(start_wp.transform.location)
            if cur_distance > threshold_distance:
                continue
            else:
                connections.append(wps)

        if direction == 'any':
            return choice(connections)[1]

        if visual_manager is not None:
            for connection in connections:
                visual_manager.show_waypoints(connection, color=carla.Color(255, 0, 0))

        forward_vector = start_wp.transform.get_forward_vector()
        for connection in connections:
            connection_vector = connection[1].transform.location - connection[0].transform.location
            dot_value = dot_product(connection_vector, forward_vector, norm=True)

            if direction == 'left' and dot_value > 0.1:
                
                return connection[1]
            elif direction == 'right' and dot_value < -0.1:

                return connection[1]
            elif direction == 'straight' and abs(dot_value) < 0.1:
                return connection[1]

        if debug:
            print(f'get_junction_turn_after_wp: can not find a waypoint in direction {direction} after wp {start_wp}')
        return None


def has_pedestrian_in_front(pedestrians, ego, distance=40.0, lane_width=3.5) -> bool:
    # Desc: 判断前方是否有行人
    # Special: 感知两个车道的宽度
    if distance < 0.1:
        distance = 10
    ego_forward_vec = ego.get_transform().get_forward_vector()
    ego_loc = ego.get_location()
    min_cos = math.cos(math.atan((lane_width * 1.0) / distance))
    max_distance = math.sqrt(distance ** 2 + lane_width ** 2)
    for pedestrian, _ in pedestrians:
        ped_loc = pedestrian.get_location()
        if ped_loc.distance(ego_loc) > max_distance:
            continue
        ego2ped = pedestrian.get_location() - ego_loc
        cos_value = calc_cos_between_vector(ego_forward_vec, ego2ped)
        if cos_value > min_cos:
            return True
    return False


def get_lane_info(ori_wp):
    # TODO: 潮汐车道未处理
    # Desc: 计算waypoint所在的车道编号
    if ori_wp.lane_type != carla.LaneType.Driving:
        return None

    # 先定位到最右侧车道
    wp = ori_wp
    while wp.get_right_lane() is not None and wp.get_right_lane().lane_type == carla.LaneType.Driving:
        wp = wp.get_right_lane()

    # 从右至左遍历车道
    lane_ids = []
    while True:
        # 如果当前车道已经在列表中，则停止遍历
        if wp.lane_id in lane_ids:
            break
        lane_ids.append(wp.lane_id)
        # 如果左侧没有车道或者左侧车道不是行驶车道，则停止遍历
        if wp.get_left_lane() is None or wp.get_left_lane().lane_type != carla.LaneType.Driving:
            break
        # 判断对向车道
        lr_lane = wp.get_left_lane().get_right_lane()
        if lr_lane is None or lr_lane.lane_type != carla.LaneType.Driving or lr_lane.lane_id not in lane_ids:
            break
        wp = wp.get_left_lane()

    # 切换至从左至右的顺序
    lane_ids = list(reversed(lane_ids))

    left_lanemarking = ori_wp.left_lane_marking
    right_lanemarking = ori_wp.right_lane_marking
    if str(left_lanemarking.type) == 'BrokenSolid' or 'Broken' in str(left_lanemarking.type):
        left_change = True
    else:
        left_change = False
    if str(right_lanemarking.type) == 'SolidBroken' or 'Broken' in str(right_lanemarking.type):
        right_change = True
    else:
        right_change = False

    lane_info = {
        'num': len(lane_ids),
        'l2r': lane_ids.index(ori_wp.lane_id) + 1,
        'r2l': len(lane_ids) - lane_ids.index(ori_wp.lane_id),
        'lm': left_lanemarking.type,
        'rm': right_lanemarking.type,
        'lchange': left_change,
        'rchange': right_change
    }
    return Edict(lane_info)


def find_spectator(world):
    spec = None
    for item in world.get_actors():
        if item.type_id == 'spectator':
            spec = item
            break

    if spec is None:
        print('look_actor: can not find spectator')
        return None
    return spec


def actor_front_of_ego(data):
    return data['cos_value'] > 0


def actor_right_of_ego(data):
    return data['cross_value'] >= 0


def get_blueprint(world, name, excludes=None):
    # Desc: 获取蓝图
    if excludes is None:
        excludes = []
    blueprints = world.get_blueprint_library().filter(name)
    blueprint = choice([x for x in blueprints if x.id not in excludes])
    return blueprint.id


def look_actor(world, actor):
    spec = find_spectator(world)
    look_trans = carla.Transform(actor.get_location() + carla.Location(z=10), carla.Rotation(pitch=-90))
    spec.set_transform(look_trans)


def look_waypoint(world, wp):
    spec = find_spectator(world)
    look_trans = carla.Transform(wp.transform.location + carla.Location(z=10), carla.Rotation(pitch=-90))
    spec.set_transform(look_trans)


def look_transform(world, transform):
    spec = find_spectator(world)
    look_trans = carla.Transform(transform.location + carla.Location(z=10), carla.Rotation(pitch=-90))
    spec.set_transform(look_trans)


def look_location(world, location):
    spec = find_spectator(world)
    look_trans = carla.Transform(location + carla.Location(z=10), carla.Rotation(pitch=-90))
    spec.set_transform(look_trans)


def debug_loc(world, item):
    if isinstance(item, carla.Waypoint):
        loc = item.transform.location
    elif isinstance(item, carla.Location):
        loc = item
    elif isinstance(item, carla.Transform):
        loc = item.location
    else:
        raise NotImplementedError

    debug = world.debug
    debug.draw_point(loc, size=0.2, color=carla.Color(255, 0, 0), life_time=10000)
    look_location(world, loc)


class ShowText(object):
    def __init__(self, world):
        self.text = ''
        self.last_text = ''
        self.world = world
        self.show = True

    def set_text(self, text):
        self.text = text

    def get_show_loc(self):
        spec = find_spectator(self.world)
        spec_forward_vec = spec.get_transform().get_forward_vector()
        spec_right_vec = spec.get_transform().get_right_vector()
        spec_up_vec = spec.get_transform().get_up_vector()
        distance = [0.2, 0, 0.1]
        draw_loc = spec.get_transform().location + spec_forward_vec * distance[0] + spec_right_vec * distance[1] + spec_up_vec * distance[2]
        return draw_loc

    def show(self):
        draw_loc = self.get_show_loc()
        self.world.debug.draw_string(draw_loc, self.text, draw_shadow=True, color=carla.Color(255, 0, 0), life_time=0.2)


def hue_calculate(round1, round2, delta, add_num):
    return (((round1 - round2) / delta) * 60 + add_num) % 360


def rgb_to_hsv(rgb_seq):
    r, g, b = rgb_seq
    r_round = float(r) / 255
    g_round = float(g) / 255
    b_round = float(b) / 255
    max_c = max(r_round, g_round, b_round)
    min_c = min(r_round, g_round, b_round)
    delta = max_c - min_c

    h = None
    if delta == 0:
        h = 0
    elif max_c == r_round:
        h = hue_calculate(g_round, b_round, delta, 360)
    elif max_c == g_round:
        h = hue_calculate(b_round, r_round, delta, 120)
    elif max_c == b_round:
        h = hue_calculate(r_round, g_round, delta, 240)
    if max_c == 0:
        s = 0
    else:
        s = (delta / max_c) * 100
    v = max_c * 100
    return h, s, v


def color_cls(rgb_seq):
    """
    将rgb转为hsv之后根据h和v寻找色系
    :param rgb_seq:
    :return:
    """
    h, s, v = rgb_to_hsv(rgb_seq)
    cs = None
    if 30 < h <= 90:
        cs = 'yellow'
    elif 90 < h <= 150:
        cs = 'green'
    elif 150 < h <= 210:
        cs = 'cyan'
    elif 210 < h <= 270:
        cs = 'blue'
    elif 270 < h <= 330:
        cs = 'purple'
    elif h > 330 or h <= 30:
        cs = 'red'

    if s < 10:  # 色相太淡时，显示什么颜色主要由亮度来决定
        if v <= 100 / 3 * 1:
            cs = 'black'
        elif v <= 100 / 3 * 2:
            cs = 'gray'
        else:
            cs = 'white'
    return cs


def fast_debug(input_data):
    client = carla.Client('localhost', 2000)

    if isinstance(input_data, carla.Waypoint):
        transform = input_data.transform
    elif isinstance(input_data, carla.Transform):
        transform = input_data
    elif isinstance(input_data, carla.Location):
        transform = carla.Transform(input_data, carla.Rotation())
    else:
        raise NotImplementedError

    world = client.get_world()
    spec = find_spectator(world)
    spec.set_transform(transform)

    debug = world.debug
    debug.draw_point(transform.location, size=0.2, color=carla.Color(0, 0, 255), life_time=10000)

    import pdb
    pdb.set_trace(context=10)
    pause = 1


def calc_cos_between_vector(carla_v1, carla_v2):
    # Desc: 计算两个carla.Vector3D之间的夹角cos
    v1 = np.array([carla_v1.x, carla_v1.y])
    v2 = np.array([carla_v2.x, carla_v2.y])
    v1 = normalize(v1)
    v2 = normalize(v2)
    cos_value = np.dot(v1, v2)
    return cos_value


def do_sample(data_dict):
    return random.choices(list(data_dict.keys()), weights=list(data_dict.values()), k=1)[0]


def spectator_focus_actor(spec, actor):
    transform = actor.get_transform()
    backward_vec = transform.get_forward_vector() * -1
    spec_loc = transform.location + carla.Location(z=5) + backward_vec * 7
    spec_rot = carla.Rotation(pitch=-30, yaw=transform.rotation.yaw)
    spec.set_transform(carla.Transform(spec_loc, spec_rot))


def spectator_focus_wp(spec, wp):
    transform = wp.transform
    backward_vec = transform.get_forward_vector() * -1
    spec_loc = transform.location + carla.Location(z=5) + backward_vec * 7
    spec_rot = carla.Rotation(pitch=-30, yaw=transform.rotation.yaw)
    spec.set_transform(carla.Transform(spec_loc, spec_rot))


def distance_between_waypoints(wp1, wp2):
    return wp1.transform.location.distance(wp2.transform.location)


def calculate_y_for_given_x(x, point1, point2):
    # 计算斜率
    m = (point2[1] - point1[1]) / (point2[0] - point1[0])

    # 计算截距
    b = point1[1] - m * point1[0]

    # 计算y
    y = m * x + b
    return y

def calculate_speed(current_distance, initial_distance, initial_speed, min_distance):
    """
    根据当前距离计算速度。

    参数:
    current_distance (float): 当前距离
    initial_distance (float): 初始距离
    initial_speed (float): 初始速度
    min_distance (float): 最小距离，达到该距离时速度为0

    返回:
    float: 当前速度
    """
    if current_distance <= min_distance:
        return 0
    return initial_speed * (current_distance - min_distance) / (initial_distance - min_distance)

def position_with_respect_to_vector_2d(_point, _vector):
    vector = np.array([_vector.x, _vector.y])
    point = np.array([_point.x, _point.y])
    x1, y1 = vector[0], vector[1]
    a, b = point[0], point[1]

    # 在2D空间中，点的相对位置有可能正好在向量上，也就是垂足
    # 计算垂足
    foot = ((a * x1 + b * y1) / (x1 * x1 + y1 * y1)) * vector

    # 计算垂足和点之间的向量
    point_vector = point - foot

    # 计算向量与点之间的点积
    _dot_product = sum(p * q for p, q in zip([x1, y1], point_vector))

    # 如果点积大于0，那么点在向量的前方
    # 如果点积小于0，那么点在向量的后方
    # 如果点积等于0，那么点在向量上
    if _dot_product >= 0:
        # "在前方"
        return 'front'
    else:
        # "在后方"
        return 'back'
