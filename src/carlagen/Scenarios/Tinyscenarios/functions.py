# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 下午9:29
# @Author  : Hcyang
# @File    : functions.py
# @Desc    : TODO:

import math
import random
import carla
import numpy as np

from easydict import EasyDict as Edict


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


def move_waypoint_forward_through_junction(wp, distance):
    # Desc: 将waypoint沿着前进方向移动一定距离，如果遇到路口则直行
    dist = 0
    next_wp = wp
    while dist < distance:
        next_wps = next_wp.next(1)
        if not next_wps:
            break
        if next_wps[0].is_junction:
            next_wp = get_junction_turn_after_wp(next_wps[0].get_junction(), next_wp, 'straight')
        else:
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


def angle_to_azimuth_of_pedestrian(angle):
    return angle_to_azimuth_of_obstacle(angle)


def normalize(v):
    # Desc: 将向量归一化
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


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


def azimuth_diff(a1, a2):
    # Desc: 计算两个方位角的差值
    diff1 = abs(a1 - a2)
    if a1 < a2:
        diff2 = abs(a1 + 360 - a2)
    else:
        diff2 = abs(a2 + 360 - a1)
    return min(diff1, diff2)


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


def hue_calculate(round1, round2, delta, add_num):
    return (((round1 - round2) / delta) * 60 + add_num) % 360


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
        return random.choice(connections)[1]

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
