

import math
import random
# # import ipdb

import numpy as np
from easydict import EasyDict as Edict
import pdb

# 剩下的都是汽车
VEHICLE_TYPE = {
    '卡车': ['vehicle.carlamotors.carlacola', 'vehicle.carlamotors.european_hgv', 'vehicle.tesla.cybertruck'],
    '货车': ['vehicle.mercedes.sprinter', 'vehicle.volkswagen.t2', 'vehicle.volkswagen.t2_2021'],
    '公共汽车': ['vehicle.mitsubishi.fusorosa'],
    '摩托车': ['vehicle.harley-davidson.low_rider', 'vehicle.kawasaki.ninja', 'vehicle.vespa.zx125', 'vehicle.yamaha.yzf'],
    '自行车': ['vehicle.bh.crossbike', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets'],
    '消防车': ['vehicle.carlamotors.firetruck'],
    '救护车': ['vehicle.ford.ambulance'],
    '警车': ['vehicle.dodge.charger_police', 'vehicle.dodge.charger_police_2020'],
    '出租车': ['vehicle.ford.crown']
}

ENG_VEHICLE_TYPE = {
    '卡车': 'truck', '货车': 'wagon', '公共汽车': 'bus', '摩托车': 'motorbike', '自行车': 'bike', '消防车': 'fire engine',
    '救护车': 'ambulance', '警车': 'police car', '出租车': 'taxi', '轿车': 'car',
}
ENG_COLOR = {
    '白色': 'white', '红色': 'red', '绿色': 'green', '蓝色': 'blue', '黄色': 'yellow',
    '品红色': 'magenta', '青色':'cyan', '黑色': 'black', '未知颜色': '',
}
ENG_SPEED = {'静止': 'stopped', '缓慢': 'slow-moving', '低速': 'slow-moving', '高速': 'fast-moving'}
ENG_DIRECTION = {
    # at the xxx of ego
    '正前方': 'front ahead', '左前方': 'front left', '右前方': 'front right',
    '正后方': 'right back', '左后方': 'rear left', '右后方': 'rear right',
    '右侧': 'right side', '左侧': 'left side',
}
ENG_DISTANCE = {
    '近': 'near', '较近': 'slightly near', '远': 'far away from', '较远': 'slightly far away from',
}


def _normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def calc_head_lane_cos(wmap, actor):
    # Desc: 计算车头朝向与当前最近waypoint的夹角cos
    # Special: 但是最后没用上
    wp = wmap.get_waypoint(actor.get_location())
    wp_forward_vec = wp.transform.get_forward_vector()
    wp_forward_vec = np.array([wp_forward_vec.x, wp_forward_vec.y])
    wp_forward_vec = _normalize(wp_forward_vec)
    actor_forward_vec = actor.get_transform().get_forward_vector()
    actor_forward_vec = np.array([actor_forward_vec.x, actor_forward_vec.y])
    actor_forward_vec = _normalize(actor_forward_vec)
    cos_value = np.dot(wp_forward_vec, actor_forward_vec)
    return cos_value


def calc_cos_between_vector(carla_v1, carla_v2):
    # Desc: 计算两个carla.Vector3D之间的夹角cos
    v1 = np.array([carla_v1.x, carla_v1.y])
    v2 = np.array([carla_v2.x, carla_v2.y])
    v1 = _normalize(v1)
    v2 = _normalize(v2)
    cos_value = np.dot(v1, v2)
    return cos_value


def _calc_static_data(ego, actor, eng=False):
    """
    计算静态物体相对于自车的数据
    """
    ego_forward_vec = ego.get_transform().get_forward_vector()
    ego_loc = ego.get_location()
    actor_loc = actor.get_location()
    actor2ego_vec = actor_loc - ego_loc
    actor2ego_vec = np.array([actor2ego_vec.x, actor2ego_vec.y])
    actor2ego_vec = _normalize(actor2ego_vec)
    ego_forward_vec = np.array([ego_forward_vec.x, ego_forward_vec.y])

    # 计算前后左右
    cos_value = np.dot(ego_forward_vec, actor2ego_vec)
    cross_value = np.cross(ego_forward_vec, actor2ego_vec)
    distance = actor_loc.distance(ego.get_location())

    description = {}
    # 生成方向描述
    direction = []
    if abs(cos_value) > 0.965:
        if cos_value > 0:
            direction.append('正前方')
        else:
            direction.append('正后方')
    else:
        if cross_value < 0:
            direction.append('左')
        elif cross_value >= 0:
            direction.append('右')

        if abs(cos_value) < 0.25:
            direction.append('侧')
        else:
            if cos_value > 0:
                direction.append('前方')
            else:
                direction.append('后方')
    direction = ''.join(direction)
    if eng:
        direction = ENG_DIRECTION[direction]
    description['direction'] = direction

    # 生成距离描述
    if distance < 10:
        distance_desc = '近'
    elif distance < 20:
        distance_desc = '较近'
    elif distance < 30:
        distance_desc = '较远'
    else:
        distance_desc = '远'
    if eng:
        distance_desc = ENG_DISTANCE[distance_desc]
    description['distance'] = distance_desc

    if eng:
        description['type'] = 'obstacle'
    else:
        description['type'] = '障碍物'

    # cos_value转为float
    cos_value = float(cos_value)
    cross_value = float(cross_value)

    data = {
        'type_id': actor.type_id,
        'cos_value': cos_value,
        'cross_value': cross_value,
        'location': {'x': actor_loc.x, 'y': actor_loc.y, 'z': actor_loc.z},
        'distance': distance,
        'description': description
    }

    return data


def _calc_interactive_data(ego, actor, eng=False):
    """
    计算其他车辆相对于自车的数据
    """
    actor_speed = math.sqrt(actor.get_velocity().x ** 2 + actor.get_velocity().y ** 2) * 3.6
    
    ego_forward_vec = ego.get_transform().get_forward_vector()
    ego_loc = ego.get_location()
    actor_loc = actor.get_location()
    actor2ego_vec = actor_loc - ego_loc
    actor2ego_vec = np.array([actor2ego_vec.x, actor2ego_vec.y])
    actor2ego_vec = _normalize(actor2ego_vec)
    ego_forward_vec = np.array([ego_forward_vec.x, ego_forward_vec.y])

    # 计算前后左右
    cos_value = np.dot(ego_forward_vec, actor2ego_vec)
    cross_value = np.cross(ego_forward_vec, actor2ego_vec)

    distance = actor_loc.distance(ego.get_location())

    description = {}
    # 生成方向描述
    direction = []
    if abs(cos_value) > 0.965:
        if cos_value > 0:
            direction.append('正前方')
        else:
            direction.append('正后方')
    else:
        if cross_value < 0:
            direction.append('左')
        elif cross_value >= 0:
            direction.append('右')

        if abs(cos_value) < 0.25:
            direction.append('侧')
        else:
            if cos_value > 0:
                direction.append('前方')
            else:
                direction.append('后方')
    direction = ''.join(direction)
    if eng:
        direction = ENG_DIRECTION[direction]
    description['direction'] = direction

    # 生成速度描述
    if actor_speed < 0.1:
        speed_desc = '静止'
    elif actor_speed <= 10:
        speed_desc = '缓慢'
    elif actor_speed <= 30:
        speed_desc = '低速'
    else:
        speed_desc = '高速'
    if eng:
        speed_desc = ENG_SPEED[speed_desc]
    description['speed'] = speed_desc

    # 生成距离描述
    if distance < 10:
        distance_desc = '近'
    elif distance < 20:
        distance_desc = '较近'
    elif distance < 30:
        distance_desc = '较远'
    else:
        distance_desc = '远'
    if eng:
        distance_desc = ENG_DISTANCE[distance_desc]
    description['distance'] = distance_desc

    # 生成车型描述
    actor_type_id = actor.type_id.lower()
    type_desc = '轿车'
    for key, value in VEHICLE_TYPE.items():
        if actor_type_id in value:
            type_desc = key
            break
    if eng:
        type_desc = ENG_VEHICLE_TYPE[type_desc]
    description['type'] = type_desc

    # 生成车辆颜色
    if "color" not in actor.attributes:
        if actor.type_id == 'vehicle.tesla.cybertruck':
            rgb = "0,0,0"
        else:
            pdb.set_trace()
    else:
        rgb = actor.attributes['color']
    r = float(rgb.split(',')[0]) / 255.0
    g = float(rgb.split(',')[1]) / 255.0
    b = float(rgb.split(',')[2]) / 255.0
    rgb = Edict({'r': r, 'g': g, 'b': b})
    # 按照rgb生成颜色描述
    if rgb.r > 0.5 and rgb.g > 0.5 and rgb.b > 0.5:
        color_desc = '白色'
    elif rgb.r > 0.5 > rgb.g and rgb.b < 0.5:
        color_desc = '红色'
    elif rgb.r < 0.5 < rgb.g and rgb.b < 0.5:
        color_desc = '绿色'
    elif rgb.r < 0.5 < rgb.b and rgb.g < 0.5:
        color_desc = '蓝色'
    elif rgb.r > 0.5 > rgb.b and rgb.g > 0.5:
        color_desc = '黄色'
    elif rgb.r > 0.5 > rgb.g and rgb.b > 0.5:
        color_desc = '品红色'
    elif rgb.r < 0.5 < rgb.g and rgb.b > 0.5:
        color_desc = '青色'
    elif rgb.r < 0.5 and rgb.g < 0.5 and rgb.b < 0.5:
        color_desc = '黑色'
    else:
        color_desc = '未知颜色'
    if eng:
        color_desc = ENG_COLOR[color_desc]
    description['color'] = color_desc

    # cos_value转为float
    cos_value = float(cos_value)
    cross_value = float(cross_value)

    data = {
        'type_id': actor.type_id,
        'cos_value': cos_value,
        'cross_value': cross_value,
        'speed': actor_speed,
        'location': {'x': actor_loc.x, 'y': actor_loc.y, 'z': actor_loc.z},
        'distance': distance,
        'description': description
    }

    return data


def build_actor_data(ego, other_actors, eng=False):
    """
    构建场景中进行交互的其他车辆的数据（尽量简洁）
    """
    data = {}
    for actor in other_actors:
        if not actor.is_alive:
            continue
        if 'vehicle' in actor.type_id:
            actor_data = _calc_interactive_data(ego, actor, eng=eng)
            data[actor.id] = actor_data
        elif 'prop' in actor.type_id:
            actor_data = _calc_static_data(ego, actor, eng=eng)
            data[actor.id] = actor_data
        else:
            continue
    return data


def get_actor_lane_id(world, actor, default_equal=False):
    """
    获取车辆所在的车道id
    """
    # 获取车辆的transform
    actor_transform = actor.get_transform()
    # 获取车辆的location
    actor_location = actor_transform.location
    # 获取车辆的waypoint
    actor_waypoint = world.get_map().get_waypoint(actor_location)

    if actor.get_location().distance(actor_waypoint.transform.location) > actor_waypoint.lane_width / 2:
        if default_equal:
            return 0
        else:
            return random.randint(100, 99999999)

    # 获取车辆的lane id
    actor_lane_id = actor_waypoint.lane_id
    return actor_lane_id


def get_actor_lane_width(world, actor):
    """
    获取车辆所在的车道id
    """
    # 获取车辆的transform
    actor_transform = actor.get_transform()
    # 获取车辆的location
    actor_location = actor_transform.location
    # 获取车辆的waypoint
    actor_waypoint = world.get_map().get_waypoint(actor_location)

    # 获取车辆的lane id
    actor_lane_width = actor_waypoint.lane_width
    return actor_lane_width


def judge_point_in_lr(point, line_start_point, line_vec):
    point = np.array([point.x, point.y])
    line_start_point = np.array([line_start_point.x, line_start_point.y])
    line_start_point2point_vec = point - line_start_point
    line_vec = np.array([line_vec.x, line_vec.y])
    cross_value = np.cross(line_vec, line_start_point2point_vec)
    if cross_value < 0:
        return 'left'
    else:
        return 'right'


def get_actor_waypoint(world, actor):
    """
    获取车辆所在的waypoint
    """
    # 获取车辆的transform
    actor_transform = actor.get_transform()
    # 获取车辆的location
    actor_location = actor_transform.location
    # 获取车辆的waypoint
    actor_waypoint = world.get_map().get_waypoint(actor_location)
    return actor_waypoint
