# -*- coding: utf-8 -*-
# @Time    : 2023/11/8 下午2:42
# @Author  : Hcyang
# @File    : generate_xml.py
# @Desc    : xxx



import os
import argparse
import json
import xml.etree.ElementTree as ET
from random import shuffle
from xml.dom import minidom
from tqdm import tqdm
import carla
# import ipdb
import importlib


FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--scenario_name', type=str, required=True, help='场景名字（请保证大小写一致）')
    parser.add_argument('-ip', type=str, default='127.0.0.1', help='carla server ip，默认本机')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='debug模式，将在地图上可视化所有的trigger points')
    parser.add_argument('-t', '--town', type=str, default='', help='仅在debug模式下有效，只可视化指定town的trigger points')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    client = carla.Client(args.ip, 2000)
    client.set_timeout(60.0)

    scenario_cfg, condition_class = load_condition_class(args)

    all_trigger_wps = {}
    for town in scenario_cfg['maps']:
        if args.debug and args.town and town != args.town:
            continue

        client.load_world(town)
        world = client.get_world()
        wmap = world.get_map()
        debug = world.debug

        condition_obj = condition_class()
        random_wps, town_trigger_wps = generate_trigger_points_by_random_sample_points(wmap, condition_obj, distance=scenario_cfg['wp_interval'])

        if args.debug:
            for wp in town_trigger_wps:
                debug.draw_point(wp.transform.location + carla.Location(z=0.3), size=0.15, color=carla.Color(0, 255, 0), life_time=999)
            spectator = world.get_spectator()
            # 设置为俯视
            spectator.set_transform(carla.Transform(carla.Location(x=0, y=0, z=350), carla.Rotation(pitch=-90)))
            # ipdb.set_trace(context=10)
            pause = 1

        all_trigger_wps[town] = town_trigger_wps

    generate_trigger_points_xml(all_trigger_wps, args.scenario_name)


def load_condition_class(args):
    # Desc: 从配置文件中加载场景配置，以及对应的Condition类
    config_file_path = os.path.join(FILE_DIR, 'Files', 'Config', 'xml_generate_cfg.json')
    if not os.path.exists(config_file_path):
        print(f'配置文件不存在: {config_file_path}')
        exit(0)

    with open(config_file_path, 'r') as f:
        config = json.load(f)

    if args.scenario_name not in config:
        print(f'场景名{args.scenario_name}不在配置文件中，请先配置')
        exit(0)

    scenario_cfg = config[args.scenario_name]
    print(f'触发点之间间距: {scenario_cfg["wp_interval"]}m')
    if 'maps' not in scenario_cfg or len(scenario_cfg['maps']) == 0:
        scenario_cfg['maps'] = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']
        print(f'将使用所有地图: {scenario_cfg["maps"]}')
        print(f'\nPress Enter to continue')
        input()
    else:
        print(f'将使用地图: {scenario_cfg["maps"]}')
        print(f'\nPress Enter to continue')
        input()

    scenario_condition_class_name = args.scenario_name + 'Condition'
    # 从Scenarios/Allscenarios/{scenario_name}.py中import {scenario_condition_class_name}
    module_name = f'Scenarios.Allscenarios.{args.scenario_name.lower()}'
    module = importlib.import_module(module_name)
    condition_class = getattr(module, scenario_condition_class_name)

    return scenario_cfg, condition_class


def set_node_attributes_from_dict(node, info_dict):
    for k, v in info_dict.items():
        node.set(k, v)


def generate_trigger_points_xml(all_trigger_wps, scenario_name):
    # Desc: 生成xml文件
    root_output = ET.Element("routes")
    route_idx = 0
    for town, trigger_wps in tqdm(all_trigger_wps.items(), leave=False, desc=scenario_name):
        for wp in tqdm(trigger_wps, leave=False, desc=town):
            wp_data = {'x': str(wp.transform.location.x), 'y': str(wp.transform.location.y), 'z': str(wp.transform.location.z), 'yaw': str(wp.transform.rotation.yaw)}

            try:
                end_wp = wp.next(2)[0]
            except Exception as e:
                end_wp = wp.previous(2)[0]
            end_wp_data = {'x': str(end_wp.transform.location.x), 'y': str(end_wp.transform.location.y), 'z': str(end_wp.transform.location.z), 'yaw': str(end_wp.transform.rotation.yaw)}

            info = {
                'weather': {'route_percentage': '0', 'cloudiness': '5.0', 'precipitation': '0.0', 'precipitation_deposits': '0.0', 'wetness': '0.0', 'wind_intensity': '10.0', 'sun_azimuth_angle': '-1.0', 'sun_altitude_angle': '90.0', 'fog_density': '2.0', 'fog_distance': '0.75', 'fog_falloff': '0.1', 'scattering_intensity': '1.0', 'mie_scattering_scale': '0.03'},
                'scenario_attrib': {'name': f'{scenario_name}_0', 'type': scenario_name},
                'scenario_child_attrib': {'trigger_point': wp_data},
                'start_wp': wp_data,
                'end_wp': end_wp_data,
                'town_name': town
            }

            output_route = ET.SubElement(root_output, 'route')
            set_node_attributes_from_dict(output_route, {'id': str(route_idx), 'town': info['town_name']})

            # 首先处理天气
            output_weathers = ET.SubElement(output_route, 'weathers')
            cur_node = ET.SubElement(output_weathers, 'weather')
            set_node_attributes_from_dict(cur_node, info['weather'])

            # 遍历scenarios
            output_scenarios = ET.SubElement(output_route, 'scenarios')
            cur_scenario_node = ET.SubElement(output_scenarios, 'scenario')
            set_node_attributes_from_dict(cur_scenario_node, info['scenario_attrib'])
            for cur_scenario_child_node_key, cur_scenario_child_node_value in info['scenario_child_attrib'].items():
                cur_scenario_child_node = ET.SubElement(cur_scenario_node, cur_scenario_child_node_key)
                set_node_attributes_from_dict(cur_scenario_child_node, cur_scenario_child_node_value)

            # 开始处理waypoints
            output_waypoints = ET.SubElement(output_route, 'waypoints')
            start_wp = ET.SubElement(output_waypoints, 'position')
            set_node_attributes_from_dict(start_wp, info['start_wp'])
            end_wp = ET.SubElement(output_waypoints, 'position')
            set_node_attributes_from_dict(end_wp, info['end_wp'])
            route_idx += 1

        output_xml_path = os.path.join(FILE_DIR, 'Files', 'Xml', f'{scenario_name}.xml')
        xmlstr = minidom.parseString(ET.tostring(root_output)).toprettyxml(indent="   ")
        with open(output_xml_path, "w") as f:
            f.write(xmlstr)


def generate_trigger_points_by_random_sample_points(wmap, condition_obj, distance=100):
    # Desc: 从地图中随机采样点，检查是否满足condition_obj的条件，满足则作为trigger point
    random_wps = generate_random_wps_from_topology(wmap)

    trigger_wps = []
    for trigger_wp in tqdm(random_wps):
        wp_validity = condition_obj.check_condition_by_waypoint(trigger_wp)
        if wp_validity:
            if satisfy_test_trigger_point_rule(trigger_wp, trigger_wps, distance=distance):
                trigger_wps.append(trigger_wp)
    return random_wps, trigger_wps


def generate_random_wps_from_topology(wmap):
    # Desc: 从地图中的所有非交叉口的起始点中随机选择
    topologies = wmap.get_topology()

    all_candidates = []
    for topology in topologies:
        if is_junction_topology(topology):
            continue
        else:
            start_wp = topology[0]
            cur_candidates = start_wp.next_until_lane_end(1)
            all_candidates.extend(cur_candidates)
    print(f'len(all_candidates): {len(all_candidates)}')
    shuffle(all_candidates)
    return all_candidates


def is_junction_topology(topology):
    start_wp = topology[0]
    count = 10
    junction_wp = 0
    while count > 0:
        if start_wp.is_junction:
            junction_wp += 1
        start_wp = start_wp.next(1)[0]
        count -= 1
    if junction_wp > 5:
        return True
    else:
        return False


def satisfy_test_trigger_point_rule(trigger_wp, trigger_wps, distance=100):
    # Desc: 当前trigger point是否满足与已有trigger points的距离要求
    for existed_trigger_point in trigger_wps:
        if trigger_wp.road_id == existed_trigger_point.road_id:
            if trigger_wp.lane_id == existed_trigger_point.lane_id:
                if calc_distance_between_wps(trigger_wp, existed_trigger_point) < distance:
                    return False
    return True


def calc_distance_between_wps(wp1, wp2):
    return wp1.transform.location.distance(wp2.transform.location)


if __name__ == '__main__':
    main()
