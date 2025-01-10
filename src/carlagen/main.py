# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 下午4:33
# @Author  : Hcyang
# @File    : single_main.py
# @Desc    : 数据采集入口


import os
import sys
import traceback
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import argparse
from time import sleep

from Modes import XmlMode, DynamicMode
from Initialization import MainInitializer
from Execution import MainExecution
from Logger import *

def parse_args():
    parser = argparse.ArgumentParser(description='指定场景使用Autopilot采集UniAD数据')
    parser.add_argument(dest='scenario', type=str, help='场景名')
    parser.add_argument('-i', '--ip', type=str, default='localhost', help='Carla服务器IP')
    parser.add_argument('-p', '--port', type=int, default=2000, help='Carla服务器端口')
    parser.add_argument('-tp', '--traffic_port', type=int, default=8000, help='Carla TrafficManager端口')
    parser.add_argument('-o', '--output', dest='save_path', type=str, required=True, help='数据保存根目录')
    parser.add_argument('-fv', '--front-video', dest='front_video', action='store_true', default=False, help='是否将前置摄像头图像合成视频')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='是否开启debug模式')
    parser.add_argument('-a', '--aug', action='store_true', default=False, help='是否添加相机扰动')

    parser.add_argument('-sa', '--saver-config', default="./Config/saver.json", help='数据保存配置')
    parser.add_argument('-se', '--sensor-config', default="./Config/sensor.json", help='传感器配置')
    parser.add_argument('-si', '--simulator-config', default="./Config/simulator.json", help='仿真配置')

    add_sub_parsers(parser)
    args = parser.parse_args()

    return args


def add_sub_parsers(parser):
    subparsers = parser.add_subparsers(dest="mode", help='采集模式')
    subparser_xml = subparsers.add_parser('xml', help='xml文件采集模式')
    group = subparser_xml.add_mutually_exclusive_group()
    group.add_argument('--index', type=int, dest='xml_index', help='指定xml文件中的索引')
    group.add_argument('--random', action='store_true', dest='xml_random', help='随机选择xml文件中的索引')
    subparser_xml.add_argument('--town', type=str, dest='xml_town', default='random', help='指定town，仅在--random下有效')
    subparser_dynamic = subparsers.add_parser('dynamic', help='动态模式')


def main():
    args = parse_args()
    if args.mode == 'xml':
        mode_instance = XmlMode(args)
    elif args.mode == 'dynamic':
        mode_instance = DynamicMode(args)
    else:
        raise NotImplementedError

    sleep(1)
    try:
        scenario_config = mode_instance.run()

        info(f'[MAIN] 开始初始化...')
        main_initializer = MainInitializer(args, scenario_config)
        main_initializer.run()
        success(f'[MAIN] 初始化完成')
        info(f'[MAIN] 开始执行...')
        main_executor = MainExecution(args, main_initializer)
        main_executor.run()
        success(f'[MAIN] 执行完成')
        main_initializer.set_async_mode()
        message(f'>> 异步模式已设置')
        print('#&--success--&#')
        print(f'data_save_path {main_initializer.saver.get_save_path()}')
        special(f'>> 采集完成!')

    except Exception as e:
        info('#&--failed--&#')
        traceback.print_exc()
        print(e)


if __name__ == '__main__':
    main()
