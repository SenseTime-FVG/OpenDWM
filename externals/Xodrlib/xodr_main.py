# -*- coding: utf-8 -*-
# @Time    : 2024/3/11 下午4:02
# @Author  : Hcyang
# @File    : xodr_main.py
# @Desc    : xodr解析库用法与测试

import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import warnings
import numpy as np

# 忽略 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
from OpenDriveLibrary import MapParser

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description='xodr解析库用法与测试')
    parser.add_argument('-x', '--xodr', type=str, default='', help='xodr文件绝对路径')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.xodr != '':
        xodr_file_path = args.xodr
    else:
        xodr_file_path = os.path.join(FILE_DIR, 'opendrive_maps', 'Town05.xodr')

    if not os.path.exists(xodr_file_path):
        print(f'xodr文件不存在: {xodr_file_path}')
        exit(0)

    map_parser = MapParser(xodr_file_path)

    map_parser.car_init((-226.5, -80.9, 10.0), -90.8/180*3.14, 50, 0.005, 5)

    seg_res,line_type,line_color = map_parser.get_segs()  # for lanes
    map_parser.scopemap_init()  # 只画视野内的roads
    scope_res,seg_res = map_parser.get_scope(segs=seg_res)  # 旋转到车头方向朝上，裁剪到真正的视野
    # np.save("tp.npy",np.array([scope_res,seg_res],dtype=object))
    # map_parser.visualize()
    # 查看2d地图
    # map_parser.visualize_elevation()
    # 查看3d地图


if __name__ == '__main__':
    main()
