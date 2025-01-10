# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 下午4:44
# @Author  : Hcyang
# @File    : Xml.py
# @Desc    : 加载XML文件

import os
import random
import sys
import json
# import ipdb
import pickle
import argparse

# from tqdm import tqdm
from datetime import datetime
import xml.etree.ElementTree as ET
from Logger import *

from .Basic import BasicMode


class XmlMode(BasicMode):
    def __init__(self, args):
        super(XmlMode, self).__init__(args)

    def run(self):
        # 加载XML文件
        xml_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Files', 'Xml'))
        xml_path = os.path.join(xml_root, f"{self.args.scenario}.xml")

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML文件不存在: {xml_path}")

        tree = ET.parse(xml_path)

        all_routes = tree.findall('route')

        if self.args.xml_index is not None:
            index = min(max(0, self.args.xml_index), len(all_routes) - 1)
        elif self.args.xml_random:
            if self.args.xml_town == 'random':
                index = random.randint(0, len(all_routes) - 1)
            else:
                candidates = [i for i, route in enumerate(all_routes) if route.attrib['town'].lower() == self.args.xml_town.lower()]
                if len(candidates) == 0:
                    error(f'没有{self.args.xml_town}相关的路线')
                    exit(0)
                index = random.choice(candidates)
        else:
            index = random.randint(0, len(all_routes) - 1)
        print(index)
        route = all_routes[index]
        scenario_config = self.parse_route(route)

        return scenario_config

    def parse_route(self, route):
        town = route.attrib['town']
        trigger_point = route.find('scenarios').find('scenario').find('trigger_point')
        xyz = trigger_point.attrib
        # save_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
        save_path = self.args.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path):
            error(f'无法创建数据保存目录: {save_path}')
            exit(0)
        scenario_config = {
            'scenario_name': self.args.scenario,
            'town': town,
            'trigger_point': xyz,
            'date': datetime.now(),
            'save_path': save_path
        }
        return scenario_config

