# -*- coding: utf-8 -*-
# @Time    : 2023/11/12 下午7:28
# @Author  : Hcyang
# @File    : lane_desc.py
# @Desc    : xxx


import sys
import os
import argparse
import pickle
import json
from .basic_desc_utils import *
from Scenarios.Tinyscenarios.functions import *


class LaneDesc(BasicDesc):
    def __init__(self, world, ego):
        super(LaneDesc, self).__init__(world, ego)
        self.wmap = world.get_map()

    def _gen_atomic_desc(self):
        self.ego_loc = self.ego.get_location()
        self.ego_wp = self.wmap.get_waypoint(self.ego_loc)
        atomic_desc = self._gen_lane_desc()
        return atomic_desc

    def _gen_lane_desc(self):
        """
        Desc: 生成车道描述和原因
        """
        if self.ego_wp.is_junction or self.ego_wp.is_intersection:
            return {}

        lane_info = get_lane_info(self.ego_wp)
        lane_num = lane_info['num']
        r2l = lane_info['r2l']
        l2r = lane_info['l2r']
        lm = 'dotted line' if lane_info['lchange'] else 'solid line'
        rm = 'dotted line' if lane_info['rchange'] else 'solid line'
        if lane_info['lchange'] and not lane_info['rchange']:
            lc = 'left', 'can change lane to the left'
        elif lane_info['rchange'] and not lane_info['lchange']:
            lc = 'right', 'can change lane to the right'
        elif lane_info['lchange'] and lane_info['rchange']:
            lc = 'left or right', 'can change lane to the left or right'
        else:
            lc = '', 'can not change lane'

        desc = {
            'lane_num': [f'{lane_num} lanes', f'there are {lane_num} lanes'],
            'lane_r2l': [f'lane {r2l}', f'currently in the lane {r2l} from right to left'],
            'lane_l2r': [f'lane {l2r}', f'currently in the lane {l2r} from left to right'],
            'lane_lm': [lm, f'The left lane marking is {lm}'],
            'lane_rm': [rm, f'The right lane marking is {rm}'],
            'lane_lc': lc
        }
        return desc

