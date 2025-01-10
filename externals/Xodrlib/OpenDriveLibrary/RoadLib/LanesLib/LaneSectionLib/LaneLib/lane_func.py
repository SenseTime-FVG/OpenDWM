# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 下午4:55
# @Author  : Hcyang
# @File    : lane_func.py
# @Desc    : TODO:
import random

# import ipdb

import json
import numpy as np
from OpenDriveLibrary.BasicLib import OpenDriveParser
from .WidthLib.width_func import WidthParser, WidthListParser
from .RoadMarkLib.road_mark_func import RoadMarkParser, RoadMarkListParser
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import cv2
# import shapely.geometry as geo


class LaneParser(object):
    # Desc: 道路车道
    def __init__(self, lane: OpenDriveParser, lane_section):
        self.lane = lane
        self.lane_section = lane_section
        self.id = int(lane.attrib.id)
        self.type = lane.attrib.type
        self.level = lane.attrib.level
        self.width_list = None
        self.roadmark_list = None
        self.neighboring_lane = None
        self._parse_widths()
        self._parse_roadmarks()

        # Special: LaneSection内车道的车道线数据
        self.lanemark_coords_3d = {
            'usable': False,
            'sd': [],
            'xyz': [],
        }
        # Special: LaneSection内车道的可行驶区域相关数据: p1和p4为self车道线的起点和终点，正方向为 p4 -> p1
        self.drivable_area_points_3d = {
            'usable': False,
            'p1': [],
            'p2': [],
            'p3': [],
            'p4': [],
        }
        self.left_lower = []
        self.right_upper = []

    def _parse_widths(self):
        widths = []
        for width in self.lane.width_s:
            width_obj = WidthParser(width, self)
            widths.append(width_obj)
        widths = sorted(widths, key=lambda x: x.s0)
        self.width_list = WidthListParser(widths, self)

    def _parse_roadmarks(self):
        roadmarks = []
        for roadmark in self.lane.roadMark_s:
            roadmark_obj = RoadMarkParser(roadmark, self)
            roadmarks.append(roadmark_obj)
        roadmarks = sorted(roadmarks, key=lambda x: x.s0)
        self.roadmark_list = RoadMarkListParser(roadmarks, self)

    def list_init(self):
        self.width_list.list_init()
        for width in self.width_list:
            width.list_init()
        self.roadmark_list.list_init()
        for roadmark in self.roadmark_list:
            roadmark.list_init()

    def s2width(self, s):
        d = self.width_list.s2width(s)
        return d

    def set_lanemark_coords_3d(self, sampled_s, sampled_d, corres_x, corres_y, corres_z):
        if len(sampled_s) > 0:
            self.lanemark_coords_3d['sd'] = [
                sampled_s,
                sampled_d,
            ]
            self.lanemark_coords_3d['xyz'] = [
                corres_x,
                corres_y,
                corres_z
            ]
            self.lanemark_coords_3d['usable'] = True
            if len(self.left_lower) < 1:
                self.left_lower.append(np.min(corres_x))
                self.left_lower.append(np.min(corres_y))
                self.left_lower.append(np.min(corres_z))
                self.right_upper.append(np.max(corres_x))
                self.right_upper.append(np.max(corres_y))
                self.right_upper.append(np.max(corres_z))
            else:
                self.left_lower[0] = min(np.min(corres_x), self.left_lower[0])
                self.left_lower[1] = min(np.min(corres_y), self.left_lower[1])
                self.left_lower[2] = min(np.min(corres_z), self.left_lower[2])
                self.right_upper[0] = max(np.max(corres_x), self.right_upper[0])
                self.right_upper[1] = max(np.max(corres_y), self.right_upper[1])
                self.right_upper[2] = max(np.max(corres_z), self.right_upper[2])
            for roadmark in self.roadmark_list:
                roadmark.set_lanemark_coords_3d(sampled_s, sampled_d, corres_x, corres_y, corres_z)

    def set_drivable_area_points_3d(self, neighboring_lane):
        if self.lanemark_coords_3d['usable']:
            self.drivable_area_points_3d['p1'] = [self.lanemark_coords_3d['xyz'][0][0],
                                                  self.lanemark_coords_3d['xyz'][1][0],
                                                  self.lanemark_coords_3d['xyz'][2][0]]
            self.drivable_area_points_3d['p4'] = [self.lanemark_coords_3d['xyz'][0][-1],
                                                  self.lanemark_coords_3d['xyz'][1][-1],
                                                  self.lanemark_coords_3d['xyz'][2][-1]]
            self.drivable_area_points_3d['p2'] = [neighboring_lane.lanemark_coords_3d['xyz'][0][0],
                                                  neighboring_lane.lanemark_coords_3d['xyz'][1][0],
                                                  neighboring_lane.lanemark_coords_3d['xyz'][2][0]]
            self.drivable_area_points_3d['p3'] = [neighboring_lane.lanemark_coords_3d['xyz'][0][-1],
                                                  neighboring_lane.lanemark_coords_3d['xyz'][1][-1],
                                                  neighboring_lane.lanemark_coords_3d['xyz'][2][-1]]
            self.neighboring_lane = neighboring_lane
            self.drivable_area_points_3d['usable'] = True

    def visualize(self, ax, cr='gray'):
        # For: 可视化可行驶区域
        if self.drivable_area_points_3d['usable']:
            drivable_points = []
            # 正序
            for x, y in zip(self.neighboring_lane.lanemark_coords_3d['xyz'][0],
                            self.neighboring_lane.lanemark_coords_3d['xyz'][1]):
                drivable_points.append([x, y])
            # 倒序
            for x, y in zip(reversed(self.lanemark_coords_3d['xyz'][0]), reversed(self.lanemark_coords_3d['xyz'][1])):
                drivable_points.append([x, y])
            polygon = Polygon(np.array(drivable_points), closed=True, color='gray', edgecolor='white')
            ax.add_patch(polygon)

        if self.lane_section.lanes.road.road.attrib.junction == '-1':
            for roadmark in self.roadmark_list:
                roadmark.visualize(ax)

    def draw_lane(self, image, x_offset, y_offset, quality):
        if self.drivable_area_points_3d['usable']:
            drivable_points = []
            for x, y in zip(self.neighboring_lane.lanemark_coords_3d['xyz'][0],
                            self.neighboring_lane.lanemark_coords_3d['xyz'][1]):
                drivable_points.append([(x - x_offset) * quality, (y - y_offset) * quality])
            for x, y in zip(reversed(self.lanemark_coords_3d['xyz'][0]),
                            reversed(self.lanemark_coords_3d['xyz'][1])):
                drivable_points.append([(x - x_offset) * quality, (y - y_offset) * quality])

            # poly = geo.Polygon(np.array(drivable_points, np.int32))
            # hull = poly.convex_hull
            # coords = list(hull.exterior.coords)
            # 将 drivable_points转为 numpy array，并且调整作为cv2.fillPoly所需的形状
            drivable_points = np.array(drivable_points, np.int32).reshape((-1, 1, 2))

            # 使用黑色填充这个多边形
            cv2.fillPoly(image, [drivable_points], 128)
            cv2.polylines(image, [drivable_points], isClosed=True, color=255, thickness=3)
            # return coords


    def visualize_3d(self, ax, cr='gray'):
        # For: 可视化可行驶区域
        if self.drivable_area_points_3d['usable']:
            drivable_points = []
            # 正序
            for x, y, z in zip(self.neighboring_lane.lanemark_coords_3d['xyz'][0],
                               self.neighboring_lane.lanemark_coords_3d['xyz'][1],
                               self.neighboring_lane.lanemark_coords_3d['xyz'][2]):
                drivable_points.append([x, y, z])

            # 倒序
            for x, y, z in zip(reversed(self.lanemark_coords_3d['xyz'][0]),
                               reversed(self.lanemark_coords_3d['xyz'][1]),
                               reversed(self.lanemark_coords_3d['xyz'][2])):
                drivable_points.append([x, y, z])

            ax.add_collection3d(
                Poly3DCollection([np.array(drivable_points)], edgecolors='white', linewidths=1, alpha=.25,
                                 facecolors='gray'))

    def get_direction(self):
        if self in self.lane_section.left_lanes:
            return 'left'
        elif self in self.lane_section.right_lanes:
            return 'right'
        else:
            return 'center'

    def __str__(self):
        return str(self.lane)

    def __repr__(self):
        return self.__str__()
