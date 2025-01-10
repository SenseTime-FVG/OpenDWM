# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 下午4:50
# @Author  : Hcyang
# @File    : road_func.py
# @Desc    : TODO:
# import ipdb

import sys
from os import path
# print(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from OpenDriveLibrary.BasicLib import OpenDriveParser
from .GeometryLib import *
from .LanesLib import *
from .ElevationLib import *


class RoadParser(object):
    # Desc: 单条道路
    def __init__(self, road: OpenDriveParser):
        self.road = road
        self.geometry_list = None
        self.elevation_list = None
        self.lanes = None

        self.junction = float(self.road.attrib.junction)

        self._parse_geometries()
        self._parse_elevations()
        self._parse_lanes()

    def _parse_geometries(self):
        geometries = []
        for geometry in self.road.planView.geometry_s:
            geometry_obj = GeometryParser(geometry, self)
            geometries.append(geometry_obj)
        self.geometry_list = GeometryListParser(geometries, self)

    def _parse_elevations(self):
        elevations = []
        for elevation in self.road.elevationProfile.elevation_s:
            elevation_obj = ElevationParser(elevation, self)
            elevations.append(elevation_obj)
        self.elevation_list = ElevationListParser(elevations, self)

    def _parse_lanes(self):
        self.lanes = LanesParser(self.road.lanes, self)

    def list_init(self):
        self.lanes.list_init()

    def calc_lanemarks_3d(self):
        self.lanes.calc_lanemarks_3d()

    def calc_drivable_area_3d(self):
        self.lanes.calc_drivable_area_3d()

    def visualize(self, ax):
        # self.geometry_list.visualize(ax)
        self.lanes.visualize(ax)

    def visualize_3d(self, ax):
        # self.geometry_list.visualize(ax)
        self.lanes.visualize_3d(ax)

    def draw_road(self, image, x_offset, y_offset, quality):
        self.lanes.draw_lanes(image, x_offset, y_offset, quality)
        # return coords
    def __str__(self):
        return str(self.road)

    def __repr__(self):
        return self.__str__()
