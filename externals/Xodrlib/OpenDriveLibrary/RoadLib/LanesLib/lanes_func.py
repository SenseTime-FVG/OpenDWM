# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 下午4:53
# @Author  : Hcyang
# @File    : lanes_func.py
# @Desc    : TODO:
# import ipdb

from OpenDriveLibrary.BasicLib import OpenDriveParser
from .LaneSectionLib import LaneSectionListParser, LaneSectionParser
from .LaneOffsetLib import LaneOffsetParser, LaneOffsetListParser


class LanesParser(object):
    # Desc: 道路车道
    def __init__(self, lanes: OpenDriveParser, road):
        self.lanes = lanes
        self.road = road
        self.lane_section_list = None
        self.lane_offset_list = None

        self._parse_lane_offsets()
        self._parse_lane_sections()

    def _parse_lane_offsets(self):
        lane_offsets = []
        for lane_offset in self.lanes.laneOffset_s:
            lane_offset_obj = LaneOffsetParser(lane_offset, self)
            lane_offsets.append(lane_offset_obj)
        lane_offsets = sorted(lane_offsets, key=lambda x: x.s0)
        self.lane_offset_list = LaneOffsetListParser(lane_offsets, self)

    def _parse_lane_sections(self):
        lane_sections = []
        for lane_section in self.lanes.laneSection_s:
            lane_section_obj = LaneSectionParser(lane_section, self)
            lane_sections.append(lane_section_obj)
        self.lane_section_list = LaneSectionListParser(lane_sections, self)

    def list_init(self):
        self.lane_offset_list.list_init()
        for lane_offset in self.lane_offset_list:
            lane_offset.list_init()
        self.lane_section_list.list_init()
        for lane_section in self.lane_section_list:
            lane_section.list_init()

    def calc_lanemarks_3d(self):
        for lane_section in self.lane_section_list:
            lane_section.calc_lanemarks_3d()

    def calc_drivable_area_3d(self):
        for lane_section in self.lane_section_list:
            lane_section.calc_drivable_area_3d()

    def visualize(self, ax):
        self.lane_section_list.visualize(ax)

    def visualize_3d(self, ax):
        self.lane_section_list.visualize_3d(ax)

    def draw_lanes(self,image, x_offset, y_offset, quality):
        self.lane_section_list.draw_lane_sections(image, x_offset, y_offset, quality)
        # return coords
    def __str__(self):
        return str(self.lanes)

    def __repr__(self):
        return self.__str__()

