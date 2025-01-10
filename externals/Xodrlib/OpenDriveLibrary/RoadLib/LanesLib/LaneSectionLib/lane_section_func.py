# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 下午4:54
# @Author  : Hcyang
# @File    : lane_section_func.py
# @Desc    : TODO:
# import ipdb
import numpy as np
from OpenDriveLibrary.BasicLib import OpenDriveParser
from .LaneLib import LaneParser
from typing import List


class LaneSectionParser(object):
    # Desc: 道路车道
    def __init__(self, lane_section: OpenDriveParser, lanes):
        self.lane_section = lane_section
        self.lanes = lanes
        self.s0 = float(lane_section.attrib.s)
        self.s1 = None

        self.left_lanes = []
        self.center_lane = None
        self.right_lanes = []

        self.drivable_lane_type = ['driving', 'bidirectional', 'median', 'parking']
        self.vis_lane_types = ['driving', 'bidirectional', 'median', 'parking']

        self._parse_lanes()

    def _parse_lanes(self):
        if 'left' in self.lane_section:
            for lane in self.lane_section.left.lane_s:
                lane_obj = LaneParser(lane, self)
                self.left_lanes.append(lane_obj)
            # Special: 1, 2, 3, 4
            self.left_lanes = sorted(self.left_lanes, key=lambda x: x.id)
        if 'center' in self.lane_section:
            self.center_lane = LaneParser(self.lane_section.center.lane, self)
        if 'right' in self.lane_section:
            for lane in self.lane_section.right.lane_s:
                lane_obj = LaneParser(lane, self)
                self.right_lanes.append(lane_obj)
            # Special: -1, -2, -3, -4
            self.right_lanes = sorted(self.right_lanes, key=lambda x: x.id, reverse=True)

    def list_init(self):
        for lane in self.left_lanes:
            lane.list_init()
        if self.center_lane:
            self.center_lane.list_init()
        for lane in self.right_lanes:
            lane.list_init()

    def set_s1(self, s1):
        self.s1 = s1

    def visualize(self, ax):
        last_vis_sign = 0
        for lane in reversed(self.left_lanes):
            if lane.type in self.vis_lane_types:
                lane.visualize(ax)
                if last_vis_sign == 0:
                    last_vis_sign = 1
            else:
                if last_vis_sign == 1:
                    lane.visualize(ax)
                    last_vis_sign = 2

        if self.center_lane:
            self.center_lane.visualize(ax)

        last_vis_sign = 0
        for lane in reversed(self.right_lanes):
            if lane.type in self.vis_lane_types:
                lane.visualize(ax)
                if last_vis_sign == 0:
                    last_vis_sign = 1
            else:
                if last_vis_sign == 1:
                    lane.visualize(ax)
                    last_vis_sign = 2

    def draw_lanesec(self,image, x_offset, y_offset, quality):
        last_vis_sign = 0
        coords = []
        for lane in reversed(self.left_lanes):
            if lane.type in self.vis_lane_types:
                p=lane.draw_lane(image, x_offset, y_offset, quality)
                coords.append(p)
                if last_vis_sign == 0:
                    last_vis_sign = 1
            else:
                if last_vis_sign == 1:
                    p=lane.draw_lane(image, x_offset, y_offset, quality)
                    coords.append(p)
                    last_vis_sign = 2

        if self.center_lane:
            p=self.center_lane.draw_lane(image, x_offset, y_offset, quality)
            coords.append(p)

        last_vis_sign = 0
        for lane in reversed(self.right_lanes):
            if lane.type in self.vis_lane_types:
                p=lane.draw_lane(image, x_offset, y_offset, quality)
                coords.append(p)
                if last_vis_sign == 0:
                    last_vis_sign = 1
            else:
                if last_vis_sign == 1:
                    p=lane.draw_lane(image, x_offset, y_offset, quality)
                    coords.append(p)
                    last_vis_sign = 2
        return coords

    def calc_lanemarks_3d(self):
        start_s = self.s0
        end_s = self.s1
        sampled_s = np.linspace(start_s, end_s, int((end_s - start_s) * 10))
        sampled_d = {
            'left': {},
            'center': {},
            'right': {},
        }
        # Special: 参考线(geometry line)和中心线(center line)的d差值
        d_diff_ref_center = []

        for lane in self.left_lanes:
            sampled_d['left'][lane.id] = []
        sampled_d['center'][self.center_lane.id] = []
        for lane in self.right_lanes:
            sampled_d['right'][lane.id] = []

        for s in sampled_s:
            # if self.lanes.road.road.attrib.id == '31':
            #     ipdb.set_trace(context=10)
            d_diff = self.lanes.lane_offset_list.s2width(s)
            d_diff_ref_center.append(d_diff)

            left_accumulate_d = d_diff
            for lane in self.left_lanes:
                d = lane.s2width(s)
                left_accumulate_d += d
                sampled_d['left'][lane.id].append(left_accumulate_d)

            sampled_d['center'][self.center_lane.id].append(-d_diff)

            right_accumulate_d = -d_diff
            for lane in self.right_lanes:
                d = lane.s2width(s)
                right_accumulate_d += d
                sampled_d['right'][lane.id].append(right_accumulate_d)

        geometry_list = self.lanes.road.geometry_list
        elevation_list = self.lanes.road.elevation_list
        for key in sampled_d:
            if key == 'left':
                process_lanes = self.left_lanes
            elif key == 'right':
                process_lanes = self.right_lanes
            elif key == 'center':
                process_lanes = [self.center_lane]
            else:
                raise NotImplementedError
            for cur_lane in process_lanes:
                corres_s = []
                corres_d = []
                corres_x = []
                corres_y = []
                corres_z = []
                z_index = 0
                max_zi = len(elevation_list.elevations)
                for s, d in zip(sampled_s, sampled_d[key][cur_lane.id]):
                    corres_s.append(s)
                    corres_d.append(d)
                    x, y = geometry_list.sd2xy(s, d, cur_lane)
                    if z_index < max_zi-1:
                        if s > elevation_list.elevations[z_index+1].s0:
                            z_index += 1
                    z = elevation_list.elevations[z_index].s2z(s)
                    corres_x.append(x)
                    corres_y.append(y)
                    corres_z.append(z)
                cur_lane.set_lanemark_coords_3d(corres_s, corres_d, corres_x, corres_y, corres_z)

    def calc_drivable_area_3d(self):
        for lane_index in range(len(self.left_lanes) - 1, -1, -1):
            cur_lane = self.left_lanes[lane_index]
            if cur_lane.type in self.drivable_lane_type:
                if lane_index != 0:
                    cur_lane.set_drivable_area_points_3d(self.left_lanes[lane_index - 1])
                else:
                    cur_lane.set_drivable_area_points_3d(self.center_lane)
        for lane_index in range(len(self.right_lanes) - 1, -1, -1):
            cur_lane = self.right_lanes[lane_index]
            if cur_lane.type in self.drivable_lane_type:
                if lane_index != 0:
                    cur_lane.set_drivable_area_points_3d(self.right_lanes[lane_index - 1])
                else:
                    cur_lane.set_drivable_area_points_3d(self.center_lane)

    def visualize_3d(self, ax):
        last_vis_sign = 0
        for lane in reversed(self.left_lanes):
            if lane.type in self.vis_lane_types:
                lane.visualize_3d(ax)
                if last_vis_sign == 0:
                    last_vis_sign = 1
            else:
                if last_vis_sign == 1:
                    lane.visualize_3d(ax)
                    last_vis_sign = 2

        if self.center_lane:
            self.center_lane.visualize_3d(ax)

        last_vis_sign = 0
        for lane in reversed(self.right_lanes):
            if lane.type in self.vis_lane_types:
                lane.visualize_3d(ax)
                if last_vis_sign == 0:
                    last_vis_sign = 1
            else:
                if last_vis_sign == 1:
                    lane.visualize_3d(ax)
                    last_vis_sign = 2

    def __str__(self):
        return str(self.lane_section)

    def __repr__(self):
        return self.__str__()


class LaneSectionListParser(object):
    # Desc: 道路区域集合
    def __init__(self, lane_sections: List[LaneSectionParser], lanes):
        self.lane_sections = lane_sections
        self.lanes = lanes

    def list_init(self):
        road_end_s = self.lanes.road.geometry_list.s1
        for i in range(len(self.lane_sections) - 1):
            self.lane_sections[i].set_s1(self.lane_sections[i + 1].s0)
        if len(self.lane_sections) >= 1:
            self.lane_sections[-1].set_s1(road_end_s)

    def visualize(self, ax):
        for lane_section in self.lane_sections:
            lane_section.visualize(ax)

    def visualize_3d(self, ax):
        for lane_section in self.lane_sections:
            lane_section.visualize_3d(ax)

    def draw_lane_sections(self, image, x_offset, y_offset, quality):
        # coords=None
        for lane_section in self.lane_sections:
            # if coords is None:
            lane_section.draw_lanesec(image, x_offset, y_offset, quality)
            # else:
            #     coords += lane_section.draw_lanesec(image, x_offset, y_offset, quality)
        # return coords
    def __iter__(self):
        for lane_section in self.lane_sections:
            yield lane_section

    def __str__(self):
        info = ''
        for index, lane_section in enumerate(self.lane_sections):
            info += f'{index}: {lane_section}\n'
        return info

    def __repr__(self):
        return self.__str__()
