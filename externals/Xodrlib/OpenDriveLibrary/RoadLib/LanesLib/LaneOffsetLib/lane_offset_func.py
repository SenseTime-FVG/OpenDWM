# -*- coding: utf-8 -*-
# @Time    : 2024/3/27 下午4:43
# @Author  : Hcyang
# @File    : lane_offset_func.py
# @Desc    : TODO:


from OpenDriveLibrary.BasicLib import OpenDriveParser


class LaneOffsetParser(object):
    # Desc: 参考线与中心线的偏移（Frenet坐标系）
    def __init__(self, lane_offset: OpenDriveParser, lane):
        self.lane_offset = lane_offset
        self.lane = lane
        self.s0 = float(lane_offset.attrib.s)
        self.s1 = None
        self.a = float(lane_offset.attrib.a)
        self.b = float(lane_offset.attrib.b)
        self.c = float(lane_offset.attrib.c)
        self.d = float(lane_offset.attrib.d)

    def list_init(self):
        pass

    def set_s1(self, s1):
        assert s1 is not None
        self.s1 = s1

    def __str__(self):
        return str(self.lane_offset)

    def __repr__(self):
        return self.__str__()


class LaneOffsetListParser(object):
    # Desc: 车道线列表
    def __init__(self, lane_offsets, lanes):
        self.lane_offsets = lane_offsets
        self.lanes = lanes

    def list_init(self):
        road_end_s = self.lanes.road.geometry_list.s1
        for i in range(len(self.lane_offsets) - 1):
            self.lane_offsets[i].set_s1(self.lane_offsets[i + 1].s0)
        if len(self.lane_offsets) >= 1:
            self.lane_offsets[-1].set_s1(road_end_s)

    def s2width(self, s):
        # Desc: 给定Frenet坐标系下的s，计算车道宽度
        for lane_offset in self.lane_offsets:
            if lane_offset.s0 <= s <= lane_offset.s1:
                tmp = lane_offset.a + lane_offset.b * s + lane_offset.c * s ** 2 + lane_offset.d * s ** 3
                return tmp
        raise ValueError('s is out of range')

    def __iter__(self):
        for lane_offset in self.lane_offsets:
            yield lane_offset

    def __str__(self):
        info = ''
        for index, lane_offset in enumerate(self.lane_offsets):
            info += f'{index}: {lane_offset}\n'
        return info

    def __repr__(self):
        return self.__str__()
