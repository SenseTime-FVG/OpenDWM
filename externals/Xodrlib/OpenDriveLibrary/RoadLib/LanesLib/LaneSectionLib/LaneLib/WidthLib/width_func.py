# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 下午4:55
# @Author  : Hcyang
# @File    : road_mark_func.py
# @Desc    : TODO:
# import ipdb

from OpenDriveLibrary.BasicLib import OpenDriveParser


class WidthParser(object):
    # Desc: 道路宽度（Frenet坐标系）
    def __init__(self, width: OpenDriveParser, lane):
        self.width = width
        self.lane = lane
        self.s0 = float(width.attrib.sOffset)
        self.s1 = None
        self.a = float(width.attrib.a)
        self.b = float(width.attrib.b)
        self.c = float(width.attrib.c)
        self.d = float(width.attrib.d)

    def list_init(self):
        pass

    def set_s1(self, s1):
        assert s1 is not None
        self.s1 = s1

    def __str__(self):
        return str(self.width)

    def __repr__(self):
        return self.__str__()


class WidthListParser(object):
    # Desc: 道路宽度列表
    def __init__(self, widths, lane):
        self.widths = widths
        self.lane = lane

    def list_init(self):
        for i in range(len(self.widths) - 1):
            self.widths[i].set_s1(self.widths[i + 1].s0)
        if len(self.widths) >= 1:
            self.widths[-1].set_s1(self.lane.lane_section.s1)

    def __iter__(self):
        for width in self.widths:
            yield width

    def s2width(self, s):
        # Desc: 给定Frenet坐标系下的s，计算车道宽度
        for width in self.widths:
            s1 = s - width.lane.lane_section.s0
            if width.s0 <= s1 <= width.s1:
                s2 = s1 - width.s0
                tmp = width.a + width.b * s2 + width.c * s2 ** 2 + width.d * s2 ** 3
                return tmp
        raise ValueError('s is out of range')

    def __str__(self):
        info = ''
        for index, width in enumerate(self.widths):
            info += f'{index}: {width}\n'
        return info

    def __repr__(self):
        return self.__str__()
