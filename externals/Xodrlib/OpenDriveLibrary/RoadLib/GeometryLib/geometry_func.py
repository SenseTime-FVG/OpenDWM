# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 下午4:51
# @Author  : Hcyang
# @File    : geometry_func.py
# @Desc    : TODO:
# import ipdb

import random
from OpenDriveLibrary.BasicLib import OpenDriveParser, OpenDriveParserList
import numpy as np
from typing import List


class GeometryParser(object):
    # Desc: 道路单个几何中心线
    def __init__(self, geometry: OpenDriveParser, road):
        self.geometry = geometry
        self.road = road
        self.s0 = float(geometry.attrib.s)
        self.s1 = None
        self.x0 = float(geometry.attrib.x)
        self.y0 = float(geometry.attrib.y)
        self.hdg = float(geometry.attrib.hdg)
        self.length = float(geometry.attrib.length)

        if 'line' in geometry:
            self.type = 'line'
            self._line_calc_end()
        elif 'arc' in geometry:
            self.type = 'arc'
            self._arc_calc_end()
        else:
            raise NotImplementedError

    def _line_calc_end(self):
        self.x1 = self.x0 + np.cos(self.hdg) * self.length
        self.y1 = self.y0 + np.sin(self.hdg) * self.length

    def _arc_calc_end(self):
        self.curvature = float(self.geometry.arc.attrib.curvature)
        self.R = abs(1 / self.curvature)
        self.theta = self.length / self.R
        if self.curvature < 0:
            # Special: 顺时针弧
            self.cx = self.x0 + self.R * np.sin(self.hdg)
            self.cy = self.y0 - self.R * np.cos(self.hdg)
            self.start_angle = self.hdg + np.pi / 2
            self.end_angle = self.start_angle - self.theta
        else:
            # Special: 逆时针弧
            self.cx = self.x0 - self.R * np.sin(self.hdg)
            self.cy = self.y0 + self.R * np.cos(self.hdg)
            self.start_angle = self.hdg - np.pi / 2
            self.end_angle = self.start_angle + self.theta

        # 保证start_angle和end_angle大于0
        # while self.start_angle < 0 or self.end_angle < 0:
        #     self.start_angle += 2 * np.pi
        #     self.end_angle += 2 * np.pi

        self.x1 = self.cx + self.R * np.cos(self.end_angle)
        self.y1 = self.cy + self.R * np.sin(self.end_angle)

    def set_s1(self, s1):
        self.s1 = s1

    def visualize(self, ax):
        if self.type == 'line':
            self._visualize_line(ax)
        elif self.type == 'arc':
            self._visualize_arc(ax)

    def _visualize_line(self, ax):
        ax.plot([self.x0, self.x1], [self.y0, self.y1], color='blue')

    def _visualize_arc(self, ax):
        angles = np.linspace(self.start_angle, self.end_angle, 100)
        xs = self.cx + self.R * np.cos(angles)
        ys = self.cy + self.R * np.sin(angles)

        # color = random.choice(['green', 'blue', 'black', 'purple'])
        color = 'blue'
        ax.plot(xs, ys, label="Arc Path", color=color)
        # ax.scatter([self.x0], [self.y0], color='red', label="Start Point")

    def calc_tangent_angle(self, x, y):
        # Desc: 计算笛卡尔坐标系下横坐标为x的切线与x轴的夹角
        if self.type == 'line':
            return self.hdg
        elif self.type == 'arc':
            # if self.x0 < x < self.x1:
            alpha = np.arctan2(y - self.cy, x - self.cx) + np.pi / 2
            return alpha
            # else:
            #     raise ValueError('x is out of range')

    def __str__(self):
        return str(self.geometry)

    def __repr__(self):
        return self.__str__()


class GeometryListParser(object):
    # Desc: 道路几何中心线集合
    def __init__(self, geometries: List[GeometryParser], road):
        self.geometries = geometries
        self.road = road
        self._init()

    def _init(self):
        self.s0 = self.geometries[0].s0
        self.s1 = self.geometries[-1].s0 + self.geometries[-1].length

        for i in range(len(self.geometries) - 1):
            self.geometries[i].set_s1(self.geometries[i + 1].s0)
        if len(self.geometries) >= 1:
            self.geometries[-1].set_s1(self.s1)

    def s2x(self, s, geo=False):
        # Desc: 给定Frenet坐标系下的(s, 0)，计算x
        x = None
        corres_geometry = None
        for geometry in self.geometries:
            if geometry.s0 <= s <= geometry.s1:
                if geometry.type == 'line':
                    x = geometry.x0 + np.cos(geometry.hdg) * (s - geometry.s0)
                    corres_geometry = geometry
                elif geometry.type == 'arc':
                    theta = (s - geometry.s0) / geometry.R
                    x = geometry.cx + geometry.R * np.cos(geometry.start_angle + theta)
                    corres_geometry = geometry
        if geo:
            return x, corres_geometry
        else:
            return x

    def s2y(self, s, geo=False):
        # Desc: 给定Frenet坐标系下的(s, 0)，计算y
        y = None
        corres_geometry = None
        for geometry in self.geometries:
            if geometry.s0 <= s <= geometry.s1:
                if geometry.type == 'line':
                    y = geometry.y0 + np.sin(geometry.hdg) * (s - geometry.s0)
                    corres_geometry = geometry
                elif geometry.type == 'arc':
                    theta = (s - geometry.s0) / geometry.R
                    y = geometry.cy + geometry.R * np.sin(geometry.start_angle + theta)
                    corres_geometry = geometry
        if geo:
            return y, corres_geometry
        else:
            return y

    def s2xy(self, s, geo=False):
        # Desc: 给定Frenet坐标系下的(s, 0)，计算x, y
        x, y = None, None
        corres_geometry = None
        for geometry in self.geometries:
            if geometry.s0 <= s <= geometry.s1:
                if geometry.type == 'line':
                    x = geometry.x0 + np.cos(geometry.hdg) * (s - geometry.s0)
                    y = geometry.y0 + np.sin(geometry.hdg) * (s - geometry.s0)
                    corres_geometry = geometry
                    break
                elif geometry.type == 'arc':
                    theta = (s - geometry.s0) / geometry.R
                    if geometry.curvature < 0:
                        x = geometry.cx + geometry.R * np.cos(geometry.start_angle - theta)
                        y = geometry.cy + geometry.R * np.sin(geometry.start_angle - theta)
                    else:
                        x = geometry.cx + geometry.R * np.cos(geometry.start_angle + theta)
                        y = geometry.cy + geometry.R * np.sin(geometry.start_angle + theta)
                    corres_geometry = geometry
                    break
        if geo:
            return x, y, corres_geometry
        else:
            return x, y

    def sd2xy(self, s, d, lane):
        # Desc: 给定Frenet坐标系下的(s, d)，计算x, y
        # if s > 12.90:
        #     ipdb.set_trace(context=10)
        x3, y3, geometry = self.s2xy(s, geo=True)
        alpha = geometry.calc_tangent_angle(x3, y3)
        if lane.id <= 0:
            if geometry.type == 'arc':
                if geometry.curvature < 0:
                    x4 = x3 + d * np.cos(alpha + np.pi / 2)
                    y4 = y3 + d * np.sin(alpha + np.pi / 2)
                else:
                    x4 = x3 + d * np.cos(alpha - np.pi / 2)
                    y4 = y3 + d * np.sin(alpha - np.pi / 2)
            else:
                x4 = x3 + d * np.cos(alpha - np.pi / 2)
                y4 = y3 + d * np.sin(alpha - np.pi / 2)
        elif lane.id > 0:
            if geometry.type == 'arc':
                if geometry.curvature < 0:
                    x4 = x3 + d * np.cos(alpha - np.pi / 2)
                    y4 = y3 + d * np.sin(alpha - np.pi / 2)
                else:
                    x4 = x3 + d * np.cos(alpha + np.pi / 2)
                    y4 = y3 + d * np.sin(alpha + np.pi / 2)
            else:
                x4 = x3 + d * np.cos(alpha + np.pi / 2)
                y4 = y3 + d * np.sin(alpha + np.pi / 2)
        else:
            raise NotImplementedError
        return x4, y4

    def visualize(self, ax):
        for geometry in self.geometries:
            geometry.visualize(ax)

    def __str__(self):
        info = ''
        for index, geometry in enumerate(self.geometries):
            info += f'{index}: {geometry}\n'
        return info

    def __repr__(self):
        return self.__str__()


def td(hudu):
    return hudu * 180 / np.pi
