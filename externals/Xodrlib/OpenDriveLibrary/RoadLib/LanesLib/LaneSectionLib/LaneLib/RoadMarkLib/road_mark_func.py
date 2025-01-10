# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 下午4:55
# @Author  : Hcyang
# @File    : road_mark_func.py
# @Desc    : TODO:

import numpy as np
from OpenDriveLibrary.BasicLib import OpenDriveParser


class RoadMarkParser(object):
    # Desc: 车道线（Frenet坐标系）
    def __init__(self, roadmark: OpenDriveParser, lane):
        self.roadmark = roadmark
        self.lane = lane
        self.s0 = float(roadmark.attrib.sOffset)
        self.s1 = None
        self.type = roadmark.attrib.type
        if 'color' in roadmark.attrib:
            self.color = roadmark.attrib.color
            if self.color == 'white':
                self.color = 'gainsboro'
        else:
            self.color = 'none'
        self.lane_change = roadmark.attrib.laneChange

        # Special: 车道中某一段车道线的数据

        self.lanemark_coords_3d = {
            'usable': False,
            'sd': [],
            'xyz': [],
        }

        self.vis_roadmark_types = ['solid', 'broken', 'solid solid', 'solid broken', 'broken solid']

    def list_init(self):
        pass

    def set_s1(self, s1):
        assert s1 is not None
        self.s1 = s1

    def set_lanemark_coords_3d(self, sampled_s, sampled_d, corres_x, corres_y, corres_z):
        ss = []
        ds = []
        xs = []
        ys = []
        zs = []
        for s, d, x, y, z in zip(sampled_s, sampled_d, corres_x, corres_y, corres_z):
            if self.s0 <= s <= self.s1:
                ss.append(s)
                ds.append(d)
                xs.append(x)
                ys.append(y)
                zs.append(z)
        if len(ss) > 0:
            self.lanemark_coords_3d['sd'] = [ss, ds]
            self.lanemark_coords_3d['xyz'] = [xs, ys, zs]
            self.lanemark_coords_3d['usable'] = True

    def visualize(self, ax):
        if self.lanemark_coords_3d['usable'] and self.color != 'none' and self.type in self.vis_roadmark_types:
            if self.type == 'solid':
                xs = self.lanemark_coords_3d['xyz'][0]
                ys = self.lanemark_coords_3d['xyz'][1]
                ax.plot(xs, ys, color=self.color)
            elif self.type == 'broken':
                xs = self.lanemark_coords_3d['xyz'][0]
                ys = self.lanemark_coords_3d['xyz'][1]
                ax.plot(xs, ys, color=self.color, linestyle='--')
            elif self.type == 'solid solid':
                xs = self.lanemark_coords_3d['xyz'][0]
                ys = self.lanemark_coords_3d['xyz'][1]
                ax.plot(xs, ys, color=self.color, linestyle='-.')
            elif self.type == 'solid broken':
                xs = self.lanemark_coords_3d['xyz'][0]
                ys = self.lanemark_coords_3d['xyz'][1]
                ax.plot(xs, ys, color=self.color, linestyle=':')
            elif self.type == 'broken solid':
                xs = self.lanemark_coords_3d['xyz'][0]
                ys = self.lanemark_coords_3d['xyz'][1]
                ax.plot(xs, ys, color=self.color, linestyle=':')

    def __str__(self):
        return str(self.roadmark)

    def __repr__(self):
        return self.__str__()


class RoadMarkListParser(object):
    # Desc: 车道线列表
    def __init__(self, roadmarks, lane):
        self.roadmarks = roadmarks
        self.lane = lane

    def list_init(self):
        for i in range(len(self.roadmarks) - 1):
            self.roadmarks[i].set_s1(self.roadmarks[i + 1].s0)
        if len(self.roadmarks) >= 1:
            self.roadmarks[-1].set_s1(self.lane.lane_section.s1)

    def __iter__(self):
        for roadmark in self.roadmarks:
            yield roadmark

    def __str__(self):
        info = ''
        for index, roadmark in enumerate(self.roadmarks):
            info += f'{index}: {roadmark}\n'
        return info

    def __repr__(self):
        return self.__str__()


class BevRectangleView(object):
    def __init__(self, cx, cy, vx, vy, view_scope):
        self.cx = cx
        self.cy = cy
        self.vx = vx
        self.vy = vy
        self.view_scope = view_scope

        self.points = calc_rect_4points(cx, cy, vx, vy, view_scope)
        # 左上角
        self.p1 = self.points['p1']
        # 右上角
        self.p2 = self.points['p2']
        # 右下角
        self.p3 = self.points['p3']
        # 左下角
        self.p4 = self.points['p4']

        # 直线方程
        self.line1 = calc_rect_line(self.p1, self.p2)
        self.line2 = calc_rect_line(self.p2, self.p3)
        self.line3 = calc_rect_line(self.p3, self.p4)
        self.line4 = calc_rect_line(self.p4, self.p1)
        self.lines = {
            'line1': (self.p1, self.p2, self.line1),
            'line2': (self.p2, self.p3, self.line2),
            'line3': (self.p3, self.p4, self.line3),
            'line4': (self.p4, self.p1, self.line4),
        }

        # 向量
        self.v12 = np.array(self.p2) - np.array(self.p1)
        self.v23 = np.array(self.p3) - np.array(self.p2)
        self.v34 = np.array(self.p4) - np.array(self.p3)
        self.v41 = np.array(self.p1) - np.array(self.p4)
        self.vectors = {
            'v12': (self.p1, self.p2, self.v12),
            'v23': (self.p2, self.p3, self.v23),
            'v34': (self.p3, self.p4, self.v34),
            'v41': (self.p4, self.p1, self.v41),
        }

    def calc_nearest_line(self, px, py):
        # Desc: 计算点(px, py)到矩形四条边的最近的一条的key
        d1 = calc_point2line(px, py, self.line1)
        d2 = calc_point2line(px, py, self.line2)
        d3 = calc_point2line(px, py, self.line3)
        d4 = calc_point2line(px, py, self.line4)
        # 计算最小距离对应的Line的索引
        d_list = [d1, d2, d3, d4]
        min_d = min(d_list)
        return 'line' + str(d_list.index(min_d) + 1)

    def is_point_in_rect(self, px, py):
        cross_results = []
        for vertex, _, vector in self.vectors.values():
            point2vertex = np.array(vertex) - np.array([px, py])
            cross_result = cross_product(vector, point2vertex)
            cross_results.append(cross_result)

        if all([cross_result >= 0 for cross_result in cross_results]) or all(
                [cross_result <= 0 for cross_result in cross_results]):
            return True
        return False


def calc_rect_4points(cx, cy, vx, vy, view_scope):
    # Desc: 计算矩形四个顶点坐标，顶点顺序为顺时针，从左上角开始：p1->p2->p3->p4
    top_v = np.array([vx, vy])
    top_v = normalize(top_v) * view_scope
    # 顺时针旋转90度
    right_v = np.array([top_v[1], -top_v[0]])
    right_v = normalize(right_v) * view_scope

    center_point = np.array([cx, cy])
    top_left_point = center_point + top_v - right_v
    top_right_point = center_point + top_v + right_v
    bottom_right_point = center_point - top_v + right_v
    bottom_left_point = center_point - top_v - right_v

    # 转为list
    return {
        'p1': top_left_point.tolist(),
        'p2': top_right_point.tolist(),
        'p3': bottom_right_point.tolist(),
        'p4': bottom_left_point.tolist()
    }


def calc_rect_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    if x2 - x1 == 0:  # 避免除以0
        k = float('inf')
        a, b, c = 1, 0, -x1
        print(f'警告: 两点{x1, y1}, {x2, y2}连线斜率为无穷大')
    else:
        k = (y2 - y1) / (x2 - x1)
        a, b, c = k, -1, y1 - k * x1

    return a, b, c


def calc_point2line(px, py, line):
    a, b, c = line
    return abs(a * px + b * py + c) / (a ** 2 + b ** 2) ** 0.5


def normalize(v):
    # Desc: 将向量归一化
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def cross_product(v1, v2):
    # Desc: 计算两个向量的叉乘
    return v1[0] * v2[1] - v1[1] * v2[0]
