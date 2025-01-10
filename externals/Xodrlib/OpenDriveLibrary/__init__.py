# -*- coding: utf-8 -*-
# @Time    : 2024/3/12 下午4:45
# @Author  : Hcyang
# @File    : __init__.py.py
# @Desc    : TODO:
# import ipdb
from tqdm import tqdm
from .RoadLib import *
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt

import matplotlib.path as mplpath
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np


class MapParser(object):
    def __init__(self, xodr_file_path):
        self.xodr_file_path = xodr_file_path
        self.opendrive = None
        self.road_list = []

        self.car = {
            'carx': 0.0,
            'cary': 0.0,
            'carz': 0.0,
            'z_lim': 5,
            'carhdg': 0.0,  # rad
            'dis': 50.0,  # m
            'theta': 0.005,  # about 0.3 degree
            'lazyflag': True,
            'box': [],  # [minx,maxx,miny,maxy,s_coords]
            'quality': 1e2
        }

        self._init()
        self.roads_in_scope = []

        self.mapfig = {
            'map': np.zeros((1, 1)),
            'x_offset': 0,
            'y_offset': 0,
            'z_offset': 0,
            'quality': 1e2
        }

    def _init(self):
        xodr_data = ET.parse(self.xodr_file_path)
        self.opendrive = OpenDriveParser(xodr_data.getroot())

        for road in tqdm(self.opendrive.road_s):
            road = RoadParser(road)
            road.list_init()
            road.calc_lanemarks_3d()
            road.calc_drivable_area_3d()
            self.road_list.append(road)



    def visualize_elevation(self):
        print('visualize 3d')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for road in self.road_list:
            road.visualize_3d(ax)
        ax.set_xlim([-270, 270])
        ax.set_ylim([-220, 220])
        ax.set_zlim([-1, 150])
        ax.set_facecolor('black')
        ax.grid(False)
        ax.axis('off')
        plt.show()
        plt.close()
        # exit()

    def car_init(self, car_coords=(0.0, 0.0, 0.0), car_hdg=0.0,
                 scope_r=50.0, theta=0.005, z_lim=5, quality=1e2):
        self.car['carx'] = car_coords[0]
        self.car['cary'] = car_coords[1]
        self.car['carz'] = car_coords[2]
        self.car['z_lim'] = z_lim
        self.car['carhdg'] = car_hdg
        self.car['dis'] = scope_r
        self.car['theta'] = theta
        self.car['quality'] = quality
        self.car['lazyflag'] = True

    def scopemap_init(self):
        # 初始化画布大小，留出车的视野的冗余边界
        quality = self.car['quality']
        x_offset = None
        y_offset = None
        x_ru = None
        y_ru = None
        points=[]
        # 寻找整个地图xy平面的左下角与右上角，在lanelib里有记录
        for road in self.road_list:
            for lane_section in road.lanes.lane_section_list.lane_sections:
                all_lane = lane_section.left_lanes + [lane_section.center_lane] + lane_section.right_lanes
                for cur_lane in all_lane:
                    if len(cur_lane.left_lower) > 1:
                        if x_offset is None:
                            x_offset = cur_lane.left_lower[0]
                            y_offset = cur_lane.left_lower[1]

                            x_ru = cur_lane.right_upper[0]
                            y_ru = cur_lane.right_upper[1]
                        else:
                            x_offset = min(x_offset, cur_lane.left_lower[0])
                            y_offset = min(y_offset, cur_lane.left_lower[1])
                            x_ru = max(cur_lane.right_upper[0], x_ru)
                            y_ru = max(cur_lane.right_upper[1], y_ru)
        margin = self.car['dis'] * 1.5
        x_ru = int((x_ru - x_offset + margin * 2) * quality)
        y_ru = int((y_ru - y_offset + margin * 2) * quality)

        mapfig = np.full((y_ru, x_ru), 255, np.uint8)

        for road in self.roads_in_scope:
        # for road in self.road_list:
            road.draw_road(mapfig, x_offset - margin, y_offset - margin, quality)

        self.mapfig['map'] = mapfig
        self.mapfig['x_offset'] = x_offset - margin
        self.mapfig['y_offset'] = y_offset - margin
        self.mapfig['quality'] = quality

    def cal_car(self):
        carx = self.car['carx']
        cary = self.car['cary']
        carhdg = (90-self.car['carhdg']) / 180 * np.pi
        dis = self.car['dis']

        x1, x2, y1, y2 = self.half_scope_square(carx + dis * np.cos(carhdg),
                                                cary + dis * np.sin(carhdg),
                                                carx, cary, dis)
        x3, x4, y3, y4 = self.half_scope_square(carx - dis * np.cos(carhdg),
                                                cary - dis * np.sin(carhdg),
                                                carx, cary, dis)
        # ccw_sort 现在这个框可以是顶点顺序混乱的非凸多边形了
        s_coords = self.ccw_sort([[x1, y1], [x2, y2], [x4, y4], [x3, y3]])

        minx = np.min([x1, x2, x3, x4])
        maxx = np.max([x1, x2, x3, x4])
        miny = np.min([y1, y2, y3, y4])
        maxy = np.max([y1, y2, y3, y4])
        self.car['box'] = [minx, maxx, miny, maxy, s_coords]
        self.car['lazyflag'] = False

        del self.roads_in_scope[:]

        return minx, maxx, miny, maxy, s_coords

    def get_segs(self):
        if self.car['lazyflag']:
            minx, maxx, miny, maxy, s_coords = self.cal_car()
        else:
            minx, maxx, miny, maxy, s_coords = self.car['box']

        theta = self.car['theta']
        z_min, z_max = self.car['carz'] - self.car['z_lim'], self.car['carz'] + self.car['z_lim']
        polyg = mplpath.Path(s_coords)
        all_local_segms = None
        linetypes=None
        linecolors=None
        for road in self.road_list:
            if float(road.road.attrib['length']) < 0.1:
                continue
            # print(road)
            # segms, local_segms = self.seg_one_road(road, polyg, theta)  # 备用绘图
            local_segms = self.seg_one_road(road, polyg, theta, z_min, z_max)
            if type(local_segms) is int:
                self.roads_in_scope.append(road)
                continue
            else:
                local_segms, tps, crs = local_segms
            if len(local_segms) < 1:  # 没在scope里
                continue
            if road.road.attrib['junction'] == '-1':
                if all_local_segms is None:
                    all_local_segms = local_segms
                    linetypes = tps
                    linecolors = crs
                else:
                    all_local_segms += local_segms
                    linetypes += tps
                    linecolors += crs
                # all_local_segms.append(np.array(local_segms, dtype=object))
            self.roads_in_scope.append(road)
            # 以下废弃作图步骤，留作备用
            # ax.scatter(px, py, c='y')  # 不同road的端点
            # for lanesegs in segms:  # 所有分段
            #     for segs in lanesegs:
            #         s, e = list(zip(*segs))
            #         ax.plot(s, e, linestyle='solid')
            # 以上废弃可视化，需要确定分段正常的时候可用，需要在函数调用中增加参数ax
        # ipdb.set_trace()

        return all_local_segms, linetypes, linecolors

    def get_scope(self, scale=None, segs=None,step=-1):
        # 最新结论：carla get_location 的 y 与 xodr地图的 y 相反

        if self.car['lazyflag']:
            minx, maxx, miny, maxy, s_coords = self.cal_car()
        else:
            minx, maxx, miny, maxy, s_coords = self.car['box']
        minx = int((minx - self.mapfig['x_offset']) * self.mapfig['quality']+0.5)
        maxx = int((maxx - self.mapfig['x_offset']) * self.mapfig['quality']+0.5)
        miny = int((miny - self.mapfig['y_offset']) * self.mapfig['quality']+0.5)
        maxy = int((maxy - self.mapfig['y_offset']) * self.mapfig['quality']+0.5)
        res = self.mapfig['map'][miny:maxy + 1, minx:maxx + 1]
        _, res = cv2.threshold(res,
                               254, 255, cv2.THRESH_BINARY_INV)
        # res = cv2.flip(res, 0)
        # cv2.imwrite('cvbi.png', res)  # 很大的图，如果需要check，存下来
        centre = (int((self.car['cary'] - self.mapfig['y_offset']) * self.mapfig['quality']+0.5 - miny),
                                    int((self.car['carx'] - self.mapfig['x_offset']) * self.mapfig['quality']+0.5 - minx))
        dispx = int(self.car['dis'] * self.mapfig['quality'])
        # angle = 180 if self.car['carhdg']<0 else 0
        # angle = 0
        angle = 90-self.car['carhdg']
        # if step==30 or step==190:
        #     print(self.car)
        res,segs = self.rotate_and_crop(res, segs, angle,
                                   centre,
                                   scale=scale,
                                   dis=dispx)
        # cv2.imwrite('cvres.png', res)  # 很大的图，如果需要check，存下来
        # res[res == 255] = 1  # 蒙版化
        # _, res = cv2.threshold(res,
        #                        254, 255, cv2.THRESH_BINARY_INV)
        # if self.car['carhdg']>0:
        res = cv2.flip(res, 1)  #
        res = cv2.flip(res, 0)  #

        contours, _ = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull_list = []
        for i in range(len(contours)):
            hullp = []
            for p in contours[i]:
                px = (p[0][0]-dispx+0.5).astype(np.float32)/self.mapfig['quality']
                py = (p[0][1]-dispx+0.5).astype(np.float32)/self.mapfig['quality']
                hullp.append([px,py])
            hull_list.append(hullp)

        if segs is not None:
            return segs, hull_list #res
        else:
            return res

    def seg_one_road(self, cur_road, polyg, theta, z_min, z_max):
        # 处理特定road的车道线分段
        # candidates = []
        # inside_points = []
        # inside_sd = []
        # segms = []  # 绘图备用列表
        local_segms = []
        road_linetypes=[]
        road_linecolors=[]
        # 枚举每一条车道，并且确保可行驶，如果不是路口就枚举其车道线坐标，否则枚举其车道坐标
        for lane_section in cur_road.lanes.lane_section_list.lane_sections:
            all_lane = lane_section.left_lanes + [lane_section.center_lane] + lane_section.right_lanes
            for cur_lane in all_lane:
                if cur_lane.lane_section.lanes.road.road.attrib.junction == '-1':
                    for roadmark in cur_lane.roadmark_list:
                        if roadmark.lanemark_coords_3d['usable'] \
                                and roadmark.type in roadmark.vis_roadmark_types:
                            # 沿着车道线s路径
                            x_coords, y_coords, z_coords = roadmark.lanemark_coords_3d['xyz']
                            ps = roadmark.lanemark_coords_3d['sd'][0][0]  # 当前点的s坐标
                            pre = -1  # 前一个端点
                            # lane_segms = []
                            local_ls = []
                            st = -1  # 当前起点索引

                            for i in range(len(x_coords)):
                                p = (x_coords[i], y_coords[i])
                                # 若当前点在车的视野方框内且符合高程限制，此时若存在s坐标的大间隔，
                                # 存在前一个点，则说明这条车道线穿出视野又穿回视野，记录该对出点和入点，更新起点索引
                                if polyg.contains_point(p) and z_max >= z_coords[i] >= z_min:
                                    # inside_points.append(p)
                                    cur_s = roadmark.lanemark_coords_3d['sd'][0][i]
                                    if cur_s - ps > 2:
                                        if pre >= 0:
                                            # candidates.append([(x_coords[pre],
                                            #                     y_coords[pre]),
                                            #                    (cur_lane.lanemark_coords['sd'][0][pre],
                                            #                     cur_lane.lanemark_coords['sd'][1][pre])
                                            #                    ])
                                            # lane_segms.append([(x_coords[st], y_coords[st]),
                                            #                    (x_coords[pre], y_coords[pre])])
                                            local_ls.append(
                                                    [x_coords[st] - self.car['carx'], y_coords[st] - self.car['cary']])
                                            local_ls.append(
                                                [x_coords[pre] - self.car['carx'],
                                                          y_coords[pre] - self.car['cary']])
                                        # candidates.append([p, (cur_s, cur_d)])
                                        st = i
                                    else:
                                        # s坐标连续，计算 当前点与当前分段起点连线 和 前一点与当前分段起点连线 的 夹角
                                        if st == -1:
                                            st = i
                                        elif st != pre:
                                            pst = np.array(p) - np.array((x_coords[st], y_coords[st]))
                                            prest = np.array((x_coords[st + 1],
                                                              y_coords[st + 1])) - np.array(
                                                (x_coords[st], y_coords[st]))
                                            dot = np.dot(pst, prest)
                                            cosvalue = dot / np.sqrt((pst * pst).sum()) / np.sqrt((prest * prest).sum())
                                            if cosvalue >= 1:
                                                angle = np.nan
                                            else:
                                                angle = np.arccos(cosvalue)
                                            if abs(angle) > theta:  # np.nan !>!< theta
                                                local_ls.append(
                                                    [x_coords[st] - self.car['carx'],
                                                              y_coords[st] - self.car['cary']])
                                                local_ls.append(
                                                    [x_coords[pre] - self.car['carx'],
                                                              y_coords[pre] - self.car['cary']])
                                                # lane_segms.append([(x_coords[st], y_coords[st]),
                                                #                    (x_coords[pre], y_coords[pre])])
                                                st = pre
                                    ps = cur_s
                                    # inside_sd.append((cur_s, cur_d))
                                    pre = i
                                if (i == 0 or i == len(x_coords) - 1) and pre >= 0:
                                    # 包括了section起点终点在中间的情况
                                    # candidates.append([(x_coords[pre],
                                    #                     y_coords[pre]),
                                    #                    (cur_lane.lanemark_coords['sd'][0][pre],
                                    #                     cur_lane.lanemark_coords['sd'][1][pre])
                                    #                    ])
                                    if i == 0:
                                        st = i
                                    else:
                                        local_ls.append(
                                                [x_coords[st] - self.car['carx'], y_coords[st] - self.car['cary']])
                                        local_ls.append(
                                                [x_coords[pre] - self.car['carx'], y_coords[pre] - self.car['cary']])

                                        # lane_segms.append([(x_coords[st], y_coords[st]),
                                        #                    (x_coords[pre], y_coords[pre])])
                            # if len(lane_segms) < 1:
                            #     continue
                            if len(local_ls) < 1:
                                continue
                            # segms.append(lane_segms)
                            local_segms.append(local_ls)
                            road_linetypes.append('kDASHED' if 'broken' in roadmark.roadmark.attrib.type else 'kSOLID')
                            road_linecolors.append('kWHITE' if 'white' in roadmark.roadmark.attrib.color else 'kYELLOW')
                else:
                    if cur_lane.lanemark_coords_3d['usable'] \
                            and cur_lane.type in ['driving', 'bidirectional', 'median', 'parking']:
                        x_coords, y_coords, z_coords = cur_lane.lanemark_coords_3d['xyz']
                        for i in range(len(x_coords)):
                            p = (x_coords[i], y_coords[i])
                            if polyg.contains_point(p) and z_max >= z_coords[i] >= z_min:
                                # 是路口的话，只要有一个点出现在视野中都需要完全画出来，再由后续步骤裁剪，
                                # 因为路口没有清楚的车道线
                                return int(cur_road.road.attrib['id'])
        # ipdb.set_trace()
        # px = []
        # py = []
        # for can in candidates:
        #     px.append(can[0][0])
        #     py.append(can[0][1])
        # print(inside_points)

        # return segms, local_segms
        return local_segms, road_linetypes, road_linecolors

    def ccw_sort(self, p):
        # 确保scope框是矩形
        p = np.array(p)
        mean = np.mean(p, axis=0)
        d = p - mean
        s = np.arctan2(d[:, 0], d[:, 1])
        return p[np.argsort(s), :]

    def half_scope_square(self, mx, my, carx, cary, dis):
        # 计算上/下半矩形的顶点坐标
        k = (mx - carx) / (cary - my)
        b = my - k * mx
        A = 1 + np.power(k, 2)
        B = 2 * (b * k - mx - my * k)
        C = np.power(mx, 2) + np.power(b - my, 2) - np.power(dis, 2)
        if np.power(B, 2) - 4 * A * C < 0:
            print(A, B, C)
            print('delta error')
            # exit()
        delta = np.sqrt(np.power(B, 2) - 4 * A * C)

        x1 = (-B - delta) / (2 * A)
        x2 = (-B + delta) / (2 * A)
        y1 = k * x1 + b
        y2 = k * x2 + b

        return x1, x2, y1, y2

    def rotate_and_crop(self, img, segs, angle, car, scale=None, dis=50):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((car[0], car[1]), angle, 1.0)
        res = cv2.warpAffine(img, M, (h, w))

        crop = res[car[0] - dis+1:car[0] + dis+1, car[1] - dis+1:car[1] + dis+1]
        if scale is not None:
            crop = cv2.resize(crop, scale, interpolation=cv2.INTER_LINEAR)

        M = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
        # new_segs=[]
        if segs is not None:
                for lane in segs:
                    # new_lane=[]
                    for i in range(len(lane)):
                        ori = np.array([lane[i][0], lane[i][1], 1])
                        x1, y1 = M @ ori
                        # lane[i] = [-x1,y1]
                        lane[i] = [-x1,-y1]
                        # new_lane.append([x1, y1])
                    # new_segs.append(new_lane)

        return crop, segs

    def visualize(self):
        fig, ax = plt.subplots()
        ax.set_facecolor('black')
        for road in self.road_list:
            road.visualize(ax)

        plt.show()

        plt.close()

    def generate_scope_data(self, x, y):
        pass
