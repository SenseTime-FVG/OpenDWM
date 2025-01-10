# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 下午10:24
# @Author  : Hcyang
# @File    : functions.py
# @Desc    : TODO:

import cv2
import os
import sys
import json
import h5py
import carla
import pickle
import argparse

# import ipdb
import numpy as np
from PIL import Image
from collections import deque
from pathlib import Path

current_dir = Path(__file__).resolve().parent
target_dir = current_dir
for _ in range(4):
    target_dir = target_dir.parent
sys.path.append(os.path.dirname(target_dir))
from Xodrlib.OpenDriveLibrary import MapParser

# from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


class ImageSaver:
    def __init__(self, save_path, all_sensors):
        self.save_path = save_path
        self.all_sensors = all_sensors

    def save_image(self, sensor_id, frame, image):
        # 保存图像
        save_path = os.path.join(self.save_path, sensor_id, "rgb",f"{frame:.5f}.jpg")
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image)
        # print(f"Saved: {save_path}")

    def save_rgb_images(self, tick_data, frame):
        with ThreadPoolExecutor() as executor:
            # 启动多线程保存任务
            futures = [
                executor.submit(self.save_image, sensor['id'], frame, Image.fromarray(tick_data[sensor['id']]))
                for sensor in self.all_sensors if sensor['type'] == 'sensor.camera.rgb'
            ]
            # 等待所有保存任务完成
            for future in futures:
                future.result()


class ObsManager:
    def __init__(self):
        """
        obs_configs:
            birdview:
                width_in_pixels: 500
                pixels_ev_to_bottom: 250
                pixels_per_meter: 5.0
                scale_bbox: true
                scale_mask_col: 1.0
        """

        self._width = 500
        self._pixels_ev_to_bottom = 250
        self._pixels_per_meter = 5.0
        self._scale_bbox = True
        self._scale_mask_col = 1.0

        self._image_channels = 3
        self._parent_actor = None
        self._world = None

        self._map_dir = Path(__file__).resolve().parent / 'maps'

        self._distance_threshold = None
        self._world_offset = None
        self._lane_marking_white_broken = None
        self._road = None
        self._lane_marking_all = None

        self.map_parser = None

        super(ObsManager, self).__init__()

    def attach_ego_vehicle(self, ego_vehicle):
        self._parent_actor = ego_vehicle
        self._world = self._parent_actor.get_world()

        maps_h5_path = self._map_dir / (self._world.get_map().name.split('/')[-1] + '.h5')  # WY NOTE: 这里确定用哪个h5地图
        with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf:
            self._road = np.array(hf['road'], dtype=np.uint8)
            self._lane_marking_all = np.array(hf['lane_marking_all'], dtype=np.uint8)
            self._lane_marking_white_broken = np.array(hf['lane_marking_white_broken'], dtype=np.uint8)

            self._world_offset = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)
            assert np.isclose(self._pixels_per_meter, float(hf.attrs['pixels_per_meter']))

        xodr_path = self._map_dir / (self._world.get_map().name.split('/')[-1] + '.xodr')

        # xindan version
        if not os.path.exists(xodr_path):
            # print(xodr_path)
            self._world.get_map().save_to_disk(path=str(xodr_path))
        if not os.path.exists(xodr_path):
            print(xodr_path)
            raise NotImplementedError("map")
        self.map_parser = MapParser(xodr_path)
        self._distance_threshold = np.ceil(self._width / self._pixels_per_meter)

    def get_seg_mask(self,step):
        cur_loc = self._parent_actor.get_transform()
        # 2405: map get transform y = xodr y * -1
        self.map_parser.car_init(car_coords=(cur_loc.location.x,-cur_loc.location.y, cur_loc.location.z),
                                 car_hdg=cur_loc.rotation.yaw ,# 180 * np.pi,
                                 scope_r=self._distance_threshold / 2.0, quality=10)
                                 # theta = 0.002)
                                 # theta=0.005, z_lim=5
        seg_res, seg_types, seg_colors = self.map_parser.get_segs()
        
        # 注释 debug
        # self.map_parser.scopemap_init()
        # # ipdb.set_trace()
        # # if seg_res is None:
        # #     scope_res = self.map_parser.get_scope(segs=seg_res)
        # # else:
        # seg_res, scope_res = self.map_parser.get_scope(segs=seg_res,step=step)  # (_distance_threshold * quality) ^2
        # # scope_res = self.map_parser.get_scope()  # (_distance_threshold * quality) ^2
        # debug
        scope_res = None
        
        return seg_res, scope_res, seg_types, seg_colors


    def get_observation(self):
        ev_transform = self._parent_actor.get_transform()
        ev_loc = ev_transform.location
        ev_rot = ev_transform.rotation

        M_warp = self._get_warp_transform(ev_loc, ev_rot)

        # road_mask, lane_mask
        road_mask = cv2.warpAffine(self._road, M_warp, (self._width, self._width)).astype(np.bool)
        lane_mask_all = cv2.warpAffine(self._lane_marking_all, M_warp, (self._width, self._width)).astype(np.bool)
        lane_mask_broken = cv2.warpAffine(self._lane_marking_white_broken, M_warp, (self._width, self._width)).astype(
            np.bool)

        # masks
        c_road = road_mask * 255
        c_lane = lane_mask_all * 255
        c_lane[lane_mask_broken] = 120

        # Special: 只保留roadline和drivable area
        masks = np.stack((c_road, c_lane), axis=2)
        masks = np.transpose(masks, [2, 0, 1])



        return masks

    def _get_warp_transform(self, ev_loc, ev_rot):
        ev_loc_in_px = self._world_to_pixel(ev_loc)
        yaw = np.deg2rad(ev_rot.yaw)

        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        right_vec = np.array([np.cos(yaw + 0.5 * np.pi), np.sin(yaw + 0.5 * np.pi)])

        bottom_left = ev_loc_in_px - self._pixels_ev_to_bottom * forward_vec - (0.5 * self._width) * right_vec
        top_left = ev_loc_in_px + (self._width - self._pixels_ev_to_bottom) * forward_vec - (0.5 * self._width) * right_vec
        top_right = ev_loc_in_px + (self._width - self._pixels_ev_to_bottom) * forward_vec + (0.5 * self._width) * right_vec

        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.array([[0, self._width - 1],
                            [0, 0],
                            [self._width - 1, 0]], dtype=np.float32)
        return cv2.getAffineTransform(src_pts, dst_pts)

    def _world_to_pixel(self, location, projective=False):
        """Converts the world coordinates to pixel coordinates"""
        x = self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self._pixels_per_meter * (location.y - self._world_offset[1])

        if projective:
            p = np.array([x, y, 1], dtype=np.float32)
        else:
            p = np.array([x, y], dtype=np.float32)
        return p

    def _world_to_pixel_width(self, width):
        """Converts the world units to pixel units"""
        return self._pixels_per_meter * width
