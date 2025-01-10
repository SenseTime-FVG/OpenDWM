# -*- coding: utf-8 -*-
# @Time    : 2024/3/8 下午3:34
# @Author  : Hcyang
# @File    : Sensor_initializer.py
# @Desc    : TODO:

import os
import sys
import json
# import ipdb
import carla
import pickle
import copy
import argparse

import numpy as np
# from tqdm import tqdm

from .Basic_initializer import BasicInitializer
from Initialization.Standardimport.scenariomanager.carla_data_provider import CarlaDataProvider
from Logger import *


class SensorInitializer(BasicInitializer):
    def __init__(self, ego, saver):
        self.ego = ego
        self.saver = saver
        self.sensors_list = []

    def run(self, *args, **kwargs):
        self.setup_sensors()

    def setup_sensors(self):
        bp_library = CarlaDataProvider.get_world().get_blueprint_library()
        sensors = self.saver.sensors()

        for sensor_spec in sensors:
            bp = bp_library.find(str(sensor_spec['type']))

            sensor_location = carla.Location(
                x=sensor_spec['x'],
                y=sensor_spec['y'],
                z=sensor_spec['z']
            )
            sensor_rotation = carla.Rotation(
                pitch=sensor_spec['pitch'],
                roll=sensor_spec['roll'],
                yaw=sensor_spec['yaw']
            )

            if sensor_spec['type'].startswith('sensor.camera.semantic_segmentation'):
                bp.set_attribute('image_size_x', str(sensor_spec['width']))
                bp.set_attribute('image_size_y', str(sensor_spec['height']))
                bp.set_attribute('fov', str(sensor_spec['fov']))

            elif sensor_spec['type'].startswith('sensor.camera.depth'):
                bp.set_attribute('image_size_x', str(sensor_spec['width']))
                bp.set_attribute('image_size_y', str(sensor_spec['height']))
                bp.set_attribute('fov', str(sensor_spec['fov']))

            elif sensor_spec['type'].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(sensor_spec['width']))
                bp.set_attribute('image_size_y', str(sensor_spec['height']))
                bp.set_attribute('fov', str(sensor_spec['fov']))
                # For: 保留畸变
                # bp.set_attribute('lens_circle_multiplier', str(3.0))
                # bp.set_attribute('lens_circle_falloff', str(3.0))
                # bp.set_attribute('chromatic_aberration_intensity', str(0.5))
                # bp.set_attribute('chromatic_aberration_offset', str(0))
                # For: 去除畸变
                bp.set_attribute('lens_circle_multiplier', str(0.0))
                bp.set_attribute('lens_circle_falloff', str(0.00001))
                bp.set_attribute('chromatic_aberration_intensity', str(0.5))
                bp.set_attribute('chromatic_aberration_offset', str(0))
                bp.set_attribute('role_name', str(sensor_spec["id"]))

            elif sensor_spec['type'].startswith('sensor.lidar.ray_cast_semantic'):
                bp.set_attribute('range', str(85))
                bp.set_attribute('rotation_frequency', str(10))  # default: 10, change to 20 for old lidar models
                bp.set_attribute('channels', str(64))
                bp.set_attribute('upper_fov', str(10))
                bp.set_attribute('lower_fov', str(-30))
                bp.set_attribute('points_per_second', str(600000))

            elif sensor_spec['type'].startswith('sensor.lidar'):
                bp.set_attribute('range', str(85))
                bp.set_attribute('rotation_frequency', str(10))  # default: 10, change to 20 to generate 360 degree LiDAR point cloud
                bp.set_attribute('channels', str(64))
                bp.set_attribute('upper_fov', str(10))
                bp.set_attribute('lower_fov', str(-30))
                bp.set_attribute('points_per_second', str(600000))
                bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
                bp.set_attribute('dropoff_general_rate', str(0.45))
                bp.set_attribute('dropoff_intensity_limit', str(0.8))
                bp.set_attribute('dropoff_zero_intensity', str(0.4))

            elif sensor_spec['type'].startswith('sensor.other.radar'):
                bp.set_attribute('horizontal_fov', str(sensor_spec['fov']))  # degrees
                bp.set_attribute('vertical_fov', str(sensor_spec['fov']))  # degrees
                bp.set_attribute('points_per_second', '1500')
                bp.set_attribute('range', '100')  # meters

            elif sensor_spec['type'].startswith('sensor.other.gnss'):
                # bp.set_attribute('noise_alt_stddev', str(0.000005))
                # bp.set_attribute('noise_lat_stddev', str(0.000005))
                # bp.set_attribute('noise_lon_stddev', str(0.000005))
                bp.set_attribute('noise_alt_bias', str(0.0))
                bp.set_attribute('noise_lat_bias', str(0.0))
                bp.set_attribute('noise_lon_bias', str(0.0))
                sensor_rotation = carla.Rotation()

            elif sensor_spec['type'].startswith('sensor.other.imu'):
                bp.set_attribute('noise_accel_stddev_x', str(0.001))
                bp.set_attribute('noise_accel_stddev_y', str(0.001))
                bp.set_attribute('noise_accel_stddev_z', str(0.015))
                bp.set_attribute('noise_gyro_stddev_x', str(0.001))
                bp.set_attribute('noise_gyro_stddev_y', str(0.001))
                bp.set_attribute('noise_gyro_stddev_z', str(0.001))

            # create sensor
            sensor_transform = carla.Transform(sensor_location, sensor_rotation)
            sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, self.ego)
            # setup callback
            sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self.saver.sensor_interface))
            self.sensors_list.append(sensor)

    def stop_all(self):
        for sensor in self.sensors_list:
            sensor.stop()


class CallBack(object):
    def __init__(self, tag, sensor_type, sensor, data_provider):
        self._tag = tag
        self._data_provider = data_provider

        self._data_provider.register_sensor(tag, sensor_type, sensor)
        print(f'注册： {tag}  类型： {sensor_type}')

    def __call__(self, data):
        # special(f'++ 接收到传感器数据：{self._tag}')
        if isinstance(data, carla.libcarla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.SemanticLidarMeasurement):
            self._parse_semantic_lidar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.RadarMeasurement):
            self._parse_radar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.IMUMeasurement):
            self._parse_imu_cb(data, self._tag)
        else:
            print('No callback method for this sensor.')

    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image, tag):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        self._data_provider.update_sensor(tag, array, image.frame)

    def _parse_lidar_cb(self, lidar_data, tag):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self._data_provider.update_sensor(tag, points, lidar_data.frame)

    def _parse_semantic_lidar_cb(self, semantic_lidar_data, tag):
        points = np.frombuffer(semantic_lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        self._data_provider.update_sensor(tag, points, semantic_lidar_data.frame)

    def _parse_radar_cb(self, radar_data, tag):
        # [depth, azimuth, altitude, velocity]
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        self._data_provider.update_sensor(tag, points, radar_data.frame)

    def _parse_gnss_cb(self, gnss_data, tag):
        array = np.array([gnss_data.latitude, gnss_data.longitude, gnss_data.altitude], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, gnss_data.frame)

    def _parse_imu_cb(self, imu_data, tag):
        array = np.array([
            imu_data.accelerometer.x,
            imu_data.accelerometer.y,
            imu_data.accelerometer.z,
            imu_data.gyroscope.x,
            imu_data.gyroscope.y,
            imu_data.gyroscope.z,
            imu_data.compass
        ], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, imu_data.frame)
