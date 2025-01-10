# -*- coding: utf-8 -*-
# @Time    : 2024/3/8 下午3:53
# @Author  : Hcyang
# @File    : Basic_saver.py
# @Desc    : TODO:

import os
import sys
import json
import carla
import pickle
import argparse

# from tqdm import tqdm
from queue import Queue
from queue import Empty

from Initialization.Standardimport.scenariomanager.timer import GameTime


class BasicSaver(object):
    def __init__(self):
        self.sensor_interface = SensorInterface()
        self.wallclock_t0 = None

    def sensors(self):  # pylint: disable=no-self-use
        raise NotImplementedError

    def run_step(self, input_data, timestamp, *args, **kwargs):
        raise NotImplementedError

    def destroy(self):
        pass

    def __call__(self, sensors, *args, **kwargs):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """

        input_data = self.sensor_interface.get_data()

        timestamp = GameTime.get_time()

        self.run_step(sensors, input_data, timestamp, *args, **kwargs)

    def get_debug_image_dir(self):
        raise NotImplementedError

    def check_data_is_empty(self):
        raise NotImplementedError

    def calc_critical_waypoint(self):
        raise NotImplementedError

    def calc_max_velocity(self):
        raise NotImplementedError

    def save_data_tags(self, data_tags):
        raise NotImplementedError


class SensorInterface(object):
    def __init__(self):
        self._sensors_objects = {}
        self._data_buffers = {}
        self._new_data_buffers = Queue()
        self._queue_timeout = 10  # default: 10

        # Only sensor that doesn't get the data on tick, needs special treatment
        self._opendrive_tag = None

    def register_sensor(self, tag, sensor_type, sensor):
        if tag not in self._sensors_objects:
            # raise NotImplementedError("Duplicated sensor tag [{}]".format(tag))
            self._sensors_objects[tag] = sensor

        if sensor_type == 'sensor.opendrive_map':
            self._opendrive_tag = tag

    def update_sensor(self, tag, data, timestamp):
        # print("Updating {} - {}".format(tag, timestamp))
        if tag not in self._sensors_objects:
            raise NotImplementedError("The sensor with tag [{}] has not been created!".format(tag))

        self._new_data_buffers.put((tag, timestamp, data))

    def get_data(self):
        data_dict = {}
        try:
            while len(data_dict.keys()) < len(self._sensors_objects.keys()):

                # Don't wait for the opendrive sensor
                if self._opendrive_tag and self._opendrive_tag not in data_dict.keys() \
                        and len(self._sensors_objects.keys()) == len(data_dict.keys()) + 1:
                    # print("Ignoring opendrive sensor")
                    break

                sensor_data = self._new_data_buffers.get(True, self._queue_timeout)
                data_dict[sensor_data[0]] = ((sensor_data[1], sensor_data[2]))

        except Empty:
            print(f'A sensor took too long to send their data')
            print(f'sensors: {self._sensors_objects.keys()}')
            print(f'data: {data_dict.keys()}')
            for k in self._sensors_objects.keys():
                if k not in data_dict:
                    print(f"Sensor {k} fails provide data")
                    data_dict[k] = 0
        return data_dict

