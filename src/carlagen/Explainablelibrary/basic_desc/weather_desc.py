# -*- coding: utf-8 -*-
# @Time    : 2023/11/10 下午4:56
# @Author  : Hcyang
# @File    : weather_desc.py
# @Desc    : xxx


import sys
import os
import argparse
import pickle
import json
# # import ipdb
from .basic_desc_utils import *


class WeatherDesc(BasicDesc):
    def __init__(self, world, ego):
        super(WeatherDesc, self).__init__(world, ego)
        self.weather = None
        self.sun_arg = None

    def _gen_env_desc(self, short=True, long=False):
        # Desc: 生成环境描述
        # Special: [短描述，长描述，反面描述]
        essential_atomic_keys = ['sun', 'rain', 'ground_water', 'fog']
        if short:
            desc = [self.atomic_desc[key][0] for key in essential_atomic_keys]
        elif long:
            desc = [self.atomic_desc[key][1] for key in essential_atomic_keys]
        else:
            raise ValueError("short and long cannot be both False")
        return [item.capitalize() for item in desc if item.strip()]

    def _gen_atomic_desc(self):
        self.weather = self.world.get_weather()
        atomic_desc = {
            'cloud': self._cloud_desc(),
            'rain': self._rain_desc(),
            'ground_water': self._ground_water_desc(),
            'wind': self._wind_desc(),
            'sun': self._sun_desc(randomly=True),
            'fog': self._fog_desc()
        }
        return atomic_desc

    def _cloud_desc(self):
        """
        Desc: 生成云层描述
        Special: [短词用于插入，长句完整描述]
        0: clear sky
        100: completely covered with clouds
        """
        cloudiness = self.weather.cloudiness
        if cloudiness < 20:
            desc = ["clear sky", "the sky is clear"]
        elif cloudiness < 50:
            desc = ["partly cloudy", "the sky is partly cloudy"]
        else:
            desc = ["completely covered with clouds", "the sky is completely covered with clouds"]
        return desc

    def _rain_desc(self):
        """
        Desc: 生成降雨描述
        Special: [短词用于插入，长句完整描述]
        0: no rain
        100: heavy rain
        """
        precipitation = self.weather.precipitation
        if precipitation <= 1:
            # desc = ["no rain", "there is no rain"]
            desc = ["", ""]
        else:
            if precipitation < 70:
                desc = ["light rain", "it is drizzling"]
            else:
                desc = ["heavy rain", "it is pouring rain"]
            self.add_speed_reason(DEC, 'rain')
        return desc

    def _ground_water_desc(self):
        """
        Desc: 生成地面积水描述
        Special: [短词用于插入，长句完整描述]
        0: none
        100: a road completely capped with water
        """
        precipitation_deposits = self.weather.precipitation_deposits
        if precipitation_deposits < 15:
            # desc = ["no water on the ground", "there is no water on the ground"]
            desc = ["", ""]
        else:
            if precipitation_deposits < 50:
                desc = ["some water on the ground", "there is some water on the ground"]
            else:
                desc = ["a road completely capped with water", "the road is completely capped with water"]
            self.add_path_reason(DEC, 'water')
        return desc

    def _wind_desc(self):
        """
        Desc: 生成风描述
        Special: [短词用于插入，长句完整描述]
        0: no wind
        100: strong wind
        """
        wind_intensity = self.weather.wind_intensity
        if wind_intensity < 20:
            # desc = ["no wind", "there is no wind"]
            desc = ["", ""]
        elif wind_intensity < 80:
            desc = ["some wind", "there is some wind"]
        else:
            desc = ["strong wind", "there is strong wind"]
        return desc

    def _sun_desc(self, morning=False, evening=False, randomly=False):
        """
        Desc: 生成太阳描述
        Special: [短词用于插入，长句完整描述]
        -90: midnight
        90: midday
        """
        sun_altitude_angle = self.weather.sun_altitude_angle

        if sun_altitude_angle < 0:
            desc = ["night", "it is night"]
        else:
            desc = ["daylight", "it is daylight"]

        # if randomly:
        #     if self.sun_arg is None:
        #         self.sun_arg = choice(['morning', 'evening'])
        #     if self.sun_arg == 'morning':
        #         morning = True
        #         evening = False
        #     else:
        #         evening = True
        #         morning = False
        #
        # if morning:
        #     if sun_altitude_angle < -1:
        #         desc = ["night", "it is night"]
        #         self.add_speed_reason(DEC, 'sun')
        #     elif sun_altitude_angle < 1:
        #         desc = ["sunrise", "the sun is rising"]
        #     elif sun_altitude_angle < 10:
        #         desc = ["morning", "it is morning"]
        #     else:
        #         # desc = ["midday", "it is midday"]
        #         desc = ["", ""]
        # elif evening:
        #     if sun_altitude_angle < -1:
        #         desc = ["night", "it is night"]
        #         self.add_speed_reason(DEC, 'sun')
        #     elif sun_altitude_angle < 1:
        #         desc = ["sunset", "the sun is setting"]
        #     elif sun_altitude_angle < 10:
        #         desc = ["nightfall", "it is nightfall"]
        #     else:
        #         # desc = ["midday", "it is midday"]
        #         desc = ["", ""]
        # else:
        #     raise ValueError("morning and evening cannot be both False")

        return desc

    def _fog_desc(self):
        """
        Desc: 生成雾描述
        Special: [短词用于插入，长句完整描述]
        0: no fog
        100: thick fog
        """
        fog_density = self.weather.fog_density
        if fog_density < 35:
            # desc = ["no fog", "there is no fog"]
            desc = ["", ""]
        else:
            if fog_density < 70:
                desc = ["some fog", "there is some fog"]
            else:
                desc = ["thick fog", "there is thick fog"]
            self.add_path_reason(DEC, 'fog')
        return desc
