# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 下午5:51
# @Author  : Hcyang
# @File    : Scenario_initializer.py
# @Desc    : TODO:

import re
import carla
import random

# import ipdb

from .Basic_initializer import BasicInitializer
from Logger import *


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

class WeatherInitializer(BasicInitializer):
    def __init__(self, world, ego, weather_config):
        self.world = world
        self.ego = ego
        self.data_tags = {}
        self.weather_config = weather_config

    def run(self):
        if self.weather_config["type"] == "random":
            self.world.set_weather(getattr(carla.WeatherParameters), do_sample(self.weather_config["presets"]))
        else:
            self.world.set_weather(getattr(carla.WeatherParameters, self.weather_config["type"]))
        cur_weather = self.world.get_weather()
        info(f'设置天气为: {cur_weather}')

        sun_altitude_angle = cur_weather.sun_altitude_angle
        while sun_altitude_angle > 180 or sun_altitude_angle < -180:
            if sun_altitude_angle > 180:
                sun_altitude_angle -= 360
            elif sun_altitude_angle < -180:
                sun_altitude_angle += 360

        self.data_tags['sun'] = sun_altitude_angle
        self.data_tags['rain'] = cur_weather.precipitation
        self.data_tags['fog'] = cur_weather.fog_density
        self.data_tags['dust'] = cur_weather.dust_storm
        if sun_altitude_angle > 10:
            message(f'light off')
            self.ego.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position))
        else:
            message(f'light on')
            self.ego.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))

    def set_daylight(self):
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def get_data_tags(self):
        return self.data_tags


def do_sample(data_dict):
    return random.choices(list(data_dict.keys()), weights=list(data_dict.values()), k=1)[0]
