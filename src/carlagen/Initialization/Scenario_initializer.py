# -*- coding: utf-8 -*-
# @Time    : 2024/3/7 下午5:51
# @Author  : Hcyang
# @File    : Scenario_initializer.py
# @Desc    : TODO:

import carla
import importlib

from .Basic_initializer import BasicInitializer
from Initialization.Standardimport.scenariomanager.carla_data_provider import CarlaDataProvider


class ScenarioInitializer(BasicInitializer):
    def __init__(self, scenario_name, client, world, traffic_manager, ego, scenario_config, tm_port):
        self.scenario_name = scenario_name
        self.client = client
        self.world = world
        self.traffic_manager = traffic_manager
        self.scenario_config = scenario_config
        self.ego = ego
        self.tm_port = tm_port

        self.scenario = None
        self.behavior = None

        self.necessary_config = CustomEdict({
            'weather': carla.WeatherParameters(),
            'friction': None,
            'trigger_points': [],
            'name': 'Test',
            'other_parameters': {},
            'route': False,
            'start_distance': 0,
            'random': True,
            'wp_route': None,
            'route_var_name': None,
        })

    def run(self, trigger_transform):
        self._carla_provider_init()

        self.necessary_config.trigger_points = [trigger_transform]

        scenario_module = importlib.import_module(f'Scenarios.Allscenarios.{self.scenario_name.lower()}')
        scenario_class = getattr(scenario_module, self.scenario_name)
        self.scenario = scenario_class(self.world, [self.ego], self.necessary_config, criteria_enable=False, uniad=False, interfuser=True)

        self.scenario.set_tf(self.traffic_manager)
        self.behavior = self.scenario._create_behavior()

    def _carla_provider_init(self):
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.register_actor(self.ego)
        CarlaDataProvider.set_ego(self.ego)
        CarlaDataProvider.set_trafficmanager(self.traffic_manager)
        CarlaDataProvider.set_traffic_manager_port(self.tm_port)

        # wp_route = self.scenario_config['wp_route']
        # CarlaDataProvider.set_ego_vehicle_route([(item[0].transform.location, item[1]) for item in wp_route])

    def get_data_tags(self):
        return self.scenario.get_data_tags()


class CustomEdict(dict):
    def __init__(self, d=None, **kwargs):
        super().__init__()
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(CustomEdict, self).__setattr__(name, value)
        super(CustomEdict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(CustomEdict, self).pop(k, d)

