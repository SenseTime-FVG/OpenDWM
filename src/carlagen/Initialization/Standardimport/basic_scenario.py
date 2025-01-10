#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provide BasicScenario, the basic class of all the scenarios.
"""

from __future__ import print_function

import time
import operator
import py_trees

import carla
import math
from random import choice

from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_trigger_conditions import (WaitForBlackboardVariable, InTimeToArrivalToLocation)
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_behaviors import WaitForever
from Initialization.Standardimport.scenariomanager.carla_data_provider import CarlaDataProvider
from Initialization.Standardimport.scenariomanager.timer import TimeOut
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_behaviors import UpdateAllActorControls
from Initialization.Standardimport.scenariomanager.scenarioatomics.atomic_criteria import Criterion
from Initialization.Standardimport.agents.navigation.global_route_planner import GlobalRoutePlanner


class BasicScenario(object):

    """
    Base class for user-defined scenario
    """

    def __init__(self, name, ego_vehicles, config, world, randomize=False,
                 debug_mode=False, terminate_on_failure=False, criteria_enable=False, uniad=False, interfuser=False, assign_distribution=None):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self.name = name
        self.ego_vehicles = ego_vehicles
        self.other_actors = []
        self.parking_slots = []
        self.config = config
        self.world = world
        self.debug_mode = debug_mode
        self.terminate_on_failure = terminate_on_failure
        self.criteria_enable = criteria_enable
        self.traffic_manager = None
        self.assign_distribution = assign_distribution

        self.default_settings = {}
        self._update_scenario_settings()

        self.route_mode = bool(config.route)
        self.behavior_tree = None
        self.criteria_tree = None
        self.uniad = uniad
        self.interfuser = interfuser
        self.route_planner = GlobalRoutePlanner(self.world.get_map(), 1.0)

        # If no timeout was provided, set it to 60 seconds
        if not hasattr(self, 'timeout'):
            self.timeout = 60 
        if debug_mode:
            self.debug_time = time.time()
        #     py_trees.logging.level = py_trees.logging.Level.DEBUG

        self._initialize_environment(world)
        self._initialize_actors(config)

        if CarlaDataProvider.is_sync_mode():
            world.tick()
        else:
            world.wait_for_tick()

        # Main scenario tree
        self.scenario_tree = py_trees.composites.Parallel(name, policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # Add a trigger and end condition to the behavior to ensure it is only activated when it is relevant
        self.behavior_tree = py_trees.composites.Sequence()

        trigger_behavior = self._setup_scenario_trigger(config)
        if trigger_behavior:
            self.behavior_tree.add_child(trigger_behavior)

        scenario_behavior = self._create_behavior()
        self.behavior_tree.add_child(scenario_behavior)
        self.behavior_tree.name = scenario_behavior.name

        end_behavior = self._setup_scenario_end(config)
        if end_behavior:
            self.behavior_tree.add_child(end_behavior)

        # Create the lights behavior
        lights = self._create_lights_behavior()
        if lights:
            self.scenario_tree.add_child(lights)

        # Create the weather behavior
        weather = self._create_weather_behavior()
        if weather:
            self.scenario_tree.add_child(weather)

        # And then add it to the main tree
        self.scenario_tree.add_child(self.behavior_tree)

        # Create the criteria tree (if needed)
        if self.criteria_enable:
            criteria = self._create_test_criteria()

            # All the work is done, thanks!
            if isinstance(criteria, py_trees.composites.Composite):
                self.criteria_tree = criteria
                self.debug_route_completion_criteria = None

            # Lazy mode, but its okay, we'll create the parallel behavior tree for you.
            elif isinstance(criteria, list):
                self.debug_route_completion_criteria = criteria[0]
                for criterion in criteria:
                    criterion.terminate_on_failure = terminate_on_failure

                self.criteria_tree = py_trees.composites.Parallel(name="Test Criteria",
                                                                  policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
                self.criteria_tree.add_children(criteria)
                self.criteria_tree.setup(timeout=1)

            else:
                raise ValueError("WARNING: Scenario {} couldn't be setup, make sure the criteria is either "
                                 "a list or a py_trees.composites.Composite".format(self.name))

            self.scenario_tree.add_child(self.criteria_tree)

        # Create the timeout behavior
        self.timeout_node = self._create_timeout_behavior()
        if self.timeout_node:
            self.scenario_tree.add_child(self.timeout_node)

        # Add other nodes
        self.scenario_tree.add_child(UpdateAllActorControls())

        self.scenario_tree.setup(timeout=1)

        # 为了避免重复设置速度（如果速度和上一次的一样，就不设置）仅在autopilot模式下有效
        self.current_ego_speed = -1
        self.current_ego_offset = 0
        self.current_ego_force_go = 0
        self.ego_is_autopilot = True

        # 为了避免重复设置他车速度（如果速度和上一次的一样，就不设置）仅在autopilot模式下有效
        self.other_actor_dict = {}

        self.route_int2str = {
            0: 'Right',
            1: 'Left',
            2: 'Straight',
        }

    def _update_scenario_settings(self):
        pass

    def get_data_tags(self):
        raise NotImplementedError

    def _check_tf(self):
        if self.traffic_manager is None:
            print(f'Please set traffic manager by scenario.set_tf(tf_instance) first!')
            return False
        return True

    def set_tf(self, tf_instance):
        self.traffic_manager = tf_instance

    def _tf_set_ego_speed(self, speed):
        if self._check_tf():
            if self.current_ego_speed != speed:
                self.current_ego_speed = speed
                self.traffic_manager.set_desired_speed(self.ego_vehicles[0], speed)

    def _tf_set_actor_speed(self, speed, actor):
        if actor not in self.other_actor_dict:
            self.other_actor_dict[actor] = {
                'cur_speed': -1,
            }
        if self.other_actor_dict[actor]['cur_speed'] != speed:
            self.other_actor_dict[actor]['cur_speed'] = speed
            self.traffic_manager.set_desired_speed(actor, speed)

    def _tf_set_ego_offset(self, offset):
        if self._check_tf():
            if self.current_ego_offset != offset:
                self.current_ego_offset = offset
                self.traffic_manager.vehicle_lane_offset(self.ego_vehicles[0], offset)

    def _tf_set_actor_offset(self, offset,actor):
        if self._check_tf():
            # if self.current_ego_offset != offset:
            #     self.current_ego_offset = offset
                self.traffic_manager.vehicle_lane_offset(actor, offset)

    def _tf_set_ego_force_go(self, perc):
        if self._check_tf():
            if self.current_ego_force_go != perc:
                self.current_ego_force_go = perc
                self.traffic_manager.ignore_vehicles_percentage(self.ego_vehicles[0], perc)
                self.traffic_manager.ignore_signs_percentage(self.ego_vehicles[0], perc)
                self.traffic_manager.ignore_lights_percentage(self.ego_vehicles[0], perc)
                self.traffic_manager.ignore_walkers_percentage(self.ego_vehicles[0], perc)

    def _tf_switch_ego_autopilot(self):
        if self._check_tf():
            if self.ego_is_autopilot:
                self.ego_is_autopilot = False
                self.ego_vehicles[0].set_autopilot(False)
            else:
                self.ego_is_autopilot = True
                self.ego_vehicles[0].set_autopilot(True)

    def _tf_set_ego_route(self, route_int):
        if self._check_tf():
            if isinstance(route_int[0], int):
                route = [self.route_int2str[item] for item in route_int]
            else:
                route = route_int
            self.traffic_manager.set_route(self.ego_vehicles[0], route)

    def _tf_set_actor_route(self, route,actor):
        if self._check_tf():
            self.traffic_manager.set_route(actor, route)

    def _tf_disable_ego_auto_lane_change(self):
        if self._check_tf():
            self.traffic_manager.auto_lane_change(self.ego_vehicles[0], False)

    def _set_ego_autopilot(self):
        self.ego_vehicles[0].set_autopilot(enabled=True, tm_port=CarlaDataProvider.get_traffic_manager_port())

    def _force_ego_lanechange_left(self):
        if self._check_tf():
            self.traffic_manager.force_lane_change(self.ego_vehicles[0], False)

    def _force_ego_lanechange_right(self):
        if self._check_tf():
            self.traffic_manager.force_lane_change(self.ego_vehicles[0], True)

    def _tf_set_ego_ignore_signs(self):
        if self._check_tf():
            self.traffic_manager.ignore_signs_percentage(self.ego_vehicles[0], 100)

    def _tf_set_ego_ignore_lights(self):
        if self._check_tf():
            self.traffic_manager.ignore_lights_percentage(self.ego_vehicles[0], 100)

    def tick_autopilot(self, *args, **kwargs):
        if self.uniad:
            assert self.interfuser is False
            data = self.uniad_tick_autopilot(*args, **kwargs)
            cur_wp = data['cur_wp']
            navi_locs = self.get_navi_route(cur_wp)
            return {'navi_locs': navi_locs, **data}
        elif self.interfuser:
            return self.interfuser_tick_autopilot(*args, **kwargs)
        else:
            raise NotImplementedError

    def interfuser_tick_autopilot(self, *args, **kwargs):
        raise NotImplementedError

    def uniad_tick_autopilot(self, *args, **kwargs):
        raise NotImplementedError

    def is_success(self):
        raise NotImplementedError("This function is re-implemented by all scenarios")

    def start_npc(self):
        raise NotImplementedError("This function is re-implemented by all scenarios")

    def get_navi_route(self, cur_wp):
        start_wp, dest_wp = self._get_navi_route(cur_wp)
        navi_route = self.route_planner.trace_route(start_wp.transform.location, dest_wp.transform.location)
        if self.debug_mode:
            cur_debug_time = time.time()
            if cur_debug_time - self.debug_time > 1.0:
                color = choice([carla.Color(255, 0, 0), carla.Color(0, 255, 0), carla.Color(0, 0, 255), carla.Color(255, 255, 0), carla.Color(0, 255, 255), carla.Color(255, 0, 255)])
                for navi_wp in navi_route:
                    debug = self.world.debug
                    debug.draw_point(navi_wp[0].transform.location + carla.Location(z=0.1), size=0.15, color=color, life_time=1.0)
                self.debug_time = cur_debug_time

        navi_route = [item[0].transform.location for item in navi_route]
        return navi_route

    def _get_navi_route(self, cur_wp):
        raise NotImplementedError("This function is re-implemented by all scenarios")

    def _initialize_environment(self, world):
        """
        Default initialization of weather and road friction.
        Override this method in child class to provide custom initialization.
        """

        # Set the appropriate weather conditions
        # world.set_weather(self.config.weather)

        # Set the appropriate road friction
        if self.config.friction is not None:
            friction_bp = world.get_blueprint_library().find('static.trigger.friction')
            extent = carla.Location(1000000.0, 1000000.0, 1000000.0)
            friction_bp.set_attribute('friction', str(self.config.friction))
            friction_bp.set_attribute('extent_x', str(extent.x))
            friction_bp.set_attribute('extent_y', str(extent.y))
            friction_bp.set_attribute('extent_z', str(extent.z))

            # Spawn Trigger Friction
            transform = carla.Transform()
            transform.location = carla.Location(-10000.0, -10000.0, 0.0)
            world.spawn_actor(friction_bp, transform)

    def _initialize_actors(self, config):
        """
        Default initialization of other actors.
        Override this method in child class to provide custom initialization.
        """
        if config.other_actors:
            new_actors = CarlaDataProvider.request_new_actors(config.other_actors)
            if not new_actors:
                raise Exception("Error: Unable to add actors")

            for new_actor in new_actors:
                self.other_actors.append(new_actor)

    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.

        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """
        if config.trigger_points and config.trigger_points[0]:
            start_location = config.trigger_points[0].location
        else:
            return None

        # Scenario is not part of a route, wait for the ego to move
        if not self.route_mode or config.route_var_name is None:
            return InTimeToArrivalToLocation(self.ego_vehicles[0], 2.0, start_location)

        # Scenario is part of a route.
        check_name = "WaitForBlackboardVariable: {}".format(config.route_var_name)
        return WaitForBlackboardVariable(config.route_var_name, True, False, name=check_name)

    def _setup_scenario_end(self, config):
        """
        This function adds and additional behavior to the scenario, which is triggered
        after it has ended. The Blackboard variable is set to False to indicate the scenario has ended.
        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """
        if not self.route_mode or config.route_var_name is None:
            return None

        # Scenario is part of a route.
        end_sequence = py_trees.composites.Sequence()
        name = "Reset Blackboard Variable: {} ".format(config.route_var_name)
        end_sequence.add_child(py_trees.blackboard.SetBlackboardVariable(name, config.route_var_name, False))
        end_sequence.add_child(WaitForever())  # scenario can't stop the route

        return end_sequence

    def _create_behavior(self):
        """
        Pure virtual function to setup user-defined scenario behavior
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios"
            "If this error becomes visible the class hierarchy is somehow broken")

    def _create_test_criteria(self):
        """
        Pure virtual function to setup user-defined evaluation criteria for the
        scenario
        """
        raise NotImplementedError(
            "This function is re-implemented by all scenarios"
            "If this error becomes visible the class hierarchy is somehow broken")

    def _create_weather_behavior(self):
        """
        Default empty initialization of the weather behavior,
        responsible of controlling the weather during the simulation.
        Override this method in child class to provide custom initialization.
        """
        pass

    def _create_lights_behavior(self):
        """
        Default empty initialization of the lights behavior,
        responsible of controlling the street lights during the simulation.
        Override this method in child class to provide custom initialization.
        """
        pass

    def _create_timeout_behavior(self):
        """
        Default initialization of the timeout behavior.
        Override this method in child class to provide custom initialization.
        """
        return TimeOut(self.timeout, name="TimeOut")  # Timeout node

    def change_control(self, control):  # pylint: disable=no-self-use
        """
        This is a function that changes the control based on the scenario determination
        :param control: a carla vehicle control
        :return: a control to be changed by the scenario.

        Note: This method should be overriden by the user-defined scenario behavior
        """
        return control

    def get_criteria(self):
        """
        Return the list of test criteria, including all the leaf nodes.
        Some criteria might have trigger conditions, which have to be filtered out.
        """
        criteria = []
        if not self.criteria_tree:
            return criteria

        criteria_nodes = self._extract_nodes_from_tree(self.criteria_tree)
        for criterion in criteria_nodes:
            if isinstance(criterion, Criterion):
                criteria.append(criterion)

        return criteria

    def _extract_nodes_from_tree(self, tree):  # pylint: disable=no-self-use
        """
        Returns the list of all nodes from the given tree
        """
        node_list = [tree]
        more_nodes_exist = True
        while more_nodes_exist:
            more_nodes_exist = False
            for node in node_list:
                if node.children:
                    node_list.remove(node)
                    more_nodes_exist = True
                    for child in node.children:
                        node_list.append(child)

        if len(node_list) == 1 and isinstance(node_list[0], py_trees.composites.Parallel):
            return []

        return node_list

    def terminate(self):
        """
        This function sets the status of all leaves in the scenario tree to INVALID
        """
        # Get list of all nodes in the tree
        node_list = self._extract_nodes_from_tree(self.scenario_tree)

        # Set status to INVALID
        for node in node_list:
            node.terminate(py_trees.common.Status.INVALID)

        # Cleanup all instantiated controllers
        actor_dict = {}
        try:
            check_actors = operator.attrgetter("ActorsWithController")
            actor_dict = check_actors(py_trees.blackboard.Blackboard())
        except AttributeError:
            pass
        for actor_id in actor_dict:
            actor_dict[actor_id].reset()
        py_trees.blackboard.Blackboard().set("ActorsWithController", {}, overwrite=True)

    def remove_all_actors(self):
        """
        Remove all actors
        """
        if not hasattr(self, 'other_actors'):
            return
        for i, _ in enumerate(self.other_actors):
            if self.other_actors[i] is not None:
                if CarlaDataProvider.actor_id_exists(self.other_actors[i].id):
                    CarlaDataProvider.remove_actor_by_id(self.other_actors[i].id)
                self.other_actors[i] = None
        self.other_actors = []

    def get_parking_slots(self):
        """
        Returns occupied parking slots.
        """
        return self.parking_slots

    def update_assign_distribution(self, target_dist):
        if isinstance(self.assign_distribution, dict):
            return deep_update(target_dist, self.assign_distribution)
        return target_dist

def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict):
            # 如果值是字典类型，则获取对应的字典，并递归更新
            source[key] = deep_update(source.get(key, {}), value)
        else:
            source[key] = value
    return source
