import argparse
import carla
import dwm.common
import json
import random
import time


def create_parser():
    parser = argparse.ArgumentParser(
        description="The tool to setup the Carla simulation.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The path of the simulation config about the environment, ego "
        "vehicle, and scenario.")
    parser.add_argument(
        "--host", default="127.0.0.1", type=str,
        help="The host address of the Carla simulator.")
    parser.add_argument(
        "-p", "--port", default=2000, type=int,
        help="The port of the Carla simulator.")
    parser.add_argument(
        "-tp", "--traffic-port", default=8000, type=int,
        help="The port of the Traffic manager.")
    parser.add_argument(
        "--client-timeout", default=10.0, type=float,
        help="The timeout of the Carla client.")
    parser.add_argument(
        "--step-sleep", default=0.0, type=float,
        help="The time to sleep for each step.")
    return parser


def make_actor(
    world: carla.World, blueprint_library: carla.BlueprintLibrary,
    spawn_points: list, actor_config: dict, random_state: random.Random,
    attach_to=None
):
    # prepare the blueprint
    if "pattern" in actor_config:
        bp_list = blueprint_library.filter(actor_config["pattern"])
        bp = (
            bp_list[actor_config["matched_index"]]
            if "matched_index" in actor_config
            else random_state.choice(bp_list)
        )
    else:
        bp = blueprint_library.find(actor_config["id"])

    for k, v in actor_config.get("attributes", {}).items():
        bp.set_attribute(k, v)

    # prepare the spawn location for vehicles, pedestrians, cameras
    if "spawn_index" in actor_config:
        spawn_transform = spawn_points[
            actor_config["spawn_index"] % len(spawn_points)
        ]
    elif "spawn_from_navigation" in actor_config:
        location = world.get_random_location_from_navigation()
        spawn_transform = carla.Transform(location, carla.Rotation(0, 0, 0))
    else:
        spawn_transform = actor_config["spawn_transform"]
        spawn_transform = carla.Transform(
            carla.Location(*spawn_transform.get("location", [0, 0, 0])),
            carla.Rotation(*spawn_transform.get("rotation", [0, 0, 0])))

    # instantiate the actor, set attributes and apply custom setup functions
    actor = world.try_spawn_actor(bp, spawn_transform, attach_to)

    if actor is not None:
        if actor.attributes.get("role_name") == "autopilot":
            actor.set_autopilot(True)

    if actor is None:
        print("Warning: failed to spawn {}".format(bp.id))
        return None, None, None

    if actor_config.get("report_actor_id", False):
        attributes = actor.attributes
        report_text = "{}{}: {}".format(
            actor_config["id"],
            " ({})".format(attributes["role_name"])
            if "role_name" in attributes else "", actor.id)
        print(report_text)

    if actor_config.get("report_actor_attributes", False):
        print("{}: {}".format(actor.type_id, actor.attributes))

    if "state_machine" in actor_config:
        _class = dwm.common.get_class(actor_config["state_machine"])
        state_machine = _class(
            actor, **actor_config.get("state_machine_args", {}))
    else:
        state_machine = None

    children = [
        make_actor(
            world, blueprint_library, spawn_points, i, rs, actor)
        for i in actor_config["child_configs"]
    ] if "child_configs" in actor_config else None

    return actor, state_machine, children


def update_actor_state(actors: list):
    for _, state_machine, children in actors:
        if state_machine is not None:
            state_machine.update()

        if children is not None:
            update_actor_state(children)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    rs = random.Random(config.get("seed", None))
    client = carla.Client(args.host, args.port, 1)
    client.set_timeout(args.client_timeout)

    # This script just use the activated map. To config the map, please use the
    # {CarlaRoot}/PythonAPI/util/config.py
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(args.traffic_port)

    if config.get("master", False):
        traffic_manager.set_synchronous_mode(True)

    if "world_settings" in config:
        settings = world.get_settings()
        for k, v in config["world_settings"].items():
            setattr(settings, k, v)

        world.apply_settings(settings)

    if "traffic_manager_settings" in config:
        for k, v in config["traffic_manager_settings"].items():
            getattr(traffic_manager, k)(v)

    actors = [
        make_actor(
            world, world.get_blueprint_library(),
            world.get_map().get_spawn_points(), i, rs)
        for i in config["actor_configs"]
    ]

    step = 0
    total_steps = config.get("total_steps", -1)
    while total_steps == -1 or step < total_steps:
        if args.step_sleep > 0.0:
            time.sleep(args.step_sleep)

        if config.get("master", False):
            world.tick()
        else:
            world.wait_for_tick()

        update_actor_state(actors)
        step += 1
