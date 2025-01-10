from easydict import EasyDict as Edict
import carla
import pickle
import os
#import ipdb

def parse_bounding_box_info(bbox, transform=carla.Transform()):
    vertices = bbox.get_world_vertices(transform)
    world_bboxes = []
    for v in vertices:
        world_bboxes.append(parse_location_info(v))
    data = Edict({
        'vertices': world_bboxes,
        'extent': parse_location_info(bbox.extent),
        'location': parse_location_info(bbox.location),
        'rotation': parse_rotation_info(bbox.rotation),
    })
    return data


def parse_location_info(loc):
    data = Edict({
        'x': loc.x,
        'y': loc.y,
        'z': loc.z,
    })
    return data


def parse_rotation_info(rot):
    data = Edict({
        'yaw': rot.yaw,
        'pitch': rot.pitch,
        'roll': rot.roll,
    })
    return data


def save_static(world, town):
    save_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(save_path, 'Files', 'static_bbs')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    filename = os.path.join(save_path, f'{town.lower()}.pkl')
    if os.path.exists(filename):
        return

    level_bbs = Edict({'vehicles_car': [parse_bounding_box_info(item) for item in
                                        world.get_level_bbs(carla.CityObjectLabel.Car)],
                       'vehicles_bus': [parse_bounding_box_info(item) for item in
                                        world.get_level_bbs(carla.CityObjectLabel.Bus)],
                       'vehicles_truck': [parse_bounding_box_info(item) for item in
                                          world.get_level_bbs(carla.CityObjectLabel.Truck)],
                       'vehicles_motorcycle': [parse_bounding_box_info(item) for item in
                                               world.get_level_bbs(carla.CityObjectLabel.Motorcycle)],
                       'vehicles_bicycle': [parse_bounding_box_info(item) for item in
                                            world.get_level_bbs(carla.CityObjectLabel.Bicycle)],
                       'vehicles_train': [parse_bounding_box_info(item) for item in
                                          world.get_level_bbs(carla.CityObjectLabel.Train)],
                       'pedestrians': [parse_bounding_box_info(item) for item in
                                       world.get_level_bbs(carla.CityObjectLabel.Pedestrians)],
                       'rider': [parse_bounding_box_info(item) for item in
                                 world.get_level_bbs(carla.CityObjectLabel.Rider)],
                       })

    # find_inside(world, level_bbs)
    # exit()
    # 只有town10hd里有建筑物内静态车辆 vehicles_car index：5，6，13，18，23，26，27
    if town.lower() == 'town10hd':
        filted=[]
        for i in range(len(level_bbs['vehicles_car'])):
            if i in [5,6,13,18,23,26,27]:
                continue
            filted.append(level_bbs['vehicles_car'][i])
        level_bbs['vehicles_car'] = filted

    with open(filename, 'wb') as f:
        pickle.dump(level_bbs, f)


def find_inside(world, bbs):
    debug = world.debug
    spectator = world.get_spectator()
    # 设置为俯视
    spectator.set_transform(carla.Transform(carla.Location(x=0, y=0, z=350), carla.Rotation(pitch=-90)))

    for key in bbs.keys():
        print(key)
        for i in range(len(bbs[key])):
            bb = bbs[key][i]
            loc = carla.Location(x=bb['location'].x,
                                 y=bb['location'].y,
                                 z=bb['location'].z)
            print(bb['location'])
            debug.draw_point(loc + carla.Location(z=0.3), size=0.1,
                             color=carla.Color(0, 255, 0),life_time=999)

            debug.draw_point(loc + carla.Location(z=3), size=0.1,
                             color=carla.Color(255, 0, 0),life_time=999)

            debug.draw_point(loc + carla.Location(z=15), size=0.1,
                             color=carla.Color(255, 180, 255),life_time=999)


            debug.draw_point(loc + carla.Location(z=30), size=0.1,
                             color=carla.Color(0, 0, 255),life_time=999)
            # ipdb.set_trace(context=1)
