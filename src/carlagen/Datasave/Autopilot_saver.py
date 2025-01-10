import pathlib
import time

import random

import cv2

# import ipdb
import skimage

from easydict import EasyDict as Edict
from copy import deepcopy
from Initialization.Standardimport.scenariomanager.carla_data_provider import (
    CarlaDataProvider,
)
from Datasave.Saveutils.functions import *
from .Basic_saver import BasicSaver

from Explainablelibrary.prompt import PromptGen

SEM_COLORS = {
    0: (0, 0, 0), # Unlabeled
    1: (128, 64, 128), # Roads
    2: (244, 35, 232), # SideWalks
    3: (70, 70, 70), # Building
    4: (102, 102, 156), # Wall
    5: (190, 153, 153), # Fence
    6: (153, 153, 153), # Pole
    7: (250, 170, 30), # TrafficLight
    8: (220, 220, 0), # TrafficSign
    9: (107, 142, 35), # Vegetation
    10: (152, 251, 152), # Terrain
    11: (70, 130, 180), # Sky
    12: (220, 20, 60), # Pedestrian
    13: (255, 0, 0), # Rider
    14: (0, 0, 142), # Car
    15: (0, 0, 70), # Truck
    16: (0, 60, 100), # Bus
    17: (0, 60, 100), # Train
    18: (0, 0, 230), # Motorcycle
    19: (119, 11, 32), # Bicycle
    20: (110, 190, 160), # Static
    21: (170, 120, 50), # Dynamic
    22: (55, 90, 80), # Other
    23: (45, 60, 150), # Water
    24: (0, 255, 0), # RoadLine
    25: (81, 0, 81), # Ground
    26: (150, 100, 100), # Bridge
    27: (230, 150, 140), # RailTrack
    28: (180, 165, 180) # GuardRail
}

SEM_TEXT = {
    0: "Unlabeled",
    1: "Roads",
    2: "SideWalks",
    3: "Building",
    4: "Wall",
    5: "Fence",
    6: "Pole",
    7: "TrafficLight",
    8: "TrafficSign",
    9: "Trees",
    10: "Terrain",
    11: "Sky",
    12: "Pedestrian",
    13: "Rider",
    14: "Car",
    15: "Truck",
    16: "Bus",
    17: "Train",
    18: "Motorcycle",
    19: "Bicycle",
    20: "Static",
    21: "Dynamic",
    22: "Other",
    23: "Water",
    24: "RoadLine",
    25: "Ground",
    26: "Bridge",
    27: "RailTrack",
    28: "GuardRail"
}




def interpolate_points(points, num_interp_points):

    (x1, y1) = points[-2]
    (x2, y2) = points[-1]
    
    x_vals = np.linspace(x1, x2, num_interp_points)
    y_vals = np.linspace(y1, y2, num_interp_points)
    
    interpolated_points = [[x, y] for x, y in zip(x_vals, y_vals)]
    
    return interpolated_points


def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False


def get_image_point(loc, K, w2c):

    if isinstance(loc, tuple):
        point = np.array([loc[0], loc[1], loc[2], 1])
    else:
        point = np.array([loc.x, loc.y, loc.z, 1])

    point_camera = np.dot(w2c, point)
    # (x, y ,z) -> (y, -z, x)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


class AutopilotSaver(BasicSaver):
    def __init__(self, scenario_config, sensors_config, saver_config, debug=False):  #aug=True 如果加相机扰动
        self._static_info_saved_flag = None
        self.scenario_config = scenario_config
        self.sensors_config = sensors_config
        self.saver_config = saver_config

        self.birdview = None

        self.debug = debug 

        super().__init__()
        self._setup()

        self.global_image_saver = None


    def _setup(self):
        self.step = -1
        self.cnt = -1
        self.initialized = False
        self._3d_bb_distance = 50
        self.prev_lidar = None
        self.save_path = None
        
        self.data_dict = {}

        save_path = self.scenario_config["save_path"]
        if save_path is not None:
            # town12_Autopilot_FrontBrake_20210603153000
            string = self.scenario_config["date"].strftime("%Y%m%d%H%M%S%f")[:17]
            string += (
                "_"
                + self.scenario_config["scenario_name"]
                + "_"
            )
            string += self.scenario_config["town"].lower()

            self.save_path = pathlib.Path(save_path) / string
            if os.path.isdir(str(self.save_path)):
                print(f"Warning: {str(self.save_path)} already exists, deleting")
                cmd = f"rm -rf {str(self.save_path)}"
                print(f"Executing: {cmd}")
                os.system(cmd)

            self.save_path.mkdir(parents=True, exist_ok=False)

        for sensor in self.sensors():
            if sensor["type"] == "sensor.camera.rgb":
                (self.save_path / sensor["id"]).mkdir()
                (self.save_path / sensor["id"] / "rgb").mkdir()
                (self.save_path / sensor["id"] / "3dbox").mkdir()
                (self.save_path / sensor["id"] / "hdmap").mkdir()
                (self.save_path / sensor["id"] / "debug").mkdir()
            if sensor["type"] == "sensor.camera.semantic_segmentation":
                (self.save_path / sensor["id"].replace("_S", "") / "sem").mkdir()


    def get_save_path(self):
        # 返回字符串
        return str(self.save_path)


    def _init(self):
        # Special: 第一次tick的时候调用
        self._world = CarlaDataProvider.get_world()
        self._map = self._world.get_map()
        self._ego_vehicle = CarlaDataProvider.get_ego()
        self.birdview_obs_manager = ObsManager()
        self.birdview_obs_manager.attach_ego_vehicle(self._ego_vehicle)

        self.PromptGen = PromptGen(self._world, self._ego_vehicle)

        # DWM 需要人行道
        crosswalks_list = self._map.get_crosswalks()
        cw_list = zip(*(iter(crosswalks_list),) * 5)
        cw_polygons = [list(i) for i in cw_list]
        count = len(crosswalks_list) % 2
        cw_polygons.append(crosswalks_list[-count:]) if count != 0 else cw_polygons
        self.crosswalks = cw_polygons

        self.initialized = True
        all_sensors = self.sensors()
        self.all_sensors_id = [sensors["id"] for sensors in all_sensors]
        self.global_image_saver = ImageSaver(self.save_path, all_sensors)
        print("initialized")

    def sensors(self):
        
        sensors = self.sensors_config["main"]

        return sensors


    def run_step(
        self,
        sensors,
        input_data,
        timestamp,
        *args,
        data_save_sign=True,
        frame_data=None,
        **kwargs,
    ):
        if not data_save_sign:
            return

        if not self.initialized:
            self._init()

        data = {}
        data["camera_infos"] = {}

        time_record = []

        self.step += 1
        frame = self.step  
        
        snapshot = self._world.get_snapshot()
        _timestamp = snapshot.timestamp.elapsed_seconds 
        data["timestamp"] = _timestamp
        frame = _timestamp

        ego_transform = self._ego_vehicle.get_transform()
        ego_location = ego_transform.location
        data["ego_pose"] = np.array(ego_transform.get_matrix()).tolist()

        world_2_ego = np.array(ego_transform.get_inverse_matrix())

        tick_data, seg_res, road_drivable_mask, seg_types, seg_colors = self.tick(
            input_data, timestamp
        )


        nearby_actors = self.get_surround_actors_info(object_type="vehicle")
        nearby_walkers = self.get_surround_actors_info(object_type="walker")
        nearby_actors += nearby_walkers

        # nearby_objects = self.get_surround_objects_info(object_type="Any")

        for idx, sensor in enumerate(sensors):

            if sensor.type_id == "sensor.camera.rgb":
                
                cam_transform = sensor.get_transform()
                world_2_camera = np.array(cam_transform.get_inverse_matrix())
                cam_pos = np.array(cam_transform.get_matrix())
                forward_vec = cam_transform.get_forward_vector()

                sensor_id = self.all_sensors_id[idx]
                semantic_id = self.all_sensors_id[idx+1]
                data["camera_infos"][sensor_id] = {}
                time = self.PromptGen.time
                weather = self.PromptGen.weather 
                env = self.PromptGen.env
                twe_prompt = f"{time}. {weather}. {env}"
                
                data["camera_infos"][sensor_id]['extrin'] = np.dot(cam_pos, world_2_ego).tolist()

                image_width = int(sensor.attributes['image_size_x'])
                image_height = int(sensor.attributes['image_size_y'])
                fov = float(sensor.attributes['fov']) 
                K = build_projection_matrix(image_width, image_height, fov)
                K_b = build_projection_matrix(image_width, image_height, fov, is_behind_camera=True)
                data["camera_infos"][sensor_id]['intrin'] = K.tolist()

                img = tick_data[sensor_id]
                img_sem = tick_data[semantic_id][1]
                object_prompt_list = []
                for cla, color in SEM_COLORS.items():
                    mask = img_sem[:, :, 2] == cla
                    true_count = np.sum(mask)
                    img_sem[mask] = color + (255,)

                    if true_count > 200 and cla in [1, 3, 4, 5, 7, 8, 9, 12, 13, 14, 15, 16, 18, 19]:
                        object_prompt_list.append(SEM_TEXT[cla])
                
                obj_prompt = ",".join(object_prompt_list)

                data["camera_infos"][sensor_id]["image_description"] = f"{twe_prompt} {obj_prompt}."
                sem_path = os.path.join(self.save_path, sensor_id, "sem", f"{frame:.5f}.jpg")
                cv2.imwrite(sem_path, img_sem)

                if self.saver_config["save_debug"]:
                    canvas_debug = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
                    canvas_debug[:, :, 0:3] = img
                    canvas_debug[:, :, 3] = 255
                else:
                    canvas_debug = None

                canvas_3dbox = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8) 
                canvas_3dbox[:, :, 3] = 255

                canvas_lane = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8) 
                canvas_lane[:, :, 3] = 255


                # 3dbox
                npc_dict = {}
                for npc in nearby_actors:

                    bb = npc.bounding_box
                    npc_type = npc.type_id
                    box_color = tuple(self.saver_config["3dbox"]["color"].get(npc_type, (255, 0, 0))) # 小汽车默认蓝色 
                    if "walker" in npc_type:
                        box_color = (0, 0, 255) # 行人默认红色

                    # TODO npc_dict 加入各种描述

                    # CALRA 0.9.14及更旧的版本中二轮车的Box有bug，
                    min_extent=0.75
                    if (bb.extent.x * bb.extent.y * bb.extent.z == 0):
                        bb.location = carla.Location(0, 0, max(bb.extent.z, min_extent))
                        bb.extent.x = 0.8
                        bb.extent.y = 0.3
                        bb.extent.z = 0.75

                    npc_transform = npc.get_transform()
                    npc_location = npc_transform.location
                    ray = npc_location - ego_location

                    if forward_vec.dot(ray) > 0:
                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                        verts = bb.get_world_vertices(npc.get_transform())

                        for edge in self.saver_config["3dbox"]["edges"]:
                            p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                            p2 = get_image_point(verts[edge[1]], K, world_2_camera)

                            p1_in_canvas = point_in_canvas(p1, image_height, image_width)
                            p2_in_canvas = point_in_canvas(p2, image_height, image_width) 
                        
                            if not p1_in_canvas and not p2_in_canvas:
                                continue

                            ray0 = verts[edge[0]] - cam_transform.location
                            ray1 = verts[edge[1]] - cam_transform.location

                            if not (forward_vec.dot(ray0) > 0):
                                p1 = get_image_point(verts[edge[0]], K_b, world_2_camera)
                            if not (forward_vec.dot(ray1) > 0):
                                p2 = get_image_point(verts[edge[1]], K_b, world_2_camera)

                            line_thickness = self.saver_config["3dbox"]["thickness"]

                            if canvas_3dbox is not None:
                                cv2.line(canvas_3dbox, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), box_color+(255,), line_thickness)
                            if canvas_debug is not None:
                                cv2.line(canvas_debug, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), box_color+(255,), line_thickness)
                
                # lane
                for si, seg_line in enumerate(seg_res):
                    lane_color = tuple(self.saver_config["lane"]["color"].get(seg_types[si], [0, 0, 255]))
                    
                    line = []
                    for i in range(1, len(seg_line)):

                        p2 = seg_line[i]
                        p1 = seg_line[i-1]

                        if np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) > 1:
                            new_p_list = interpolate_points([p1, p2], num_interp_points=20)
                            line += new_p_list
                        else:
                            line += [p1, p2]

                    for i in range(1, len(line)):
                        cur_p = line[i]
                        pre_p = line[i-1]
                        
                        cur_p = carla.Location(x=cur_p[0] + ego_location.x, y= ego_location.y - cur_p[1], z= ego_location.z)
                        pre_p = carla.Location(x=pre_p[0] + ego_location.x, y= ego_location.y - pre_p[1], z= ego_location.z)

                        ray_cur = cur_p - ego_location
                        ray_pre = pre_p - ego_location

                        dis_cur = ray_cur.length()
                        dis_pre = ray_pre.length()
                        
                        if forward_vec.dot(ray_cur) >= 0 and forward_vec.dot(ray_pre) >= 0:
                            p1 = get_image_point(cur_p, K, world_2_camera)
                            p2 = get_image_point(pre_p, K, world_2_camera)

                            p1_in_canvas = point_in_canvas(p1, image_height, image_width)
                            p2_in_canvas = point_in_canvas(p2, image_height, image_width) 
                        
                            if (p1_in_canvas) or (p2_in_canvas):
                                
                                ray0 = cur_p - cam_transform.location
                                ray1 = pre_p - cam_transform.location

                                if not (forward_vec.dot(ray0) > 0):
                                    p1 = get_image_point(cur_p, K_b, world_2_camera)
                                if not (forward_vec.dot(ray1) > 0):
                                    p2 = get_image_point(pre_p, K_b, world_2_camera)
                                
                                line_thickness = self.saver_config["lane"]["thickness"]

                                if canvas_lane is not None:
                                    cv2.line(canvas_lane, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), lane_color+(255,), line_thickness)
                                if canvas_debug is not None:
                                    cv2.line(canvas_debug, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), lane_color+(255,), line_thickness)

                
                # crosswalks
                for cw in self.crosswalks:
                    cw_color = tuple(self.saver_config["crosswalks"]["color"])

                    p_c = carla.Location(
                        np.mean([p.x for p in cw[0:4]]),
                        np.mean([p.y for p in cw[0:4]]),
                        np.mean([p.z for p in cw[0:4]]),
                    )

                    dist = p_c.distance(ego_location)
                    ray = p_c - ego_location

                    if forward_vec.dot(ray) > 0:
                        if dist <= self.saver_config["crosswalks"]["detect_distance"]:
                            for cw_edge in self.saver_config["crosswalks"]["edges"]:
                                p1 = get_image_point(cw[cw_edge[0]], K, world_2_camera)
                                p2 = get_image_point(cw[cw_edge[1]], K, world_2_camera)

                                p1_in_canvas = point_in_canvas(p1, image_height, image_width)
                                p2_in_canvas = point_in_canvas(p2, image_height, image_width) 
                            
                                if not p1_in_canvas and not p2_in_canvas:
                                    continue

                                ray0 = cw[cw_edge[0]] - cam_transform.location
                                ray1 = cw[cw_edge[1]] - cam_transform.location

                                if not (forward_vec.dot(ray0) > 0):
                                    p1 = get_image_point(cw[cw_edge[0]], K_b, world_2_camera)
                                if not (forward_vec.dot(ray1) > 0):
                                    p2 = get_image_point(cw[cw_edge[1]], K_b, world_2_camera)
                                
                                line_thickness = self.saver_config["crosswalks"]["thickness"]

                                if canvas_lane is not None:
                                    cv2.line(canvas_lane, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), cw_color+(255,), line_thickness)
                                if canvas_debug is not None:
                                    cv2.line(canvas_debug, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), cw_color+(255,), line_thickness) 

                data["camera_infos"][sensor_id]["rgb"] = os.path.join(self.save_path, sensor_id, "rgb", f"{frame:.5f}.jpg")
                _3dbox_path = os.path.join(self.save_path, sensor_id, "3dbox", f"{frame:.5f}.jpg")
                cv2.imwrite(_3dbox_path, canvas_3dbox)

                data["camera_infos"][sensor_id]["3dbox"] = _3dbox_path
                hdmap_path = os.path.join(self.save_path, sensor_id, "hdmap", f"{frame:.5f}.jpg")
                data["camera_infos"][sensor_id]["hdmap"] = hdmap_path
                cv2.imwrite(hdmap_path, canvas_lane)

                if canvas_debug is not None:
                    debug_save_path = os.path.join(self.save_path, sensor_id, "debug", f"{frame:.5f}.jpg")
                    cv2.imwrite(debug_save_path, canvas_debug)

        with open(os.path.join(self.save_path, "data_notext.json"), 'a') as f:
            json.dump(data, f)
            f.write('\n')

        # rgb
        self.global_image_saver.save_rgb_images(tick_data, frame)


    def tick(self, input_data, timestamp):
        # xindan version
        seg_res, road_drivable_mask, seg_types, seg_colors = (
            self.birdview_obs_manager.get_seg_mask(self.step)
        )
       
        rgb_dict = {}
        all_sensors = self.sensors()
        for sensor in all_sensors:
            if sensor["type"] == "sensor.camera.rgb":
                rgb_dict[sensor["id"]] = cv2.cvtColor(
                    input_data[sensor["id"]][1][:, :, :3], cv2.COLOR_BGR2RGB
                )
            if sensor["type"] == "sensor.camera.semantic_segmentation":
                rgb_dict[sensor["id"]] = input_data[sensor["id"]]

        return rgb_dict, seg_res, road_drivable_mask, seg_types, seg_colors

    def get_surround_objects_info(self, surround_dist=100, object_types="Any"):
        ego = self._ego_vehicle
        objects = []
        for type in object_types:
            objects.extend(self._world.get_environment_objects(type))
        nearby_objects = []

        for object in objects:
            cur_dist = object.get_transform().location.distance(
                self._ego_vehicle.get_location()
            )
            if cur_dist < surround_dist:
                nearby_objects.append(object)

        return nearby_objects

    def get_surround_actors_info(self, surround_dist=40, object_type="vehicle"):

        ego = self._ego_vehicle
        actors = self._world.get_actors().filter(f"*{object_type}*")
        nearby_actors = []
        nearby_actors_dist = []
        for actor in actors:
            if actor == ego:
                print(f"WENYANG DEBUG: ego is not counted as surround actor")
                continue
            cur_dist = actor.get_transform().location.distance(
                self._ego_vehicle.get_location()
            )
            if cur_dist < surround_dist:
                nearby_actors.append(actor)
                nearby_actors_dist.append(cur_dist)

        if len(nearby_actors_dist) > 1:
            sort_id = np.argsort(nearby_actors_dist)
            nearby_actors = [
                nearby_actors[_s_id] for _s_id in sort_id
            ]  
        return nearby_actors


    def get_lane_info(self):
        current_wp = self._map.get_waypoint(self._ego_vehicle.get_location())
        lane_info = {
            "current": parse_lane_info_by_waypoint(current_wp),
            "left": [],
            "right": [],
        }
        # 判断当前是几车道
        next_lane = current_wp
        lane_ids = [current_wp.lane_id]
        for i in range(10):
            next_lane = next_lane.get_right_lane()
            if next_lane is None:
                break
            else:
                if next_lane.lane_id in lane_ids:
                    break
                else:
                    lane_ids.append(next_lane.lane_id)
                    lane_info["right"].append(parse_lane_info_by_waypoint(next_lane))

        next_lane = current_wp
        lane_ids = [current_wp.lane_id]
        lane_sign = 0
        for i in range(10):
            if lane_sign == 0:
                next_lane = next_lane.get_left_lane()
            else:
                next_lane = next_lane.get_right_lane()

            if next_lane is None:
                break
            else:
                if next_lane.lane_id in lane_ids:
                    if lane_sign == 0:
                        next_lane = next_lane.get_left_lane()
                    else:
                        next_lane = next_lane.get_right_lane()
                    lane_sign = 1
                    continue
                else:
                    lane_ids.append(next_lane.lane_id)
                    lane_info["left"].append(parse_lane_info_by_waypoint(next_lane))
        return lane_info


    def _get_nearby_actors(self, object_type):
        if object_type == "traffic_sign":
            actors = [
                item
                for item in self._world.get_actors().filter("*traffic*")
                if "traffic_light" not in item.type_id
            ]
        else:
            # ipdb.set_trace(context=10) # CHUANYANG
            actors = self._world.get_actors().filter(f"*{object_type}*")

        ego_loc = self._ego_vehicle.get_location()

        nearby_actors = []
        for actor in actors:
            actor_loc = actor.get_location()
            if actor_loc.distance(ego_loc) <= 50:
                nearby_actors.append(actor)

            # 储存数据的时候不需要判断距离（所有的actor都存）
            # actor_coord_point = [actor.get_location().x, actor.get_location().y]
            # actor2world = actor.get_transform().get_matrix()
            # actor_world_coord_point = np.dot(actor2world, np.array([actor_coord_point[0], actor_coord_point[1], 0, 1])).tolist()[:-1]
            #
            # ego_coord_point = [self._ego_vehicle.get_location().x, self._ego_vehicle.get_location().y]
            # ego2world = self._ego_vehicle.get_transform().get_matrix()
            # ego_world_coord_point = np.dot(ego2world, np.array([ego_coord_point[0], ego_coord_point[1], 0, 1])).tolist()[:-1]
            #
            # if np.sqrt((actor_world_coord_point[0]-ego_world_coord_point[0])**2+(actor_world_coord_point[1]-ego_world_coord_point[1])**2) < 50:
            # 	nearby_actors.append(actor)

        return nearby_actors


