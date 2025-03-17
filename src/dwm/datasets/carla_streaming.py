import bisect
import carla
import collections
import dwm.datasets.common
import numpy as np
import math
from PIL import Image, ImageDraw
import time
import torch
import xml.etree.ElementTree as ET


class CarlaCallbackWithSensor:
    def __init__(self, sensor: carla.Actor, callback):
        self.sensor = sensor
        self.callback = callback

    def __call__(self, data: carla.SensorData):
        self.callback(self.sensor, data)


class StreamingDataAdapter:

    point_keys = ["x", "y", "z"]

    extrinsic_correction = [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]
    default_rear_vehicle_center = [
        [1, 0, 0, -1.5],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    lane_types_to_extract = [
        "curb", "solid", "broken", "solid solid", "broken solid",
        "solid broken", "broken broken", "crosswalk"
    ]
    default_3dbox_color_table = {
        "pedestrian": (255, 0, 0),
        "bicycle": (128, 255, 0),
        "motorcycle": (0, 255, 128),
        "bus": (128, 0, 255),
        "van": (0, 0, 255),
        "car": (0, 0, 255),
        "truck": (255, 255, 0)
    }
    default_hdmap_color_table = {
        "curb": (0, 0, 255),
        "solid": (0, 255, 0),
        "broken": (0, 255, 0),
        "solid solid": (0, 255, 0),
        "broken solid": (0, 255, 0),
        "solid broken": (0, 255, 0),
        "broken broken": (0, 255, 0),
        "crosswalk": (255, 0, 0)
    }
    default_3dbox_corner_template = [
        [-1, -1, -1, 1], [-1, -1, 1, 1],
        [-1, 1, -1, 1], [-1, 1, 1, 1],
        [1, -1, -1, 1], [1, -1, 1, 1],
        [1, 1, -1, 1], [1, 1, 1, 1]
    ]
    default_3dbox_edge_indices = [
        (0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (1, 5),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        (6, 3), (6, 5)
    ]
    prompt_cityscape_semantic_label = {
        1: "roads",
        3: "building",
        4: "wall",
        5: "fence",
        7: "traffic light",
        8: "traffic sign",
        9: "trees",
        12: "pedestrian",
        13: "rider",
        14: "car",
        15: "truck",
        16: "bus",
        17: "train",
        18: "motorcycle",
        19: "bicycle",
        26: "bridge"
    }

    @staticmethod
    def make_image_from_sensor_data(data: carla.SensorData):
        array = np.frombuffer(data.raw_data, dtype=np.uint8)\
            .reshape(data.height, data.width, 4)[..., 2::-1]
        return Image.fromarray(array, "RGB")

    @staticmethod
    def make_object_prompt_from_segm_sensor_data(
        data: carla.SensorData, min_pixel_ratio: float = 0.005
    ):
        min_pixel_count = data.height * data.width * min_pixel_ratio
        segm_labels = np.frombuffer(data.raw_data, dtype=np.uint8)\
            .reshape(data.height, data.width, 4)[..., 2].flatten().tolist()
        segm_label_count = collections.Counter(segm_labels)
        object_list = [
            StreamingDataAdapter.prompt_cityscape_semantic_label[k]
            for k, v in segm_label_count.items()
            if (
                v > min_pixel_count and
                k in StreamingDataAdapter.prompt_cityscape_semantic_label
            )
        ]

        return ", ".join(object_list)

    @staticmethod
    def make_camera_transforms(sensors: list, rear_vehicle_center: list):
        rh_from_lh = lh_from_rh = np.diag([1, -1, 1, 1])
        inv_ec = np.linalg.inv(
            np.array(StreamingDataAdapter.extrinsic_correction))
        inv_rvc = np.linalg.inv(np.array(rear_vehicle_center))

        camera_transform_list = []
        for i in sensors:
            lh_world_from_ego = np.array(i.parent.get_transform().get_matrix())
            lh_world_from_sensor = np.array(i.get_transform().get_matrix())
            lh_ego_from_sensor = np.linalg.solve(
                lh_world_from_ego, lh_world_from_sensor)

            rh_ego_from_sensor = \
                rh_from_lh @ lh_ego_from_sensor @ lh_from_rh
            camera_transform = inv_rvc @ rh_ego_from_sensor @ inv_ec
            camera_transform_list.append(camera_transform)

        return np.stack(camera_transform_list)

    @staticmethod
    def make_camera_intrinsics(
        width: np.array, height: np.array, fov_x: np.array
    ):
        focal = width / (2.0 * np.tan(0.5 * np.deg2rad(fov_x)))
        ones = np.ones(width.shape)
        zeros = np.zeros(width.shape)
        return np.stack([
            np.stack([focal, zeros, 0.5 * width], -1),
            np.stack([zeros, focal, 0.5 * height], -1),
            np.stack([zeros, zeros, ones], -1)
        ], -2)

    @staticmethod
    def make_ego_transform(ego: carla.Actor, rear_vehicle_center: list):
        rvc = np.array(rear_vehicle_center)
        rh_from_lh = lh_from_rh = np.diag([1, -1, 1, 1])
        lh_world_from_ego = np.array(ego.get_transform().get_matrix())
        return rh_from_lh @ lh_world_from_ego @ lh_from_rh @ rvc

    @staticmethod
    def get_3dbox_image(
        pvb: list, sensor: carla.Sensor, _3dbox_image_settings: dict
    ):
        # options
        max_distance = _3dbox_image_settings.get("max_distance", 80.0)
        pen_width = _3dbox_image_settings.get("pen_width", 3)
        color_table = _3dbox_image_settings.get(
            "color_table", StreamingDataAdapter.default_3dbox_color_table)
        corner_templates = _3dbox_image_settings.get(
            "corner_templates",
            StreamingDataAdapter.default_3dbox_corner_template)
        edge_indices = _3dbox_image_settings.get(
            "edge_indices", StreamingDataAdapter.default_3dbox_edge_indices)

        # get the transform from the world space to the image space
        image_size = (
            int(sensor.attributes["image_size_x"]),
            int(sensor.attributes["image_size_y"])
        )
        intrinsic = np.eye(4)
        focal = float(sensor.attributes["image_size_x"]) /\
            (2.0 * np.tan(0.5 * np.deg2rad(float(sensor.attributes["fov"]))))
        intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
            [focal, focal], [0.5 * image_size[0], 0.5 * image_size[1]])

        rh_from_lh = lh_from_rh = np.diag([1, -1, 1, 1])
        ec = np.array(StreamingDataAdapter.extrinsic_correction)

        lh_sensor_from_lh_world = np.array(
            sensor.get_transform().get_inverse_matrix())
        rh_sensor_from_lh_world = rh_from_lh @ lh_sensor_from_lh_world
        image_from_lh_world = intrinsic @ ec @ rh_sensor_from_lh_world

        # draw annotations to the image
        def list_annotation():
            sensor_location = sensor.get_location()
            for i in pvb:
                if sensor_location.distance(i.get_location()) <= max_distance:
                    yield i

        def get_world_transform(i):
            lh_world_from_lh_model = np.array(i.get_transform().get_matrix())
            lh_model = np.diag(
                [
                    getattr(i.bounding_box.extent, j)
                    for j in StreamingDataAdapter.point_keys
                ] + [1])
            lh_model[:3, 3] = [
                getattr(i.bounding_box.location, j)
                for j in StreamingDataAdapter.point_keys
            ]
            return lh_world_from_lh_model @ lh_model @ lh_from_rh

        def get_annotation_label(i):
            if i.type_id.startswith("vehicle"):
                return i.attributes["base_type"].lower()
            elif i.type_id.startswith("walker.pedestrian"):
                return "pedestrian"
            else:
                return ""

        image = Image.new("RGB", image_size)
        draw = ImageDraw.Draw(image)
        dwm.datasets.common.draw_3dbox_image(
            draw, image_from_lh_world, list_annotation, get_world_transform,
            get_annotation_label, pen_width, color_table,
            corner_templates, edge_indices)

        return image

    @staticmethod
    def xodr_get_offset(item: ET.Element, ds: float):
        param_keys = ["a", "b", "c", "d"]
        a, b, c, d = [float(item.attrib[i]) for i in param_keys]
        return a + b * ds + c * (ds ** 2) + d * (ds ** 3)

    @staticmethod
    def xodr_get_lane_t(
        lane_offset_list: list, lane_offset_s: list, lane_dict: dict,
        lane_id: int, s_begin: float, s_offset: float
    ):
        s = s_begin + s_offset
        lane_offset = lane_offset_list[
            bisect.bisect_right(lane_offset_s, s) - 1
        ]
        t = StreamingDataAdapter.xodr_get_offset(
            lane_offset, s - float(lane_offset.attrib["s"]))
        direction = 1 if lane_id >= 0 else -1
        for i in range(0, lane_id + direction, direction):
            lane = lane_dict[i]
            if len(lane["width_s"]) > 0:
                width = lane["width"][
                    bisect.bisect_right(lane["width_s"], s_offset) - 1
                ]
                t += direction * StreamingDataAdapter.xodr_get_offset(
                    width, s_offset - float(width.attrib["sOffset"]))

        return t

    @staticmethod
    def xodr_transform_from_road_to_world(
        s: float, t: float, geometry_list: list, geometry_s: list,
        elevation_list: list, elevation_s: list
    ):
        param_keys = ["x", "y", "hdg"]
        geometry = geometry_list[
            bisect.bisect_right(geometry_s, s) - 1
        ]
        s_g = s - float(geometry.attrib["s"])
        x0, y0, hdg = [float(geometry.attrib[i]) for i in param_keys]
        sin_hdg, cos_hdg = math.sin(hdg), math.cos(hdg)
        r = (
            cos_hdg, -sin_hdg,
            sin_hdg, cos_hdg
        )

        child = geometry.find("*")
        if child.tag == "line":
            x = x0 + s_g * r[0] + t * r[1]
            y = y0 + s_g * r[2] + t * r[3]
        elif child.tag == "arc":
            curv = float(child.attrib["curvature"])
            radius = 1 / curv
            theta = s_g * curv
            s_ = (radius - t) * math.sin(theta)
            t_ = radius - (radius - t) * math.cos(theta)
            x = x0 + s_ * r[0] + t_ * r[1]
            y = y0 + s_ * r[2] + t_ * r[3]
        else:
            raise Exception("Unsupported geometry {}".format(child.tag))

        if len(elevation_list) > 0:
            elevation = elevation_list[
                bisect.bisect_right(elevation_s, s) - 1
            ]
            z = StreamingDataAdapter.xodr_get_offset(
                elevation, s - float(elevation.attrib["s"]))
        else:
            z = 0

        return x, y, z

    @staticmethod
    def extract_object_points(
        obj: ET.Element, result: dict, geometry_list: list, geometry_s: list,
        elevation_list: list, elevation_s: list
    ):
        obj_id = obj.attrib["id"]
        obj_type = obj.attrib["type"]
        obj_param_keys = ["s", "t", "zOffset", "hdg"]
        corner_param_keys = ["u", "v", "z"]
        s, t, z_offset, hdg = [float(obj.attrib[i]) for i in obj_param_keys]
        sin_hdg, cos_hdg = math.sin(hdg), math.cos(hdg)
        r = (
            cos_hdg, -sin_hdg,
            sin_hdg, cos_hdg
        )

        for i in obj.findall("outline/cornerLocal"):
            u, v, z = [float(i.attrib[j]) for j in corner_param_keys]
            s1 = s + u * r[0] + v * r[1]
            t1 = t + u * r[2] + v * r[3]
            x, y, z_local = StreamingDataAdapter\
                .xodr_transform_from_road_to_world(
                    s1, t1, geometry_list, geometry_s, elevation_list,
                    elevation_s)
            p = (x, y, z + z_offset + z_local)
            if obj_id not in result[obj_type]:
                result[obj_type][obj_id] = [p]
            else:
                result[obj_type][obj_id].append(p)

    @staticmethod
    def extract_lines(
        roads: list, result: dict, interval: float = 1.0
    ):
        for i in roads:
            geometry_list = i.findall("planView/geometry")
            elevation_list = i.findall("elevationProfile/elevation")
            geometry_s = [float(j.attrib["s"]) for j in geometry_list]
            elevation_s = [float(j.attrib["s"]) for j in elevation_list]

            lanes = i.find("lanes")
            lane_offset_list = lanes.findall("laneOffset")
            lane_section = lanes.findall("laneSection")
            lane_offset_s = [float(j.attrib["s"]) for j in lane_offset_list]
            for j_id, j in enumerate(lane_section):
                lane_dict = {
                    int(k.attrib["id"]): {
                        "lane": k,
                        "width": k.findall("width"),
                        "width_s": [
                            float(j.attrib["sOffset"])
                            for j in k.findall("width")
                        ]
                    }
                    for k in j.findall("*/lane")
                }

                is_last = j_id + 1 == len(lane_section)
                s_begin = float(j.attrib["s"])
                s_end = float(
                    lane_section[j_id + 1].attrib["s"]
                    if not is_last
                    else i.attrib["length"]
                )
                s_offset_length = s_end - s_begin
                for k_id, k in lane_dict.items():
                    roadmarks = k["lane"].findall("roadMark")
                    for l_id, l in enumerate(roadmarks):
                        if l.attrib["type"] not in result:
                            continue

                        s_offset = float(l.attrib["sOffset"])
                        s_offset_end = (
                            float(roadmarks[l_id + 1].attrib["sOffset"])
                            if l_id + 1 < len(roadmarks)
                            else s_offset_length
                        )
                        points = []
                        while s_offset < s_offset_end:
                            t = StreamingDataAdapter.xodr_get_lane_t(
                                lane_offset_list, lane_offset_s, lane_dict,
                                k_id, s_begin, s_offset)
                            p = StreamingDataAdapter\
                                .xodr_transform_from_road_to_world(
                                    s_begin + s_offset, t, geometry_list,
                                    geometry_s, elevation_list, elevation_s)
                            points.append(p)
                            s_offset += interval

                        if is_last:
                            t = StreamingDataAdapter.xodr_get_lane_t(
                                lane_offset_list, lane_offset_s, lane_dict,
                                k_id, s_begin, s_offset_end)
                            p = StreamingDataAdapter\
                                .xodr_transform_from_road_to_world(
                                    s_begin + s_offset_end, t, geometry_list,
                                    geometry_s, elevation_list, elevation_s)
                            points.append(p)

                        rm_id = "{}_{}_{}_{}".format(
                            i.attrib["id"], j_id, k_id, l_id)
                        result[l.attrib["type"]][rm_id] = points

            for j in i.findall("objects/object"):
                if j.attrib["type"] not in result:
                    continue

                StreamingDataAdapter.extract_object_points(
                    j, result, geometry_list, geometry_s, elevation_list,
                    elevation_s)

    @staticmethod
    def get_hdmap_image(
        map_lines: dict, sensor: carla.Sensor, hdmap_image_settings: dict
    ):
        # options
        max_distance = hdmap_image_settings.get("max_distance", 65.0)
        pen_width = hdmap_image_settings.get("pen_width", 3)
        color_table = hdmap_image_settings.get(
            "color_table", StreamingDataAdapter.default_hdmap_color_table)

        # get the transform from the world space to the image space
        image_size = (
            int(sensor.attributes["image_size_x"]),
            int(sensor.attributes["image_size_y"])
        )
        intrinsic = np.eye(4)
        focal = float(sensor.attributes["image_size_x"]) /\
            (2.0 * np.tan(0.5 * np.deg2rad(float(sensor.attributes["fov"]))))
        intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
            [focal, focal], [0.5 * image_size[0], 0.5 * image_size[1]])

        rh_from_lh = lh_from_rh = np.diag([1, -1, 1, 1])
        ec = np.array(StreamingDataAdapter.extrinsic_correction)

        lh_sensor_from_lh_world = np.array(
            sensor.get_transform().get_inverse_matrix())
        rh_sensor_from_rh_world = \
            rh_from_lh @ lh_sensor_from_lh_world @ lh_from_rh
        image_from_world = intrinsic @ ec @ rh_sensor_from_rh_world

        # draw map elements to the image
        image = Image.new("RGB", image_size)
        draw = ImageDraw.Draw(image)
        for lane_type, lane_dict in map_lines.items():
            if lane_type in color_table:
                pen_color = tuple(color_table[lane_type])
                for i in lane_dict.values():
                    lane_points = np.array([
                        j + (1,) for j in i
                    ]).transpose()
                    p = image_from_world @ lane_points
                    for j in range(1, p.shape[1]):
                        xy = dwm.datasets.common.project_line(
                            p[:, j - 1], p[:, j], far_z=max_distance)
                        if xy is not None:
                            draw.line(xy, fill=pen_color, width=pen_width)

        return image

    def __init__(
        self, client, sensor_channels, transform_list: list, pop_list=None,
        collate_fn=None, master: bool = True,
        environment_description: str = "urban street scene.", fps=None,
        enable_images: bool = False, rear_vehicle_center=None,
        _3dbox_image_settings=None, hdmap_image_settings=None,
        min_pixel_ratio: float = 0.005
    ):
        self.client = client
        self.sensor_channels = sensor_channels
        self.transform_list = transform_list
        self.pop_list = pop_list
        self.collate_fn = collate_fn or torch.utils.data.default_collate
        self.master = master
        self.environment_description = environment_description
        self.world = client.get_world()
        settings = self.world.get_settings()

        xodr_map = ET.fromstring(self.world.get_map().to_opendrive())
        self.map_lines = {
            i: {} for i in StreamingDataAdapter.lane_types_to_extract
        }
        StreamingDataAdapter.extract_lines(
            xodr_map.findall("road"), self.map_lines)

        if fps is None:
            assert settings.fixed_delta_seconds != 0.0

        self.fps = fps or (1.0 / settings.fixed_delta_seconds)
        self.enable_images = enable_images
        self.rear_vehicle_center = (
            rear_vehicle_center or
            StreamingDataAdapter.default_rear_vehicle_center
        )
        self._3dbox_image_settings = _3dbox_image_settings or {}
        self.hdmap_image_settings = hdmap_image_settings or {}
        self.min_pixel_ratio = min_pixel_ratio

        if self.master:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        actors = self.world.get_actors()
        pvb_prefix = ["vehicle", "walker.pedestrian"]
        self.ego = None
        self.pvb = []
        self.sensors = []
        for i in actors:
            if i.attributes.get("role_name", "") == "hero":
                self.ego = i
            elif (
                i.type_id.startswith("sensor") and i.parent is not None and
                i.parent.attributes.get("role_name", "") == "hero"
            ):
                self.sensors.append(i)
            elif any([i.type_id.startswith(j) for j in pvb_prefix]):
                self.pvb.append(i)

        self.sensor_count = {}
        for i in self.sensors:
            if (
                i.type_id == "sensor.camera.rgb" or
                i.type_id == "sensor.camera.semantic_segmentation"
            ):
                i.listen(CarlaCallbackWithSensor(i, self.camera_callback))

            if i.type_id in self.sensor_count:
                self.sensor_count[i.type_id] += 1
            else:
                self.sensor_count[i.type_id] = 1

    def __del__(self):
        for i in self.sensors:
            if i.is_listening:
                i.stop()

    def camera_callback(self, sensor: carla.Actor, data: carla.SensorData):
        if not hasattr(self, "buffer"):
            return

        if sensor.type_id not in self.buffer:
            self.buffer[sensor.type_id] = {}

        self.buffer[sensor.type_id][sensor.attributes["role_name"]] = data

    def is_complete(self):
        return all([
            self.sensor_count[k] == len(v)
            for k, v in self.buffer.items()
        ])

    def query_data(self):
        self.buffer = {i: {} for i in self.sensor_count.keys()}
        if self.master:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        while not self.is_complete():
            time.sleep(0.001)

        carla_weather = self.world.get_weather()
        time_prompt = (
            "daytime" if carla_weather.sun_altitude_angle > 5.0
            else (
                "sunset" if carla_weather.sun_altitude_angle > -5
                else "night"
            )
        )
        weather_prompt = (
            "rainy" if carla_weather.precipitation > 30
            else (
                "foggy" if carla_weather.fog_density > 50
                else (
                    "overcast" if carla_weather.cloudiness > 70
                    else (
                        "cloudy" if carla_weather.cloudiness > 30
                        else "clear sky"
                    )
                )
            )
        )
        base_prompt = "{}. {}. {}".format(
            time_prompt, weather_prompt, self.environment_description)
        segm_buffer = self.buffer["sensor.camera.semantic_segmentation"]
        intrinsic_params = ["width", "height", "fov"]

        item = {
            "fps": torch.tensor(self.fps, dtype=torch.float32),
            "pts": torch.zeros(
                (1, self.sensor_count["sensor.camera.rgb"]),
                dtype=torch.float32),
            "camera_transforms": torch.tensor(
                StreamingDataAdapter.make_camera_transforms(
                    [
                        j
                        for i in self.sensor_channels
                        for j in self.sensors
                        if (
                            j.type_id == "sensor.camera.rgb" and
                            j.attributes["role_name"] == i
                        )
                    ],
                    self.rear_vehicle_center),
                dtype=torch.float32).unsqueeze(0),
            "camera_intrinsics": torch.tensor(
                StreamingDataAdapter.make_camera_intrinsics(*[
                    np.array([
                        getattr(self.buffer["sensor.camera.rgb"][j], i)
                        for j in self.sensor_channels
                    ])
                    for i in intrinsic_params
                ]),
                dtype=torch.float32).unsqueeze(0),
            "image_size": torch.tensor([[
                [
                    self.buffer["sensor.camera.rgb"][i].width,
                    self.buffer["sensor.camera.rgb"][i].height
                ]
                for i in self.sensor_channels
            ]], dtype=torch.float32),
            "ego_transforms": torch.tensor(
                StreamingDataAdapter.make_ego_transform(
                    self.ego, self.default_rear_vehicle_center),
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)
            .repeat(1, len(self.sensor_channels), 1, 1),
            "3dbox_images": [[
                StreamingDataAdapter.get_3dbox_image(
                    self.pvb, j, self._3dbox_image_settings)
                for i in self.sensor_channels
                for j in self.sensors
                if (
                    j.type_id == "sensor.camera.rgb" and
                    j.attributes["role_name"] == i
                )
            ]],
            "hdmap_images": [[
                StreamingDataAdapter.get_hdmap_image(
                    self.map_lines, j, self.hdmap_image_settings)
                for i in self.sensor_channels
                for j in self.sensors
                if (
                    j.type_id == "sensor.camera.rgb" and
                    j.attributes["role_name"] == i
                )
            ]],
            "image_description": [[
                "{} {}.".format(
                    base_prompt,
                    StreamingDataAdapter
                        .make_object_prompt_from_segm_sensor_data(
                            segm_buffer[i], self.min_pixel_ratio))
                for i in self.sensor_channels
            ]]
        }

        if self.enable_images:
            item["images"] = [[
                StreamingDataAdapter.make_image_from_sensor_data(
                    self.buffer["sensor.camera.rgb"][i])
                for i in self.sensor_channels
            ]]

        for i in self.transform_list:
            item[i["new_key"]] = dwm.datasets.common.DatasetAdapter\
                .apply_transform(
                    i["transform"], item[i["old_key"]],
                    i["stack"] if "stack" in i else True)

        if self.pop_list is not None:
            for i in self.pop_list:
                if i in item:
                    item.pop(i)

        return self.collate_fn([item])


if __name__ == "__main__":
    import torchvision

    client = carla.Client("127.0.0.1", 2000, 1)
    sda = StreamingDataAdapter(
        client, ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
        [
            {
                "old_key": "images",
                "new_key": "vae_images",
                "transform": torchvision.transforms.Compose([
                    torchvision.transforms.Resize((176, 304)),
                    torchvision.transforms.ToTensor()
                ])
            },
            {
                "old_key": "3dbox_images",
                "new_key": "3dbox_images",
                "transform": torchvision.transforms.Compose([
                    torchvision.transforms.Resize((176, 304)),
                    torchvision.transforms.ToTensor()
                ])
            },
            {
                "old_key": "hdmap_images",
                "new_key": "hdmap_images",
                "transform": torchvision.transforms.Compose([
                    torchvision.transforms.Resize((176, 304)),
                    torchvision.transforms.ToTensor()
                ])
            },
            {
                "old_key": "image_description",
                "new_key": "clip_text",
                "transform": dwm.datasets.common.Copy(),
                "stack": False
            }
        ],
        ["images"],
        dwm.datasets.common.CollateFnIgnoring(["clip_text"]),
        enable_images=True)

    for i in range(100):
        data = sda.query_data()
        print(i)
