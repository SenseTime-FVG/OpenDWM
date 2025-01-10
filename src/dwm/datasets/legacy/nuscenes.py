import bisect
import dwm.common
import dwm.datasets.common
import einops
import fsspec
import json
import numpy as np
import os
from PIL import Image, ImageDraw
import torch
import torchvision.transforms.functional
# PYTHONPATH includes ${workspaceFolder}/externals/nuscenes-devkit/python-sdk
# or install the pip of nuscenes-devkit
import nuscenes.utils.splits


class CanBusExtension:
    def __init__(
        self, fs: fsspec.AbstractFileSystem,
        table_names: list = ["steeranglefeedback", "vehicle_monitor", "pose"]
    ):
        self.fs = fs
        self.table_names = table_names

        files = self.fs.ls("can_bus/")
        self.tables = {i: {} for i in self.table_names}
        self.indices = {i: {} for i in self.table_names}
        for i in files:
            if i["type"] != "file":
                continue

            path = i["name"]
            dataset, filename_ext = path.split("/")
            if dataset != "can_bus" or filename_ext == "":
                continue

            filename, _ = os.path.splitext(filename_ext)
            sp_index = filename.index("_")
            scene_name = filename[:sp_index]
            table_name = filename[sp_index + 1:]
            if table_name in self.table_names:
                table = json.loads(self.fs.cat_file(path).decode())
                self.tables[table_name][scene_name] = table
                self.indices[table_name][scene_name] = list([
                    j["utime"] for j in table])

    def query_and_interpolate(self, table_name, scene_name, utime, column):
        if scene_name not in self.tables[table_name]:
            if table_name == 'pose':
                if column == 'pos':
                    return [[0., 0., 0.], [0., 0., 0.]]
                elif column == 'orientation':
                    return [[0., 0., 0., 0.], [0., 0., 0., 0.]]
                else:
                    return [0., 0., 0.]
            return None

        table = self.tables[table_name][scene_name]
        index = bisect.bisect_left(
            self.indices[table_name][scene_name], utime)

        if index == len(self.indices[table_name][scene_name]):
            if table_name == 'pose':
                if column == 'pos':
                    return [[0., 0., 0.], [0., 0., 0.]]
                elif column == 'orientation':
                    return [[0., 0., 0., 0.], [0., 0., 0., 0.]]
                else:
                    return [0., 0., 0.]
            return None
        elif index == 0:
            if utime == table[0]["utime"]:
                return table[0][column]
            else:
                if table_name == 'pose':
                    if column == 'pos':
                        return [[0., 0., 0.], table[0][column]]
                    elif column == 'orientation':
                        return [[0., 0., 0., 0.], table[0][column]]
                    else:
                        return table[0][column]
                else:
                    return None
        else:
            item = table[index]
            if column in ['pos', 'orientation']:
                last_item = table[index - 1]  # last utime item
                return [last_item[column], item[column]]
            elif column in ['accel', 'rotation_rate', 'vel']:
                return item[column]
            item_1 = table[index - 1]
            alpha = (utime - item_1["utime"]) / \
                (item["utime"] - item_1["utime"])
            return alpha * item[column] + (1 - alpha) * item_1[column]


class MotionDataset(torch.utils.data.Dataset):
    """The motion frame data of the nuScenes dataset for video clips.

    Args:
        fs (fsspec.AbstractFileSystem): The file system for the dataset table
            and content files.
        dataset_name (str): The nuScenes dataset name such as "v1.0-mini",
            "v1.0-trainval".
        sequence_length (int): The frame count of each video clips extracted
            from the dataset, also the "T" of the video tensor shape
            [T, C, H, W]. For the multimodal mode, the video tensor is
            returned in the shape of [T, V, C, H, W].
        fps_stride_tuples (list): The list of tuples in the form of
            (FPS, stride). If the FPS > 0, the stride is the second count
            of the beginning time between 2 adjacent video clips, else the
            stride is the index count of the beginning between 2 adjacent
            video clips.
        split (str): The dataset split different purpose of training,
            validation, test. Should be one of "train", "val", "mini_train",
            "mini_val".
        sensor_channels (list): The string list of required views in
            "LIDAR_TOP", "CAM_FRONT", "CAM_BACK", "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT".
        keyframe_only (bool): The flag to return the key frame only, and the
            key frames of the nuScenes dataset are with annotations.
        is_multimodal (bool): The flag to return the items in multimodal
            format, which means different views are fused into the same item.
        can_bus_table_names (list): The tables of CAN bus extension to read.
        enable_scene_description (bool): The flag to return the text of the
            scene description by the key "scene_description".
        enable_camera_transforms (bool): The flag to return the 4x4 transform
            matrix to the world from camera (by the key "camera_transforms")
            and from lidar (by the key "lidar_transforms"), the the 3x3 camera
            intrinsic transform matrix (by the key "camera_intrinsics"), and
            the image size (tuple of (width, height), by the key "image_size").
        enable_ego_transforms (bool): The flag to return the 4x4 transform
            matrix to the world of the ego vehicle by the key "ego_transforms".
        enable_sample_data (bool): The flag to return sample data objects.
        _3dbox_image_settings (dict): The settings to return and control the 3D
            box images by the key "3dbox_images".
        hdmap_image_settings (dict): The settings to return and control the HD
            map images by the key "hdmap_images".
        foreground_region_image_settings (dict): The settings to return the
            foreground region image by the key "foreground_region_images".
        _3dbox_bev_settings (dict): The settings to return and control the 3D
            box BEV images by the key "3dbox_bev_images".
        hdmap_bev_settings (dict): The settings to return and control the HD
            map BEV images by the key "hdmap_bev_images".
        stub_key_data_dict (dict): The dict of stub key and data, mainly for
            aligning other dataset with missing key and data in the current
            dataset.
    """

    table_names = [
        "calibrated_sensor", "category", "ego_pose", "instance", "log", "map",
        "sample", "sample_annotation", "sample_data", "scene", "sensor"
    ]
    index_names = [
        "calibrated_sensor.token", "category.token", "ego_pose.token",
        "instance.token", "log.token", "map.token", "sample.token",
        "sample_data.sample_token", "sample_data.token",
        "sample_annotation.sample_token", "sample_annotation.token",
        "scene.token", "sensor.token"
    ]
    serialized_table_names = [
        "sample", "sample_annotation", "sample_data", "scene"
    ]

    default_3dbox_color_table = {
        "human.pedestrian": (255, 0, 0),
        "vehicle.bicycle": (0, 255, 0),
        "vehicle.motorcycle": (0, 255, 0),
        "vehicle.bus": (0, 0, 255),
        "vehicle.car": (0, 0, 255),
        "vehicle.construction": (0, 0, 255),
        "vehicle.emergency": (0, 0, 255),
        "vehicle.trailer": (0, 0, 255),
        "vehicle.truck": (0, 0, 255)
    }
    default_hdmap_color_table = {
        "ped_crossing": (255, 0, 0),
        "lane": (0, 255, 0),
        "drivable_area": (0, 0, 255),
    }
    default_3dbox_corner_template = [
        [-0.5, -0.5, -0.5, 1], [-0.5, -0.5, 0.5, 1],
        [-0.5, 0.5, -0.5, 1], [-0.5, 0.5, 0.5, 1],
        [0.5, -0.5, -0.5, 1], [0.5, -0.5, 0.5, 1],
        [0.5, 0.5, -0.5, 1], [0.5, 0.5, 0.5, 1]
    ]
    default_3dbox_edge_indices = [
        (0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (1, 5),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
    ]
    default_bev_from_ego_transform = [
        [6.4, 0, 0, 320],
        [0, -6.4, 0, 320],
        [0, 0, -6.4, 0],
        [0, 0, 0, 1]
    ]
    default_bev_3dbox_corner_template = [
        [-0.5, -0.5, 0, 1], [-0.5, 0.5, 0, 1],
        [0.5, -0.5, 0, 1], [0.5, 0.5, 0, 1]
    ]
    default_bev_3dbox_edge_indices = [(0, 2), (2, 3), (3, 1), (1, 0)]
    default_image_description_keys = [
        "time", "weather", "environment", "objects", "image_description"
    ]

    @staticmethod
    def get_dict_indices(tables: dict, index_name: str):
        table_name, column_name = index_name.split(".")
        return dwm.common.ReadonlyDictIndices(
            [i[column_name] for i in tables[table_name]])

    @staticmethod
    def load_tables(
        fs: fsspec.AbstractFileSystem, dataset_name: str, table_names: list,
        index_names: list
    ):
        tables = {
            i: json.loads(
                fs.cat_file("{}/{}.json".format(dataset_name, i)).decode())
            for i in table_names
        }
        indices = {
            i: MotionDataset.get_dict_indices(tables, i)
            for i in index_names
        }
        return tables, indices

    @staticmethod
    def query(
        tables: dict, indices: dict, table_name: str, key: str,
        column_name: str = "token"
    ):
        i = indices["{}.{}".format(table_name, column_name)][key]
        return tables[table_name][i]

    @staticmethod
    def query_range(
        tables: dict, indices: dict, table_name: str, key: str,
        column_name: str = "token"
    ):
        all_indices = indices["{}.{}".format(table_name, column_name)]\
            .get_all_indices(key)
        table = tables[table_name]
        return [table[i] for i in all_indices]

    @staticmethod
    def get_scene_samples(tables: dict, indices: dict, scene: dict):
        result = []
        i = scene["first_sample_token"]
        while i != "":
            sample = MotionDataset.query(tables, indices, "sample", i)
            result.append(sample)
            i = sample["next"]

        return result

    @staticmethod
    def check_sensor(
        tables: dict, indices: dict, sample_data: dict, channel: str = None,
        modality: str = None
    ):
        calibrated_sensor = MotionDataset.query(
            tables, indices, "calibrated_sensor",
            sample_data["calibrated_sensor_token"])
        sensor = MotionDataset.query(
            tables, indices, "sensor", calibrated_sensor["sensor_token"])
        is_channel = channel is None or sensor["channel"] == channel
        is_modality = modality is None or sensor["modality"] == modality
        return is_channel and is_modality

    @staticmethod
    def enumerate_segments(
        sample_data_list: list, sequence_length: int, fps, stride
    ):
        # stride == 0: all segments are begin with key frames.
        # stride > 0:
        #   * FPS == 0: offset between segment beginings are by index.
        #   * FPS > 0: offset between segment beginings are by second.

        sdl = sample_data_list
        if fps == 0:
            # frames are extracted by the index.
            for t in range(0, len(sdl) - sequence_length + 1, max(1, stride)):
                if stride != 0 or sdl[t]["is_key_frame"]:
                    yield [
                        sdl[i]["token"] for i in range(t, t+sequence_length)
                    ]

        else:
            # frames are extracted by the timestamp.
            def enumerate_begin_time(sdl, sequence_duration, stride):
                s = sdl[-1]["timestamp"] / 1000000 - sequence_duration
                if stride == 0:
                    for i in sdl:
                        t = i["timestamp"] / 1000000
                        if i["is_key_frame"] and t <= s:
                            yield t

                else:
                    t = sdl[0]["timestamp"] / 1000000
                    while t <= s:
                        yield t
                        t += stride

            timestamp_list = [i["timestamp"] for i in sdl]
            for t in enumerate_begin_time(sdl, sequence_length / fps, stride):
                expected_times = [
                    (t + i / fps) * 1000000
                    for i in range(sequence_length)
                ]
                candidates = [
                    dwm.datasets.common.find_sample_data_of_nearest_time(
                        sdl, timestamp_list, i)["token"]
                    for i in expected_times
                ]
                max_time_error = max([
                    abs(i0["timestamp"] - i1)
                    for i0, i1 in zip(candidates, expected_times)
                ])
                if max_time_error <= 500000 / fps:
                    yield candidates

    @staticmethod
    def enumerate_multimodal_segments(
        channel_sample_data_list: list, sequence_length: int, fps, stride
    ):
        # stride == 0: all segments are begin with key frames.
        # stride > 0:
        #   * FPS == 0: offset between segment beginings are by index.
        #   * FPS > 0: offset between segment beginings are by second.

        csdl = channel_sample_data_list
        channel_timestamp_list = [
            [i["timestamp"] for i in sdl] for sdl in csdl
        ]
        channel_key_frame_timestamp_list = [
            [i["timestamp"] for i in sdl if i["is_key_frame"]]
            for sdl in csdl
        ]
        if fps == 0:
            # frames are extracted by the index.
            channel_key_frame_index_list = [
                [i_id for i_id, i in enumerate(sdl) if i["is_key_frame"]]
                for sdl in csdl
            ]
            for t in range(0, len(csdl[0]), max(1, stride)):
                # find the indices of the first frame of channels matching the
                # given timestamp
                ct0 = [
                    dwm.datasets.common.find_sample_data_of_nearest_time(
                        None, tl, csdl[0][t]["timestamp"])
                    for tl in channel_timestamp_list
                ] if stride != 0 else [
                    kfil[
                        dwm.datasets.common.find_sample_data_of_nearest_time(
                            None, kftl, csdl[0][t]["timestamp"])]
                    for kfil, kftl in zip(
                        channel_key_frame_index_list,
                        channel_key_frame_timestamp_list)
                ]

                if (stride != 0 or csdl[0][t]["is_key_frame"]) and all([
                    t0 + sequence_length <= len(sdl)
                    for t0, sdl in zip(ct0, csdl)
                ]):
                    yield [
                        [sdl[t0 + i]["token"] for t0, sdl in zip(ct0, csdl)]
                        for i in range(sequence_length)
                    ]

        else:
            # frames are extracted by the timestamp.
            def enumerate_begin_time(sdl, sequence_duration, stride):
                s = sdl[-1]["timestamp"] / 1000000 - sequence_duration
                if stride == 0:
                    for i in sdl:
                        t = i["timestamp"] / 1000000
                        if i["is_key_frame"] and t <= s:
                            yield t

                else:
                    t = sdl[0]["timestamp"] / 1000000
                    while t <= s:
                        yield t
                        t += stride

            channel_key_frame_list = [
                [i for i in sdl if i["is_key_frame"]]
                for sdl in csdl
            ]
            for t in enumerate_begin_time(
                csdl[0], sequence_length / fps, stride
            ):
                # find the indices of the first frame of channels matching the
                # given timestamp
                ct0 = [t * 1000000 for _ in csdl] if stride != 0 else [
                    dwm.datasets.common.find_sample_data_of_nearest_time(
                        kfl, kftl, t)["timestamp"]
                    for kfl, kftl in zip(
                        channel_key_frame_list,
                        channel_key_frame_timestamp_list)
                ]

                channel_expected_times = [
                    [t0 + i / fps * 1000000 for i in range(sequence_length)]
                    for t0 in ct0
                ]
                channel_candidates = [
                    [
                        dwm.datasets.common.find_sample_data_of_nearest_time(
                            sdl, timestamp_list, i)
                        for i in expected_times
                    ]
                    for sdl, timestamp_list, expected_times in zip(
                        csdl, channel_timestamp_list, channel_expected_times)
                ]
                max_time_error = max([
                    abs(i0["timestamp"] - i1)
                    for candidates, expected_times in zip(
                        channel_candidates, channel_expected_times)
                    for i0, i1 in zip(candidates, expected_times)
                ])
                if max_time_error <= 500000 / fps:
                    yield [
                        [
                            candidates[i]["token"]
                            for candidates in channel_candidates
                        ]
                        for i in range(sequence_length)
                    ]

    @staticmethod
    def get_transform(
        tables: dict, indices: dict, table_name: str, queried_key: str,
        output_type: str = "np"
    ):
        posed_object = MotionDataset.query(
            tables, indices, table_name, queried_key)
        return dwm.datasets.common.get_transform(
            posed_object["rotation"], posed_object["translation"], output_type)

    @staticmethod
    def draw_lines_to_image(
        nodes: list, draw: ImageDraw, transform: np.array,
        max_distance: float, pen_color: tuple, pen_width: int
    ):
        if len(nodes) == 0:
            return

        polygon_nodes = np.array(nodes).transpose().reshape(4, -1)
        p = (transform @ polygon_nodes).reshape(4, 2, -1)
        m = p.shape[-1]
        for i in range(m):
            xy = dwm.datasets.common.project_line(
                p[:, 0, i], p[:, 1, i], far_z=max_distance)
            if xy is not None:
                draw.line(xy, fill=pen_color, width=pen_width)

    @staticmethod
    def draw_polygon_to_bev_image(
        polygon: dict, nodes: list, draw: ImageDraw, transform: np.array,
        pen_color: tuple, pen_width: int, solid: bool = False
    ):
        polygon_nodes = np.array([
            [nodes[i]["x"], nodes[i]["y"], 0, 1]
            for i in polygon["exterior_node_tokens"]
        ]).transpose()
        p = transform @ polygon_nodes
        draw.polygon(
            [(p[0, i], p[1, i]) for i in range(p.shape[1])],
            fill=pen_color if solid else None,
            outline=None if solid else pen_color, width=pen_width)

        for i in polygon["holes"]:
            hole_nodes = np.array([
                [nodes[j]["x"], nodes[j]["y"], 0, 1] for j in i["node_tokens"]
            ]).transpose()
            p = transform @ hole_nodes
            draw.polygon(
                [(p[0, i], p[1, i]) for i in range(p.shape[1])],
                fill=(0, 0, 0) if solid else None,
                outline=None if solid else pen_color, width=pen_width)

    @staticmethod
    def get_images_and_lidar_points(
        fs: fsspec.AbstractFileSystem, tables: dict, indices: dict,
        sample_data_list: list
    ):
        images = []
        lidar_points = []
        for i in sample_data_list:
            if MotionDataset.check_sensor(
                    tables, indices, i, modality="camera"):
                with fs.open(i["filename"]) as f:
                    image = Image.open(f)
                    image.load()

                images.append(image)

            elif MotionDataset.check_sensor(
                    tables, indices, i, modality="lidar"):
                point_data = np.frombuffer(
                    fs.cat_file(i["filename"]), dtype=np.float32)
                lidar_points.append(
                    torch.tensor(point_data.reshape((-1, 5))[:, :3]))

        return images, lidar_points

    @staticmethod
    def get_image_description(
        fs: fsspec.AbstractFileSystem, fn2token: dict,
        sample_data: dict, all_sample_data: list
    ):
        path = "caption/{}.json".format(sample_data["token"])
        if path not in fs._belongs_to:
            all_timestamps = [i["timestamp"] for i in all_sample_data]
            index = bisect.bisect_left(all_timestamps, sample_data["timestamp"])
            while path not in fs._belongs_to:
                if fn2token is None:
                    path = "caption/{}.json".format(
                        all_sample_data[index]["token"])
                elif all_sample_data[index]["filename"] in fn2token:
                    path = "caption/{}.json".format(
                        fn2token[all_sample_data[index]["filename"]])

                index -= 1

        with fs.open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def get_3dbox_image(
        tables: dict, indices: dict, sample_data: dict, _3dbox_image_settings: dict
    ):
        # options
        pen_width = _3dbox_image_settings.get("pen_width", 4)
        color_table = _3dbox_image_settings.get(
            "color_table", MotionDataset.default_3dbox_color_table)
        corner_templates = _3dbox_image_settings.get(
            "corner_templates", MotionDataset.default_3dbox_corner_template)
        edge_indices = _3dbox_image_settings.get(
            "edge_indices", MotionDataset.default_3dbox_edge_indices)

        # get the transform from the referenced ego space to the image space
        calibrated_sensor = MotionDataset.query(
            tables, indices, "calibrated_sensor",
            sample_data["calibrated_sensor_token"])
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = np.array(calibrated_sensor["camera_intrinsic"])

        ego_from_camera = dwm.datasets.common.get_transform(
            calibrated_sensor["rotation"], calibrated_sensor["translation"])
        world_from_ego = MotionDataset.get_transform(
            tables, indices, "ego_pose", sample_data["ego_pose_token"])
        camera_from_world = np.linalg.inv(world_from_ego @ ego_from_camera)
        image_from_world = intrinsic @ camera_from_world

        # draw annotations to the image
        image = Image.new("RGB", (sample_data["width"], sample_data["height"]))
        if not sample_data["is_key_frame"]:
            return image

        draw = ImageDraw.Draw(image)

        corner_templates_np = np.array(corner_templates).transpose()
        for sa in MotionDataset.query_range(
                tables, indices, "sample_annotation",
                sample_data["sample_token"], column_name="sample_token"):
            instance = MotionDataset.query(
                tables, indices, "instance", sa["instance_token"])
            category = MotionDataset.query(
                tables, indices, "category", instance["category_token"])

            # check the category from the color table
            color = None
            for i, c in color_table.items():
                if category["name"].startswith(i):
                    color = c if isinstance(c, tuple) else tuple(c)
                    break

            if color is None:
                continue

            # get the transform from the annotation template to the world space
            scale = np.diag([sa["size"][1], sa["size"][0], sa["size"][2], 1])
            world_from_annotation = dwm.datasets.common.get_transform(
                sa["rotation"], sa["translation"])

            # project and render lines
            image_corners = image_from_world @ world_from_annotation @ \
                scale @ corner_templates_np
            for a, b in edge_indices:
                xy = dwm.datasets.common.project_line(
                    image_corners[:, a], image_corners[:, b])
                if xy is not None:
                    draw.line(xy, fill=color, width=pen_width)

        return image

    @staticmethod
    def get_neighbor_indices(x, y, index):
        i = index // y
        j = index % y

        indices = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                ni, nj = i + dx, j + dy

                if 0 <= ni < x and 0 <= nj < y:
                    flat_index = ni * y + nj
                    indices.append(flat_index)

        return indices

    @staticmethod
    def get_hdmap_image(
        map_expansion: dict, map_expansion_dict: dict, tables: dict,
        indices: dict, sample_data: dict, hdmap_image_settings: dict
    ):
        # options
        max_distance = hdmap_image_settings.get("max_distance", 65.0)
        pen_width = hdmap_image_settings.get("pen_width", 4)
        color_table = hdmap_image_settings.get(
            "color_table", MotionDataset.default_hdmap_color_table)

        # get the transform from the world (map) space to the image space
        calibrated_sensor = MotionDataset.query(
            tables, indices, "calibrated_sensor",
            sample_data["calibrated_sensor_token"])
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = np.array(calibrated_sensor["camera_intrinsic"])
        ego_from_camera = dwm.datasets.common.get_transform(
            calibrated_sensor["rotation"], calibrated_sensor["translation"])
        ego_pose = MotionDataset.query(
            tables, indices, "ego_pose", sample_data["ego_pose_token"])
        world_from_ego = dwm.datasets.common.get_transform(
            ego_pose["rotation"], ego_pose["translation"])
        camera_from_world = np.linalg.inv(world_from_ego @ ego_from_camera)
        image_from_world = intrinsic @ camera_from_world

        # draw map elements to the image
        image = Image.new("RGB", (sample_data["width"], sample_data["height"]))
        draw = ImageDraw.Draw(image)

        sample = MotionDataset.query(
            tables, indices, "sample", sample_data["sample_token"])
        scene = MotionDataset.query(
            tables, indices, "scene", sample["scene_token"])
        log = MotionDataset.query(tables, indices, "log", scene["log_token"])
        map = map_expansion[log["location"]]
        map_dict = map_expansion_dict[log["location"]]
        nodes = map_dict["node"]
        polygons = map_dict["polygon"]

        blocks = map["blocks"]
        choose_block = np.argmin(np.linalg.norm(
            np.array(blocks["center_points"]) - world_from_ego[:, 3], axis=1))
        choose_block_index = MotionDataset.get_neighbor_indices(
            blocks["len_x"], blocks["len_y"], choose_block)

        if "lane" in color_table and "lane" in map:
            pen_color = tuple(color_table["lane"])
            choose_nodes = list(set([(tuple(i[0]), tuple(i[1])) for j in choose_block_index
                                for i in map["lane_blocks"][str(j)]]))

            MotionDataset.draw_lines_to_image(
                choose_nodes, draw, image_from_world, max_distance, pen_color, pen_width)

        if "drivable_area" in color_table and "drivable_area" in map:
            pen_color = tuple(color_table["drivable_area"])
            choose_nodes = list(set([(tuple(i[0]), tuple(i[1])) for j in choose_block_index
                                for i in map["drivable_area_blocks"][str(j)]]))

            MotionDataset.draw_lines_to_image(
                choose_nodes, draw, image_from_world, max_distance, pen_color, pen_width)

        if "ped_crossing" in color_table and "ped_crossing" in map:
            pen_color = tuple(color_table["ped_crossing"])
            choose_nodes = list(set([(tuple(i[0]), tuple(i[1])) for j in choose_block_index
                                for i in map["ped_crossing_blocks"][str(j)]]))

            MotionDataset.draw_lines_to_image(
                choose_nodes, draw, image_from_world, max_distance, pen_color, pen_width)

        return image

    @staticmethod
    def get_foreground_region_image(
        tables: dict, indices: dict, sample_data: dict,
        foreground_region_image_settings: dict
    ):
        # options
        foreground_color = tuple(
            foreground_region_image_settings.get(
                "foreground_color", [255, 255, 255]))
        background_color = tuple(
            foreground_region_image_settings.get(
                "background_color", [0, 0, 0]))
        foreground_categories = foreground_region_image_settings.get(
            "categories", MotionDataset.default_3dbox_color_table.keys())
        corner_templates = foreground_region_image_settings.get(
            "corner_templates", MotionDataset.default_3dbox_corner_template)

        # get the transform from the referenced ego space to the image space
        calibrated_sensor = MotionDataset.query(
            tables, indices, "calibrated_sensor",
            sample_data["calibrated_sensor_token"])
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = np.array(calibrated_sensor["camera_intrinsic"])

        ego_from_camera = dwm.datasets.common.get_transform(
            calibrated_sensor["rotation"], calibrated_sensor["translation"])
        world_from_ego = MotionDataset.get_transform(
            tables, indices, "ego_pose", sample_data["ego_pose_token"])
        camera_from_world = np.linalg.inv(world_from_ego @ ego_from_camera)
        image_from_world = intrinsic @ camera_from_world

        # draw annotations to the image
        image = Image.new(
            "RGB", (sample_data["width"], sample_data["height"]),
            background_color)
        if not sample_data["is_key_frame"]:
            return image

        draw = ImageDraw.Draw(image)

        corner_templates_np = np.array(corner_templates).transpose()
        for sa in MotionDataset.query_range(
                tables, indices, "sample_annotation",
                sample_data["sample_token"], column_name="sample_token"):
            instance = MotionDataset.query(
                tables, indices, "instance", sa["instance_token"])
            category = MotionDataset.query(
                tables, indices, "category", instance["category_token"])

            # check the category from the color table
            out_of_categories = True
            for i in foreground_categories:
                if category["name"].startswith(i):
                    out_of_categories = False
                    break

            if out_of_categories:
                continue

            # get the transform from the annotation template to the world space
            scale = np.diag([sa["size"][1], sa["size"][0], sa["size"][2], 1])
            world_from_annotation = dwm.datasets.common.get_transform(
                sa["rotation"], sa["translation"])

            # project and render lines
            image_corners = image_from_world @ world_from_annotation @ \
                scale @ corner_templates_np

            # All points are in the front of the camera
            if np.min(image_corners[2], -1) > 0:
                p = image_corners[:2] / image_corners[2]
                top_left = np.min(p, -1)
                bottom_right = np.max(p, -1)
                draw.rectangle(
                    tuple(np.concatenate([top_left, bottom_right]).tolist()),
                    fill=foreground_color)

        return image

    @staticmethod
    def get_3dbox_bev_image(
        tables: dict, indices: dict, sample_data: dict,
        _3dbox_bev_settings: dict
    ):
        # options
        pen_width = _3dbox_bev_settings.get("pen_width", 2)
        bev_size = _3dbox_bev_settings.get("bev_size", [640, 640])
        bev_from_ego_transform = _3dbox_bev_settings.get(
            "bev_from_ego_transform",
            MotionDataset.default_bev_from_ego_transform)
        fill_box = _3dbox_bev_settings.get("fill_box", False)
        color_table = _3dbox_bev_settings.get(
            "color_table", MotionDataset.default_3dbox_color_table)
        corner_templates = _3dbox_bev_settings.get(
            "corner_templates",
            MotionDataset.default_bev_3dbox_corner_template)
        edge_indices = _3dbox_bev_settings.get(
            "edge_indices", MotionDataset.default_bev_3dbox_edge_indices)

        # get the transform from the world space to the BEV space
        world_from_ego = MotionDataset.get_transform(
            tables, indices, "ego_pose", sample_data["ego_pose_token"])
        ego_from_world = np.linalg.inv(world_from_ego)
        bev_from_ego = np.array(bev_from_ego_transform, np.float32)
        bev_from_world = bev_from_ego @ ego_from_world

        # draw annotations to the image
        image = Image.new("RGB", tuple(bev_size))
        if not sample_data["is_key_frame"]:
            return image

        draw = ImageDraw.Draw(image)

        corner_templates_np = np.array(corner_templates).transpose()
        for sa in MotionDataset.query_range(
                tables, indices, "sample_annotation",
                sample_data["sample_token"], column_name="sample_token"):
            instance = MotionDataset.query(
                tables, indices, "instance", sa["instance_token"])
            category = MotionDataset.query(
                tables, indices, "category", instance["category_token"])

            # check the category from the color table
            color = None
            for i, c in color_table.items():
                if category["name"].startswith(i):
                    color = c if isinstance(c, tuple) else tuple(c)
                    break

            if color is None:
                continue

            # get the transform from the annotation template to the world space
            scale = np.diag([sa["size"][1], sa["size"][0], sa["size"][2], 1])
            world_from_annotation = dwm.datasets.common.get_transform(
                sa["rotation"], sa["translation"])

            # project and render lines
            image_corners = bev_from_world @ world_from_annotation @ scale @ \
                corner_templates_np
            p = image_corners[:2]
            if fill_box:
                draw.polygon(
                    [(p[0, a], p[1, a]) for a, _ in edge_indices],
                    fill=color, width=pen_width)
            else:
                for a, b in edge_indices:
                    draw.line(
                        (p[0, a], p[1, a], p[0, b], p[1, b]),
                        fill=color, width=pen_width)

        return image

    @staticmethod
    def get_hdmap_bev_image(
        map_expansion: dict, map_expansion_dict: dict, tables: dict,
        indices: dict, sample_data: dict, hdmap_bev_settings: dict
    ):
        # options
        pen_width = hdmap_bev_settings.get("pen_width", 2)
        bev_size = hdmap_bev_settings.get("bev_size", [640, 640])
        bev_from_ego_transform = hdmap_bev_settings.get(
            "bev_from_ego_transform",
            MotionDataset.default_bev_from_ego_transform)
        color_table = hdmap_bev_settings.get(
            "color_table", MotionDataset.default_hdmap_color_table)

        # get the transform from the world (map) space to the BEV space
        world_from_ego = MotionDataset.get_transform(
            tables, indices, "ego_pose", sample_data["ego_pose_token"])
        bev_from_ego = np.array(bev_from_ego_transform, np.float32)
        bev_from_world = bev_from_ego @ np.linalg.inv(world_from_ego)

        # draw map elements to the image
        image = Image.new("RGB", tuple(bev_size))
        draw = ImageDraw.Draw(image)

        sample = MotionDataset.query(
            tables, indices, "sample", sample_data["sample_token"])
        scene = MotionDataset.query(
            tables, indices, "scene", sample["scene_token"])
        log = MotionDataset.query(tables, indices, "log", scene["log_token"])
        map = map_expansion[log["location"]]
        map_dict = map_expansion_dict[log["location"]]
        nodes = map_dict["node"]
        polygons = map_dict["polygon"]

        if "drivable_area" in color_table and "drivable_area" in map:
            pen_color = tuple(color_table["drivable_area"])
            for i in map["drivable_area"]:
                for polygon_token in i["polygon_tokens"]:
                    MotionDataset.draw_polygon_to_bev_image(
                        polygons[polygon_token], nodes, draw, bev_from_world,
                        (0, 0, 255), pen_width, solid=True)

        if "ped_crossing" in color_table and "ped_crossing" in map:
            pen_color = tuple(color_table["ped_crossing"])
            for i in map["ped_crossing"]:
                MotionDataset.draw_polygon_to_bev_image(
                    polygons[i["polygon_token"]], nodes, draw, bev_from_world,
                    (255, 0, 0), pen_width, solid=True)

        if "lane" in color_table and "lane" in map:
            pen_color = tuple(color_table["lane"])
            for i in map["lane"]:
                MotionDataset.draw_polygon_to_bev_image(
                    polygons[i["polygon_token"]], nodes, draw, bev_from_world,
                    pen_color, pen_width)

        return image

    @staticmethod
    def get_segmentation_image(
        fs: fsspec.AbstractFileSystem, sample_data: dict,
        image_segmentation_settings: dict
    ):
        gw = image_segmentation_settings.get("gw", 4)
        gh = image_segmentation_settings.get("gh", 2)
        total_channels = image_segmentation_settings.get("total_channels", 19)
        path = "{}.png".format(sample_data["filename"])
        with fs.open(path) as f:
            image = Image.open(f)
            return einops.rearrange(
                torchvision.transforms.functional.to_tensor(image),
                "c (gh h) (gw w) -> (gh gw c) h w", gh=gh, gw=gw
            )[:total_channels]

    def __init__(
        self, fs: fsspec.AbstractFileSystem, dataset_name: str,
        sequence_length: int, fps_stride_tuples: list, split: str = None,
        sensor_channels: list = ["CAM_FRONT"],
        keyframe_only: bool = False,
        is_multimodal: bool = False,
        can_bus_table_names: list = None,
        enable_scene_description: bool = False,
        enable_camera_transforms: bool = False,
        enable_ego_transforms: bool = False,
        enable_sample_data: bool = False,
        _3dbox_image_settings: dict = None,
        hdmap_image_settings: dict = None,
        foreground_region_image_settings: dict = None,
        _3dbox_bev_settings: dict = None,
        hdmap_bev_settings: dict = None,
        image_description_settings: dict = None,
        image_segmentation_settings: dict = None,
        stub_key_data_dict: dict = None
    ):
        self.fs = fs
        self.tables, self.indices = MotionDataset.load_tables(
            fs, dataset_name, MotionDataset.table_names,
            MotionDataset.index_names)

        self.sequence_length = sequence_length
        self.fps_stride_tuples = fps_stride_tuples
        self.is_multimodal = is_multimodal
        self.enable_scene_description = enable_scene_description
        self.enable_camera_transforms = enable_camera_transforms
        self.enable_ego_transforms = enable_ego_transforms
        self.enable_sample_data = enable_sample_data
        self._3dbox_image_settings = _3dbox_image_settings
        self.hdmap_image_settings = hdmap_image_settings
        self.foreground_region_image_settings = \
            foreground_region_image_settings
        self._3dbox_bev_settings = _3dbox_bev_settings
        self.hdmap_bev_settings = hdmap_bev_settings
        self.image_description_settings = image_description_settings
        self.image_segmentation_settings = image_segmentation_settings

        # load filename to token mapping in image description
        self.fn2token = None
        if (
            self.image_description_settings is not None and
            self.image_description_settings.get("fn2token_path")
        ):
            with open(self.image_description_settings.get(
                    "fn2token_path"), "r") as json_file:
                self.fn2token = json.load(json_file)

        # cache the map data
        if self.hdmap_image_settings is not None or \
                self.hdmap_bev_settings is not None:
            assert self.hdmap_image_settings is None or \
                "path" in self.hdmap_image_settings
            self.map_expansion = {}
            self.map_expansion_dict = {}
            for i in self.tables["log"]:
                to_dict = ["node", "polygon"]
                if i["location"] not in self.map_expansion:
                    name = "expansion/{}.json".format(i["location"])
                    self.map_expansion[i["location"]] = json.loads(
                        fs.cat_file(name).decode())
                    self.map_expansion_dict[i["location"]] = {}
                    for j in to_dict:
                        self.map_expansion_dict[i["location"]][j] = {
                            k["token"]: k
                            for k in self.map_expansion[i["location"]][j]
                        }

            if self.hdmap_image_settings is not None:
                with open(
                    self.hdmap_image_settings.get("path"), 'r'
                ) as json_file:
                    add_map_expansion = json.load(json_file)
                    for key in add_map_expansion:
                        self.map_expansion[key].update(add_map_expansion[key])

        # for the ego speed and steering
        self.can_bus_extension = CanBusExtension(fs, can_bus_table_names) \
            if can_bus_table_names is not None else None
        self.stub_key_data_dict = {} if stub_key_data_dict is None \
            else stub_key_data_dict
        if self.can_bus_extension is not None:
            self.stub_key_data_dict["ego_speed"] = \
                ("tensor", (sequence_length,), -1000)
            self.stub_key_data_dict["ego_steering"] = \
                ("tensor", (sequence_length,), -1000)

        scenes = self.tables["scene"]
        if split is not None:
            scene_subset = getattr(nuscenes.utils.splits, split)
            scenes = [
                i for i in self.tables["scene"] if i["name"] in scene_subset]

        key_filter = (lambda i: i["is_key_frame"]) if keyframe_only \
            else (lambda _: True)

        # [scene_count, channel_count, sample_data_count]
        scene_channel_sample_data = [
            (scene, [
                sorted([
                    sample_data
                    for sample in MotionDataset.get_scene_samples(
                        self.tables, self.indices, scene)
                    for sample_data in MotionDataset.query_range(
                        self.tables, self.indices, "sample_data",
                        sample["token"], column_name="sample_token")
                    if MotionDataset.check_sensor(
                        self.tables, self.indices, sample_data, channel) and
                    key_filter(sample_data)
                ], key=lambda x: x["timestamp"])
                for channel in sensor_channels
            ])
            for scene in scenes
        ]
        if self.is_multimodal:
            self.items = dwm.common.SerializedReadonlyList([
                {"segment": segment, "fps": fps, "scene": scene["token"]}
                for scene, channel_sample_data in scene_channel_sample_data
                for fps, stride in self.fps_stride_tuples
                for segment in MotionDataset.enumerate_multimodal_segments(
                    channel_sample_data, self.sequence_length, fps, stride)
            ])
        else:
            self.items = dwm.common.SerializedReadonlyList([
                {"segment": segment, "fps": fps, "scene": scene["token"]}
                for scene, channel_sample_data in scene_channel_sample_data
                for sample_data_list in channel_sample_data
                for fps, stride in self.fps_stride_tuples
                for segment in MotionDataset.enumerate_segments(
                    sample_data_list, self.sequence_length, fps, stride)
            ])

        self.sample_data_table = {
            scene["token"]: channel_sample_data
            for scene, channel_sample_data in scene_channel_sample_data
        }

        for i in MotionDataset.serialized_table_names:
            self.tables[i] = dwm.common.SerializedReadonlyList(self.tables[i])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        scene = MotionDataset.query(
            self.tables, self.indices, "scene", item["scene"])
        if self.is_multimodal:
            segment = [
                [
                    MotionDataset.query(
                        self.tables, self.indices, "sample_data", j)
                    for j in i
                ]
                for i in item["segment"]
            ]
        else:
            segment = [
                MotionDataset.query(
                    self.tables, self.indices, "sample_data", i)
                for i in item["segment"]
            ]

        result = {
            "fps": torch.tensor(item["fps"], dtype=torch.float32)
        }

        if self.enable_scene_description:
            result["scene_description"] = scene["description"]

        if self.enable_sample_data:
            result["sample_data"] = segment

        if self.is_multimodal:
            result["pts"] = torch.tensor([
                [
                    (j["timestamp"] - segment[0][0]["timestamp"] + 500)
                    // 1000
                    for j in i
                ]
                for i in segment
            ], dtype=torch.float32)
            images, lidar_points = [], []
            for i in segment:
                images_i, lidar_points_i = self.get_images_and_lidar_points(
                    self.fs, self.tables, self.indices, i)
                if len(images_i) > 0:
                    images.append(images_i)
                if len(lidar_points_i) > 0:
                    lidar_points.append(lidar_points_i[0])

            if len(images) > 0:
                result["images"] = images  # [sequence_length, view_count]
            if len(lidar_points) > 0:
                result["lidar_points"] = lidar_points  # [sequence_length]

        else:
            result["pts"] = torch.tensor([
                (i["timestamp"] - segment[0]["timestamp"] + 500) // 1000
                for i in segment
            ])
            images, lidar_points = self.get_images_and_lidar_points(
                self.fs, self.tables, self.indices, segment)
            if len(images) > 0:
                result["images"] = images  # [sequence_length]
            if len(lidar_points) > 0:
                result["lidar_points"] = lidar_points  # [sequence_length]

        if self.enable_camera_transforms:
            if self.is_multimodal:
                if "images" in result:
                    result["camera_transforms"] = torch.stack([
                        torch.stack([
                            MotionDataset.get_transform(
                                self.tables, self.indices, "calibrated_sensor",
                                j["calibrated_sensor_token"], "pt")
                            for j in i
                            if MotionDataset.check_sensor(
                                self.tables, self.indices, j,
                                modality="camera")
                        ])
                        for i in segment
                    ])
                    result["camera_intrinsics"] = torch.stack([
                        torch.stack([
                            torch.tensor(
                                MotionDataset.query(
                                    self.tables, self.indices,
                                    "calibrated_sensor",
                                    j["calibrated_sensor_token"]
                                )["camera_intrinsic"], dtype=torch.float32)
                            for j in i
                            if MotionDataset.check_sensor(
                                self.tables, self.indices, j,
                                modality="camera")
                        ])
                        for i in segment
                    ])
                    result["image_size"] = torch.stack([
                        torch.stack([
                            torch.tensor(
                                [j["width"], j["height"]], dtype=torch.long)
                            for j in i
                            if MotionDataset.check_sensor(
                                self.tables, self.indices, j,
                                modality="camera")
                        ])
                        for i in segment
                    ])

                if "lidar_points" in result:
                    result["lidar_transforms"] = torch.stack([
                        torch.stack([
                            MotionDataset.get_transform(
                                self.tables, self.indices, "calibrated_sensor",
                                j["calibrated_sensor_token"], "pt")
                            for j in i
                            if MotionDataset.check_sensor(
                                self.tables, self.indices, j, modality="lidar")
                        ])
                        for i in segment
                    ])

            else:
                if "images" in result:
                    result["camera_transforms"] = torch.stack([
                        MotionDataset.get_transform(
                            self.tables, self.indices, "calibrated_sensor",
                            i["calibrated_sensor_token"], "pt")
                        for i in segment
                        if MotionDataset.check_sensor(
                            self.tables, self.indices, i, modality="camera")
                    ])
                    result["camera_intrinsics"] = torch.stack([
                        torch.tensor(
                            MotionDataset.query(
                                self.tables, self.indices, "calibrated_sensor",
                                i["calibrated_sensor_token"]
                            )["camera_intrinsic"], dtype=torch.float32)
                        for i in segment
                        if MotionDataset.check_sensor(
                            self.tables, self.indices, i, modality="camera")
                    ])
                    result["image_size"] = torch.stack([
                        torch.tensor(
                            [i["width"], i["height"]], dtype=torch.float32)
                        for i in segment
                        if MotionDataset.check_sensor(
                            self.tables, self.indices, i, modality="camera")
                    ])

                if "lidar_points" in result:
                    result["lidar_transforms"] = torch.stack([
                        MotionDataset.get_transform(
                            self.tables, self.indices, "calibrated_sensor",
                            i["calibrated_sensor_token"], "pt")
                        for i in segment
                        if MotionDataset.check_sensor(
                            self.tables, self.indices, i, modality="lidar")
                    ])

        if self.enable_ego_transforms:
            if self.is_multimodal:
                result["ego_transforms"] = torch.stack([
                    torch.stack([
                        MotionDataset.get_transform(
                            self.tables, self.indices, "ego_pose",
                            j["ego_pose_token"], "pt")
                        for j in i
                    ])
                    for i in segment
                ])
            else:
                result["ego_transforms"] = torch.stack([
                    MotionDataset.get_transform(
                        self.tables, self.indices, "ego_pose",
                        i["ego_pose_token"], "pt")
                    for i in segment
                ])

            # NOTE add rotation and translation for BEVFormer usage
            ego_pose = [
                MotionDataset.query(
                    self.tables, self.indices, "ego_pose",
                    i[0]["ego_pose_token"])
                for i in segment
            ]

            result["bev_translation"] = torch.tensor(
                ego_pose[0]['translation'])
            result["bev_rotation"] = torch.tensor(ego_pose[0]['rotation'])

        if self._3dbox_image_settings is not None:
            if self.is_multimodal:
                result["3dbox_images"] = [
                    [
                        MotionDataset.get_3dbox_image(
                            self.tables, self.indices, j,
                            self._3dbox_image_settings)
                        for j in i
                        if MotionDataset.check_sensor(
                            self.tables, self.indices, j, modality="camera")
                    ]
                    for i in segment
                ]
            else:
                result["3dbox_images"] = [
                    MotionDataset.get_3dbox_image(
                        self.tables, self.indices, i,
                        self._3dbox_image_settings)
                    for i in segment
                    if MotionDataset.check_sensor(
                        self.tables, self.indices, i, modality="camera")
                ]

        if self.hdmap_image_settings is not None:
            if self.is_multimodal:
                result["hdmap_images"] = [
                    [
                        MotionDataset.get_hdmap_image(
                            self.map_expansion, self.map_expansion_dict,
                            self.tables, self.indices, j,
                            self.hdmap_image_settings)
                        for j in i
                        if MotionDataset.check_sensor(
                            self.tables, self.indices, j, modality="camera")
                    ]
                    for i in segment
                ]
            else:
                result["hdmap_images"] = [
                    MotionDataset.get_hdmap_image(
                        self.map_expansion, self.map_expansion_dict,
                        self.tables, self.indices, i,
                        self.hdmap_image_settings)
                    for i in segment
                    if MotionDataset.check_sensor(
                        self.tables, self.indices, i, modality="camera")
                ]

        if self.image_segmentation_settings is not None:
            if self.is_multimodal:
                result["segmentation_images"] = [
                    [
                        MotionDataset.get_segmentation_image(
                            self.fs, j, self.image_segmentation_settings)
                        for j in i
                        if MotionDataset.check_sensor(
                            self.tables, self.indices, j, modality="camera")
                    ]
                    for i in segment
                ]
            else:
                result["segmentation_images"] = [
                    MotionDataset.get_segmentation_image(
                        self.fs, i, self.image_segmentation_settings)
                    for i in segment
                    if MotionDataset.check_sensor(
                        self.tables, self.indices, i, modality="camera")
                ]

        if self.foreground_region_image_settings is not None:
            if self.is_multimodal:
                result["foreground_region_images"] = [
                    [
                        MotionDataset.get_foreground_region_image(
                            self.tables, self.indices, j,
                            self.foreground_region_image_settings)
                        for j in i
                        if MotionDataset.check_sensor(
                            self.tables, self.indices, j, modality="camera")
                    ]
                    for i in segment
                ]
            else:
                result["foreground_region_images"] = [
                    MotionDataset.get_foreground_region_image(
                        self.tables, self.indices, i,
                        self.foreground_region_image_settings)
                    for i in segment
                    if MotionDataset.check_sensor(
                        self.tables, self.indices, i, modality="camera")
                ]

        if self._3dbox_bev_settings is not None:
            if self.is_multimodal:
                result["3dbox_bev_images"] = [
                    MotionDataset.get_3dbox_bev_image(
                        self.tables, self.indices, j, self._3dbox_bev_settings)
                    for i in segment
                    for j in i
                    if MotionDataset.check_sensor(
                        self.tables, self.indices, j, modality="lidar")
                ]
            else:
                result["3dbox_bev_images"] = [
                    MotionDataset.get_3dbox_bev_image(
                        self.tables, self.indices, i, self._3dbox_bev_settings)
                    for i in segment
                    if MotionDataset.check_sensor(
                        self.tables, self.indices, i, modality="lidar")
                ]

        if self.hdmap_bev_settings is not None:
            if self.is_multimodal:
                result["hdmap_bev_images"] = [
                    MotionDataset.get_hdmap_bev_image(
                        self.map_expansion, self.map_expansion_dict,
                        self.tables, self.indices, j, self.hdmap_bev_settings)
                    for i in segment
                    for j in i
                    if MotionDataset.check_sensor(
                        self.tables, self.indices, j, modality="lidar")
                ]
            else:
                result["hdmap_bev_images"] = [
                    MotionDataset.get_hdmap_bev_image(
                        self.map_expansion, self.map_expansion_dict,
                        self.tables, self.indices, i, self.hdmap_bev_settings)
                    for i in segment
                    if MotionDataset.check_sensor(
                        self.tables, self.indices, i, modality="lidar")
                ]

        # extension part for CAN bus
        if self.can_bus_extension is not None:
            scene_name = scene["name"]
            ego_speed = [
                self.can_bus_extension.query_and_interpolate(
                    "vehicle_monitor", scene_name, i[0]["timestamp"] if
                    self.is_multimodal else i["timestamp"], "vehicle_speed")
                for i in segment
            ]
            if all([i is not None for i in ego_speed]):
                result["ego_speed"] = torch.tensor(ego_speed)

            ego_steering = [
                self.can_bus_extension.query_and_interpolate(
                    "steeranglefeedback", scene_name, i[0]["timestamp"] if
                    self.is_multimodal else i["timestamp"], "value")
                for i in segment
            ]
            if all([i is not None for i in ego_steering]):
                result["ego_steering"] = torch.tensor(ego_steering)

            # NOTE add pose
            ego_pos = [
                self.can_bus_extension.query_and_interpolate(
                    "pose", scene_name, i[0]["timestamp"] if
                    self.is_multimodal else i["timestamp"], "pos")
                for i in segment
            ]

            if all([i is not None for i in ego_pos]):
                result["ego_pos"] = torch.tensor(ego_pos)

            # NOTE add orient
            ego_orient = [
                self.can_bus_extension.query_and_interpolate(
                    "pose", scene_name, i[0]["timestamp"] if
                    self.is_multimodal else i["timestamp"], "orientation")
                for i in segment
            ]
            if all([i is not None for i in ego_orient]):
                result["ego_orient"] = torch.tensor(ego_orient)

            # NOTE add accel, rotation_rate, vel
            ego_accel = [
                self.can_bus_extension.query_and_interpolate(
                    "pose", scene_name, i[0]["timestamp"] if
                    self.is_multimodal else i["timestamp"], "accel")
                for i in segment
            ]
            if all([i is not None for i in ego_accel]):
                result["ego_accel"] = torch.tensor(ego_accel)
            ego_rotation_rate = [
                self.can_bus_extension.query_and_interpolate(
                    "pose", scene_name, i[0]["timestamp"] if
                    self.is_multimodal else i["timestamp"], "rotation_rate")
                for i in segment
            ]
            if all([i is not None for i in ego_rotation_rate]):
                result["ego_rotation_rate"] = torch.tensor(ego_rotation_rate)
            ego_vel = [
                self.can_bus_extension.query_and_interpolate(
                    "pose", scene_name, i[0]["timestamp"] if
                    self.is_multimodal else i["timestamp"], "vel")
                for i in segment
            ]
            if all([i is not None for i in ego_vel]):
                result["ego_vel"] = torch.tensor(ego_vel)

        if self.image_description_settings is not None:
            selected_keys = self.image_description_settings.get(
                "selected_keys", MotionDataset.default_image_description_keys)
            if self.is_multimodal:
                image_captions = [
                    [
                        MotionDataset.get_image_description(
                            self.fs, self.fn2token, j,
                            self.sample_data_table[scene["token"]][channel_index])
                        for channel_index, j in enumerate(i)
                        if MotionDataset.check_sensor(
                            self.tables, self.indices, j, modality="camera")
                    ]
                    for i in segment
                ]
                result["image_description"] = [
                    [". ".join([j[k] for k in selected_keys]) for j in i]
                    for i in image_captions
                ]
            else:
                image_captions = [
                    MotionDataset.get_image_description(
                        self.fs, self.fn2token, i,
                        self.sample_data_table[scene["token"]])
                    for i in segment
                    if MotionDataset.check_sensor(
                        self.tables, self.indices, i, modality="camera")
                ]
                result["image_description"] = [
                    ". ".join([i[j] for j in selected_keys])
                    for i in image_captions
                ]

        # add stub values for heterogeneous dataset merging
        for key, data in self.stub_key_data_dict.items():
            if key not in result.keys():
                if data[0] == "tensor":
                    shape, value = data[1:]
                    result[key] = value * torch.ones(shape)
                else:
                    result[key] = data[1]

        return result
