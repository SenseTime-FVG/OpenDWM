import dwm.common
import dwm.datasets.common
import dwm.datasets.nuscenes_common
import einops
import fsspec
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms.functional


class MotionDataset(torch.utils.data.Dataset):
    """The motion data loaded from the nuScenes dataset.

    Args:
        fs (fsspec.AbstractFileSystem): The file system for the dataset table
            and content files.
        dataset_name (str): The nuScenes dataset name such as "v1.0-mini",
            "v1.0-trainval".
        sequence_length (int): The frame count of the temporal sequence.
        fps_stride_tuples (list): The list of tuples in the form of
            (FPS, stride). If the FPS > 0, stride is the begin time in second
            between 2 adjacent video clips, else the stride is the index count
            of the beginning between 2 adjacent video clips.
        split (str or None): The split in one of "train", "val", "mini_train",
            "mini_val", following the official split definition of the nuScenes
            dataset, or None for the whole data.
        sensor_channels (list): The string list of required views in
            "LIDAR_TOP", "CAM_FRONT", "CAM_BACK", "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", following
            the nuScenes sensor name.
        keyframe_only (bool): If set to True, only the key frames with complete
            annotation information will be included by the data items.
        enable_synchronization_check (bool): When this feature is enabled, if
            the timestamp error of a certain frame in the read video clip
            exceeds half of the frame interval, this video clip will be
            discarded. True as default.
        enable_scene_description (bool): If set to True, the data item will
            include the text of the scene description by "scene_description".
        enable_camera_transforms (bool): If set to True, the data item will
            include the "camera_transforms", "camera_intrinsics", "image_size"
            if camera modality exists, and include "lidar_transforms" if LiDAR
            modality exists. For a detailed definition of transforms, please
            refer to the dataset README.
        enable_ego_transforms (bool): If set to True, the data item will
            include the "ego_transforms". For a detailed definition of
            transforms, please refer to the dataset README.
        enable_sample_data (bool): If set to True, the data item will include
            the "sample_data" for nuScenes sample data objects.
        _3dbox_image_settings (dict or None): If set, the data item will
            include the "3dbox_images".
        hdmap_image_settings (dict or None): If set, the data item will include
            the "hdmap_images".
        image_segmentation_settings (dict or None): If set, the data item will
            include the "segmentation_images".
        foreground_region_image_settings (dict or None): If set, the data item
            will include the "foreground_region_images".
        _3dbox_bev_settings (dict or None): If set, the data item will include
            the "3dbox_bev_images".
        hdmap_bev_settings (dict or None): If set, the data item will include
            the "hdmap_bev_images".
        image_description_settings (dict or None): If set, the data item will
            include the "image_description". The "path" in the setting is for
            the content JSON file. The "time_list_dict_path" in the setting is
            for the file to seek the nearest labelled time points. Please refer
            to dwm.datasets.common.make_image_description_string() for details.
        stub_key_data_dict (dict or None): The dict of stub key and data, to
            align with other datasets with keys and data missing in this
            dataset. Please refer to dwm.datasets.common.add_stub_key_data()
            for details.
    """

    table_names = [
        "calibrated_sensor", "category", "ego_pose", "instance", "log", "map",
        "sample", "sample_annotation", "sample_data", "scene", "sensor"
    ]
    prune_table_plan = [
        ("sample", "scene_token", "scene"),
        ("sample_data", "sample_token", "sample"),
        ("sample_annotation", "sample_token", "sample")
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
        "vehicle.bicycle": (128, 255, 0),
        "vehicle.motorcycle": (0, 255, 128),
        "vehicle.bus": (128, 0, 255),
        "vehicle.car": (0, 0, 255),
        "vehicle.construction": (128, 128, 255),
        "vehicle.emergency": (255, 128, 128),
        "vehicle.trailer": (255, 255, 255),
        "vehicle.truck": (255, 255, 0)
    }
    default_hdmap_color_table = {
        "drivable_area": (0, 0, 255),
        "lane": (0, 255, 0),
        "ped_crossing": (255, 0, 0)
    }
    default_3dbox_corner_template = [
        [-0.5, -0.5, -0.5, 1], [-0.5, -0.5, 0.5, 1],
        [-0.5, 0.5, -0.5, 1], [-0.5, 0.5, 0.5, 1],
        [0.5, -0.5, -0.5, 1], [0.5, -0.5, 0.5, 1],
        [0.5, 0.5, -0.5, 1], [0.5, 0.5, 0.5, 1]
    ]
    default_3dbox_edge_indices = [
        (0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (1, 5),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        (6, 3), (6, 5)
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

    @staticmethod
    def prune_table(table: list, foreign_key: str, referenced_table: list):
        if referenced_table is None:
            return table
        else:
            referenced_tokens = set(i["token"] for i in referenced_table)
            return [i for i in table if i[foreign_key] in referenced_tokens]

    @staticmethod
    def get_dict_indices(tables: dict, index_name: str):
        table_name, column_name = index_name.split(".")
        return dwm.common.ReadonlyDictIndices(
            [i[column_name] for i in tables[table_name]])

    @staticmethod
    def load_tables(
        fs: fsspec.AbstractFileSystem, dataset_name: str, table_names: list,
        prune_table_plan: list, index_names: list, split=None
    ):
        tables = {
            i: json.loads(
                fs.cat_file("{}/{}.json".format(dataset_name, i)).decode())
            for i in table_names
        }
        if split is not None:
            scene_subset = getattr(dwm.datasets.nuscenes_common, split)
            tables["scene"] = [
                i for i in tables["scene"] if i["name"] in scene_subset
            ]

            for i in prune_table_plan:
                table_name, foreign_key, ref_table_name = i
                tables[table_name] = MotionDataset.prune_table(
                    tables[table_name], foreign_key, tables[ref_table_name])

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
    def get_sensor(tables: dict, indices: dict, sample_data: dict):
        calibrated_sensor = MotionDataset.query(
            tables, indices, "calibrated_sensor",
            sample_data["calibrated_sensor_token"])
        return MotionDataset.query(
            tables, indices, "sensor", calibrated_sensor["sensor_token"])

    @staticmethod
    def check_sensor(
        tables: dict, indices: dict, sample_data: dict, channel=None,
        modality=None
    ):
        sensor = MotionDataset.get_sensor(tables, indices, sample_data)
        is_channel = channel is None or sensor["channel"] == channel
        is_modality = modality is None or sensor["modality"] == modality
        return is_channel and is_modality

    @staticmethod
    def enumerate_segments(
        channel_sample_data_list: list, sequence_length: int, fps, stride,
        enable_synchronization_check: bool
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
                    dwm.datasets.common.find_nearest(
                        tl, csdl[0][t]["timestamp"])
                    for tl in channel_timestamp_list
                ] if stride != 0 else [
                    kfil[
                        dwm.datasets.common.find_nearest(
                            kftl, csdl[0][t]["timestamp"])]
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
                    kfl[dwm.datasets.common.find_nearest(kftl, t)]["timestamp"]
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
                        sdl[dwm.datasets.common.find_nearest(timestamps, i)]
                        for i in expected_times
                    ]
                    for sdl, timestamps, expected_times in zip(
                        csdl, channel_timestamp_list, channel_expected_times)
                ]
                max_time_error = max([
                    abs(i0["timestamp"] - i1)
                    for candidates, expected_times in zip(
                        channel_candidates, channel_expected_times)
                    for i0, i1 in zip(candidates, expected_times)
                ])
                if (
                    not enable_synchronization_check or
                    max_time_error <= 500000 / fps
                ):
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
    def get_3dbox_image(
        tables: dict, indices: dict, sample_data: dict, _3dbox_image_settings: dict
    ):
        # options
        pen_width = _3dbox_image_settings.get("pen_width", 8)
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
        world_from_ego = dwm.datasets.common.get_transform(
            sample_data["rotation"], sample_data["translation"])
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
    def draw_polygen_to_image(
        polygon: dict, nodes: list, draw: ImageDraw, transform: np.array,
        max_distance: float, pen_color: tuple, pen_width: int
    ):
        polygon_nodes = np.array([
            [nodes[i]["x"], nodes[i]["y"], 0, 1]
            for i in polygon["exterior_node_tokens"]
        ]).transpose()
        p = transform @ polygon_nodes
        m = len(polygon["exterior_node_tokens"])
        for i in range(m):
            xy = dwm.datasets.common.project_line(
                p[:, i], p[:, (i + 1) % m], far_z=max_distance)
            if xy is not None:
                draw.line(xy, fill=pen_color, width=pen_width)

        for i in polygon["holes"]:
            hole_nodes = np.array([
                [nodes[j]["x"], nodes[j]["y"], 0, 1] for j in i["node_tokens"]
            ]).transpose()
            p = transform @ hole_nodes
            m = len(i["node_tokens"])
            for j in range(m):
                xy = dwm.datasets.common.project_line(
                    p[:, j], p[:, (j + 1) % m], far_z=max_distance)
                if xy is not None:
                    draw.line(xy, fill=pen_color, width=pen_width)

    @staticmethod
    def get_hdmap_image(
        map_expansion: dict, map_expansion_dict: dict, tables: dict,
        indices: dict, sample_data: dict, hdmap_image_settings: dict
    ):
        # options
        max_distance = hdmap_image_settings.get("max_distance", 65.0)
        pen_width = hdmap_image_settings.get("pen_width", 8)
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
        world_from_ego = dwm.datasets.common.get_transform(
            sample_data["rotation"], sample_data["translation"])
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

        if "lane" in color_table and "lane" in map:
            pen_color = tuple(color_table["lane"])
            for i in map["lane"]:
                MotionDataset.draw_polygen_to_image(
                    polygons[i["polygon_token"]], nodes, draw,
                    image_from_world, max_distance, pen_color, pen_width)

        if "drivable_area" in color_table and "drivable_area" in map:
            pen_color = tuple(color_table["drivable_area"])
            for i in map["drivable_area"]:
                for polygon_token in i["polygon_tokens"]:
                    MotionDataset.draw_polygen_to_image(
                        polygons[polygon_token], nodes, draw, image_from_world,
                        max_distance, pen_color, pen_width)

        if "ped_crossing" in color_table and "ped_crossing" in map:
            pen_color = tuple(color_table["ped_crossing"])
            for i in map["ped_crossing"]:
                MotionDataset.draw_polygen_to_image(
                    polygons[i["polygon_token"]], nodes, draw,
                    image_from_world, max_distance, pen_color, pen_width)

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
        world_from_ego = dwm.datasets.common.get_transform(
            sample_data["rotation"], sample_data["translation"])
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
        world_from_ego = dwm.datasets.common.get_transform(
            sample_data["rotation"], sample_data["translation"])
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
        fill_map = hdmap_bev_settings.get("fill_map", True)
        # get the transform from the world (map) space to the BEV space
        world_from_ego = dwm.datasets.common.get_transform(
            sample_data["rotation"], sample_data["translation"])
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
                        (0, 0, 255), pen_width, solid=fill_map)

        if "ped_crossing" in color_table and "ped_crossing" in map:
            pen_color = tuple(color_table["ped_crossing"])
            for i in map["ped_crossing"]:
                MotionDataset.draw_polygon_to_bev_image(
                    polygons[i["polygon_token"]], nodes, draw, bev_from_world,
                    (255, 0, 0), pen_width, solid=fill_map)

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

    @staticmethod
    def get_image_description(
        tables: dict, indices: dict, image_descriptions: dict,
        time_list_dict: dict, scene: str, sample_data: dict
    ):
        sensor = MotionDataset.get_sensor(tables, indices, sample_data)
        scene_camera = "{}|{}".format(scene, sensor["channel"])
        time_list = time_list_dict[scene_camera]
        nearest_time = dwm.datasets.common.find_nearest(
            time_list, sample_data["timestamp"], return_item=True)
        return image_descriptions["{}|{}".format(scene_camera, nearest_time)]

    def __init__(
        self, fs: fsspec.AbstractFileSystem, dataset_name: str,
        sequence_length: int, fps_stride_tuples: list, split=None,
        sensor_channels: list = ["CAM_FRONT"], keyframe_only: bool = False,
        enable_synchronization_check: bool = True,
        enable_scene_description: bool = False,
        enable_camera_transforms: bool = False,
        enable_ego_transforms: bool = False, enable_sample_data: bool = False,
        _3dbox_image_settings=None, hdmap_image_settings=None,
        image_segmentation_settings=None,
        foreground_region_image_settings=None, _3dbox_bev_settings=None,
        hdmap_bev_settings=None, image_description_settings=None,
        stub_key_data_dict=None
    ):
        self.fs = fs
        tables, self.indices = MotionDataset.load_tables(
            fs, dataset_name, MotionDataset.table_names,
            MotionDataset.prune_table_plan, MotionDataset.index_names, split)

        self.sequence_length = sequence_length
        self.fps_stride_tuples = fps_stride_tuples
        self.enable_scene_description = enable_scene_description
        self.enable_camera_transforms = enable_camera_transforms
        self.enable_ego_transforms = enable_ego_transforms
        self.enable_sample_data = enable_sample_data
        self._3dbox_image_settings = _3dbox_image_settings
        self.hdmap_image_settings = hdmap_image_settings
        self.image_segmentation_settings = image_segmentation_settings
        self.foreground_region_image_settings = \
            foreground_region_image_settings
        self._3dbox_bev_settings = _3dbox_bev_settings
        self.hdmap_bev_settings = hdmap_bev_settings
        self.image_description_settings = image_description_settings
        self.stub_key_data_dict = stub_key_data_dict

        # cache the map data
        if (
            self.hdmap_image_settings is not None or
            self.hdmap_bev_settings is not None
        ):
            self.map_expansion = {}
            self.map_expansion_dict = {}
            for i in tables["log"]:
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

        key_filter = (lambda i: i["is_key_frame"]) if keyframe_only \
            else (lambda _: True)

        # Merge ego_pose into sample_data to reduce the memory usage of
        # multiple data workers.
        if "ego_pose" in tables:
            for i in tables["sample_data"]:
                pose = MotionDataset.query(
                    tables, self.indices, "ego_pose", i["ego_pose_token"])
                i.update({k: v for k, v in pose.items() if k not in i})

            tables.pop("ego_pose")
            self.indices.pop("ego_pose.token")

        # [scene_count, channel_count, sample_data_count]
        scene_channel_sample_data = [
            (scene, [
                sorted([
                    sample_data
                    for sample in MotionDataset.get_scene_samples(
                        tables, self.indices, scene)
                    for sample_data in MotionDataset.query_range(
                        tables, self.indices, "sample_data", sample["token"],
                        column_name="sample_token")
                    if MotionDataset.check_sensor(
                        tables, self.indices, sample_data, channel) and
                    key_filter(sample_data)
                ], key=lambda x: x["timestamp"])
                for channel in sensor_channels
            ])
            for scene in tables["scene"]
        ]
        self.items = dwm.common.SerializedReadonlyList([
            {"segment": segment, "fps": fps, "scene": scene["token"]}
            for scene, channel_sample_data in scene_channel_sample_data
            for fps, stride in self.fps_stride_tuples
            for segment in MotionDataset.enumerate_segments(
                channel_sample_data, self.sequence_length, fps, stride,
                enable_synchronization_check)
        ])

        if image_description_settings is not None:
            with open(
                image_description_settings["path"], "r", encoding="utf-8"
            ) as f:
                self.image_descriptions = json.load(f)

            self.image_desc_rs = np.random.RandomState(
                image_description_settings["seed"]
                if "seed" in image_description_settings else None)

            with open(
                image_description_settings["time_list_dict_path"], "r",
                encoding="utf-8"
            ) as f:
                self.time_list_dict = json.load(f)

        self.tables = {
            k: (
                dwm.common.SerializedReadonlyList(v)
                if k in MotionDataset.serialized_table_names else v
            )
            for k, v in tables.items()
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        scene = MotionDataset.query(
            self.tables, self.indices, "scene", item["scene"])
        segment = [
            [
                MotionDataset.query(
                    self.tables, self.indices, "sample_data", j)
                for j in i
            ]
            for i in item["segment"]
        ]

        result = {
            "fps": torch.tensor(item["fps"], dtype=torch.float32)
        }

        if self.enable_scene_description:
            result["scene_description"] = scene["description"]

        if self.enable_sample_data:
            result["sample_data"] = segment
            result["scene"] = scene

        result["pts"] = torch.tensor([
            [
                (j["timestamp"] - segment[0][0]["timestamp"] + 500) // 1000
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

        if self.enable_camera_transforms:
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

        if self.enable_ego_transforms:
            result["ego_transforms"] = torch.stack([
                torch.stack([
                    dwm.datasets.common.get_transform(
                        j["rotation"], j["translation"], "pt")
                    for j in i
                ])
                for i in segment
            ])

        if self._3dbox_image_settings is not None:
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

        if self.hdmap_image_settings is not None:
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

        if self.image_segmentation_settings is not None:
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

        if self.foreground_region_image_settings is not None:
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

        if self._3dbox_bev_settings is not None:
            result["3dbox_bev_images"] = [
                MotionDataset.get_3dbox_bev_image(
                    self.tables, self.indices, j, self._3dbox_bev_settings)
                for i in segment
                for j in i
                if MotionDataset.check_sensor(
                    self.tables, self.indices, j, modality="lidar")
            ]

        if self.hdmap_bev_settings is not None:
            result["hdmap_bev_images"] = [
                MotionDataset.get_hdmap_bev_image(
                    self.map_expansion, self.map_expansion_dict,
                    self.tables, self.indices, j, self.hdmap_bev_settings)
                for i in segment
                for j in i
                if MotionDataset.check_sensor(
                    self.tables, self.indices, j, modality="lidar")
            ]

        if self.image_description_settings is not None:
            image_captions = [
                dwm.datasets.common.align_image_description_crossview([
                    MotionDataset.get_image_description(
                        self.tables, self.indices, self.image_descriptions,
                        self.time_list_dict, scene["token"], j)
                    for j in i
                    if MotionDataset.check_sensor(
                        self.tables, self.indices, j, modality="camera")
                ], self.image_description_settings)
                for i in segment
            ]
            result["image_description"] = [
                [
                    dwm.datasets.common.make_image_description_string(
                        j, self.image_description_settings, self.image_desc_rs)
                    for j in i
                ]
                for i in image_captions
            ]

        dwm.datasets.common.add_stub_key_data(self.stub_key_data_dict, result)

        return result
