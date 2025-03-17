import dwm.common
import dwm.datasets.common
import dwm.datasets.waymo_common as wc
import fsspec
import io
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
import transforms3d
import waymo_open_dataset.dataset_pb2 as waymo_pb
import zlib


class MotionDataset(torch.utils.data.Dataset):
    """The motion data loaded from the Waymo Perception dataset.

    Args:
        fs (fsspec.AbstractFileSystem): The file system for the data records.
        info_dict_path (str): The path to the info dict file, which contains
            the offset of the data at each timestamp in the record relative to
            the beginning of the file, is used for fast seek during random
            access.
        sequence_length (int): The frame count of the temporal sequence.
        fps_stride_tuples (list): The list of tuples in the form of
            (FPS, stride). If the FPS > 0, stride is the begin time in second
            between 2 adjacent video clips, else the stride is the index count
            of the beginning between 2 adjacent video clips.
        sensor_channels (list): The string list of required views in
            "LIDAR_TOP", "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_SIDE_LEFT", "CAM_SIDE_RIGHT", following the Waymo sensor name.
        enable_camera_transforms (bool): If set to True, the data item will
            include the "camera_transforms", "camera_intrinsics", "image_size"
            if camera modality exists, and include "lidar_transforms" if LiDAR
            modality exists. For a detailed definition of transforms, please
            refer to the dataset README.
        enable_ego_transforms (bool): If set to True, the data item will
            include the "ego_transforms". For a detailed definition of
            transforms, please refer to the dataset README.
        _3dbox_image_settings (dict or None): If set, the data item will
            include the "3dbox_images".
        hdmap_image_settings (dict or None): If set, the data item will include
            the "hdmap_images".
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

    sensor_name_id_dict = {
        "CAM_FRONT": 1,
        "CAM_FRONT_LEFT": 2,
        "CAM_FRONT_RIGHT": 3,
        "CAM_SIDE_LEFT": 4,
        "CAM_SIDE_RIGHT": 5,
        "LIDAR_TOP": 1,
        "LIDAR_FRONT": 2,
        "LIDAR_SIDE_LEFT": 3,
        "LIDAR_SIDE_RIGHT": 4,
        "LIDAR_REAR": 5
    }
    box_type_dict = {
        "VEHICLE": 1,
        "PEDESTRIAN": 2,
        "SIGN": 3,
        "CYCLIST": 4
    }
    map_element_type_dict = {
        "road_line": "polyline",
        "lane": "polyline",
        "road_edge": "polyline",
        "crosswalk": "polygon",
        "driveway": "polygon",
        "speed_bump": "polygon"
    }

    extrinsic_correction = [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ]
    default_3dbox_color_table = {
        "PEDESTRIAN": (255, 0, 0),
        "CYCLIST": (0, 255, 0),
        "VEHICLE": (0, 0, 255)
    }
    default_hdmap_color_table = {
        "crosswalk": (255, 0, 0),
        "road_edge": (0, 0, 255),
        "road_line": (0, 255, 0)
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
    def find_by_name(list, queried_name):
        return [i for i in list if i.name == queried_name][0]

    @staticmethod
    def enumerate_segments(
        sample_list: list, sequence_length: int, fps, stride
    ):
        # enumerate segments for each scene
        timestamps = [i[0] for i in sample_list]
        if fps == 0:
            # frames are extracted by the index.
            stop = len(timestamps) - sequence_length + 1
            for t in range(0, stop, max(1, stride)):
                yield timestamps[t:t+sequence_length]

        else:
            # frames are extracted by the timestamp.
            def enumerate_begin_time(timestamps, sequence_duration, stride):
                s = timestamps[-1] / 1000000 - sequence_duration
                t = timestamps[0] / 1000000
                while t <= s:
                    yield t
                    t += stride

            for t in enumerate_begin_time(
                timestamps, sequence_length / fps, stride
            ):
                candidates = [
                    dwm.datasets.common.find_nearest(
                        timestamps, (t + i / fps) * 1000000, return_item=True)
                    for i in range(sequence_length)
                ]
                yield candidates

    @staticmethod
    def get_images_and_lidar_points(
        sensor_channels: list, frame: waymo_pb.Frame
    ):
        images = []
        lidar_points = []
        for i in sensor_channels:
            if i.startswith("LIDAR"):
                laser = MotionDataset.find_by_name(
                    frame.lasers, MotionDataset.sensor_name_id_dict[i])
                range_image = waymo_pb.MatrixFloat()
                range_image.ParseFromString(
                    zlib.decompress(
                        laser.ri_return1.range_image_compressed))
                range_image = np.array(range_image.data, np.float32)\
                    .reshape(range_image.shape.dims)

                laser_calibration = wc.laser_calibration_to_dict(
                    MotionDataset.find_by_name(
                        frame.context.laser_calibrations,
                        MotionDataset.sensor_name_id_dict[i]))

                if i == "LIDAR_TOP":
                    range_image_top_pose = waymo_pb.MatrixFloat()
                    range_image_top_pose.ParseFromString(
                        zlib.decompress(
                            laser.ri_return1.range_image_pose_compressed))
                    range_image_top_pose = np\
                        .array(range_image_top_pose.data, np.float32)\
                        .reshape(range_image_top_pose.shape.dims)
                    frame_pose = np.array(frame.pose.transform, np.float32)\
                        .reshape(4, 4)
                else:
                    range_image_top_pose = None
                    frame_pose = None

                lidar_points.append(
                    torch.tensor(
                        wc.convert_range_image_to_cartesian(
                            range_image, laser_calibration,
                            range_image_top_pose, frame_pose),
                        dtype=torch.float32))

            elif i.startswith("CAM"):
                frame_image = MotionDataset.find_by_name(
                    frame.images, MotionDataset.sensor_name_id_dict[i])
                with io.BytesIO(frame_image.image) as f:
                    image = Image.open(f)
                    image.load()

                images.append(image)

        return images, lidar_points

    @staticmethod
    def get_3dbox_image(
        laser_labels, camera_calibration, _3dbox_image_settings: dict
    ):
        # options
        pen_width = _3dbox_image_settings.get("pen_width", 8)
        color_table = _3dbox_image_settings.get(
            "color_table", MotionDataset.default_3dbox_color_table)
        native_color_table = {
            MotionDataset.box_type_dict[k]: v for k, v in color_table.items()
        }

        corner_templates = _3dbox_image_settings.get(
            "corner_templates", MotionDataset.default_3dbox_corner_template)
        edge_indices = _3dbox_image_settings.get(
            "edge_indices", MotionDataset.default_3dbox_edge_indices)

        # get the transform from the ego space to the image space
        image_size = (camera_calibration.width, camera_calibration.height)
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
            camera_calibration.intrinsic[0:2],
            camera_calibration.intrinsic[2:4])
        ec = np.array(MotionDataset.extrinsic_correction)
        ego_from_camera = np.array(
            camera_calibration.extrinsic.transform).reshape(4, 4)
        image_from_ego = intrinsic @ ec @ np.linalg.inv(ego_from_camera)

        # draw annotations to the image
        def list_annotation():
            for i in laser_labels:
                yield i

        def get_world_transform(i):
            scale = np.diag([i.box.length, i.box.width, i.box.height, 1])
            ego_from_annotation = dwm.datasets.common.get_transform(
                transforms3d.euler.euler2quat(
                    0, 0, i.box.heading).tolist(),
                [i.box.center_x, i.box.center_y, i.box.center_z])
            return ego_from_annotation @ scale

        image = Image.new("RGB", image_size)
        draw = ImageDraw.Draw(image)
        dwm.datasets.common.draw_3dbox_image(
            draw, image_from_ego, list_annotation, get_world_transform,
            lambda i: i.type, pen_width, native_color_table, corner_templates,
            edge_indices)

        return image

    @staticmethod
    def draw_polygon_to_image(
        polygon: list, draw: ImageDraw, transform: np.array,
        max_distance: float, pen_color: tuple, pen_width: int
    ):
        if len(polygon) == 0:
            return

        polygon_nodes = np.array([
            [i[0], i[1], i[2], 1] for i in polygon
        ], np.float32).transpose()
        p = transform @ polygon_nodes
        m = len(polygon)
        for i in range(m):
            xy = dwm.datasets.common.project_line(
                p[:, i], p[:, (i + 1) % m], far_z=max_distance)
            if xy is not None:
                draw.line(xy, fill=pen_color, width=pen_width)

    @staticmethod
    def draw_line_to_image(
        line: list, draw: ImageDraw, transform: np.array, max_distance: float,
        pen_color: tuple, pen_width: int
    ):
        if len(line) == 0:
            return

        line_nodes = np.array([
            [i[0], i[1], i[2], 1] for i in line
        ], np.float32).transpose()
        p = transform @ line_nodes
        for i in range(1, len(line)):
            xy = dwm.datasets.common.project_line(
                p[:, i - 1], p[:, i], far_z=max_distance)
            if xy is not None:
                draw.line(xy, fill=pen_color, width=pen_width)

    @staticmethod
    def get_hdmap_image(
        map_features, camera_calibration, pose, hdmap_image_settings: dict
    ):
        max_distance = hdmap_image_settings["max_distance"] \
            if "max_distance" in hdmap_image_settings else 65.0
        pen_width = hdmap_image_settings["pen_width"] \
            if "pen_width" in hdmap_image_settings else 8
        color_table = hdmap_image_settings.get(
            "color_table", MotionDataset.default_hdmap_color_table)
        max_distance = hdmap_image_settings["max_distance"] \
            if "max_distance" in hdmap_image_settings else 65.0

        # get the transform from the world space to the image space
        image_size = (camera_calibration.width, camera_calibration.height)
        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3, :3] = dwm.datasets.common.make_intrinsic_matrix(
            camera_calibration.intrinsic[0:2],
            camera_calibration.intrinsic[2:4])
        ec = np.array(MotionDataset.extrinsic_correction, np.float32)
        ego_from_camera = np.array(
            camera_calibration.extrinsic.transform, np.float32).reshape(4, 4)
        world_from_ego = np.array(pose.transform, np.float32).reshape(4, 4)
        image_from_world = intrinsic @ ec @ \
            np.linalg.inv(world_from_ego @ ego_from_camera)

        # draw annotations to the image
        image = Image.new("RGB", image_size)
        draw = ImageDraw.Draw(image)

        type_polygons = {}
        type_polylines = {}
        for feat in map_features:
            type_ = feat.WhichOneof('feature_data')
            if (
                type_ not in color_table or
                type_ not in MotionDataset.map_element_type_dict
            ):
                continue

            type_poly = MotionDataset.map_element_type_dict[type_]
            items = getattr(getattr(feat, type_), type_poly)
            coors_3d = []
            for item in items:
                coors_3d.append([item.x, item.y, item.z])
            if len(coors_3d) > 0:
                if type_poly == "polyline":
                    if type_ not in type_polylines:
                        type_polylines[type_] = []
                    type_polylines[type_].append(coors_3d)
                else:
                    if type_ not in type_polygons:
                        type_polygons[type_] = []
                    type_polygons[type_].append(coors_3d)

        for k, v in type_polygons.items():
            if k in color_table:
                c = tuple(color_table[k])
                for i in v:
                    MotionDataset.draw_polygon_to_image(
                        i, draw, image_from_world, max_distance, c, pen_width)

        for k, v in type_polylines.items():
            if k in color_table:
                c = tuple(color_table[k])
                for i in v:
                    MotionDataset.draw_line_to_image(
                        i, draw, image_from_world, max_distance, c, pen_width)

        return image

    @staticmethod
    def get_3dbox_bev_image(laser_labels, _3dbox_bev_settings: dict):
        # options
        pen_width = _3dbox_bev_settings.get("pen_width", 2)
        bev_size = _3dbox_bev_settings.get("bev_size", [640, 640])
        bev_from_ego_transform = _3dbox_bev_settings.get(
            "bev_from_ego_transform",
            MotionDataset.default_bev_from_ego_transform)
        fill_box = _3dbox_bev_settings.get("fill_box", False)
        color_table = _3dbox_bev_settings.get(
            "color_table", MotionDataset.default_3dbox_color_table)
        native_color_table = {
            MotionDataset.box_type_dict[k]: v for k, v in color_table.items()
        }

        corner_templates = _3dbox_bev_settings.get(
            "corner_templates",
            MotionDataset.default_bev_3dbox_corner_template)
        edge_indices = _3dbox_bev_settings.get(
            "edge_indices", MotionDataset.default_bev_3dbox_edge_indices)

        # get the transform from the referenced ego space to the BEV space
        bev_from_ego = np.array(bev_from_ego_transform)

        # draw annotations to the image
        image = Image.new("RGB", bev_size)
        draw = ImageDraw.Draw(image)

        corner_templates_np = np.array(corner_templates).transpose()
        for i in laser_labels:
            category = i.type
            if category in native_color_table:
                pen_color = tuple(native_color_table[category])
                scale = np.diag([i.box.length, i.box.width, i.box.height, 1])
                ego_from_annotation = dwm.datasets.common.get_transform(
                    transforms3d.euler.euler2quat(
                        0, 0, i.box.heading).tolist(),
                    [i.box.center_x, i.box.center_y, i.box.center_z])
                p = bev_from_ego @ ego_from_annotation @ scale @ \
                    corner_templates_np
                if fill_box:
                    draw.polygon(
                        [(p[0, a], p[1, a]) for a, _ in edge_indices],
                        fill=pen_color, width=pen_width)
                else:
                    for a, b in edge_indices:
                        draw.line(
                            (p[0, a], p[1, a], p[0, b], p[1, b]),
                            fill=pen_color, width=pen_width)

        return image

    @staticmethod
    def draw_polygon_bev_to_image(
        polygon: list, draw: ImageDraw, transform: np.array, pen_color: tuple,
        pen_width: int, solid: bool = True
    ):
        if len(polygon) == 0:
            return

        polygon_nodes = np.array([
            [i[0], i[1], 0, 1] for i in polygon
        ], np.float32).transpose()
        p = transform @ polygon_nodes
        draw.polygon(
            [(p[0, i], p[1, i]) for i in range(p.shape[1])],
            fill=pen_color if solid else None,
            outline=None if solid else pen_color, width=pen_width)

    @staticmethod
    def draw_line_bev_to_image(
        line: list, draw: ImageDraw, transform: np.array, pen_color: tuple,
        pen_width: int
    ):
        if len(line) == 0:
            return

        line_nodes = np.array([
            [i[0], i[1], 0, 1] for i in line
        ], np.float32).transpose()
        p = transform @ line_nodes
        for i in range(1, len(line)):
            draw.line(
                (p[0, i - 1], p[1, i - 1], p[0, i], p[1, i]),
                fill=pen_color, width=pen_width)

    @staticmethod
    def get_hdmap_bev_image(map_features, pose, hdmap_bev_settings: dict):
        pen_width = hdmap_bev_settings.get("pen_width", 2)
        bev_size = hdmap_bev_settings.get("bev_size", [640, 640])
        bev_from_ego_transform = hdmap_bev_settings.get(
            "bev_from_ego_transform",
            MotionDataset.default_bev_from_ego_transform)
        color_table = hdmap_bev_settings.get(
            "color_table", MotionDataset.default_hdmap_color_table)

        # get the transform from the referenced ego space to the BEV space
        world_from_ego = np.array(pose.transform, np.float32).reshape(4, 4)
        bev_from_ego = np.array(bev_from_ego_transform, np.float32)
        bev_from_world = bev_from_ego @ np.linalg.inv(world_from_ego)

        # draw map elements to the image
        image = Image.new("RGB", bev_size)
        draw = ImageDraw.Draw(image)

        type_polygons = {}
        type_polylines = {}
        for feat in map_features:
            type_ = feat.WhichOneof('feature_data')
            if (
                type_ not in color_table or
                type_ not in MotionDataset.map_element_type_dict
            ):
                continue

            type_poly = MotionDataset.map_element_type_dict[type_]
            items = getattr(getattr(feat, type_), type_poly)
            coors_3d = []
            for item in items:
                coors_3d.append([item.x, item.y, item.z])
            if len(coors_3d) > 0:
                if type_poly == "polyline":
                    if type_ not in type_polylines:
                        type_polylines[type_] = []
                    type_polylines[type_].append(coors_3d)
                else:
                    if type_ not in type_polygons:
                        type_polygons[type_] = []
                    type_polygons[type_].append(coors_3d)

        for k, v in type_polygons.items():
            if k in color_table:
                c = tuple(color_table[k])
                for i in v:
                    MotionDataset.draw_polygon_bev_to_image(
                        i, draw, bev_from_world, c, pen_width)

        for k, v in type_polylines.items():
            if k in color_table:
                c = tuple(color_table[k])
                for i in v:
                    MotionDataset.draw_line_bev_to_image(
                        i, draw, bev_from_world, c, pen_width)

        return image

    @staticmethod
    def get_image_description(
        image_descriptions: dict, time_list_dict: dict, scene_key: str,
        timestamp: int, camera_id: int
    ):
        nearest_time = dwm.datasets.common.find_nearest(
            time_list_dict[scene_key], timestamp, return_item=True)
        key = "{}|{}|{}".format(scene_key, nearest_time, camera_id)
        return image_descriptions[key]

    def __init__(
        self, fs: fsspec.AbstractFileSystem, info_dict_path: str,
        sequence_length: int, fps_stride_tuples: list,
        sensor_channels: list = ["CAM_FRONT"],
        enable_camera_transforms: bool = False,
        enable_ego_transforms: bool = False,
        _3dbox_image_settings: dict = None, hdmap_image_settings: dict = None,
        _3dbox_bev_settings: dict = None, hdmap_bev_settings: dict = None,
        image_description_settings: dict = None,
        stub_key_data_dict: dict = None,
    ):
        self.fs = fs
        self.sequence_length = sequence_length
        self.fps_stride_tuples = fps_stride_tuples
        self.sensor_channels = sensor_channels
        self.enable_camera_transforms = enable_camera_transforms
        self.enable_ego_transforms = enable_ego_transforms
        self._3dbox_image_settings = _3dbox_image_settings
        self.hdmap_image_settings = hdmap_image_settings
        self._3dbox_bev_settings = _3dbox_bev_settings
        self.hdmap_bev_settings = hdmap_bev_settings
        self.image_description_settings = image_description_settings

        self.stub_key_data_dict = {} if stub_key_data_dict is None \
            else stub_key_data_dict

        # key: context_name, value: (micro_time, length, offset)
        with open(info_dict_path, 'r') as f:
            scene_sample_info = json.load(f)

        # prepend first sample info of the scene to each samples for HD map
        self.sample_info_dict = dwm.common.SerializedReadonlyDict({
            "{};{}".format(scene, sample_info[0]):
            sample_info_list[0] + sample_info
            for scene, sample_info_list in scene_sample_info.items()
            for sample_info in sample_info_list
        })

        self.items = dwm.common.SerializedReadonlyList([
            {"segment": segment, "fps": fps, "scene": scene}
            for scene, sample_info_list in scene_sample_info.items()
            for fps, stride in self.fps_stride_tuples
            for segment in MotionDataset.enumerate_segments(
                sample_info_list, self.sequence_length, fps, stride)
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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        segment = [
            self.sample_info_dict["{};{}".format(item["scene"], i)]
            for i in item["segment"]
        ]

        result = {
            "fps": torch.tensor(item["fps"]).float(),
            "pts": torch.tensor([
                [(i[3] - segment[0][3]) / 1000] * len(self.sensor_channels)
                for i in segment
            ], dtype=torch.float32)
        }

        scene_frame = waymo_pb.Frame()
        frames = [waymo_pb.Frame() for _ in segment]
        scene_path = "segment-{}_with_camera_labels.tfrecord".format(
            item["scene"])
        with self.fs.open(scene_path, "rb") as f:
            for i_id, i in enumerate(segment):
                _, length0, offset0, _, length, offset = i
                if i_id == 0 and (
                    self.hdmap_image_settings is not None or
                    self.hdmap_bev_settings is not None
                ):
                    f.seek(offset0)
                    scene_frame.ParseFromString(f.read(length0))

                f.seek(offset)
                frames[i_id].ParseFromString(f.read(length))

        images, lidar_points = [], []
        for i_id, i in enumerate(segment):
            images_i, lidar_points_i = \
                MotionDataset.get_images_and_lidar_points(
                    self.sensor_channels, frames[i_id])
            if len(images_i) > 0:
                images.append(images_i)
            if len(lidar_points_i) > 0:
                lidar_points.append(lidar_points_i[0])

        if len(images) > 0:
            result["images"] = images

        if len(lidar_points) > 0:
            result["lidar_points"] = lidar_points

        if self.enable_camera_transforms:
            if "images" in result:
                camera_calibrations = [
                    [
                        MotionDataset.find_by_name(
                            i.context.camera_calibrations,
                            MotionDataset.sensor_name_id_dict[j])
                        for j in self.sensor_channels
                        if j.startswith("CAM")
                    ]
                    for i in frames
                ]

                ec_inv = torch.linalg.inv(
                    torch.tensor(
                        MotionDataset.extrinsic_correction,
                        dtype=torch.float32))
                result["camera_transforms"] = torch.stack([
                    torch.stack([
                        torch.tensor(
                            j.extrinsic.transform,
                            dtype=torch.float32).reshape(4, 4) @ ec_inv
                        for j in i
                    ])
                    for i in camera_calibrations
                ])
                result["camera_intrinsics"] = torch.stack([
                    torch.stack([
                        dwm.datasets.common.make_intrinsic_matrix(
                            j.intrinsic[0:2], j.intrinsic[2:4], "pt")
                        for j in i
                    ])
                    for i in camera_calibrations
                ])
                result["image_size"] = torch.stack([
                    torch.stack([
                        torch.tensor([j.width, j.height], dtype=torch.long)
                        for j in i
                    ])
                    for i in camera_calibrations
                ])

            if "lidar_points" in result:
                result["lidar_transforms"] = torch.stack([
                    torch.stack([
                        torch.eye(4)
                        for j in self.sensor_channels
                        if j.startswith("LIDAR")
                    ])
                    for _ in frames
                ])

        if self.enable_ego_transforms:
            result["ego_transforms"] = torch.stack([
                torch.stack([
                    torch.tensor(
                        i.pose.transform, dtype=torch.float32).reshape(4, 4)
                    for _ in self.sensor_channels
                ])
                for i in frames
            ])

        if self._3dbox_image_settings is not None:
            result["3dbox_images"] = [
                [
                    MotionDataset.get_3dbox_image(
                        i.laser_labels,
                        MotionDataset.find_by_name(
                            i.context.camera_calibrations,
                            MotionDataset.sensor_name_id_dict[j]),
                        self._3dbox_image_settings)
                    for j in self.sensor_channels
                    if j.startswith("CAM")
                ]
                for i in frames
            ]

        if self.hdmap_image_settings is not None:
            result["hdmap_images"] = [
                [
                    MotionDataset.get_hdmap_image(
                        scene_frame.map_features,
                        MotionDataset.find_by_name(
                            i.context.camera_calibrations,
                            MotionDataset.sensor_name_id_dict[j]),
                        i.pose, self.hdmap_image_settings)
                    for j in self.sensor_channels
                    if j.startswith("CAM")
                ]
                for i in frames
            ]

        if self._3dbox_bev_settings is not None:
            result["3dbox_bev_images"] = [
                MotionDataset.get_3dbox_bev_image(
                    i.laser_labels, self._3dbox_bev_settings)
                for i in frames
                for j in self.sensor_channels
                if j.startswith("LIDAR")
            ]

        if self.hdmap_bev_settings is not None:
            result["hdmap_bev_images"] = [
                MotionDataset.get_hdmap_bev_image(
                    scene_frame.map_features, i.pose, self.hdmap_bev_settings)
                for i in frames
                for j in self.sensor_channels
                if j.startswith("LIDAR")
            ]

        if self.image_description_settings is not None:
            image_captions = [
                dwm.datasets.common.align_image_description_crossview([
                    MotionDataset.get_image_description(
                        self.image_descriptions, self.time_list_dict,
                        item["scene"], i[3],
                        MotionDataset.sensor_name_id_dict[j])
                    for j in self.sensor_channels
                    if "LIDAR" not in j
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
