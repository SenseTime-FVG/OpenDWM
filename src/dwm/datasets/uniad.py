import bisect
import dwm.common
import dwm.datasets.common
import fsspec
import json
import numpy as np
import pickle
from PIL import Image, ImageDraw
import pypcd4
import torch
import transforms3d


class MotionDataset(torch.utils.data.Dataset):
    """The motion frame data of the uniad dataset for video clips.

    Args:
        fs (fsspec.AbstractFileSystem): The file system for the dataset table
            and content files.
        sequence_length (int): The frame count of each video clips extracted
            from the dataset, also the "T" of the video tensor shape
            [T, C, H, W]. For the multimodal mode, the video tensor is
            returned in the shape of [T, V, C, H, W].
        fps_stride_tuples (list): The list of tuples in the form of
            (FPS, stride). If the FPS > 0, the stride is the second count
            of the beginning time between 2 adjacent video clips, else the
            stride is the index count of the beginning between 2 adjacent
            video clips.
        sensor_channels (list): The string list of required views in "lidar",
            "left_front_camera", "center_camera_fov120", "right_front_camera",
            "right_rear_camera", "rear_camera", "left_rear_camera".
        enable_camera_transforms (bool): The flag to return the 4x4 transform
            matrix to the world from camera (by the key "camera_transforms")
            and from lidar (by the key "lidar_transforms"), the the 3x3 camera
            intrinsic transform matrix (by the key "camera_intrinsics"), and
            the image size (tuple of (width, height), by the key "image_size").
        enable_ego_transforms (bool): The flag to return the 4x4 transform
            matrix to the world of the ego vehicle by the key "ego_transforms".
        ego_calibration_transform (list): The 4x4 transform matrix from origin
            ego space to the expected ego space.
        _3dbox_image_settings (dict): The settings to return and control the 3D
            box images by the key "3dbox_images".
        hdmap_image_settings (dict): The settings to return and control the HD
            map images by the key "hdmap_images".
        _3dbox_bev_settings (dict): The settings to return and control the 3D
            box BEV images by the key "3dbox_bev_images".
        hdmap_bev_settings (dict): The settings to return and control the HD
            map BEV images by the key "hdmap_bev_images".
        stub_key_data_dict (dict): The dict of stub key and data, mainly for
            aligning other dataset with missing key and data in the current
            dataset.
    """

    default_ex2ey = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    default_ego_calibration = [
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    default_3dbox_color_table = {
        "pedestrian": (255, 0, 0),
        "cyclist": (0, 255, 0),
        "car": (0, 0, 255),
        "truck": (0, 0, 255)
    }
    default_hdmap_color_table = {
        "kWHITE.kSOLID": (0, 255, 0),
        "kWHITE.kDASHED": (0, 255, 0),
        "kYELLOW.kSOLID": (0, 255, 0),
        "kYELLOW.kDASHED": (0, 255, 0),
        "drivable": (0, 0, 255)
    }
    default_3dbox_corner_template = [
        [-0.5, -0.5, 0, 1], [-0.5, -0.5, 1, 1],
        [-0.5, 0.5, 0, 1], [-0.5, 0.5, 1, 1],
        [0.5, -0.5, 0, 1], [0.5, -0.5, 1, 1],
        [0.5, 0.5, 0, 1], [0.5, 0.5, 1, 1]
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
        [0.5, -0.5, 0, 1], [0.5, 0.5, 0, 1],
    ]
    default_bev_3dbox_edge_indices = [(0, 2), (2, 3), (3, 1), (1, 0)]
    default_image_description_keys = [
        "time", "weather", "environment", "objects", "image_description"
    ]

    @staticmethod
    def get_transform(
        rotation: list, translation: list, left_transform=None,
        right_transform=None, output_type: str = "np"
    ):
        t = dwm.datasets.common.get_transform(rotation, translation)
        if left_transform is not None:
            t = left_transform @ t

        if right_transform is not None:
            t = t @ right_transform

        if output_type == "np":
            return t
        elif output_type == "pt":
            return torch.tensor(t, dtype=torch.float32)
        else:
            raise Exception("Unknown output type of the get_transform()")

    @staticmethod
    def enumerate_multimodal_segments(
        sample_infos: list, sample_indices: list, sequence_length: int, fps,
        stride
    ):
        # stride > 0:
        #   * FPS == 0: offset between segment beginings are by index.
        #   * FPS > 0: offset between segment beginings are by second.

        if fps == 0:
            # frames are extracted by the index.
            for t in range(0, len(sample_indices), max(1, stride)):
                if t + sequence_length <= len(sample_indices):
                    yield [
                        sample_indices[t + i] for i in range(sequence_length)
                    ]

        else:
            # frames are extracted by the timestamp.
            def enumerate_begin_time(
                sample_infos: list, sample_indices: list, sequence_duration,
                stride
            ):
                s = sample_infos[sample_indices[-1]]["timestamp"] - \
                    sequence_duration
                t = sample_infos[sample_indices[0]]["timestamp"]
                while t <= s:
                    yield t
                    t += stride

            timestamp_list = [
                sample_infos[i]["timestamp"] for i in sample_indices
            ]
            for t in enumerate_begin_time(
                sample_infos, sample_indices, sequence_length / fps, stride
            ):
                # find the indices of the first frame matching the given
                # timestamp
                yield [
                    sample_indices[
                        dwm.datasets.common.find_nearest(
                            timestamp_list, t + i / fps)
                    ]
                    for i in range(sequence_length)
                ]

    @staticmethod
    def get_images_and_lidar_points(
        fs: fsspec.AbstractFileSystem, sample: dict, sample_info: dict,
        sensor_channels: list
    ):
        images = []
        lidar_points = []
        for s in sensor_channels:
            if s == "lidar":
                if sample_info["lidar_path"].endswith("pkl"):
                    lidar_path = sample_info["lidar_path"][5:]
                    with fs.open(lidar_path) as f:
                        points = pickle.load(f)

                    np_points = np.array(points)

                else:
                    lidar_path_prefix = sample["ceph_root"][5:]
                    second_last_slash_index = lidar_path_prefix.rfind(
                        "/", 0, lidar_path_prefix.rfind("/"))
                    lidar_path_prefix = lidar_path_prefix[:second_last_slash_index + 1]
                    lidar_path = lidar_path_prefix + sample_info["lidar_path"]
                    with fs.open(lidar_path) as f:
                        points = pypcd4.PointCloud.from_fileobj(f)

                    np_points = points.numpy()[:, :3]

                lidar_points.append(
                    torch.tensor(np_points, dtype=torch.float32))

            else:
                with fs.open(sample["cams"][s]["data_path"][5:]) as f:
                    image = Image.open(f)
                    image.load()

                images.append(image)

        return images, lidar_points

    @staticmethod
    def get_image_description(
        fs: fsspec.AbstractFileSystem, token: str, sensor: str
    ):
        path = "caption/{}${}.json".format(token, sensor)
        with fs.open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def get_lidar_image(
        lidar_points: np.array, sample_info: dict, sample_data: dict,
        size: tuple
    ):
        # get the transform from the ego space to the image space
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = np.array(sample_data["cam_intrinsic"])

        ego_from_camera = dwm.datasets.common.get_transform(
            sample_data["sensor2ego_rotation"],
            sample_data["sensor2ego_translation"])
        ego_from_lidar = np.array(
            sample_info.get("ex2ey", MotionDataset.default_ex2ey))
        image_from_lidar = intrinsics @ np.linalg.solve(
            ego_from_camera, ego_from_lidar)

        # draw annotations to the image
        image = Image.new("RGBA", size)
        draw = ImageDraw.Draw(image)

        lidar_points_np = np.concatenate([
            lidar_points.numpy(),
            np.ones((lidar_points.shape[0], 1))
        ], -1).transpose()
        projected_lidar_points = image_from_lidar @ lidar_points_np
        p = projected_lidar_points[:2] / projected_lidar_points[2]

        for i in range(p.shape[1]):
            rz = projected_lidar_points[2, i] / 50.0
            if rz > 0.01 and rz < 1:
                c = int(rz * 255)
                draw.ellipse(
                    (p[0, i] - 2, p[1, i] - 2, p[0, i] + 2, p[1, i] + 2),
                    fill=(c, c, c))

        return image

    @staticmethod
    def get_3dbox_image(
        sample: dict, sample_data: dict, size: tuple,
        _3dbox_image_settings: dict
    ):
        # options
        pen_width = _3dbox_image_settings.get("pen_width", 5)
        color_table = _3dbox_image_settings.get(
            "color_table", MotionDataset.default_3dbox_color_table)
        corner_templates = _3dbox_image_settings.get(
            "corner_templates", MotionDataset.default_3dbox_corner_template)
        edge_indices = _3dbox_image_settings.get(
            "edge_indices", MotionDataset.default_3dbox_edge_indices)

        # get the transform from the ego space to the image space
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = np.array(sample_data["cam_intrinsic"])

        ego_from_camera = dwm.datasets.common.get_transform(
            sample_data["sensor2ego_rotation"],
            sample_data["sensor2ego_translation"])
        image_from_ego = intrinsics @ np.linalg.inv(ego_from_camera)

        # draw annotations to the image
        image = Image.new("RGB", size)
        draw = ImageDraw.Draw(image)

        corner_templates_np = np.array(corner_templates).transpose()
        for box, category in zip(sample["gt_boxes"], sample["gt_names"]):
            if category in color_table:
                pen_color = tuple(color_table[category])
                scale = np.diag(box[3:6] + [1])
                ego_from_annotation = dwm.datasets.common.get_transform(
                    transforms3d.euler.euler2quat(0, 0, -box[-1]).tolist(),
                    box[:3])
                p = image_from_ego @ ego_from_annotation @ scale @ \
                    corner_templates_np
                for a, b in edge_indices:
                    xy = dwm.datasets.common.project_line(p[:, a], p[:, b])
                    if xy is not None:
                        draw.line(xy, fill=pen_color, width=pen_width)

        return image

    @staticmethod
    def get_hdmap_image(
        sample: dict, sample_data: dict, size: tuple,
        hdmap_image_settings: dict
    ):
        # options
        max_distance = hdmap_image_settings.get("max_distance", 65.0)
        pen_width = hdmap_image_settings.get("pen_width", 5)
        color_table = hdmap_image_settings.get(
            "color_table", MotionDataset.default_hdmap_color_table)

        # get the transform from the ego (map) space to the image space
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = np.array(sample_data["cam_intrinsic"])

        ego_from_camera = dwm.datasets.common.get_transform(
            sample_data["sensor2ego_rotation"],
            sample_data["sensor2ego_translation"])
        image_from_ego = intrinsics @ np.linalg.inv(ego_from_camera)

        # draw map elements to the image:
        image = Image.new("RGB", size)
        draw = ImageDraw.Draw(image)

        # NOTE: the UniAD drivable area is not supported for camera condition

        map_keys = ["key_points", "colors", "linetypes"]
        if all([i in sample["maps"] for i in map_keys]):
            map_item_list = [sample["maps"][i] for i in map_keys]
            for points, color, linetype in zip(*map_item_list):
                category = "{}.{}".format(color, linetype)
                if category in color_table:
                    pen_color = tuple(color_table[category])
                    line_nodes = np.array(
                        [i + [0, 1] for i in points]).transpose()
                    p = image_from_ego @ line_nodes
                    for i in range(1, p.shape[1]):
                        xy = dwm.datasets.common.project_line(
                            p[:, i - 1], p[:, i], far_z=max_distance)
                        if xy is not None:
                            draw.line(xy, fill=pen_color, width=pen_width)

        return image

    @staticmethod
    def get_3dbox_bev_image(
        sample: dict, ego_calibration: list, _3dbox_bev_settings: dict
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

        # get the transform from the ego space to the BEV space
        bev_from_ego = np.array(bev_from_ego_transform) @ \
            np.array(ego_calibration)

        # draw annotations to the image
        image = Image.new("RGB", tuple(bev_size))
        draw = ImageDraw.Draw(image)

        corner_templates_np = np.array(corner_templates).transpose()
        for box, category in zip(sample["gt_boxes"], sample["gt_names"]):
            if category in color_table:
                pen_color = tuple(color_table[category])
                scale = np.diag(box[3:6] + [1])
                ego_from_annotation = dwm.datasets.common.get_transform(
                    transforms3d.euler.euler2quat(0, 0, -box[-1]).tolist(),
                    box[:3])
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
    def get_hdmap_bev_image(
        sample: dict, ego_calibration: list, hdmap_bev_settings: dict
    ):
        # options
        pen_width = hdmap_bev_settings.get("pen_width", 2)
        bev_size = hdmap_bev_settings.get("bev_size", [640, 640])
        bev_from_ego_transform = hdmap_bev_settings.get(
            "bev_from_ego_transform",
            MotionDataset.default_bev_from_ego_transform)
        color_table = hdmap_bev_settings.get(
            "color_table", MotionDataset.default_hdmap_color_table)

        # get the transform from the ego space to the BEV space
        bev_from_ego = np.array(bev_from_ego_transform) @ \
            np.array(ego_calibration)

        # draw map elements to the image:
        image = Image.new("RGB", tuple(bev_size))
        draw = ImageDraw.Draw(image)

        if "drivable" in color_table and "drivable" in sample:
            for points in sample["drivable"]:
                polygon_nodes = np.array(
                    [i + [0, 1] for i in points]).transpose()
                p = bev_from_ego @ polygon_nodes
                draw.polygon(
                    [(p[0, i], p[1, i]) for i in range(p.shape[1])],
                    fill=color_table["drivable"])

        map_keys = ["key_points", "colors", "linetypes"]
        if all([i in sample["maps"] for i in map_keys]):
            map_item_list = [sample["maps"][i] for i in map_keys]
            for points, color, linetype in zip(*map_item_list):
                category = "{}.{}".format(color, linetype)
                if category in color_table:
                    pen_color = tuple(color_table[category])
                    line_nodes = np.array(
                        [i + [0, 1] for i in points]).transpose()
                    p = bev_from_ego @ line_nodes
                    for i in range(1, p.shape[1]):
                        xy = (p[0, i - 1], p[1, i - 1], p[0, i], p[1, i])
                        draw.line(xy, fill=pen_color, width=pen_width)

        return image

    def __init__(
        self, pkl_path_list: str, fs: fsspec.AbstractFileSystem,
        sequence_length: int, fps_stride_tuples: list,
        sensor_channels: list = ["center_camera_fov120"],
        enable_camera_transforms: bool = False,
        enable_ego_transforms: bool = False,
        ego_calibration_transform: list = None,
        _3dbox_image_settings: dict = None,
        hdmap_image_settings: dict = None,
        _3dbox_bev_settings: dict = None,
        hdmap_bev_settings: dict = None,
        image_description_settings: dict = None,
        stub_key_data_dict: dict = None,
        filter_dict=None,
    ):
        self.fs = fs
        self.sequence_length = sequence_length
        self.fps_stride_tuples = fps_stride_tuples
        self.sensor_channels = sensor_channels
        self.enable_camera_transforms = enable_camera_transforms
        self.enable_ego_transforms = enable_ego_transforms
        self.ego_calibration = MotionDataset.default_ego_calibration if \
            ego_calibration_transform is None else ego_calibration_transform
        self._3dbox_image_settings = _3dbox_image_settings
        self.hdmap_image_settings = hdmap_image_settings
        self._3dbox_bev_settings = _3dbox_bev_settings
        self.hdmap_bev_settings = hdmap_bev_settings
        self.image_description_settings = image_description_settings
        self.pre_ignore_ind = 70

        self.stub_key_data_dict = {} if stub_key_data_dict is None \
            else stub_key_data_dict

        for i, pkl_path in enumerate(pkl_path_list):
            with open(pkl_path, "rb") as f:
                if i == 0:
                    pkl_file = pickle.load(f)
                else:
                    pkl_file["infos"] += pickle.load(f)["infos"]

        self.scene_sample_indices = dict()
        for i, sample in enumerate(pkl_file["infos"]):
            if sample["scene_token"] not in self.scene_sample_indices:
                self.scene_sample_indices[sample["scene_token"]] = []

            self.scene_sample_indices[sample["scene_token"]].append(i)

        self.camera_index = [
            i_id
            for i_id, i in enumerate(self.sensor_channels)
            if i != "lidar"
        ]

        self.items = dwm.common.SerializedReadonlyList([
            {"segment": segment, "fps": fps, "scene_token": scene_token}
            for scene_token, sample_indices in self.scene_sample_indices.items()
            for fps, stride in self.fps_stride_tuples
            for segment in MotionDataset.enumerate_multimodal_segments(
                pkl_file["infos"], sample_indices, self.sequence_length, fps,
                stride)
        ])
        self.sample_infos = dwm.common.SerializedReadonlyList(pkl_file["infos"])
        self.uniad_size = None
        assert not (self.uniad_size is not None and filter_dict is not None)
        if filter_dict is not None:
            filter_ids = self.filter_indexs(filter_dict)["default"]
            self.uniad_size = len(filter_ids)
            self.filter_ids = filter_ids
        else:
            self.filter_ids = None

    def filter_indexs(self, filter_dict=None):
        from collections import defaultdict
        assert self.image_description_settings is not None
        filter_ids = defaultdict(list)
        start, num_valid = 0, 0
        k = "default"
        if self.uniad_size is not None:
            start = len(self.items) - self.uniad_size
        if filter_dict == "auto":
            filter_dict = dict(positive=[
                "intersection", "crossing", "crossroads",           # 路口
            ],
            negative=[
                "foggy"
            ])
        print("Start: ", start)
        for index in range(start, len(self.items)):
            item = self.items[index]
            segment = [self.sample_infos[i] for i in item["segment"]]

            selected_keys = self.image_description_settings.get(
                "selected_keys", MotionDataset.default_image_description_keys)
            stride = self.image_description_settings.get("stride", 5)

            scene_token = segment[0]["scene_token"]
            sample_indices = self.scene_sample_indices[scene_token]
            index_mapping = {
                i: sample_indices[i_id // stride * stride]
                for i_id, i in enumerate(sample_indices)
            }
            image_captions = [
                [
                    MotionDataset.get_image_description(
                        self.fs, self.sample_infos[index_mapping[i]]["token"],
                        j)
                    for j in self.sensor_channels
                    if j != "lidar"
                ]
                for i in item["segment"]
            ]
            image_description = [
                [". ".join([j[k] for k in selected_keys]) for j in i]
                for i in image_captions
            ]
            valid = any([name in image_description[0][i] for i in range(len(image_description[0])) for name in filter_dict['positive']])            # skip t, k
            valid = valid and not any([name in image_description[0][i] for i in range(len(image_description[0])) for name in filter_dict['negative']])
            if "action" in filter_dict:
                assert self.enable_ego_transforms
                samples = []
                for i in segment:
                    sample_path = i["token"][self.pre_ignore_ind:]
                    with self.fs.open(sample_path, "r", encoding="utf-8") as f:
                        samples.append(json.load(f))
                ego_transforms = torch.stack([
                    MotionDataset.get_transform(
                        i["ego2global_rotation"],
                        i["ego2global_translation"],
                        right_transform=np.linalg.inv(
                            np.array(self.ego_calibration)),
                        output_type="pt")
                    .unsqueeze(0)
                    .repeat_interleave(len(self.sensor_channels), 0)
                    for i in samples
                ])
                # frame diff
                num_act = 3
                action = ego_transforms.new_zeros((ego_transforms.shape[0], ego_transforms.shape[1], num_act))
                for fid in range(ego_transforms.shape[0]-1):
                    for vid in range(ego_transforms.shape[1]):
                        tfm_inverse = torch.inverse(ego_transforms[fid+1, vid])
                        diff_pose = tfm_inverse @ ego_transforms[fid, vid]

                        action[fid, vid] = diff_pose[0:num_act, -1]
                valid = valid and (action.max(0)[0].max(0)[0] >=  action.new_tensor(filter_dict["action"])).min().item()

            if valid:
                num_valid += 1
                filter_ids[k].append(index)
            if index % 1000 == 0:
                print("Parse index: ", index, num_valid)
        return filter_ids


    def __len__(self):
        if self.uniad_size is not None:
            return self.uniad_size
        return len(self.items)

    def __getitem__(self, index: int):
        if self.filter_ids is not None:
            index = self.filter_ids[index] % len(self.items)
        item = self.items[index]
        segment = [self.sample_infos[i] for i in item["segment"]]

        result = {
            "fps": torch.tensor(item["fps"], dtype=torch.float32)
        }

        samples = []
        for i in segment:
            sample_path = i["token"][self.pre_ignore_ind:]
            with self.fs.open(sample_path, "r", encoding="utf-8") as f:
                samples.append(json.load(f))

        result["pts"] = torch.tensor([
            [
                (i["timestamp"] - segment[0]["timestamp"]) * 1000
            ] * len(self.sensor_channels)
            for i in segment
        ], dtype=torch.float32)

        images, lidar_points = [], []
        for sample, sample_info in zip(samples, segment):
            images_i, lidar_points_i = \
                MotionDataset.get_images_and_lidar_points(
                    self.fs, sample, sample_info, self.sensor_channels)

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
                            i["cams"][j]["sensor2ego_rotation"],
                            i["cams"][j]["sensor2ego_translation"],
                            left_transform=np.array(self.ego_calibration),
                            output_type="pt")
                        for j in self.sensor_channels
                        if j != "lidar"
                    ])
                    for i in samples
                ])
                result["camera_intrinsics"] = torch.stack([
                    torch.stack([
                        torch.tensor(
                            i["cams"][j]["cam_intrinsic"],
                            dtype=torch.float32)
                        for j in self.sensor_channels
                        if j != "lidar"
                    ])
                    for i in samples
                ])
                result["image_size"] = torch.stack([
                    torch.stack([
                        torch.tensor(list(j.size), dtype=torch.long)
                        for j in i
                    ])
                    for i in result["images"]
                ])

            if "lidar_points" in result:
                result["lidar_transforms"] = torch.stack([
                    torch.stack([
                        MotionDataset.get_transform(
                            i1["lidar2ego_rotation"],
                            i1["lidar2ego_translation"],
                            left_transform=np.array(self.ego_calibration),
                            right_transform=np.array(
                                i0.get("ex2ey", MotionDataset.default_ex2ey)),
                            output_type="pt")
                        for j in self.sensor_channels
                        if j == "lidar"
                    ])
                    for i0, i1 in zip(segment, samples)
                ])

        if self.enable_ego_transforms:
            result["ego_transforms"] = torch.stack([
                MotionDataset.get_transform(
                    i["ego2global_rotation"],
                    i["ego2global_translation"],
                    right_transform=np.linalg.inv(
                        np.array(self.ego_calibration)),
                    output_type="pt")
                .unsqueeze(0)
                .repeat_interleave(len(self.sensor_channels), 0)
                for i in samples
            ])

        if self._3dbox_image_settings is not None:
            if "images" not in result:
                raise Exception(
                    "At least one camera should be in the sensor channel "
                    "list.")

            result["3dbox_images"] = [
                [
                    MotionDataset.get_3dbox_image(
                        i, i["cams"][j], result["images"][i_id][
                            bisect.bisect_left(self.camera_index, j_id)].size,
                        self._3dbox_image_settings)
                    for j_id, j in enumerate(self.sensor_channels)
                    if j != "lidar"
                ]
                for i_id, i in enumerate(samples)
            ]

        if self.hdmap_image_settings is not None:
            if "images" not in result:
                raise Exception(
                    "At least one camera should be in the sensor channel "
                    "list.")

            result["hdmap_images"] = [
                [
                    MotionDataset.get_hdmap_image(
                        i, i["cams"][j], result["images"][i_id][
                            bisect.bisect_left(self.camera_index, j_id)].size,
                        self.hdmap_image_settings)
                    for j_id, j in enumerate(self.sensor_channels)
                    if j != "lidar"
                ]
                for i_id, i in enumerate(samples)
            ]

        if self._3dbox_bev_settings is not None:
            result["3dbox_bev_images"] = [
                MotionDataset.get_3dbox_bev_image(
                    i, self.ego_calibration, self._3dbox_bev_settings)
                for i in samples
                for j in self.sensor_channels
                if j == "lidar"
            ]

        if self.hdmap_bev_settings is not None:
            result["hdmap_bev_images"] = [
                MotionDataset.get_hdmap_bev_image(
                    i, self.ego_calibration, self.hdmap_bev_settings)
                for i in samples
                for j in self.sensor_channels
                if j == "lidar"
            ]

        # debug:
        # lidar_images = [
        #     [
        #         MotionDataset.get_lidar_image(
        #             result["lidar_points"][i_id], segment[i_id],
        #             i["cams"][j], result["images"][i_id][
        #                 bisect.bisect_left(self.camera_index, j_id)].size)
        #         for j_id, j in enumerate(self.sensor_channels)
        #         if j != "lidar"
        #     ]
        #     for i_id, i in enumerate(samples)
        # ]
        # for i_id, i in enumerate(lidar_images[0]):
        #     result["images"][0][i_id].save("image_{}.jpg".format(i_id))
        #     i.save("lidar_{}.png".format(i_id))

        if self.image_description_settings is not None:
            selected_keys = self.image_description_settings.get(
                "selected_keys", MotionDataset.default_image_description_keys)
            stride = self.image_description_settings.get("stride", 5)

            scene_token = segment[0]["scene_token"]
            sample_indices = self.scene_sample_indices[scene_token]
            index_mapping = {
                i: sample_indices[i_id // stride * stride]
                for i_id, i in enumerate(sample_indices)
            }
            image_captions = [
                [
                    MotionDataset.get_image_description(
                        self.fs, self.sample_infos[index_mapping[i]]["token"],
                        j)
                    for j in self.sensor_channels
                    if j != "lidar"
                ]
                for i in item["segment"]
            ]
            result["image_description"] = [
                [". ".join([j[k] for k in selected_keys]) for j in i]
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


if __name__ == "__main__":
    import dwm.fs.czip

    # debug code
    with open("configs/fs/uniad_st_local.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    fs = dwm.fs.czip.CombinedZipFileSystem(
        dwm.common.create_instance_from_config(config),
        paths=["0530_json.zip", "0530/blob_001.zip", "0530/blob_002.zip"],
        enable_cached_info=True)
    dataset = MotionDataset(
        ["/mnt/storage/user/wuzehuan/Downloads/data/uniad/0530_mini_train.pkl"],
        fs, 4, [(0, 1)], [
            "lidar", "left_front_camera", "center_camera_fov120",
            "right_front_camera", "right_rear_camera", "rear_camera",
            "left_rear_camera"
        ],
        enable_camera_transforms=True,
        enable_ego_transforms=True,
        _3dbox_image_settings={
            "edge_indices": [
                (0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (1, 5),
                (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
                (4, 7), (5, 6)
            ]
        },
        hdmap_image_settings={},
        _3dbox_bev_settings={"fill_box": True},
        hdmap_bev_settings={})
    item = dataset[6000]
    print(len(dataset))
