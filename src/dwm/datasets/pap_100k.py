import av
import dwm.common
import dwm.datasets.common
import fractions
import fsspec
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
import transforms3d


class MotionDataset(torch.utils.data.Dataset):

    default_3dbox_color_table = {
        "PEDESTRIAN_NORMAL": (255, 0, 0),
        "PEDESTRIAN_TRAFFIC_POLICE": (255, 0, 0),
        "CYCLIST_BICYCLE": (0, 255, 0),
        "CYCLIST_MOTOR": (0, 255, 0),
        "VEHICLE_BUS": (0, 0, 255),
        "VEHICLE_CAR": (0, 0, 255),
        "VEHICLE_MULTI_STAGE": (0, 0, 255),
        "VEHICLE_PICKUP": (0, 0, 255),
        "VEHICLE_POLICE": (0, 0, 255),
        "VEHICLE_SPECIAL": (0, 0, 255),
        "VEHICLE_SUV": (0, 0, 255),
        "VEHICLE_TRIKE": (0, 0, 255),
        "VEHICLE_TRUCK": (0, 0, 255)
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

    @staticmethod
    def split_scene_info(scene_info: dict):
        return {
            "camera": scene_info["cameras"]
            if "cameras" in scene_info else None,
            "lidar": scene_info["lidars"]
            if "lidars" in scene_info else None
        }

    @staticmethod
    def enumerate_segments(
        scene_info: dict, sensor_channels: list, sequence_length: int, fps,
        stride
    ):
        split_scene_info = MotionDataset.split_scene_info(scene_info)
        typed_sensor_channels = [i.split("/") for i in sensor_channels]
        time_range = [
            max([
                split_scene_info[i_t][i_s]["time_range"][0]
                for i_t, i_s in typed_sensor_channels
                if split_scene_info[i_t] is not None
            ]),
            min([
                split_scene_info[i_t][i_s]["time_range"][1]
                for i_t, i_s in typed_sensor_channels
                if split_scene_info[i_t] is not None
            ])
        ]
        duration_ms = sequence_length * 1000 // fps
        stride_ms = int(round(stride * 1000))
        for i in range(time_range[0], time_range[1] - duration_ms, stride_ms):
            yield (i, i + duration_ms)

    @staticmethod
    def read_images_with_pts_from_video(
        fs: fsspec.AbstractFileSystem, item: dict, scene_camera_info: dict,
        sequence_length: int, camera_channel: str,
        seek_correction_offset_ms: int
    ):
        segment = item["segment"]
        video_path = "{}/camera/{}.mp4".format(item["scene"], camera_channel)
        time_range = scene_camera_info[camera_channel]["time_range"]

        start_pts_ms = segment[0] - time_range[0] - 500 // item["fps"] + \
            seek_correction_offset_ms
        stop_pts_ms = segment[1] - time_range[0] + 499 // item["fps"]
        tb_ms = fractions.Fraction(1, 1000)
        frames = []
        with fs.open(video_path) as f:
            with av.open(f) as container:
                stream = container.streams.video[0]
                start_pts_stream = int(start_pts_ms * tb_ms / stream.time_base)
                container.seek(start_pts_stream, stream=stream)
                for i in container.decode(stream):

                    # rescale the pts to the unit of ms
                    i.pts = int(i.pts * stream.time_base / tb_ms)

                    if i.pts < start_pts_ms:
                        continue

                    elif i.pts >= stop_pts_ms:
                        break

                    frames.append(i)

        if len(frames) == 0:
            print(
                "Data item WARNING: Name {}, time {}, no frame".format(
                    video_path, start_pts_ms))
            frame = av.VideoFrame.from_image(
                Image.new("RGB", (1280, 720), (128, 128, 128)))
            frame.pts = start_pts_ms
            frame.time_base = tb_ms
            frames.append(frame)

        pts_list = [i.pts for i in frames]
        actual_pts_indices = [
            dwm.datasets.common.find_nearest(
                pts_list, start_pts_ms + i * 1000 // item["fps"]
            )
            for i in range(sequence_length)
        ]
        actual_images = [
            frames[i].to_image() for i in actual_pts_indices
        ]
        actual_pts_list = [
            pts_list[i] + time_range[0] for i in actual_pts_indices
        ]
        return actual_images, actual_pts_list

    @staticmethod
    def get_3dbox_image(
        scene_name: str, fs: fsspec.AbstractFileSystem, timestamp: int,
        camera_info: dict, camera_channel: str, _3dbox_time_list: list,
        annotation_cache: dict, _3dbox_image_settings: dict
    ):
        # options
        pen_width = _3dbox_image_settings.get("pen_width", 3)
        color_table = _3dbox_image_settings.get(
            "color_table", MotionDataset.default_3dbox_color_table)
        corner_templates = _3dbox_image_settings.get(
            "corner_templates", MotionDataset.default_3dbox_corner_template)
        edge_indices = _3dbox_image_settings.get(
            "edge_indices", MotionDataset.default_3dbox_edge_indices)

        annotation_index = dwm.datasets.common.find_nearest(
            _3dbox_time_list, timestamp)
        if annotation_index not in annotation_cache:
            path = "{}/{:04d}.json".format(scene_name, annotation_index)
            with fs.open(path, encoding="utf-8") as f:
                annotation_cache[annotation_index] = json.load(f)

        annotation = annotation_cache[annotation_index]

        # extra parameters for distortion
        camera = annotation["sensors"]["cameras"][camera_channel]

        # get the transform from the ego space to the image space
        image_resize_transform = np.diag([
            i_0 / i_1 for i_0, i_1 in zip(
                camera_info["frame_size"], camera_info["origin_frame_size"])
        ] + [1, 1])
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = np.array(camera_info["intrinsic"]).reshape(3, 3)
        ego_from_camera = np.array(camera_info["extrinsic"]).reshape(4, 4)
        image_from_ego = image_resize_transform @ intrinsic @ \
            np.linalg.inv(ego_from_camera)

        # draw annotations to the image
        def list_annotation():
            if "Objects" in annotation and annotation["Objects"] is not None:
                for i in annotation["Objects"]:
                    yield i

        def get_world_transform(i):
            scale = np.diag(i["bbox3d"][3:6] + [1])
            ego_from_annotation = dwm.datasets.common.get_transform(
                transforms3d.euler.euler2quat(*i["bbox3d"][6:9]).tolist(),
                i["bbox3d"][:3])
            return ego_from_annotation @ scale

        image = Image.new("RGB", tuple(camera_info["frame_size"]))
        draw = ImageDraw.Draw(image)
        dwm.datasets.common.draw_3dbox_image(
            draw, image_from_ego, list_annotation, get_world_transform,
            lambda i: i["label"], pen_width, color_table, corner_templates,
            edge_indices)

        return image

    @staticmethod
    def get_image_description(
        image_descriptions: dict, time_list_dict: dict, scene: str,
        camera: str, time: int
    ):
        scene_camera = "{}|{}".format(scene, camera)
        time_list = time_list_dict[scene_camera]
        i = dwm.datasets.common.find_nearest(time_list, time)
        nearest_time = time_list[i]
        return image_descriptions["{}|{}".format(scene_camera, nearest_time)]

    def __init__(
        self, scene_indices_path: str, fs: fsspec.AbstractFileSystem,
        sequence_length: int, fps_stride_tuples: list,
        sensor_channels: list = ["camera/center_camera_fov30"],
        enable_camera_transforms: bool = False,
        enable_ego_transforms: bool = False, _3dbox_image_settings=None,
        image_description_settings=None, seek_correction_offset_ms: int = 0,
        stub_key_data_dict=None
    ):
        with open(scene_indices_path, "r", encoding="utf-8") as f:
            self.scene_indices = json.load(f)

        self.fs = fs
        self.sequence_length = sequence_length
        self.fps_stride_tuples = fps_stride_tuples
        self.sensor_channels = sensor_channels
        self.enable_camera_transforms = enable_camera_transforms
        self.enable_ego_transforms = enable_ego_transforms
        self._3dbox_image_settings = _3dbox_image_settings
        self.image_description_settings = image_description_settings
        self.seek_correction_offset_ms = seek_correction_offset_ms
        self.stub_key_data_dict = stub_key_data_dict

        self.scene_info_dict = {}
        for i in self.scene_indices:
            scene_token = i["raw_data"]
            scene_root = scene_token.replace("s3://", "")
            scene_info_path = "{}/info.json".format(scene_root)
            with fs.open(scene_info_path, "r", encoding="utf-8") as f:
                self.scene_info_dict[scene_root] = json.load(f)
                self.scene_info_dict[scene_root].update(i)

        self.items = dwm.common.SerializedReadonlyList([
            {"segment": segment, "fps": fps, "scene": scene_root}
            for scene_root, scene_info in self.scene_info_dict.items()
            for fps, stride in self.fps_stride_tuples
            for segment in MotionDataset.enumerate_segments(
                scene_info, self.sensor_channels, self.sequence_length, fps,
                stride)
        ])

        if _3dbox_image_settings is not None:
            with open(
                _3dbox_image_settings["time_list_dict_path"], "r",
                encoding="utf-8"
            ) as f:
                self._3dbox_time_list_dict = json.load(f)

        if image_description_settings is not None:
            with open(
                image_description_settings["path"], "r", encoding="utf-8"
            ) as f:
                self.image_descriptions = json.load(f)

            # time list dict is required for sparse labelling of captions
            with open(
                image_description_settings["time_list_dict_path"], "r",
                encoding="utf-8"
            ) as f:
                self.caption_time_list_dict = json.load(f)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        scene_info = self.scene_info_dict[item["scene"]]
        result = {
            "fps": torch.tensor(item["fps"], dtype=torch.float32)
        }

        split_scene_info = MotionDataset.split_scene_info(scene_info)
        typed_sensor_channels = [i.split("/") for i in self.sensor_channels]
        pts = [[] for _ in range(self.sequence_length)]
        images = [[] for _ in range(self.sequence_length)]
        lidar_points = []
        for i_t, i_c in typed_sensor_channels:
            if i_t == "lidar":
                # TODO: implement the actual logic in future
                # a fake loading here

                for j in range(self.sequence_length):
                    lidar_points.append(
                        torch.zeros((1, 3), dtype=torch.float32))

                for j in range(self.sequence_length):
                    pts[j].append(0)

            if i_t == "camera":
                images_i, pts_i = MotionDataset\
                    .read_images_with_pts_from_video(
                        self.fs, item, split_scene_info["camera"],
                        self.sequence_length, i_c,
                        self.seek_correction_offset_ms)

                for j in range(self.sequence_length):
                    images[j].append(images_i[j])
                    pts[j].append(pts_i[j])

        result["pts"] = torch.tensor(
            [[j - pts[0][0] for j in i] for i in pts], dtype=torch.float32)
        if all([len(i) > 0 for i in images]):
            result["images"] = images

        if len(lidar_points) > 0:
            result["lidar_points"] = lidar_points  # [sequence_length]

        if self.enable_camera_transforms is not None:
            if "images" in result:
                result["camera_transforms"] = torch.stack([
                    torch.stack([
                        torch.tensor(
                            split_scene_info[j_t][j_c]["extrinsic"],
                            dtype=torch.float32).reshape(4, 4)
                        for j_t, j_c in typed_sensor_channels
                        if j_t == "camera"
                    ])
                    for _ in range(self.sequence_length)
                ])
                result["camera_intrinsics"] = torch.stack([
                    torch.stack([
                        torch.tensor(
                            split_scene_info[j_t][j_c]["intrinsic"],
                            dtype=torch.float32).reshape(3, 3)
                        for j_t, j_c in typed_sensor_channels
                        if j_t == "camera"
                    ])
                    for _ in range(self.sequence_length)
                ])
                result["image_size"] = torch.stack([
                    torch.stack([
                        torch.tensor(
                            split_scene_info[j_t][j_c]["origin_frame_size"],
                            dtype=torch.long)
                        for j_t, j_c in typed_sensor_channels
                        if j_t == "camera"
                    ])
                    for _ in range(self.sequence_length)
                ])

            if "lidar_points" in result:
                # TODO
                result["lidar_transforms"] = torch.stack([
                    torch.stack([
                        torch.eye(4, dtype=torch.float32)
                        for j_t, _ in typed_sensor_channels
                        if j_t == "lidar"
                    ])
                    for _ in range(self.sequence_length)
                ])

        if self.enable_ego_transforms:
            # TODO
            result["ego_transforms"] = torch.stack([
                torch.stack([
                    torch.eye(4, dtype=torch.float32)
                    for _ in self.sensor_channels
                ])
                for _ in range(self.sequence_length)
            ])

        if self._3dbox_image_settings is not None:
            _3dbox_time_list = self._3dbox_time_list_dict[
                scene_info["scene_name"]]
            annotation_cache = {}
            result["3dbox_images"] = [
                [
                    MotionDataset.get_3dbox_image(
                        scene_info["scene_name"],
                        self._3dbox_image_settings["fs"], pts_j,
                        split_scene_info[j_t][j_c], j_c, _3dbox_time_list,
                        annotation_cache, self._3dbox_image_settings)
                    for (j_t, j_c), pts_j in zip(typed_sensor_channels, pts_i)
                    if j_t == "camera"
                ]
                for pts_i in pts
            ]

        if self.image_description_settings is not None:
            image_captions = [
                dwm.datasets.common.align_image_description_crossview([
                    MotionDataset.get_image_description(
                        self.image_descriptions, self.caption_time_list_dict,
                        item["scene"], j_c,
                        item["segment"][0] -
                        scene_info["cameras"][j_c]["time_range"][0] +
                        i * 1000 // item["fps"])
                    for j_t, j_c in typed_sensor_channels
                    if j_t == "camera"
                ], self.image_description_settings)
                for i in range(self.sequence_length)
            ]
            result["image_description"] = [
                [
                    dwm.datasets.common.make_image_description_string(
                        j, self.image_description_settings)
                    for j in i
                ]
                for i in image_captions
            ]

        dwm.datasets.common.add_stub_key_data(self.stub_key_data_dict, result)

        return result
