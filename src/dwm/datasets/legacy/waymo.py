import bisect
import dwm.common
import fsspec
import io
import json
import numpy as np
from PIL import Image, ImageDraw
import torch

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils


class MotionDataset(torch.utils.data.Dataset):

    camera_name_id_dict = {
        "FRONT": "1",
        "FRONT_LEFT": "2",
        "SIDE_LEFT": "3",
        "FRONT_RIGHT": "4",
        "SIDE_RIGHT": "5"
    }

    default_hdmap_color_table = {
        "crosswalk": (255, 0, 0),
        "lane": (0, 255, 0),
    }
    default_image_description_keys = [
        "time", "weather", "environment", "objects", "image_description"
    ]

    def __init__(
        self, fs: fsspec.AbstractFileSystem,
        info_dict_path: str,
        sequence_length: int,
        fps_stride_tuples: list,
        sensor_channels: list = ["FRONT"],
        enable_camera_transforms: bool = False,
        enable_ego_transforms: bool = False,
        _3dbox_image_settings: dict = None,
        hdmap_image_settings: dict = None,
        _3dbox_bev_settings: dict = None,
        hdmap_bev_settings: dict = None,
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

        self.skip_points = False if "LIDAR_TOP" in sensor_channels else True

        self.camera_types = [
            sensor for sensor in sensor_channels if "LIDAR" not in sensor]

        self.stub_key_data_dict = stub_key_data_dict

        self.type_map = {
            "road_line": "polyline",
            "lane": "polyline",
            "road_edge": "polyline",
            "crosswalk": "polygon",
            "driveway": "polygon",
            "speed_bump": "polygon",
        }

        # json
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
                    dwm.datasets.common.find_sample_data_of_nearest_time(
                        timestamps, timestamps, (t + i / fps) * 1000000)
                    for i in range(sequence_length)
                ]
                yield candidates

    def get_images_and_lidar_points(self, fs, scene, sample, frame):
        type_list = [
            'UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
        ]
        selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']

        with fs.open(f"segment-{scene}_with_camera_labels.tfrecord", 'rb') as f:
            item_length, target_offset = sample[-2:]
            item_0_length, target_0_offset = sample[1:3]
            map_features_0 = []
            if target_offset != target_0_offset:
                frame_tmp = open_dataset.Frame()
                f.seek(target_0_offset)
                record_0 = f.read(item_0_length)
                frame_tmp.ParseFromString(record_0)
                map_features_0 = frame_tmp.map_features
                del record_0
                del frame_tmp
            f.seek(target_offset)
            record = f.read(item_length)
        frame.ParseFromString(record)
        images = []
        sizes = []
        _2d_labels = []
        _2dbox_images = []
        # TODO parse others from frame
        frame_infos = self.create_waymo_info_file(frame)
        instances_3dbox_info = MotionDataset.get_frame_lidar_3d_boxes(
            frame)
        if len(map_features_0) == 0:
            map_features_0 = frame.map_features  # if not exists, also []
        default_camera_channels_idx = {
            "FRONT": 0,
            "FRONT_LEFT": 1,
            "SIDE_LEFT": 2,
            "FRONT_RIGHT": 3,
            "SIDE_RIGHT": 4
        }
        for sensor in self.camera_types:
            sensor_idx = default_camera_channels_idx[sensor]
            cam_image, cam_labels = frame.images[sensor_idx], frame.camera_labels[sensor_idx]
            pil = Image.open(io.BytesIO(cam_image.image))
            images.append(pil)
            sizes.append([pil.size[0], pil.size[1]])
            if cam_labels.name != cam_image.name:
                continue
            view_labels = []
            for label in cam_labels.labels:
                my_type = type_list[label.type]
                if my_type not in selected_waymo_classes:
                    continue
                label_ = selected_waymo_classes.index(my_type)
                view_labels.append(
                    [label_, label.box.center_x, label.box.center_y, label.box.length, label.box.width])
            _2d_labels.append(view_labels)
            # 2d_boxes for each frame
            # _2d_box_image = self.get_3d_box_image_from_2d(view_labels, image_size=[pil.size[0], pil.size[1]], _3dbox_image_settings=_3dbox_image_settings)
            # _2dbox_images.append(_2d_box_image)

        points = None
        if "LIDAR_TOP" in self.sensor_channels:
            # NOTE DEBUG no points
            # import time
            # ts = time.time()
            (range_images, camera_projections, seg_labels,
             range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
            # td = time.time()
            # print(f'project time {td-ts} s')
            # ts = time.time()
            points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose)
            # td = time.time()
            # print(f'convert time {td-ts} s')
            points = points[0]
            points = torch.tensor(points)

        return images, sizes, _2dbox_images, points, frame_infos, instances_3dbox_info, map_features_0

    @staticmethod
    def get_image_description(
        fs: fsspec.AbstractFileSystem, scene_key: str, timestamp, camera_id
    ):
        path = "caption/{};{}.{}.jpg.json".format(
            scene_key, timestamp, camera_id)
        with fs.open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def cart_to_homo(mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.

        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.

        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret

    def create_waymo_info_file(self, frame):
        r"""Generate waymo train/val/test infos.

        For more details about infos, please refer to:
        https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html
        """  # noqa: E501
        frame_infos = dict()

        intrinsics, extrinsics = MotionDataset.get_frame_camera_intrinsic_extrinsic(
            frame, self.sensor_channels)
        frame_infos["camera_intrinsics"] = intrinsics
        frame_infos["camera_extrinsics"] = extrinsics.float()

        # Gather frame infos
        frame_infos['timestamp'] = frame.timestamp_micros
        frame_infos['ego2global'] = torch.tensor(np.array(frame.pose.transform).reshape(
            4, 4).astype(np.float32))
        frame_infos['context_name'] = frame.context.name

        # Gather camera infos
        frame_infos['images'] = dict()
        # waymo front camera to kitti reference camera
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
        camera_calibs = []
        Tr_velo_to_cams = []
        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4)
            T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
            Tr_velo_to_cam = \
                MotionDataset.cart_to_homo(
                    T_front_cam_to_ref) @ T_vehicle_to_cam
            Tr_velo_to_cams.append(Tr_velo_to_cam)

            # intrinsic parameters
            camera_calib = np.zeros((3, 4))
            camera_calib[0, 0] = camera.intrinsic[0]
            camera_calib[1, 1] = camera.intrinsic[1]
            camera_calib[0, 2] = camera.intrinsic[2]
            camera_calib[1, 2] = camera.intrinsic[3]
            camera_calib[2, 2] = 1
            camera_calibs.append(camera_calib)

        for i, (cam_key, camera_calib, Tr_velo_to_cam) in enumerate(
                zip(self.camera_types, camera_calibs, Tr_velo_to_cams)):
            cam_infos = dict()
            # TODO fix the image reading
            # cam_infos['img_path'] = str(sample_idx) + '.jpg'
            # # NOTE: frames.images order is different
            # for img in frame.images:
            #     if img.name == i + 1:
            #         width, height = Image.open(BytesIO(img.image)).size
            # cam_infos['height'] = height
            # cam_infos['width'] = width
            cam_infos['lidar2cam'] = Tr_velo_to_cam.astype(np.float32).tolist()
            cam_infos['cam2img'] = camera_calib.astype(np.float32).tolist()
            cam_infos['lidar2img'] = (camera_calib @ Tr_velo_to_cam).astype(
                np.float32).tolist()
            frame_infos['images'][cam_key] = cam_infos

        # Gather lidar infos
        lidar_infos = dict()
        # lidar_infos['lidar_path'] = str(sample_idx) + '.bin'
        lidar_infos['num_pts_feats'] = 6
        frame_infos['lidar_points'] = lidar_infos

        # Gather lidar sweeps and camera sweeps infos
        # TODO: Add lidar2img in image sweeps infos when we need it.
        # TODO: Consider merging lidar sweeps infos and image sweeps infos.
        lidar_sweeps_infos, image_sweeps_infos = [], []
        # for prev_offset in range(-1, -self.max_sweeps - 1, -1):
        #     prev_lidar_infos = dict()
        #     prev_image_infos = dict()
        #     if frame_idx + prev_offset >= 0:
        #         prev_frame_infos = file_infos[prev_offset]
        #         prev_lidar_infos['timestamp'] = prev_frame_infos['timestamp']
        #         prev_lidar_infos['ego2global'] = prev_frame_infos['ego2global']
        #         prev_lidar_infos['lidar_points'] = dict()
        #         lidar_path = prev_frame_infos['lidar_points']['lidar_path']
        #         prev_lidar_infos['lidar_points']['lidar_path'] = lidar_path
        #         lidar_sweeps_infos.append(prev_lidar_infos)

        #         prev_image_infos['timestamp'] = prev_frame_infos['timestamp']
        #         prev_image_infos['ego2global'] = prev_frame_infos['ego2global']
        #         prev_image_infos['images'] = dict()
        #         for cam_key in self.camera_types:
        #             prev_image_infos['images'][cam_key] = dict()
        #             img_path = prev_frame_infos['images'][cam_key]['img_path']
        #             prev_image_infos['images'][cam_key]['img_path'] = img_path
        #         image_sweeps_infos.append(prev_image_infos)
        # if lidar_sweeps_infos:
        #     frame_infos['lidar_sweeps'] = lidar_sweeps_infos
        # if image_sweeps_infos:
        #     frame_infos['image_sweeps'] = image_sweeps_infos

        # if not self.test_mode:
        # Gather instances infos which is used for lidar-based 3D detection
        # frame_infos['instances'] = self.gather_instance_info(frame)
        # Gather cam_sync_instances infos which is used for image-based
        # (multi-view) 3D detection.
        # if self.save_cam_sync_instances:
        #     frame_infos['cam_sync_instances'] = self.gather_instance_info(
        #         frame, cam_sync=True)
        # Gather cam_instances infos which is used for image-based
        # (monocular) 3D detection (optional).
        # TODO: Should we use cam_sync_instances to generate cam_instances?
        # if self.save_cam_instances:
        #     frame_infos['cam_instances'] = self.gather_cam_instance_info(
        #         copy.deepcopy(frame_infos['instances']),
        #         frame_infos['images'])
        return frame_infos

    # @staticmethod
    # def get_frame_pose(frame):
    #     import tensorflow as tf
    #     tf.config.set_visible_devices([], 'GPU')
    #     return tf.convert_to_tensor(
    #         value=np.reshape(np.array(frame.pose.transform), [4,4])
    #     )
    @staticmethod
    def get_frame_camera_intrinsic_extrinsic(frame, sensor_channels=None):
        camera_calibrations = frame.context.camera_calibrations
        intrinsics = []
        extrinsics = []

        order_dict = {
            'FRONT': [],
            'FRONT_LEFT': [],
            'FRONT_RIGHT': [],
            'SIDE_LEFT': [],
            'SIDE_RIGHT': []
        }

        for c in camera_calibrations:
            camera_name = str(c).split('\n')[0].split(':')[-1].strip()
            intrinsic = c.intrinsic
            intrinsic_ = torch.eye(3)
            intrinsic_[0, 0] = intrinsic[0]
            intrinsic_[1, 1] = intrinsic[1]
            intrinsic_[0, 2] = intrinsic[2]
            intrinsic_[1, 2] = intrinsic[3]
            extrinsic = torch.tensor(np.reshape(
                np.array(c.extrinsic.transform), (4, 4)))
            order_dict[camera_name] = [
                intrinsic_, extrinsic
            ]
        sensor_channels = ['FRONT', 'FRONT_LEFT', 'SIDE_LEFT', 'FRONT_RIGHT',
                           'SIDE_RIGHT'] if sensor_channels is None else sensor_channels
        camera_channels = [c for c in sensor_channels if "LIDAR" not in c]
        for name in camera_channels:
            intri = order_dict[name][0]
            extri = order_dict[name][1]
            intrinsics.append(intri)
            extrinsics.append(extri)

        intrinsics = torch.stack(intrinsics)
        extrinsics = torch.stack(extrinsics)

        return intrinsics, extrinsics

    @staticmethod
    def get_frame_lidar_3d_boxes(frame):
        type_list = [
            'UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
        ]
        selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
        objs = frame.laser_labels
        instance_infos = []
        for obj in objs:
            instance_info = dict()
            type = obj.type
            my_type = type_list[type]
            if my_type not in selected_waymo_classes:
                continue
            box3d = obj.box
            h, w, l = box3d.height, box3d.width, box3d.length
            x, y, z = box3d.center_x, box3d.center_y, box3d.center_z-h/2
            rotation_y = box3d.heading
            bbox_3d = np.array(
                [x, y, z, l, w, h, rotation_y]).astype(np.float32).tolist()

            label = selected_waymo_classes.index(my_type)
            instance_info['bbox_3d'] = bbox_3d
            instance_info['bbox_3d_label'] = label
            instance_infos.append(instance_info)
        return instance_infos

    def get_hdmap_bev_image(
        self, map_features, ego2global, hdmap_bev_settings: dict
    ):
        max_distance = hdmap_bev_settings["max_distance"] \
            if "max_distance" in hdmap_bev_settings else 65.0
        pen_width = hdmap_bev_settings["pen_width"] \
            if "pen_width" in hdmap_bev_settings else 2
        all_hdmap_classes = ['crosswalk', 'driveway', 'lane',
                             'road_edge', 'road_line', 'speed_bump', 'stop_sign']
        selected_hdmap_classes = ['lane']
        bev_size = hdmap_bev_settings["bev_size"] \
            if "bev_size" in hdmap_bev_settings else [640, 640]
        fill_box = hdmap_bev_settings["fill_box"] \
            if "fill_box" in hdmap_bev_settings else False
        bev_from_ego_transform = hdmap_bev_settings["bev_from_ego_transform"] \
            if "bev_from_ego_transform" in hdmap_bev_settings else [
                [6.4, 0, 0, 320],
                [0, -6.4, 0, 320],
                [0, 0, -6.4, 0],
                [0, 0, 0, 1]
        ]
        # TODO on-going
        selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
        default_color_table = {
            selected_waymo_classes[0]: (0, 0, 255),
            selected_waymo_classes[1]: (255, 0, 0),
            selected_waymo_classes[2]: (0, 255, 0)
        }
        color_table = hdmap_bev_settings["color_table"] \
            if "color_table" in hdmap_bev_settings \
            else default_color_table
        # draw 3d box
        image = Image.new("RGB", bev_size)
        draw = ImageDraw.Draw(image)

        if len(map_features) == 0:
            return image

        world_from_ego = ego2global
        ego_from_world = np.linalg.inv(world_from_ego)
        bev_from_ego = np.array(bev_from_ego_transform, np.float32)
        transform = bev_from_ego @ ego_from_world
        ego_position = np.array([
            [ego2global[0, 3]], [ego2global[1, 3]], [0],
            [1]
        ])

        type_polygons = {}
        type_polylines = {}
        for feat in map_features:
            type_ = feat.WhichOneof('feature_data')
            if type_ not in selected_hdmap_classes:
                continue
            type_poly = self.type_map[type_]
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

        # draw
        if "lane" in type_polygons:
            for i in type_polygons["lane"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_polygon_bev_to_image(
                    i, draw, transform,
                    ego_position, max_distance, (0, 255, 0), pen_width)

        if "lane" in type_polylines:
            for i in type_polylines["lane"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_line_bev_to_image(
                    i, draw, transform,
                    ego_position, max_distance, (0, 255, 0), pen_width)
        if "road_line" in type_polygons:
            for i in type_polygons["road_line"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_polygon_bev_to_image(
                    i, draw, transform,
                    ego_position, max_distance, (255, 255, 0), pen_width)
        if "road_line" in type_polylines:
            for i in type_polylines["road_line"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_line_bev_to_image(
                    i, draw, transform,
                    ego_position, max_distance, (255, 255, 0), pen_width)
        if "road_edge" in type_polygons:
            for i in type_polygons["road_edge"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_polygon_bev_to_image(
                    i, draw, transform,
                    ego_position, max_distance, (255, 255, 0), pen_width)
        if "road_edge" in type_polylines:
            for i in type_polylines["road_edge"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_line_bev_to_image(
                    i, draw, transform,
                    ego_position, max_distance, (255, 255, 0), pen_width)
        if "crosswalk" in type_polygons:
            for i in type_polygons["crosswalk"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_polygon_bev_to_image(
                    i, draw, transform,
                    ego_position, max_distance, (0, 0, 255), pen_width)
        if "crosswalk" in type_polylines:
            for i in type_polylines["crosswalk"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_line_bev_to_image(
                    i, draw, transform,
                    ego_position, max_distance, (0, 0, 255), pen_width)

        # TODO
        return image

    def get_3d_bev_image(
        self, instance_infos, image_size, _3dbox_bev_settings: dict
    ):
        def world_to_image(x, y, img_size, scale=6.4):
            img_x = img_size[0] // 2 + int(x * scale)
            img_y = img_size[1] // 2 + int(y * scale)
            return img_x, img_y

        pen_width = _3dbox_bev_settings["pen_width"] \
            if "pen_width" in _3dbox_bev_settings else 2
        bev_size = _3dbox_bev_settings["bev_size"] \
            if "bev_size" in _3dbox_bev_settings else [640, 640]
        bev_from_ego_transform = _3dbox_bev_settings["bev_from_ego_transform"] \
            if "bev_from_ego_transform" in _3dbox_bev_settings else [
                [6.4, 0, 0, 320],
                [0, -6.4, 0, 320],
                [0, 0, -6.4, 0],
                [0, 0, 0, 1]
        ]
        fill_box = _3dbox_bev_settings["fill_box"] \
            if "fill_box" in _3dbox_bev_settings else False
        selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
        default_color_table = {
            selected_waymo_classes[0]: (0, 0, 255),
            selected_waymo_classes[1]: (255, 0, 0),
            selected_waymo_classes[2]: (0, 255, 0)
        }
        color_table = _3dbox_bev_settings["color_table"] \
            if "color_table" in _3dbox_bev_settings \
            else default_color_table
        # draw 3d box
        image = Image.new("RGB", bev_size)
        draw = ImageDraw.Draw(image)
        for instance in instance_infos:
            bbox_3d = instance['bbox_3d']  # [x, y, z, l, w, h, rotation_y]
            label = instance['bbox_3d_label']  # [0, 1, 2]
            label_type = selected_waymo_classes[label]
            color = color_table[label_type]
            x, y, z, l, w, h, rotation_y = bbox_3d
            corners = np.array([
                [l / 2, w / 2],
                [l / 2, -w / 2],
                [-l / 2, -w / 2],
                [-l / 2, w / 2]
            ])
            rotation_matrix = np.array([
                [np.cos(rotation_y), np.sin(rotation_y)],
                [-np.sin(rotation_y), np.cos(rotation_y)]
            ])
            corners_rotated = np.dot(corners, rotation_matrix)
            corners_rotated[:, 0] += x
            corners_rotated[:, 1] += y
            corners_list = [world_to_image(
                corners_rotated[i, 0], corners_rotated[i, 1], image.size) for i in range(4)]
            corners_list.append(corners_list[0])
            # draw.line(corners_list, fill=color, width=2)
            draw.polygon(
                corners_list,
                outline=None if fill_box else color, width=pen_width,
                fill=color if fill_box else None)
        return image

    def get_3d_box_to_image(self, instance_infos_, cam_intrinsic, camera_from_ego, size, pen_width, _3dbox_image_settings, camera_image=None):
        selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
        default_color_table = {
            selected_waymo_classes[0]: (0, 0, 255),
            selected_waymo_classes[1]: (255, 0, 0),
            selected_waymo_classes[2]: (0, 255, 0)
        }
        color_table = _3dbox_image_settings["color_table"] \
            if "color_table" in _3dbox_image_settings \
            else default_color_table
        default_corner_templates = np.array([
            [-0.5, -0.5, 0, 1], [-0.5, -0.5, 1, 1], [-0.5, 0.5, 0, 1],
            [-0.5, 0.5, 1, 1], [0.5, -0.5, 0, 1], [0.5, -0.5, 1, 1],
            [0.5, 0.5, 0, 1], [0.5, 0.5, 1, 1]
        ]).transpose()
        corner_templates = _3dbox_image_settings["corner_templates"] \
            if "corner_templates" in _3dbox_image_settings \
            else default_corner_templates
        default_edge_indices = [
            (0, 1), (0, 2), (1, 3), (2, 3), (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 5), (4, 6), (5, 7), (6, 7)
        ]
        edge_indices = _3dbox_image_settings["edge_indices"] \
            if "edge_indices" in _3dbox_image_settings \
            else default_edge_indices

        if camera_image is None:
            camera_image = Image.new("RGB", size)
        draw = ImageDraw.Draw(camera_image)
        for idx, instance in enumerate(instance_infos_):
            # print(idx)
            bbox_3d = instance['bbox_3d']  # [x, y, z, l, w, h, rotation_y]
            label = instance['bbox_3d_label']  # [0, 1, 2]
            label_type = selected_waymo_classes[label]
            color = color_table[label_type]
            x, y, z, l, w, h, rotation_y = bbox_3d
            scale = np.diag([l, w, h, 1])
            center = (x, y, z)
            translation = np.eye(4)
            translation[:3, 3] = [x, y, z]
            rotation_y = -rotation_y
            rotation = np.array([
                [np.cos(rotation_y), np.sin(rotation_y), 0, 0],
                [-np.sin(rotation_y), np.cos(rotation_y), 0,  0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            world_from_annotation = (translation @ rotation)
            camera_from_world = np.linalg.inv(camera_from_ego.numpy())
            cam_intrinsic_expand = torch.eye(4)
            cam_intrinsic_expand[:3, :3] = cam_intrinsic
            cam_intrinsic_expand = cam_intrinsic_expand.numpy()
            tmp = np.zeros((4, 4))
            tmp[0, 1] = -1
            tmp[1, 2] = -1
            tmp[2, 0] = 1
            tmp[3, 3] = 1
            camera_corners = tmp @ camera_from_world @ world_from_annotation @ scale @ corner_templates
            image_corners = cam_intrinsic_expand @ camera_corners
            p = image_corners[:2] / image_corners[2]
            for a, b in edge_indices:
                if image_corners[2, a] > 0 and image_corners[2, b] > 0:
                    draw.line(
                        (p[0, a], p[1, a], p[0, b], p[1, b]), fill=color,
                        width=pen_width)
        return camera_image

    def get_3d_box_image(
        self, cam_intrisics, cam_extrisics, instance_infos, camera_images, image_size, _3dbox_image_settings: dict
    ):
        # note from 3d box
        selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
        # options
        pen_width = _3dbox_image_settings["pen_width"] \
            if "pen_width" in _3dbox_image_settings else 4
        project_2d_images = []
        import time
        timestep = time.time()
        for c in range(len(cam_intrisics)):
            cam_intrinsic = cam_intrisics[c]
            camera_from_ego = cam_extrisics[c]
            project_image = self.get_3d_box_to_image(
                instance_infos, cam_intrinsic, camera_from_ego, image_size[c], pen_width, _3dbox_image_settings, None)
            project_2d_images.append(project_image)
            # project_image.save(f'cam_box_{timestep}_{c}.png')
        return project_2d_images

    @staticmethod
    def draw_polygon_to_image(
        polygon, draw, transform, ego_position, max_distance,
        pen_fill: tuple, pen_width: int
    ):
        polygon_nodes = np.array([
            [i[0], i[1], i[2], 1]
            for i in polygon
        ]).transpose()
        d = np.sqrt(np.sum((polygon_nodes - ego_position) ** 2, 0))
        p = transform @ polygon_nodes
        m = len(polygon)
        polygon_points = []
        for i in range(m):
            polygon_points.append((p[0, i], p[1, i]))
        draw.polygon(
            polygon_points,
            fill=None,
            outline=pen_fill, width=pen_width)

    @staticmethod
    def draw_polygon_bev_to_image(
        polygon, draw, transform, ego_position, max_distance,
        pen_fill: tuple, pen_width: int
    ):
        polygon_nodes = np.array([
            [i[0], i[1], 0, 1]
            for i in polygon
        ]).transpose()
        d = np.sqrt(np.sum((polygon_nodes - ego_position) ** 2, 0))
        p = transform @ polygon_nodes
        m = len(polygon)
        polygon_points = []
        for i in range(m):
            polygon_points.append((p[0, i], p[1, i]))
        draw.polygon(
            polygon_points,
            fill=pen_fill,
            outline=None, width=pen_width)

    @staticmethod
    def draw_line_to_image(
        line, draw, transform, ego_position, max_distance,
        pen_fill: tuple, pen_width: int
    ):
        line_nodes = np.array([
            [i[0], i[1], i[2], 1]
            for i in line
        ]).transpose()
        d = np.sqrt(np.sum((line_nodes - ego_position) ** 2, 0))
        image_nodes = transform @ line_nodes
        p = image_nodes[:2] / image_nodes[2]
        for i in range(1, len(line)):
            if d[i - 1] <= max_distance and d[i] <= max_distance and \
                    image_nodes[2, i - 1] > 0 and image_nodes[2, i] > 0:
                draw.line(
                    (p[0, i - 1], p[1, i - 1], p[0, i], p[1, i]),
                    fill=pen_fill, width=pen_width)

    @staticmethod
    def draw_line_bev_to_image(
        line, draw, transform, ego_position, max_distance,
        pen_fill: tuple, pen_width: int
    ):
        line_nodes = np.array([
            [i[0], i[1], 0, 1]
            for i in line
        ]).transpose()
        d = np.sqrt(np.sum((line_nodes - ego_position) ** 2, 0))
        image_nodes = transform @ line_nodes
        p = image_nodes
        for i in range(1, len(line)):
            draw.line(
                (p[0, i - 1], p[1, i - 1], p[0, i], p[1, i]),
                fill=pen_fill, width=pen_width)

    def get_hdmap_to_image(
        self, map_features, cam_intrinsic, camera_from_ego, ego2global, size, pen_width, hdmap_image_settings, camera_image=None
    ):
        max_distance = hdmap_image_settings["max_distance"] \
            if "max_distance" in hdmap_image_settings else 65.0
        pen_width = hdmap_image_settings["pen_width"] \
            if "pen_width" in hdmap_image_settings else 4
        color_table = hdmap_image_settings.get(
            "color_table", MotionDataset.default_hdmap_color_table)
        selected_hdmap_classes = list(color_table.keys())
        all_hdmap_classes = ['crosswalk', 'driveway', 'lane',
                             'road_edge', 'road_line', 'speed_bump', 'stop_sign']

        max_distance = hdmap_image_settings["max_distance"] \
            if "max_distance" in hdmap_image_settings else 65.0

        if camera_image is None:
            camera_image = Image.new("RGB", size)
        if len(map_features) == 0:
            return camera_image
        draw = ImageDraw.Draw(camera_image)

        world_from_ego = ego2global
        ego_from_world = np.linalg.inv(world_from_ego)

        # camera_from_world = np.linalg.inv(camera_from_ego.numpy())
        camera_from_ego = np.linalg.inv(camera_from_ego.numpy())
        camera_from_world = camera_from_ego @ ego_from_world
        cam_intrinsic_expand = torch.eye(4)
        cam_intrinsic_expand[:3, :3] = cam_intrinsic
        cam_intrinsic_expand = cam_intrinsic_expand.numpy()
        tmp = np.zeros((4, 4))
        tmp[0, 1] = -1
        tmp[1, 2] = -1
        tmp[2, 0] = 1
        tmp[3, 3] = 1
        transform = cam_intrinsic_expand @ tmp @ camera_from_world
        ego_position = np.array([
            [ego2global[0, 3]], [ego2global[1, 3]], [0],
            [1]
        ])

        type_polygons = {}
        type_polylines = {}
        for feat in map_features:
            type_ = feat.WhichOneof('feature_data')
            if type_ not in selected_hdmap_classes:
                continue
            type_poly = self.type_map[type_]
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

        # draw
        if "lane" in type_polygons:
            c = tuple(color_table["lane"])
            for i in type_polygons["lane"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_polygon_to_image(
                    i, draw, transform,
                    ego_position, max_distance, c, pen_width)
        if "lane" in type_polylines:
            c = tuple(color_table["lane"])
            for i in type_polylines["lane"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_line_to_image(
                    i, draw, transform,
                    ego_position, max_distance, c, pen_width)
        if "road_line" in type_polygons:
            c = tuple(color_table["road_line"])
            for i in type_polygons["road_line"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_polygon_to_image(
                    i, draw, transform,
                    ego_position, max_distance, c, pen_width)
        if "road_line" in type_polylines:
            c = tuple(color_table["road_line"])
            for i in type_polylines["road_line"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_line_to_image(
                    i, draw, transform,
                    ego_position, max_distance, c, pen_width)
        if "road_edge" in type_polygons:
            c = tuple(color_table["road_edge"])
            for i in type_polygons["road_edge"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_polygon_to_image(
                    i, draw, transform,
                    ego_position, max_distance, c, pen_width)
        if "road_edge" in type_polylines:
            c = tuple(color_table["road_edge"])
            for i in type_polylines["road_edge"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_line_to_image(
                    i, draw, transform,
                    ego_position, max_distance, c, pen_width)
        if "crosswalk" in type_polygons:
            c = tuple(color_table["crosswalk"])
            for i in type_polygons["crosswalk"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_polygon_to_image(
                    i, draw, transform,
                    ego_position, max_distance, c, pen_width)
        if "crosswalk" in type_polylines:
            c = tuple(color_table["crosswalk"])
            for i in type_polylines["crosswalk"]:
                if len(i) == 0:
                    continue
                MotionDataset.draw_line_to_image(
                    i, draw, transform,
                    ego_position, max_distance, c, pen_width)

        return camera_image

    def get_hdmap_image(
        self, map_features, cam_intrisics, cam_extrisics, ego2global, camera_images, image_size, hdmap_image_settings: dict
    ):
        # options
        pen_width = hdmap_image_settings["pen_width"] \
            if "pen_width" in hdmap_image_settings else 4
        project_hdmap_images = []
        import time
        timestep = time.time()
        for c in range(len(cam_intrisics)):
            cam_intrinsic = cam_intrisics[c]
            camera_from_ego = cam_extrisics[c]
            project_hd_image = self.get_hdmap_to_image(
                map_features, cam_intrinsic, camera_from_ego, ego2global, image_size[c], pen_width, hdmap_image_settings, None)
            project_hdmap_images.append(project_hd_image)
            # project_hd_image.save(f'hdmap_0_{c}_{timestep}.png')
        return project_hdmap_images

    def get_3d_box_image_from_2d(
        self, _2d_box, image_size, _3dbox_image_settings: dict
    ):
        # note from 2d image -> 2d box
        selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
        # options
        pen_width = _3dbox_image_settings["pen_width"] \
            if "pen_width" in _3dbox_image_settings else 4
        default_color_table = {
            selected_waymo_classes[0]: (0, 0, 255),
            selected_waymo_classes[1]: (255, 0, 0),
            selected_waymo_classes[2]: (0, 255, 0)
        }
        color_table = _3dbox_image_settings["color_table"] \
            if "color_table" in _3dbox_image_settings \
            else default_color_table
        # directly draw bbox
        image = Image.new("RGB", (image_size[0], image_size[1]))
        draw = ImageDraw.Draw(image)
        for box in _2d_box:
            label = box[0]
            type = selected_waymo_classes[label]
            color = tuple(color_table[type])
            # label.box.center_x, label.box.center_y, label.box.length, label.box.width
            center_x, center_y, length, width = box[1], box[2], box[3], box[4]
            left = center_x - 0.5 * length
            top = center_y - 0.5 * width
            right = center_x + 0.5 * length
            bottom = center_y + 0.5 * width
            draw.rectangle([left, top, right, bottom],
                           outline=color, width=pen_width)
        return image

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

        frame = open_dataset.Frame()
        images, sizes, _2d_boxes, lidar_points, bev_boxes, bev_hdmaps = [], [], [], [], [], []
        frame_infos, instances_3dbox_infos = [], []
        _3dboxs, hdmaps = [], []

        projected_2d_boxes, projected_hdmaps = [[]], [[]]
        bev_box, bev_hdmap = [[]], [[]]
        for sample in segment:
            images_i, size_i, _2d_box, lidar_points_i, frame_info, instances_3dbox_info, map_features = self.get_images_and_lidar_points(
                self.fs, item["scene"], sample, frame)
            images.append(images_i)
            sizes.append(torch.tensor(size_i))
            if lidar_points_i is not None:
                lidar_points.append(lidar_points_i)

            _2d_boxes.append(_2d_box)
            frame_infos.append(frame_info)
            instances_3dbox_infos.append(instances_3dbox_info)
            # get bev image
            # ts = time.time()
            if self._3dbox_bev_settings is not None:
                bev_box = self.get_3d_bev_image(
                    instances_3dbox_info, size_i, self._3dbox_bev_settings)
            # td = time.time()
            # print(f'3dbox bev time: {td-ts} s')
            # ts = time.time()
            if self.hdmap_bev_settings is not None:
                bev_hdmap = self.get_hdmap_bev_image(
                    map_features, frame_info["ego2global"], self.hdmap_bev_settings)
            # td = time.time()
            # print(f'hdmap bev time: {td-ts} s')
            bev_boxes.append(bev_box)
            bev_hdmaps.append(bev_hdmap)
            # ts = time.time()
            if self._3dbox_image_settings is not None:
                projected_2d_boxes = self.get_3d_box_image(
                    frame_info["camera_intrinsics"], frame_info["camera_extrinsics"], instances_3dbox_info, images_i, size_i, self._3dbox_image_settings)
            # td = time.time()
            # print(f'3dbox camera time: {td-ts} s')
            # ts = time.time()
            if self.hdmap_image_settings is not None:
                projected_hdmaps = self.get_hdmap_image(
                    map_features, frame_info["camera_intrinsics"], frame_info["camera_extrinsics"], frame_info["ego2global"], images_i, size_i, self.hdmap_image_settings)
            # td = time.time()
            # print(f'hdmap camera time: {td-ts} s')
            _3dboxs.append(projected_2d_boxes)
            hdmaps.append(projected_hdmaps)

        if len(images) > 0 and len(self.camera_types) > 0:
            result["images"] = images

        if len(lidar_points) > 0:
            result["lidar_points"] = lidar_points  # [(N, 3)]

        if self._3dbox_image_settings is not None:
            result["3dbox_images"] = _3dboxs

        if self.hdmap_image_settings is not None:
            result["hdmap_images"] = hdmaps

        if self._3dbox_bev_settings is not None:
            result["3dbox_bev_images"] = bev_boxes

        if self.hdmap_bev_settings is not None:
            result["hdmap_bev_images"] = bev_hdmaps

        camera_intrinsics, camera_extrinsics, ego2globals = [], [], []
        for frame_info in frame_infos:  # each time
            camera_intrinsics.append(frame_info["camera_intrinsics"])
            camera_extrinsics.append(frame_info["camera_extrinsics"])
            ego2globals.append(frame_info["ego2global"])

        if self.enable_camera_transforms:
            if "images" in result:
                result["camera_intrinsics"] = torch.stack(camera_intrinsics)
                result["camera_transforms"] = torch.stack(camera_extrinsics)
                result["image_size"] = torch.stack(sizes).long()

            if "lidar_points" in result:
                result["lidar_transforms"] = torch.tensor(
                    np.expand_dims(
                        np.expand_dims(np.eye(4), 0),
                        0
                    ),
                    dtype=torch.float32
                ).repeat((len(segment), 1, 1, 1))

        if self.enable_ego_transforms:
            result["ego_transforms"] = torch.stack(ego2globals)\
                .unsqueeze(1).repeat_interleave(len(self.sensor_channels), 1)

            # pseudo labels
            result["bev_rotation"] = torch.randn(4)
            result["bev_translation"] = torch.randn(3)

        if self.image_description_settings is not None:
            selected_keys = self.image_description_settings.get(
                "selected_keys", MotionDataset.default_image_description_keys)
            image_captions = [
                [
                    MotionDataset.get_image_description(
                        self.image_description_settings["fs"], item["scene"], i[3],
                        MotionDataset.camera_name_id_dict[j])
                    for j in self.sensor_channels
                    if "LIDAR" not in j
                ]
                for i in segment
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


class FilterPoints():
    def __init__(self, min_distance: float = 0, max_distance: float = 1000.0):
        self.min_distance = min_distance
        self.max_distance = max_distance

    def __call__(self, a):
        distances = a[:, :3].norm(dim=-1)
        mask = torch.logical_and(
            distances >= self.min_distance, distances < self.max_distance)
        return a[mask]
