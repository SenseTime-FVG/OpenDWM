import fsspec
import numpy as np
from PIL import Image, ImageDraw
import torch
from collections import defaultdict
import xml.etree.ElementTree as ET
import os
from kitti360scripts.helpers.annotation import KITTI360Bbox3D, local2global
from kitti360scripts.helpers.labels import id2label
from transforms3d.quaternions import mat2quat, quat2mat
import dwm.datasets.common

class MotionDataset(torch.utils.data.Dataset):
    """The motion data loaded from the KITTI360 dataset.

    Args:
        fs (fsspec.AbstractFileSystem): The file system for the dataset files.
        dataset_name (str): Not used for KITTI360, kept for API compatibility.
        sequence_length (int): The frame count of the temporal sequence.
        fps_stride_tuples (list): The list of tuples in the form of
            (FPS, stride). If the FPS > 0, stride is the begin time in second
            between 2 adjacent video clips, else the stride is the index count
            of the beginning between 2 adjacent video clips.
        split (str or None): The split in one of "train", "val", following
            the split definition of the KITTI360 dataset.
        sensor_channels (list): Not used for KITTI360, kept for API compatibility.
        keyframe_only (bool): Not used for KITTI360, kept for API compatibility.
        enable_synchronization_check (bool): Not used for KITTI360, kept for API compatibility.
        enable_scene_description (bool): Not used for KITTI360, kept for API compatibility.
        enable_camera_transforms (bool): Not used for KITTI360, kept for API compatibility.
        enable_ego_transforms (bool): Not used for KITTI360, kept for API compatibility.
        enable_sample_data (bool): Not used for KITTI360, kept for API compatibility.
        _3dbox_image_settings (dict or None): Not used for KITTI360, kept for API compatibility.
        hdmap_image_settings (dict or None): Not used for KITTI360, kept for API compatibility.
        image_segmentation_settings (dict or None): Not used for KITTI360, kept for API compatibility.
        foreground_region_image_settings (dict or None): Not used for KITTI360, kept for API compatibility.
        _3dbox_bev_settings (dict or None): Not used for KITTI360, kept for API compatibility.
        hdmap_bev_settings (dict or None): Not used for KITTI360, kept for API compatibility.
        image_description_settings (dict or None): Not used for KITTI360, kept for API compatibility.
        stub_key_data_dict (dict or None): The dict of stub key and data, to
            align with other datasets with keys and data missing in this
            dataset. Please refer to dwm.datasets.common.add_stub_key_data()
            for details.
    """

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
    default_label_mapping = {
        "person": "human.pedestrian",
        "rider": "human.pedestrian",
        "car": "vehicle.car",
        "truck": "vehicle.truck",
        "bus": "vehicle.bus",
        "motorcycle": "vehicle.motorcycle",
        "bicycle": "vehicle.bicycle",
        "trailer": "vehicle.trailer"
    }
    default_bev_from_ego_transform = [
        [6.4, 0, 0, 320],
        [0, -6.4, 0, 320],
        [0, 0, -6.4, 0],
        [0, 0, 0, 1]
    ]
    default_scene_list = [
        "2013_05_28_drive_0000_sync",
        "2013_05_28_drive_0002_sync",
        "2013_05_28_drive_0003_sync",
        "2013_05_28_drive_0004_sync",
        "2013_05_28_drive_0005_sync",
        "2013_05_28_drive_0006_sync",
        "2013_05_28_drive_0007_sync",
        "2013_05_28_drive_0009_sync",
        "2013_05_28_drive_0010_sync"
    ]
    default_3dbox_corner_template = [
        [-0.5, -0.5, -0.5, 1], [-0.5, -0.5, 0.5, 1],
        [-0.5, 0.5, -0.5, 1], [-0.5, 0.5, 0.5, 1],
        [0.5, -0.5, -0.5, 1], [0.5, -0.5, 0.5, 1],
        [0.5, 0.5, -0.5, 1], [0.5, 0.5, 0.5, 1]
    ]
    default_3dbox_edge_indices = [
        (0, 2), (2, 6), (6, 4), (4, 0)
    ]
    default_gps_to_ego = [
        [1,  0,  0, -0.05],
        [0, -1,  0,  0.32],
        [0,  0, -1,  0.60],
        [0,  0,  0,  1]
    ]
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
        self.sequence_length = sequence_length
        self.fps_stride_tuples = fps_stride_tuples
        self.enable_scene_description = enable_scene_description
        self.enable_camera_transforms = enable_camera_transforms
        self.enable_ego_transforms = enable_ego_transforms
        self.enable_sample_data = enable_sample_data
        self._3dbox_image_settings = _3dbox_image_settings
        self.hdmap_image_settings = hdmap_image_settings
        self.image_segmentation_settings = image_segmentation_settings
        self.foreground_region_image_settings = foreground_region_image_settings
        self._3dbox_bev_settings = _3dbox_bev_settings
        self.hdmap_bev_settings = hdmap_bev_settings
        self.image_description_settings = image_description_settings
        self.stub_key_data_dict = stub_key_data_dict

        # Get all the bin files
        self.fs._get_dirs()
        full_list = [i for i in self.fs.dir_cache if i.endswith(".bin")]

        # Split the dataset based on the split parameter
        if split == "train":
            self.file_paths = sorted(list(filter(
                lambda folder: "0000_sync" not in folder and "0002_sync" not in folder, full_list)))
        elif split == "val":
            self.file_paths = sorted(list(filter(
                lambda folder: "0000_sync" in folder or "0002_sync" in folder, full_list)))
        else:
            self.file_paths = sorted(full_list)

        self.segments = []

        # Group files by their base directory (recording session)
        file_groups = {}
        for file_path in self.file_paths:
            # Extract the base directory containing xxxx_sync
            parts = file_path.split("/")
            for part in parts:
                if "_sync" in part:
                    base_dir = part
                    break
            else:
                # If no _sync found, use the whole path
                base_dir = os.path.dirname(file_path)

            if base_dir not in file_groups:
                file_groups[base_dir] = []
            file_groups[base_dir].append(file_path)

        self.bboxes_dict = defaultdict(dict)
        self.poses_dict = defaultdict(dict)
        # Sort files within each group
        for base_dir in file_groups:
            file_groups[base_dir] = {"lidar_files": sorted(file_groups[base_dir])}
            # load the pose data
            pose_file = f"{base_dir}/poses.txt"
            pose_data = self.fs.cat_file(pose_file).splitlines()
            all_poses = {}
            for pose_idx in range(len(pose_data)):
                split_data = pose_data[pose_idx].split()
                pose_id = int(split_data[0])
                pose = np.array(split_data[1:]).astype(
                    np.float32).reshape(3, 4)
                all_poses[pose_id] = pose
            self.poses_dict[base_dir] = all_poses
            # load the bbox data
            bbox_file = f"data_3d_bboxes/train/{base_dir}.xml"
            bbox_data = self.fs.open(bbox_file, mode="rb")
            tree = ET.parse(bbox_data)
            root = tree.getroot()
            self.bboxes_dict[base_dir] = root

        # Create segments within each group
        for base_dir, files in file_groups.items():
            for fps, stride in fps_stride_tuples:
                files_len = len(files["lidar_files"])  
                # Use time-based stride (not implemented for KITTI360)
                # For simplicity, we"ll just use index-based stride for now
                for i in range(0, files_len - sequence_length + 1, max(1, stride)):
                    self.segments.append({
                        "files": files["lidar_files"][i:i+sequence_length],
                        "scene_name": base_dir,
                        "timestamps": [j for j in range(i, i+sequence_length)],
                        "fps": fps
                    })

    def __len__(self):
        return len(self.segments)

    @staticmethod
    def interpolate_pose(pose1, pose2, num_steps):
        """Interpolate between two poses using Slerp for rotation and linear interpolation for translation.

        Args:
            pose1: Starting 4x4 pose matrix
            pose2: Ending 4x4 pose matrix
            num_steps: Number of intermediate steps to generate

        Returns:
            List of interpolated pose matrices
        """
        # Extract rotation matrices and translation vectors
        R1, t1 = pose1[:3, :3], pose1[:3, 3]
        R2, t2 = pose2[:3, :3], pose2[:3, 3]

        # Slerp quaternion interpolation
        def slerp(q1, q2, t):
            # Ensure quaternions have the same sign
            dot = np.sum(q1 * q2)
            if dot < 0:
                q2 = -q2
                dot = -dot

            # If quaternions are very close, use linear interpolation
            if dot > 0.9995:
                result = q1 + t * (q2 - q1)
                return result / np.linalg.norm(result)

            # Otherwise use spherical interpolation
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)

            theta = theta_0 * t
            sin_theta = np.sin(theta)

            s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0

            return s0 * q1 + s1 * q2

        # Convert rotation matrices to quaternions
        q1 = mat2quat(R1)
        q2 = mat2quat(R2)

        # Generate interpolated poses
        interpolated_poses = []
        for step in range(1, num_steps):
            t = step / num_steps

            # Interpolate rotation using Slerp
            q_interp = slerp(q1, q2, t)
            R_interp = quat2mat(q_interp)

            # Linearly interpolate translation
            t_interp = t1 + t * (t2 - t1)

            # Create interpolated pose matrix
            pose_interp = np.eye(4)
            pose_interp[:3, :3] = R_interp
            pose_interp[:3, 3] = t_interp
            interpolated_poses.append(pose_interp)

        return interpolated_poses

    @staticmethod
    def transform_bboxes(_3dbox_corner_template, obj, world_to_gps, gps_to_ego, bev_from_ego_transform, color_table):
        label = id2label[obj.semanticId].name
        # Exclude the unknown label
        if label not in MotionDataset.default_label_mapping:
            return None, None
        label = MotionDataset.default_label_mapping[label]
        color = color_table[label] if isinstance(
            color_table[label], tuple) else tuple(color_table[label])
        temp_to_global = np.concatenate([obj.R, obj.T[..., None]], axis=-1)
        temp_to_global = np.vstack([temp_to_global, np.array([0, 0, 0, 1])])

        vertices = temp_to_global @ _3dbox_corner_template.T # Get global coordinates of the 3dbox corners
        vertices = world_to_gps @ vertices                   # Transform to GPS coordinates               
        vertices = gps_to_ego @ vertices                     # Transform to ego coordinates
        vertices = bev_from_ego_transform @ vertices         # Transform to bev image frame
        vertices = vertices.T                                # Transpose to (N, 3)
        vertices = vertices[:, :2]

        return vertices, color

    @staticmethod
    def get_3dbox_bev_image(bboxes, gps_to_world, timestamp, _3dbox_bev_settings: dict):
        pen_width = _3dbox_bev_settings.get("pen_width", 2)
        bev_size = _3dbox_bev_settings.get("bev_size", [640, 640])
        bev_from_ego_transform = _3dbox_bev_settings.get(
            "bev_from_ego_transform",
            MotionDataset.default_bev_from_ego_transform)
        gps_to_ego = _3dbox_bev_settings.get(
            "gps_to_ego",
            MotionDataset.default_gps_to_ego)
        _3dbox_corner_template = _3dbox_bev_settings.get(
            "3dbox_corner_template",
            MotionDataset.default_3dbox_corner_template)
        edge_indices = _3dbox_bev_settings.get(
            "line_indices",
            MotionDataset.default_3dbox_edge_indices)
        bev_from_ego_transform = np.array(bev_from_ego_transform, np.float32)
        world_to_gps = np.linalg.inv(gps_to_world)
        _3dbox_corner_template = np.array(_3dbox_corner_template, np.float32)

        color_table = _3dbox_bev_settings.get(
            "color_table", MotionDataset.default_3dbox_color_table)
        fill_box = _3dbox_bev_settings.get("fill_box", False)
        image = Image.new("RGB", tuple(bev_size))
        draw = ImageDraw.Draw(image)

        bboxes_to_draw = []
        for child in bboxes:
            if child.find("transform") is None:
                continue
            obj = KITTI360Bbox3D()
            obj.parseBbox(child)
            if obj.timestamp == -1 or obj.timestamp == timestamp:
                vertices, color = MotionDataset.transform_bboxes(
                    _3dbox_corner_template, obj, world_to_gps, gps_to_ego, bev_from_ego_transform, color_table)
                if vertices is None:
                    continue
                bboxes_to_draw.append((vertices, color))

        for vertices, color in bboxes_to_draw:
            if fill_box:
                # Convert vertices to a list of tuples for polygon drawing
                polygon_points = [(vertices[0, i], vertices[1, i])
                                  for i in range(vertices.shape[1])]
                draw.polygon(polygon_points, fill=color, width=pen_width)
            else:
                for i, j in edge_indices:
                    v1 = vertices[i, :]
                    v2 = vertices[j, :]
                    line_coords = (float(v1[0]), float(
                        v1[1]), float(v2[0]), float(v2[1]))
                    draw.line(line_coords, fill=color, width=pen_width)
        return image

    def __getitem__(self, index: int):
        segment = self.segments[index]

        # Load point cloud data
        lidar_points = []
        pose_ids = []
        for file_path in segment["files"]:
            point_data = np.frombuffer(
                self.fs.cat_file(file_path), dtype=np.float32)
            # KITTI360 point clouds have x, y, z, intensity
            # Reshape to (-1, 4) and take only x, y, z
            points = torch.tensor(point_data.reshape((-1, 4))[:, :3])
            lidar_points.append(points)
            pose_ids.append(int(file_path.split("/")[-1].split(".")[0]))
        
        # Get poses for each frame
        poses = []
        scene_name = segment["scene_name"]
        pose_dict = self.poses_dict[scene_name]
        pose_dict_keys = sorted(list(pose_dict.keys()))
        interpolated_poses = None
        interpolated_poses_id = None
        for pose_id in pose_ids:
            if pose_id in pose_dict:
                # If pose exists directly, use it
                pose = pose_dict[pose_id]
                # Add homogeneous row to make it 4x4
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :] = pose
                poses.append(pose_4x4)
            else:
                # Find nearest poses for interpolation
                smaller_ids = [k for k in pose_dict_keys if k < pose_id]
                larger_ids = [k for k in pose_dict_keys if k > pose_id]
                # If we can't interpolate, use the closest available pose
                if not larger_ids:
                    closest_id = max(smaller_ids)
                    pose = pose_dict[closest_id]
                    pose_4x4 = np.eye(4)
                    pose_4x4[:3, :] = pose
                    poses.append(pose_4x4)
                elif not smaller_ids:
                    closest_id = min(larger_ids)
                    pose = pose_dict[closest_id]
                    pose_4x4 = np.eye(4)
                    pose_4x4[:3, :] = pose
                    poses.append(pose_4x4)
                else:
                    # Usually the pose data is not aligned with the lidar data. The number of pose data is less than the number of lidar data.
                    # We need to interpolate the pose data to align with the lidar data.
                    if interpolated_poses_id is None or (pose_id > interpolated_poses_id[1] \
                        or pose_id < interpolated_poses_id[0]):
                        # Get nearest poses for interpolation
                        smaller_id = max(smaller_ids)
                        larger_id = min(larger_ids)
                        pose1 = pose_dict[smaller_id]
                        pose2 = pose_dict[larger_id]
                        
                        # Convert to 4x4 matrices
                        pose1_4x4 = np.eye(4)
                        pose1_4x4[:3, :] = pose1
                        pose2_4x4 = np.eye(4)
                        pose2_4x4[:3, :] = pose2

                        # Interpolate between the two poses
                        interpolated_poses = MotionDataset.interpolate_pose( # The return is a list of poses with length of larger_id - smaller_id - 1
                            pose1_4x4, pose2_4x4, larger_id - smaller_id)
                        interpolated_poses_id = (smaller_id, larger_id)

                    interpolation_index = pose_id - interpolated_poses_id[0] - 1
                    poses.append(interpolated_poses[interpolation_index])

        # Create timestamps (dummy values based on sequence)
        timestamps = torch.tensor(
            [[i * 100 for i in range(self.sequence_length)]],
            dtype=torch.float32
        )

        # Create result dictionary
        result = {
            "fps": torch.tensor(segment["fps"], dtype=torch.float32),
            "pts": timestamps,
            "lidar_points": lidar_points,
        }
        images = torch.zeros((self.sequence_length, 3, 256, 256))
        if "lidar_points" in result:
            x_transform = 0.81 - 0.05
            y_transform = 0.
            z_transform = 1.73 - 0.3
            result["lidar_transforms"] = torch.stack([
                torch.stack([torch.eye(4, dtype=torch.float32)]) for _ in range(self.sequence_length)
            ])
            result["lidar_transforms"][0, 0, :3, 3] = torch.tensor(
                [x_transform, y_transform, z_transform])

        # Add stub data if needed
        dwm.datasets.common.add_stub_key_data(self.stub_key_data_dict, result)

        # Set all other features to None or empty lists to maintain API compatibility
        if self.enable_scene_description:
            result["scene_description"] = ""

        if self.enable_camera_transforms:
            if "images" in result:
                # Create dummy transforms
                dummy_transform = torch.eye(
                    4, dtype=torch.float32).unsqueeze(0).expand(6, -1, -1)
                result["camera_transforms"] = torch.stack([
                    torch.stack([dummy_transform]) for _ in range(self.sequence_length)
                ])

        if self.enable_ego_transforms:
            dummy_transform = torch.eye(4, dtype=torch.float32)
            result["ego_transforms"] = torch.stack([
                torch.stack([dummy_transform]) for _ in range(self.sequence_length)
            ])

        if self.enable_sample_data:
            result["sample_data"] = [[{"filename": f}
                                      for f in segment["files"]]]
            result["scene"] = {"token": "kitti360_scene"}

        if self._3dbox_bev_settings is not None:
            # get the poses
            bboxes = self.bboxes_dict[segment["scene_name"]]
            result["3dbox_bev_images"] = [
                MotionDataset.get_3dbox_bev_image(
                    bboxes, p, t, self._3dbox_bev_settings)
                for p, t in zip(poses, segment["timestamps"])
            ]


        # All other features are not supported for KITTI360
        if self._3dbox_image_settings is not None:
            raise NotImplementedError("3dbox_images are not supported for KITTI360")

        if self.hdmap_image_settings is not None:
            raise NotImplementedError("hdmap_images are not supported for KITTI360")

        if self.image_segmentation_settings is not None:
            raise NotImplementedError("segmentation_images are not supported for KITTI360")

        if self.foreground_region_image_settings is not None:
            raise NotImplementedError("foreground_region_images are not supported for KITTI360")

        if self.image_description_settings is not None:
            raise NotImplementedError("image_description is not supported for KITTI360")

        return result

if __name__ == "__main__":
    pose_1 = torch.tensor(
        [[-0.1828219793, 0.9819305714, 0.0488720448, 3418.805027],
        [0.9772070805, 0.1869485883, -0.1005810473, -2999.815943], 
        [-0.107900165, 0.0293696821, -0.993727818, 244.1487682],]
    )
    pose_2 = torch.tensor(
        [[-0.1911552129, 0.9804793214, 0.0460432947, 3418.510026],
        [0.9753168003, 0.1950124872, -0.1035725294, -2998.252617],
        [-0.1105297407, 0.02510837, -0.993555608, 243.9838356]]
    )
    MotionDataset.interpolate_pose(
        pose_1,
        pose_2,
        num_steps=10
    )
        