from collections import defaultdict
import concurrent.futures
import json
from kitti360scripts.helpers.annotation import KITTI360Bbox3D
from kitti360scripts.helpers.labels import id2label
import numpy as np
import os
from pathlib import Path
import time
import tqdm
from transforms3d.quaternions import quat2mat
from typing import List, Tuple
import xml.etree.ElementTree as ET

from dwm.datasets.kitti360 import MotionDataset as KITTI360MotionDataset

box_colormap = np.array([
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0],
]).astype(np.float32)

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


cls_attr_dist = {
    'barrier': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bicycle': {
        'cycle.with_rider': 2791,
        'cycle.without_rider': 8946,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bus': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 9092,
        'vehicle.parked': 3294,
        'vehicle.stopped': 3881,
    },
    'car': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 114304,
        'vehicle.parked': 330133,
        'vehicle.stopped': 46898,
    },
    'construction_vehicle': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 882,
        'vehicle.parked': 11549,
        'vehicle.stopped': 2102,
    },
    'ignore': {
        'cycle.with_rider': 307,
        'cycle.without_rider': 73,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 165,
        'vehicle.parked': 400,
        'vehicle.stopped': 102,
    },
    'motorcycle': {
        'cycle.with_rider': 4233,
        'cycle.without_rider': 8326,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'pedestrian': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 157444,
        'pedestrian.sitting_lying_down': 13939,
        'pedestrian.standing': 46530,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'traffic_cone': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'trailer': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 3421,
        'vehicle.parked': 19224,
        'vehicle.stopped': 1895,
    },
    'truck': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 21339,
        'vehicle.parked': 55626,
        'vehicle.stopped': 11097,
    },
}


def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: np.ndarray = np.array([[1., 0., 0.],
                                                    [0., 1., 0.],
                                                    [0., 0., 1.]]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm

def get_available_scenes(nusc, lidar_root = ""):
    available_scenes = []
    print('total scene num:', len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if lidar_root != "":
                lidar_path = os.path.join(lidar_root, os.path.basename(lidar_path))
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num:', len(available_scenes))
    return available_scenes


def get_sample_data_nuscenes(nusc, sample_data_token, selected_anntokens=None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    Args:
        nusc:
        sample_data_token: Sample_data token.
        selected_anntokens: If provided only return the selected annotation.

    Returns:

    """
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record['translation']))
        box.rotate(quat2mat(pose_record['rotation']).T)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record['translation']))
        box.rotate(quat2mat(cs_record['rotation']).T)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic

def quaternion_yaw(q: np.ndarray) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw
    

def fill_infos_nuscenes(data_path, nusc, val_scenes, test=False):
    val_nusc_infos = []
    progress_bar = tqdm.tqdm(total=len(nusc.sample), desc='create_info', dynamic_ncols=True)

    ref_chan = 'LIDAR_TOP'  # The radar channel from which we track back n sweeps to aggregate the point cloud.

    for index, sample in enumerate(nusc.sample):
        progress_bar.update()
        if sample['scene_token'] not in val_scenes:     
            continue
        ref_sd_token = sample['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])

        ref_lidar_path, ref_boxes, _ = get_sample_data_nuscenes(nusc, ref_sd_token)
        lidar_path = os.path.join(data_path, os.path.basename(ref_lidar_path)).__str__() if data_path != "" else ref_lidar_path
        if not os.path.exists(lidar_path):
            continue

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec['translation'], quat2mat(ref_pose_rec['rotation']), inverse=True,
        )

        info = {
            'lidar_path': lidar_path,
            'car_from_global': car_from_global,
        }

        if not test:
            annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]
            # the filtering gives 0.5~1 map improvement
            num_lidar_pts = np.array([anno['num_lidar_pts'] for anno in annotations])
            num_radar_pts = np.array([anno['num_radar_pts'] for anno in annotations])
            mask = (num_lidar_pts + num_radar_pts > 0)

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]  # wlh == > dxdydz (lwh)
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)
            names = np.array([b.name for b in ref_boxes])
            gt_boxes = np.concatenate([locs, dims, rots], axis=1)

            assert len(annotations) == len(gt_boxes)

            info['gt_boxes'] = gt_boxes[mask, :]
            info['gt_names'] = np.array([map_name_from_general_to_detection[name] for name in names])[mask]

            val_nusc_infos.append(info)

    progress_bar.close()
    return val_nusc_infos

def load_map(map_dataroot, map_tables):
    map_expansion = {}
    map_expansion_dict = {}
    if os.path.exists(map_dataroot):
        for i in map_tables:
            to_dict = ["node", "polygon"]
            if i["location"] not in map_expansion:
                name = os.path.join(map_dataroot, i["location"] + ".json")
                with open(name, 'r') as f:
                    map_expansion[i["location"]] = json.load(f)
                map_expansion_dict[i["location"]] = {}
                for j in to_dict:
                    map_expansion_dict[i["location"]][j] = {
                        k["token"]: k
                        for k in map_expansion[i["location"]][j]
                }
    return map_expansion, map_expansion_dict


class Box:
    """ 
    This class is modified from https://github.com/nutonomy/nuscenes-devkit
    Simple data class representing a 3d box including, label, score and velocity. 
    """

    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: np.ndarray,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == np.ndarray

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The box's rotation matrix.
        """
        return self.orientation

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion: np.ndarray) -> None:
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion, self.center)
        self.orientation = quaternion @ self.orientation
        self.velocity = np.dot(quaternion, self.velocity)

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]


class NuScenesDataset:
    """
    This class is modified from https://github.com/nutonomy/nuscenes-devkit
    Database class for nuScenes to help query and retrieve information from the database.
    """

    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/sets/nuscenes',
                 verbose: bool = True,
                ):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0", ...).
        :param dataroot: Path to the tables and data.
        :param verbose: Whether to print status messages during load.
        :param map_resolution: Resolution of maps (meters).
        """
        self.version = version
        self.dataroot = dataroot
        self.table_names = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
                            'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']

        assert os.path.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)

        start_time = time.time()
        if verbose:
            print("======\nLoading NuScenes tables for version {}...".format(self.version))

        # Explicitly assign tables to help the IDE determine valid class members.
        self.category = self.__load_table__('category')
        self.attribute = self.__load_table__('attribute')
        self.visibility = self.__load_table__('visibility')
        self.instance = self.__load_table__('instance')
        self.sensor = self.__load_table__('sensor')
        self.calibrated_sensor = self.__load_table__('calibrated_sensor')
        self.ego_pose = self.__load_table__('ego_pose')
        self.log = self.__load_table__('log')
        self.scene = self.__load_table__('scene')
        self.sample = self.__load_table__('sample')
        self.sample_data = self.__load_table__('sample_data')
        self.sample_annotation = self.__load_table__('sample_annotation')
        self.map = self.__load_table__('map')

        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

    @property
    def table_root(self) -> str:
        """ Returns the folder where the tables are stored for the relevant version. """
        return os.path.join(self.dataroot, self.version)

    def __load_table__(self, table_name) -> dict:
        """ Loads a table. """
        with open(os.path.join(self.table_root, '{}.json'.format(table_name))) as f:
            table = json.load(f)
        return table
    def __make_reverse_index__(self, verbose: bool) -> None:
        """
        De-normalizes database to create reverse indices for common cases.
        :param verbose: Whether to print outputs.
        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member['token']] = ind

        # Decorate (adds short-cut) sample_annotation table with for category name.
        for record in self.sample_annotation:
            inst = self.get('instance', record['instance_token'])
            record['category_name'] = self.get('category', inst['category_token'])['name']

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self.get('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']

        # Reverse-index samples with sample_data and annotations.
        for record in self.sample:
            record['data'] = {}
            record['anns'] = []

        for record in self.sample_data:
            if record['is_key_frame']:
                sample_record = self.get('sample', record['sample_token'])
                sample_record['data'][record['channel']] = record['token']

        for ann_record in self.sample_annotation:
            sample_record = self.get('sample', ann_record['sample_token'])
            sample_record['anns'].append(ann_record['token'])

        # Add reverse indices from log records to map records.
        if 'log_tokens' not in self.map[0].keys():
            raise Exception('Error: log_tokens not in map table. This code is not compatible with the teaser dataset.')
        log_to_map = dict()
        for map_record in self.map:
            for log_token in map_record['log_tokens']:
                log_to_map[log_token] = map_record['token']
        for log_record in self.log:
            log_record['map_token'] = log_to_map[log_record['token']]

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))

    def get(self, table_name: str, token: str) -> dict:
        """
        Returns a record from table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: Table record. See README.md for record details for each table.
        """
        assert table_name in self.table_names, "Table {} not found".format(table_name)

        return getattr(self, table_name)[self.getind(table_name, token)]
    def getind(self, table_name: str, token: str) -> int:
        """
        This returns the index of the record in a table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: The index of the record in table, table is an array.
        """
        return self._token2ind[table_name][token]
    def get_sample_data_path(self, sample_data_token: str) -> str:
        """ Returns the path to a sample_data. """

        sd_record = self.get('sample_data', sample_data_token)
        return os.path.join(self.dataroot, sd_record['filename'])

    def get_sample_data(self, sample_data_token: str,
                        # selected_anntokens: List[str] = None,
                        # use_flat_vehicle_coordinates: bool = False
                        ):
        """
        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.
        :param sample_data_token: Sample_data token.
        :param selected_anntokens: If provided only return the selected annotation.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                             aligned to z-plane in the world.
        :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
        """

        # Retrieve sensor & pose records
        sd_record = self.get('sample_data', sample_data_token)
        cs_record = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = self.get('sensor', cs_record['sensor_token'])
        pose_record = self.get('ego_pose', sd_record['ego_pose_token'])

        data_path = self.get_sample_data_path(sample_data_token)

        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        else:
            cam_intrinsic = None

        boxes = self.get_boxes(sample_data_token)

        # Make list of Box objects including coord system transforms.
        box_list = []
        for box in boxes:
            box.translate(-np.array(pose_record['translation']))
            box.rotate(quat2mat(pose_record['rotation']).T)
            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(quat2mat(cs_record['rotation']).T)

            box_list.append(box)

        return data_path, box_list, cam_intrinsic


    def get_box(self, sample_annotation_token: str) -> Box:
        """
        Instantiates a Box class from a sample annotation record.
        :param sample_annotation_token: Unique sample_annotation identifier.
        """
        record = self.get('sample_annotation', sample_annotation_token)
        return Box(record['translation'], record['size'], quat2mat(record['rotation']),
                   name=record['category_name'], token=record['token'])

    def get_boxes(self, sample_data_token: str) -> List[Box]:
        """
        Instantiates Boxes for all annotation for a particular sample_data record. If the sample_data is a
        keyframe, this returns the annotations for that sample. But if the sample_data is an intermediate
        sample_data, a linear interpolation is applied to estimate the location of the boxes at the time the
        sample_data was captured.
        :param sample_data_token: Unique sample_data identifier.
        """

        # Retrieve sensor & pose records
        sd_record = self.get('sample_data', sample_data_token)
        curr_sample_record = self.get('sample', sd_record['sample_token'])

        if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:
            # If no previous annotations available, or if sample_data is keyframe just return the current ones.
            boxes = list(map(self.get_box, curr_sample_record['anns']))

        return boxes

    def box_velocity(self, sample_annotation_token: str, max_time_diff: float = 1.5) -> np.ndarray:
        """
        Estimate the velocity for an annotation.
        If possible, we compute the centered difference between the previous and next frame.
        Otherwise we use the difference between the current and previous/next frame.
        If the velocity cannot be estimated, values are set to np.nan.
        :param sample_annotation_token: Unique sample_annotation identifier.
        :param max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.
        :return: <np.float: 3>. Velocity in x/y/z direction in m/s.
        """

        current = self.get('sample_annotation', sample_annotation_token)
        has_prev = current['prev'] != ''
        has_next = current['next'] != ''

        # Cannot estimate velocity for a single annotation.
        if not has_prev and not has_next:
            return np.array([np.nan, np.nan, np.nan])

        if has_prev:
            first = self.get('sample_annotation', current['prev'])
        else:
            first = current

        if has_next:
            last = self.get('sample_annotation', current['next'])
        else:
            last = current

        pos_last = np.array(last['translation'])
        pos_first = np.array(first['translation'])
        pos_diff = pos_last - pos_first

        time_last = 1e-6 * self.get('sample', last['sample_token'])['timestamp']
        time_first = 1e-6 * self.get('sample', first['sample_token'])['timestamp']
        time_diff = time_last - time_first

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_time_diff *= 2

        if time_diff > max_time_diff:
            # If time_diff is too big, don't return an estimate.
            return np.array([np.nan, np.nan, np.nan])
        else:
            return pos_diff / time_diff


# loading kitti360 data
def get_bboxes_kitti360(bboxes, gps_to_world, timestamp):
    gps_to_ego = [
        [1,  0,  0, -0.05],
        [0, -1,  0,  0.32],
        [0,  0, -1,  0.60],
        [0,  0,  0,  1]
    ]
    _3dbox_corner_template = KITTI360MotionDataset.default_3dbox_corner_template
    world_to_gps = np.linalg.inv(gps_to_world)
    _3dbox_corner_template = np.array(_3dbox_corner_template, np.float32)

    gt_boxes = []
    gt_names = []
    for obj in bboxes:
        if obj.timestamp == -1 or obj.timestamp == timestamp:
            temp_to_global = np.concatenate([obj.R, obj.T[..., None]], axis=-1)
            temp_to_global = np.vstack([temp_to_global, np.array([0, 0, 0, 1])])
            transform_matrix_ = gps_to_ego @ world_to_gps @ temp_to_global
            vertices = transform_matrix_ @ _3dbox_corner_template.T
            vertices = vertices[:3].T
            label = id2label[obj.semanticId].name

            if label not in KITTI360MotionDataset.default_label_mapping:
                continue
            # convert the 8 points vertices to center, size, yaw
            center = np.mean(vertices, axis=0)

            dy = np.linalg.norm(vertices[0, :2] - vertices[2, :2])
            dx = np.linalg.norm(vertices[2, :2] - vertices[6, :2])
            dz = vertices[1, 2] - vertices[0, 2]
            rot_vertex = (vertices[0, :2] + vertices[2, :2]) / 2
            yaw = np.arctan2(rot_vertex[1] - center[1], rot_vertex[0] - center[0])
            gt_boxes.append(np.concatenate([center, [dx, dy, dz], [yaw]]))
            gt_names.append(label)

    return np.array(gt_boxes), np.array(gt_names)


def fill_infos_kitti360(data_path = None, kitti360 = None):
    val_nusc_infos = []
    progress_bar = tqdm.tqdm(total=len(kitti360.sample_data), desc='create_info', dynamic_ncols=True)

    def process_sample(sample):
        if data_path is not None:
            lidar_file_name = os.path.basename(sample["files"])
            lidar_file = os.path.join(data_path, sample["scene_name"], lidar_file_name)
        else:
            lidar_file = sample["files"]
        if not os.path.exists(lidar_file):
            return None
        pose_id = int(lidar_file.split("/")[-1].split(".")[0])
        pose_dict = kitti360.poses_dict[sample["scene_name"]]
        pose_dict_keys = sorted(list(pose_dict.keys()))
        interpolated_poses = None
        interpolated_poses_id = None
        if pose_id in pose_dict:
            pose = pose_dict[pose_id]
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :] = pose
            pose = pose_4x4
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
                pose = pose_4x4
            elif not smaller_ids:
                closest_id = min(larger_ids)
                pose = pose_dict[closest_id]
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :] = pose
                pose = pose_4x4
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
                    interpolated_poses = KITTI360MotionDataset.interpolate_pose( # The return is a list of poses with length of larger_id - smaller_id - 1
                        pose1_4x4, pose2_4x4, larger_id - smaller_id)
                    interpolated_poses_id = (smaller_id, larger_id)

                interpolation_index = pose_id - interpolated_poses_id[0] - 1
                pose = interpolated_poses[interpolation_index]
        gt_boxes, gt_names = get_bboxes_kitti360(
            kitti360.bboxes_dict[sample["scene_name"]], 
            pose, 
            sample["timestamp"], 
        )
        info = {
            'lidar_path': lidar_file,
            'gt_boxes': gt_boxes,
            'gt_names': gt_names,
            'scene_name': sample["scene_name"],
        }
        return info

    with concurrent.futures.ThreadPoolExecutor(max_workers=60) as executor:
        futures = [executor.submit(process_sample, sample) for sample in kitti360.sample_data]
        for future in concurrent.futures.as_completed(futures):
            progress_bar.update()
            val_nusc_infos.append(future.result())

    progress_bar.close()
    return val_nusc_infos


class Kitti360Dataset:
    def __init__(
        self,
        dataroot: str = '/data/sets/kitti360',
        verbose: bool = True,
    ):
        self.dataroot = dataroot
        # Get all the bin files
        all_files = []
        for (dirpath, dirnames, filenames) in os.walk(dataroot):
            for filename in filenames:
                if filename.endswith(".bin"):
                    all_files.append(os.path.join(dirpath, filename))

        
        self.file_paths = sorted(list(filter(
                lambda folder: "0000_sync" in folder or "0002_sync" in folder, all_files)))
        self.sample_data = []

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
            pose_file = os.path.join(self.dataroot, "data_poses", base_dir, "poses.txt")
            with open(pose_file, "r") as f:
                pose_data = f.readlines()
            all_poses = {}
            for pose_idx in range(len(pose_data)):
                split_data = pose_data[pose_idx].split()
                pose_id = int(split_data[0])
                pose = np.array(split_data[1:]).astype(
                    np.float32).reshape(3, 4)
                all_poses[pose_id] = pose
            self.poses_dict[base_dir] = all_poses
            # load the bbox data
            bbox_file = os.path.join(self.dataroot, "data_3d_bboxes", "train", f"{base_dir}.xml")
            with open(bbox_file, "rb") as bbox_data:
                tree = ET.parse(bbox_data)
            root = tree.getroot()
            self.bboxes_dict[base_dir] = root
        self.file_groups = file_groups
        for base_dir, files in file_groups.items():
            files_len = len(files["lidar_files"])  
            # Use time-based stride (not implemented for KITTI360)
            # For simplicity, we"ll just use index-based stride for now
            for i in range(0, files_len, 1):
                self.sample_data.append({
                    "files": files["lidar_files"][i],
                    "scene_name": base_dir,
                    "timestamp": i,
                })
        for k, v in self.bboxes_dict.items():
            objs = []
            print(k)
            for child in v:
                if child.find("transform") is None:
                    continue
                obj = KITTI360Bbox3D()
                obj.parseBbox(child)
                objs.append(obj)
            self.bboxes_dict[k] = objs
        

        # self.sample_data = self.sample_data

