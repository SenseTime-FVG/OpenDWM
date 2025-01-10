import torch
import numpy as np
#import open3d as o3d
import matplotlib.pyplot as plt
from .histogram import *
import glob
import random
import os


def get_all_files_in(dir):
    all_files = glob.glob(dir + "/*")
    return all_files


def load_range_images(folder):
    files = get_all_files_in(folder)

    all_arrays = []
    for file in files:
        if '.pth' in file:
            sample = torch.load(file).numpy()
            all_arrays.append(sample[0])

    return all_arrays


def load_kitti(count, seed):
    full_list = glob.glob(os.environ.get('KITTI360_DATASET') +
                          '/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/*')
    full_list.extend(glob.glob(os.environ.get('KITTI360_DATASET') +
                     '/data_3d_raw/2013_05_28_drive_0002_sync/velodyne_points/data/*'))
    random.Random(seed).shuffle(full_list)
    full_list = full_list[0:count]

    all_arrays = []
    for file in full_list:
        range_image = point_cloud_to_range_image(file)
        range_image = np.where(range_image < 0, 0, range_image) + 0.0001
        range_image = ((np.log2(range_image+1)) / 6)

        range_image = np.clip(range_image, 0, 1)

        all_arrays.append(range_image)

    return all_arrays


def randomly_split_in_half(arr, length):
    random.shuffle(arr)
    return (arr[0:length], arr[length:length+length])


def get_histogram_for_range(range_image):
    point_cloud = range_image_to_point_cloud_fast(range_image)

    #point_cloud_3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))
    # o3d.visualization.draw_geometries([point_cloud_3d])

    histogram = point_cloud_to_histogram(160, 100, point_cloud)[0]
    return histogram

def get_histogram_for_range_nuscene(point_cloud):
    # point_cloud = range_image_to_point_cloud_fast_nuscenes(range_image)

    #point_cloud_3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))
    # o3d.visualization.draw_geometries([point_cloud_3d])

    histogram = point_cloud_to_histogram(160, 100, point_cloud)[0]
    return histogram

def load_ncsn(folder):
    files = get_all_files_in(folder)

    all_arrays = []
    for file in files:
        if '.pth' in file:
            samples = torch.load(file).numpy()
            for i in range(0, samples.shape[0]):
                all_arrays.append(samples[i][0])

    return all_arrays

def array_to_histograms(samples):
    result = []

    for sample in samples:
        hist = None
        '''
        with np.errstate(over='raise'):
            try:
                hist = get_histogram_for_range(sample)
            except:
                print('Skipping overflow')
                continue
        '''
        hist = get_histogram_for_range(sample)
        result.append(hist)

    return result

def array_to_histograms_nuscene(samples):
    result = []

    for sample in samples:
        hist = None
        '''
        with np.errstate(over='raise'):
            try:
                hist = get_histogram_for_range(sample)
            except:
                print('Skipping overflow')
                continue
        '''
        hist = get_histogram_for_range_nuscene(sample)
        result.append(hist)

    return result


def visualize_histogram(histograms):

    plt.imshow(np.log(histograms[0]+1))
    plt.show()
    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 2
    for i in range(1, columns*rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.log(histograms[i-1]+1))
    plt.show()


class NuscenesLaserScan:
    def __init__(self, project=False, H=32, W=1024, fov_up=10.0, fov_down=-30.0):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros(
            (0, 3), dtype=np.float32)        # [m, 3]: x, y, z
        self.remissions = np.zeros(
            (0, 1), dtype=np.float32)    # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)       # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def set_points(self, points, remissions=None):
        """ Set scan attributes (instead of opening from file)
        """
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute
        self.points = points    # get xyz
        if remissions is not None:
            self.remissions = remissions  # get remission
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()

    def do_range_projection(self):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W                              # in [0.0, W]
        proj_y *= self.proj_H                              # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.float32)


def load_nuscenes_laserscan(file):
    #import open3d as o3d

    raw = np.fromfile(file, dtype=np.float32)
    points = raw.reshape((-1, 5))

    #pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:,0:3]))
    # o3d.visualization.draw_geometries([pc])

    result = NuscenesLaserScan()
    result.set_points(points[:, 0:3], remissions=points[:, 4])
    result.do_range_projection()
    return result
