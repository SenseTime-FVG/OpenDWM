from chamferdist import ChamferDistance
from torch.utils.data import DataLoader
from scipy.interpolate import interp2d
import torch
import struct
import numpy as np
import copy
import json
import argparse
from tqdm import tqdm
import cv2
import concurrent.futures
from functools import partial

# ===COPILOT4D
# PC_RANGE = [-70.0, -70.0, -4.5, 70.0, 70.0, 4.5]
# PC_RANGE = [-50.0, -50.0, -5, 50.0, 50.0, 3]
# PC_RANGE = [-30.0, -30.0, -5, 30.0, 30.0, 3]

DEFAULT_HZ = {"kittiraw": 10, "nuscenes": 2}
chamfer_distance = ChamferDistance()
MAX_VALUE = 1e8


def get_nusc_ego_mask(points):
    ego_mask = np.logical_and(
        np.logical_and(-1.0 <= points[0], points[0] <= 1.0),
        np.logical_and(-1.5 <= points[1], points[1] <= 2.5),
    )
    return ego_mask


def point_at_infinity():
    return torch.from_numpy(np.array([np.inf]*3))


def clamp(pcd_, org_, return_invalid_mask=False, pc_range=[-70.0, -70.0, -4.5, 70.0, 70.0, 4.5]):
    if torch.is_tensor(pcd_):
        pcd = pcd_.cpu().numpy()
    else:
        pcd = pcd_.copy()
    if torch.is_tensor(org_):
        org = org_.cpu().numpy()
    else:
        org = org_.copy()
    # print("unique values in gt before", np.unique(pcd))
    mask1 = np.logical_and(pc_range[0] <= pcd[:, 0], pcd[:, 0] <= pc_range[3])
    mask2 = np.logical_and(pc_range[1] <= pcd[:, 1], pcd[:, 1] <= pc_range[4])
    mask3 = np.logical_and(pc_range[2] <= pcd[:, 2], pcd[:, 2] <= pc_range[5])
    inner_mask = mask1 & mask2 & mask3
    origin = org.reshape((1, 3))
    origins = np.zeros_like(pcd) + origin
    if np.logical_not(inner_mask).sum() > 0:
        pcd_outer = pcd[np.logical_not(inner_mask)]
        origin_outer, clamped_pcd_outer = _clamp(
            pcd_outer, origin, pc_range=pc_range)
        pcd[np.logical_not(inner_mask)] = clamped_pcd_outer.astype(float)
        origins[np.logical_not(inner_mask)] = origin_outer

    invalid1 = np.logical_and(
        np.isinf(pcd[:, 0]),
        np.logical_and(
            np.isinf(pcd[:, 1]),
            np.isinf(pcd[:, 2])))
    invalid2 = np.logical_and(
        np.isnan(pcd[:, 0]),
        np.logical_and(
            np.isnan(pcd[:, 1]),
            np.isnan(pcd[:, 2])))
    invalid = np.logical_or(invalid1, invalid2)

    if not return_invalid_mask:
        origins = origins[np.logical_not(invalid)]
        pcd = pcd[np.logical_not(invalid)]
        return origins, pcd
    if return_invalid_mask:
        return origins, pcd, invalid


def inside_volume(xyz, pc_range=[-70.0, -70.0, -4.5, 70.0, 70.0, 4.5]):
    x, y, z = xyz.T
    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    xmin, ymin, zmin, xmax, ymax, zmax = xmin - 0.02, ymin - \
        0.02, zmin - 0.02, xmax + 0.02, ymax + 0.02, zmax + 0.02
    return np.logical_and(
        xmin <= x,
        np.logical_and(
            x <= xmax,
            np.logical_and(
                ymin <= y,
                np.logical_and(
                    y <= ymax,
                    np.logical_and(
                        zmin <= z,
                        z <= zmax)))))


def _clamp(points, origin, pc_range=[-70.0, -70.0, -4.5, 70.0, 70.0, 4.5]):
    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    # points = torch.from_numpy(np.array([[5.99, 2.70, -4.92]]))
    new_origin = np.zeros_like(points) + origin
    # print("ray clamping", origin.tolist(), points.tolist(), end="\t")
    # ray starting point
    xo, yo, zo = origin.T
    # ray end point
    xe, ye, ze = points.T
    # degenerate ray
    mask = np.logical_and(
        (xe - xo) == 0,
        np.logical_and(
            (ye - yo) == 0,
            (ze - zo) == 0))
    if mask.sum() > 0:
        raise RuntimeError(
            "Or`igin and the end point should not be identical at", points[mask])
        # if xo == xe and yo == ye and zo == ze:
        # return (point_at_infinity(), point_at_infinity())
    # ray raw length
    l = np.sqrt((xe-xo)**2 + (ye-yo)**2 + (ze-zo)**2)
    # non-zero
    # offset along x, y, z per unit movement
    dx = (xe - xo) / l
    dy = (ye - yo) / l
    dz = (ze - zo) / l
    # unit direction vector
    d = np.stack([dx, dy, dz])  # shape: 3 x N

    # check if the origin is inside the volume
    # print("bskbvkdr", points[37], new_origin[37])
    ray_intersects_volume = np.ones(d.shape[1]).astype(bool)
    if not inside_volume(origin, pc_range=pc_range):
        # print("inside the not inside volume", points[0])
        ray_intersects_volume = np.logical_not(ray_intersects_volume)
        # distance to planes along the ray direction
        origin_to_xmin = np.where(np.isclose(
            dx, 0.0), MAX_VALUE, (xmin - xo) / dx)
        origin_to_xmax = np.where(np.isclose(
            dx, 0.0), MAX_VALUE, (xmax - xo) / dx)
        origin_to_ymin = np.where(np.isclose(
            dy, 0.0), MAX_VALUE, (ymin - yo) / dy)
        origin_to_ymax = np.where(np.isclose(
            dy, 0.0), MAX_VALUE, (ymax - yo) / dy)
        origin_to_zmin = np.where(np.isclose(
            dz, 0.0), MAX_VALUE, (zmin - zo) / dz)
        origin_to_zmax = np.where(np.isclose(
            dz, 0.0), MAX_VALUE, (zmax - zo) / dz)
        # sort distance
        origin_to_planes = np.stack([origin_to_xmin, origin_to_xmax,
                                     origin_to_ymin, origin_to_ymax,
                                     origin_to_zmin, origin_to_zmax])  # shape: 6 x N
        lambda_order = np.argsort(origin_to_planes, axis=0)
        # find each plane in order
        for j in range(lambda_order.shape[1]):
            for i in range(lambda_order.shape[0]):
                # print("hfsgjkahf", points[37], new_origin[37])
                plane = lambda_order[i][j]
                if origin_to_planes[plane][j] + 1e-4 >= 0.0:
                    intersection = origin + \
                        origin_to_planes[plane][j] * d[:, j]
                    if inside_volume(intersection, pc_range=pc_range):
                        ray_intersects_volume[j] = True
                        new_origin[j] = intersection.copy()
                        if origin_to_planes[plane][j] > l[j]:
                            points[j] = new_origin[j].copy()
                        break

    # distance to planes along the reversed ray direction
    point_to_xmin = np.where(np.isclose(
        dx, 0.0), MAX_VALUE, (xmin - xe) / (-dx))
    point_to_xmax = np.where(np.isclose(
        dx, 0.0), MAX_VALUE, (xmax - xe) / (-dx))
    point_to_ymin = np.where(np.isclose(
        dy, 0.0), MAX_VALUE, (ymin - ye) / (-dy))
    point_to_ymax = np.where(np.isclose(
        dy, 0.0), MAX_VALUE, (ymax - ye) / (-dy))
    point_to_zmin = np.where(np.isclose(
        dz, 0.0), MAX_VALUE, (zmin - ze) / (-dz))
    point_to_zmax = np.where(np.isclose(
        dz, 0.0), MAX_VALUE, (zmax - ze) / (-dz))
    # sort distance
    point_to_planes = np.stack([point_to_xmin, point_to_xmax,
                                point_to_ymin, point_to_ymax,
                                point_to_zmin, point_to_zmax])
    lambda_order = np.argsort(point_to_planes, axis=0)

    for j, point in enumerate(points):
        if not ray_intersects_volume[j]:
            new_origin[j] = point_at_infinity()
            points[j] = point_at_infinity()
        else:
            if not inside_volume(point, pc_range=pc_range):
                # find each plane in order
                touches_volume = False
                for i in range(lambda_order.shape[0]):
                    plane = lambda_order[i][j]
                    if point_to_planes[plane][j] + 1e-4 >= 0.0:
                        intersection = points[j] + \
                            point_to_planes[plane][j] * (-d[:, j])
                        if inside_volume(intersection, pc_range=pc_range):
                            touches_volume = True
                            points[j] = intersection.copy()
                            break
                assert (touches_volume)

    return (new_origin, points)


def compute_chamfer_distance_inner(pred_pcd, gt_pcd, device, pc_range=[-70.0, -70.0, -4.5, 70.0, 70.0, 4.5], savename=""):
    mask1 = np.logical_and(
        pc_range[0] <= pred_pcd[:, 0], pred_pcd[:, 0] <= pc_range[3])
    mask2 = np.logical_and(
        pc_range[1] <= pred_pcd[:, 1], pred_pcd[:, 1] <= pc_range[4])
    mask3 = np.logical_and(
        pc_range[2] <= pred_pcd[:, 2], pred_pcd[:, 2] <= pc_range[5])
    inner_mask_pred = mask1 & mask2 & mask3
    mask1 = np.logical_and(
        pc_range[0] <= gt_pcd[:, 0], gt_pcd[:, 0] <= pc_range[3])
    mask2 = np.logical_and(
        pc_range[1] <= gt_pcd[:, 1], gt_pcd[:, 1] <= pc_range[4])
    mask3 = np.logical_and(
        pc_range[2] <= gt_pcd[:, 2], gt_pcd[:, 2] <= pc_range[5])
    inner_mask_gt = mask1 & mask2 & mask3
    if inner_mask_pred.sum() == 0 or inner_mask_gt.sum() == 0:
        return 0.0
    pred_pcd_inner = pred_pcd[inner_mask_pred]
    gt_pcd_inner = gt_pcd[inner_mask_gt]
    chamfer_dist_value = chamfer_distance(
        pred_pcd_inner[None, ...].to(device),
        gt_pcd_inner[None, ...].to(device),
        bidirectional=True,
        point_reduction='mean'
    )
    return chamfer_dist_value / 2.0


def compute_chamfer_distance(pred_pcd, gt_pcd, device, savename=""):
    chamfer_dist_value = chamfer_distance(
        pred_pcd[None, ...].to(device),
        gt_pcd[None, ...].to(device),
        bidirectional=True,
        point_reduction='mean'
    )
    return chamfer_dist_value / 2.0


def spherical_projection(pcd):
    pcd = pcd.T
    d = np.sqrt(pcd[0] * pcd[0] + pcd[1] * pcd[1] + pcd[2] * pcd[2])
    azimuth = np.arctan2(pcd[0], pcd[1])
    elevation = np.arctan2(pcd[2], pcd[1])
    return azimuth, elevation, d


def compute_ray_errors(pred_pcd, gt_pcd, origin, device, return_interpolated_pcd=False, savename="",
                       pc_range=[-70.0, -70.0, -4.5, 70.0, 70.0, 4.5], pipeline=None, skip_l1_align=False):
    pred_pcd_norm = pred_pcd - origin
    gt_pcd_norm = gt_pcd - origin
    # for points with origin = [0, 0, 0], to azimuth, elevation, d
    theta_hat, phi_hat, d_hat = spherical_projection(pred_pcd_norm)
    theta, phi, d = spherical_projection(gt_pcd_norm)

    mask_hat = d_hat > 1e-2
    mask = d > 1e-2
    if skip_l1_align:
        merge_mask = mask_hat & mask
        mask_hat = merge_mask
        mask = merge_mask
    theta_hat, phi_hat, d_hat, pred_pcd = theta_hat[mask_hat], phi_hat[
        mask_hat], d_hat[mask_hat], pred_pcd[mask_hat]
    theta, phi, d, gt_pcd = theta[mask], phi[mask], d[mask], gt_pcd[mask]

    count = theta.shape[0]
    pred_spherical = torch.stack(
        [theta_hat, phi_hat, torch.ones_like(theta_hat)], axis=1)
    gt_spherical = torch.stack([theta, phi, torch.ones_like(theta)], axis=1)

    if skip_l1_align:
        pred_idx = torch.arange(gt_pcd.shape[0])[None].to(gt_pcd.device)
    else:
        _, info = chamfer_distance(
            pred_spherical[None, ...].to(device),
            gt_spherical[None, ...].to(device),
            reverse=True,
            point_reduction='mean'
            # reduction='mean'
        )      # 0.2657 / 10540
        _, pred_idx = info
    v = gt_pcd - origin[None, :]
    unit_dir = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))

    pred_idx = pred_idx.cpu()
    pred_pcd_interp = origin[None, :] + d_hat[pred_idx].T * unit_dir
    if return_interpolated_pcd:
        return pred_pcd_interp
    # draw_points(gt_pcd, pipeline, 'gt')
    # draw_points(pred_pcd_interp, pipeline, 'pred')
    # draw_points(pred_pcd, pipeline, 'pred_2')
    clamped_gt_origin, clamped_gt_pcd, invalid_mask = clamp(
        gt_pcd, origin, return_invalid_mask=True, pc_range=pc_range)
    _, clamped_pred_pcd_interp, _ = clamp(
        pred_pcd_interp, origin, return_invalid_mask=True, pc_range=pc_range)
    clamped_gt_pcd = clamped_gt_pcd[np.logical_not(invalid_mask)]
    clamped_pred_pcd_interp = clamped_pred_pcd_interp[np.logical_not(
        invalid_mask)]
    clamped_gt_origin = clamped_gt_origin[np.logical_not(invalid_mask)]
    # !!! this make it very large
    d_clamped = np.sqrt(
        ((clamped_gt_pcd - clamped_gt_origin) ** 2).sum(axis=1))
    valid = d_clamped > 0.01
    d_clamped = d_clamped[valid]
    clamped_gt_pcd = clamped_gt_pcd[valid]
    clamped_pred_pcd_interp = clamped_pred_pcd_interp[valid]
    eucl_dist = np.sqrt(
        ((clamped_gt_pcd - clamped_pred_pcd_interp) ** 2).sum(axis=1))
    l1_error = eucl_dist
    absrel_error = eucl_dist / d_clamped

    return l1_error.sum() / count, absrel_error.sum() / count, torch.median(torch.from_numpy(l1_error)), torch.median(torch.from_numpy(absrel_error))

# === Debug code


def draw_points(pts, pipeline, suffix):
    with torch.no_grad():
        voxels = pipeline.vq_point_cloud.voxelizer([[pts]])[0]
        image = (voxels.max(dim=0)[0][:, :, None].repeat(
            1, 1, 3).detach().cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(
            f'/mnt/storage/user/nijingcheng/workspace/codes/DWM/work_dirs/tmp/debug_{suffix}.jpg', image)

# === Metric ultra_lidar


def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)


def disc(samples1, samples2, kernel, is_parallel=False, *args, **kwargs):
    """Discrepancy between 2 samples"""
    d = 0

    if not is_parallel:
        for s1 in tqdm(samples1):
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=128) as executor:
            for dist in executor.map(
                kernel_parallel_worker, [(s1, samples2, partial(
                    kernel, *args, **kwargs)) for s1 in samples1]
            ):
                d += dist

    d /= len(samples1) * len(samples2)
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """MMD between two samples"""
    # normalize histograms into pmf
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]
    # print('===============================')
    # print('s1: ', disc(samples1, samples1, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('s2: ', disc(samples2, samples2, kernel, *args, **kwargs))
    # print('--------------------------')
    cross = disc(samples1, samples2, kernel, *args, **kwargs)
    print("cross: ", cross)
    print("===============================")
    return (
        disc(samples1, samples1, kernel, *args, **kwargs)
        + disc(samples2, samples2, kernel, *args, **kwargs)
        - 2 * cross
    )


def gaussian(x, y, sigma=0.5):
    support_size = max(len(x), len(y))
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    # TODO: Calculate empirical sigma by fitting dist to gaussian
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def jsd_2d(p, q):
    p = p / p.sum()
    q = q / q.sum()
    from scipy.spatial.distance import jensenshannon

    return jensenshannon(p.flatten(), q.flatten())


def point_cloud_to_histogram(field_size, bins, point_cloud):
    point_cloud_flat = point_cloud[:, 0:2]  # .cpu().detach().numpy()
    square_size = field_size / bins
    halfway_offset = 0
    if bins % 2 == 0:
        halfway_offset = (bins / 2) * square_size
    else:
        print("ERROR")
    histogram = np.histogramdd(
        point_cloud_flat, bins=bins, range=(
            [-halfway_offset, halfway_offset], [-halfway_offset, halfway_offset])
    )

    return histogram
