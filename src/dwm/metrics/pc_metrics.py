import torch
import torchmetrics
import torch.distributed
from torchmetrics import Metric

from dwm.utils.metrics_copilot4d import (
    compute_chamfer_distance,
    compute_chamfer_distance_inner,
    compute_mmd,
    jsd_2d, 
    gaussian,
    point_cloud_to_histogram
)

class PointCloudChamfer(Metric):
    def __init__(self, inner_dist=None, **kwargs):
        super().__init__(**kwargs)
        self.inner_dist = inner_dist

        self.cd_func = compute_chamfer_distance if self.inner_dist is None else compute_chamfer_distance_inner
        self.chamfer_list = []

    def update(self, pred_pcd, gt_pcd, device=None):
        for pred, gt in zip(pred_pcd, gt_pcd):
            for p, g in zip(pred, gt):
                if self.inner_dist is None:
                    cd = self.cd_func(p.to(torch.float32), g.to(torch.float32), device=device)
                else:
                    cd = self.cd_func(p.to(torch.float32), g.to(torch.float32), device=device, pc_range=[
                                    -self.inner_dist, -self.inner_dist, -3, self.inner_dist, self.inner_dist, 5])
                if not isinstance(cd, torch.Tensor):
                    cd = torch.tensor(cd).to(device)
                self.chamfer_list.append(cd)

    def compute(self):
        chamfer_list = torch.stack(self.chamfer_list, dim=0)
        world_size = torch.distributed.get_world_size() \
            if torch.distributed.is_initialized() else 1
        if world_size > 1:
            all_chamfer = chamfer_list.new_zeros(
                (len(chamfer_list)*world_size, ) + chamfer_list.shape[1:])
            torch.distributed.all_gather_into_tensor(
                all_chamfer, chamfer_list)
            chamfer_list = all_chamfer
        num_samples = (~torch.isnan(chamfer_list) & ~torch.isinf(chamfer_list)).sum()
        chamfer_list = torch.nan_to_num(chamfer_list, nan=0.0, posinf=0.0, neginf=0.0)
        return chamfer_list.sum() / num_samples

    def reset(self):
        self.chamfer_list.clear()
        super().reset()


class PointCloudMMD(Metric):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two point clouds.
    """
    def __init__(self, field_size=160, bins=100, **kwargs):
        super().__init__(**kwargs)
        self.field_size = field_size
        self.bins = bins
        self.mmd_list = []
    def update(self, pred_pcd, gt_pcd, device=None):
        pred_hist = []
        gt_hist = []
        for pred, gt in zip(pred_pcd, gt_pcd):
            for p, g in zip(pred, gt):
                p = point_cloud_to_histogram(self.field_size, self.bins, p.to(torch.float32))[0]
                g = point_cloud_to_histogram(self.field_size, self.bins, g.to(torch.float32))[0]
                pred_hist.append(p)
                gt_hist.append(g)
        mmd = compute_mmd(pred_hist, gt_hist, kernel=gaussian, is_parallel=True)
        if not isinstance(mmd, torch.Tensor):
            mmd = torch.tensor(mmd).to(device)
        self.mmd_list.append(mmd)

    def compute(self):
        mmd_list = torch.stack(self.mmd_list, dim=0)
        world_size = torch.distributed.get_world_size() \
            if torch.distributed.is_initialized() else 1
        if world_size > 1:
            all_mmd = mmd_list.new_zeros(
                (len(mmd_list)*world_size, ) + mmd_list.shape[1:])
            torch.distributed.all_gather_into_tensor(
                all_mmd, mmd_list)
            mmd_list = all_mmd
        num_samples =(~torch.isnan(mmd_list) & ~torch.isinf(mmd_list)).sum()
        mmd_list = torch.nan_to_num(mmd_list, nan=0.0, posinf=0.0, neginf=0.0)
        return mmd_list.sum() / num_samples

    def reset(self):
        self.mmd_list.clear()
        super().reset()

class PointCloudJSD(Metric):
    def __init__(self, field_size=160, bins=100, **kwargs):
        super().__init__(**kwargs)
        self.field_size = field_size
        self.bins = bins
        self.jsd_list = []

    def update(self, pred_pcd, gt_pcd, device=None):
        for pred, gt in zip(pred_pcd, gt_pcd):
            for p, g in zip(pred, gt):
                p = point_cloud_to_histogram(self.field_size, self.bins, p)[0]
                g = point_cloud_to_histogram(self.field_size, self.bins, g)[0]
                jsd = jsd_2d(p, g)
                if not isinstance(jsd, torch.Tensor):
                    jsd = torch.tensor(jsd).to(device)
                self.jsd_list.append(jsd)

    def compute(self):
        jsd_list = torch.stack(self.jsd_list, dim=0)
        world_size = torch.distributed.get_world_size() \
            if torch.distributed.is_initialized() else 1
        if world_size > 1:
            all_jsd = jsd_list.new_zeros(
                (len(jsd_list)*world_size, ) + jsd_list.shape[1:])
            torch.distributed.all_gather_into_tensor(
                all_jsd, jsd_list)
            jsd_list = all_jsd
        num_samples = (~torch.isnan(jsd_list) & ~torch.isinf(jsd_list)).sum()
        jsd_list = torch.nan_to_num(jsd_list, nan=0.0, posinf=0.0, neginf=0.0)
        return jsd_list.sum() / num_samples

    def reset(self):
        self.jsd_list.clear()
        super().reset()
