import torch
import torchmetrics
import torch.distributed
from dwm.utils.metrics_copilot4d import compute_chamfer_distance, compute_chamfer_distance_inner, compute_ray_errors


class PointCloudChamfer(torchmetrics.Metric):
    def __init__(self, inner_dist=None, **kwargs):
        super().__init__(**kwargs)
        self.inner_dist = inner_dist

        self.cd_func = compute_chamfer_distance if self.inner_dist is None else compute_chamfer_distance_inner
        self.chamfer_list = []

    def update(self, pred_pcd, gt_pcd, device):
        for pred, gt in zip(pred_pcd, gt_pcd):
            pred = pred[0] if isinstance(
                pred, tuple) or isinstance(pred, list) else pred
            gt = gt[0] if isinstance(gt, tuple) or isinstance(gt, list) else gt
            if self.inner_dist is None:
                cd = self.cd_func(pred, gt, device=device)
            else:
                cd = self.cd_func(pred, gt, device=device, pc_range=[
                                  -self.inner_dist, -self.inner_dist, -3, self.inner_dist, self.inner_dist, 5])
            if not isinstance(cd, torch.Tensor):
                cd = torch.tensor(cd).to(device)
            cd = torch.nan_to_num(cd, nan=0.0)
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
        self.num_samples = torch.count_nonzero(chamfer_list)
        return chamfer_list.sum() / self.num_samples

    def reset(self):
        self.chamfer_list.clear()
        super().reset()
