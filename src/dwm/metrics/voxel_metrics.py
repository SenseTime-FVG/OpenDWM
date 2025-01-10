import torch
import torchmetrics
import torch.distributed


class VoxelIoU(torchmetrics.Metric):
    def __init__(
        self, **kwargs
    ):
        super().__init__(**kwargs)
        self.iou_list = []

    def update(self,
               pred_voxel: torch.tensor,
               gt_voxel: torch.tensor):
        if len(gt_voxel.shape) == 3:
            self.iou_list.append(
                (pred_voxel & gt_voxel).sum() / (pred_voxel | gt_voxel).sum())
        else:
            for i, j in zip(gt_voxel, pred_voxel):
                self.iou_list.append((i & j).sum() / (i | j).sum())

    def compute(self):
        iou_list = torch.stack(self.iou_list, dim=0)
        world_size = torch.distributed.get_world_size() \
            if torch.distributed.is_initialized() else 1
        if world_size > 1:
            all_iou = iou_list.new_zeros(
                (len(iou_list)*world_size, ) + iou_list.shape[1:])
            torch.distributed.all_gather_into_tensor(
                all_iou, iou_list)
            iou_list = all_iou
        self.num_samples = len(iou_list)
        return iou_list.mean()

    def reset(self):
        self.iou_list.clear()
        super().reset()


class VoxelDiff(torchmetrics.Metric):
    def __init__(
        self, **kwargs
    ):
        super().__init__(**kwargs)
        self.diff_list = []

    def update(self,
               pred_voxel: torch.tensor,
               gt_voxel: torch.tensor):
        if len(gt_voxel.shape) == 3:
            self.diff_list.append(torch.logical_xor(
                pred_voxel, gt_voxel).sum().to(torch.float32))
        else:
            for i, j in zip(gt_voxel, pred_voxel):
                self.diff_list.append(torch.logical_xor(
                    i, j).sum().to(torch.float32))

    def compute(self):
        diff_list = torch.stack(self.diff_list, dim=0)
        world_size = torch.distributed.get_world_size() \
            if torch.distributed.is_initialized() else 1
        if world_size > 1:
            all_diff = diff_list.new_zeros(
                (len(diff_list)*world_size, ) + diff_list.shape[1:])
            torch.distributed.all_gather_into_tensor(
                all_diff, diff_list)
            diff_list = all_diff
        self.num_samples = len(diff_list)
        return diff_list.mean()

    def reset(self):
        self.diff_list.clear()
        super().reset()
