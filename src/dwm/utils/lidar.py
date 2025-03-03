import torch
import dwm.functional


def preprocess_points(batch, device):
    mhv = dwm.functional.make_homogeneous_vector
    return [
        [
            (mhv(p_j.to(device)) @ t_j.permute(1, 0))[:, :3]
            for p_j, t_j in zip(p_i, t_i.flatten(0, 1))
        ]
        for p_i, t_i in zip(
            batch["lidar_points"], batch["lidar_transforms"].to(device))
    ]


def postprocess_points(batch, ego_space_points):
    return [
        [
            (
                dwm.functional.make_homogeneous_vector(p_j.cpu()) @
                torch.linalg.inv(t_j).permute(1, 0)
            )[:, :3]
            for p_j, t_j in zip(p_i, t_i.flatten(0, 1))
        ]
        for p_i, t_i in zip(
            ego_space_points, batch["lidar_transforms"])
    ]


def voxels2points(grid_size, voxels):
    interval = torch.tensor([grid_size["interval"]])
    min = torch.tensor([grid_size["min"]])
    return [
        [
            torch.nonzero(v_j).flip(-1).cpu() * interval + min
            for v_j in v_i
        ]
        for v_i in voxels
    ]
