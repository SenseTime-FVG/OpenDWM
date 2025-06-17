
import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import Union

def resize_camera_params(camera_intrinsics: torch.tensor,
                         image_shape: Union[tuple[int], list[int]]) -> torch.tensor:
    """
    Description:
        Resize the camera intrinsics to the new image shape.
    Args:
        camera_intrinsics: (batch_size, 3, 3)
        image_shape: list or tuple of int with length 2.
    Return:
        camera_intrinsics: (batch_size, num_view, 3, 3) - Resized camera intrinsics
    """
    for i in range(camera_intrinsics.shape[0]):
        h_ratio = (image_shape[-2] // 2) / camera_intrinsics[i][1, 2]
        camera_intrinsics[i, :2, :] = camera_intrinsics[i, :2, :] * h_ratio
    return camera_intrinsics


def maybe_resize(tensor, tgt_reso, interp_mode="bilinear"): # "nearest" or "bilinear"
    if type(tgt_reso) == type([]):
        tensor = F.interpolate(
            tensor, size=tgt_reso, mode=interp_mode, antialias=interp_mode=="bilinear"
        )
    else:
        if tensor.shape[-1] != tgt_reso:
            tensor = F.interpolate(
                tensor, size=(tgt_reso, tgt_reso), mode=interp_mode, antialias=interp_mode=="bilinear"
            )
    return tensor

def get_rays(
        camera_intrinsics: torch.tensor,
        camera_transforms: torch.tensor,
        target_size: Union[int, tuple[int]],):
    ''' get rays
        Args:
            camera_transforms: (B*T*V, 4, 4), cam2world
            intrinsics: (B*T*V, 3, 3)
        Returns:
            rays_o: [B*T*V, 3]
            rays_d: [B*T*V, H, W, 3]
    '''
    device = camera_transforms.device
    dtype = camera_transforms.dtype
    camera_transforms = camera_transforms.to(dtype=torch.float32)
    camera_intrinsics = camera_intrinsics.to(dtype=torch.float32)

    H, W = (target_size, target_size) if isinstance(
        target_size, int) else target_size
    i, j = torch.meshgrid(torch.linspace(
        0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')

    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5

    zs = torch.ones_like(i)
    points_coord = torch.stack([i, j, zs])  # (H*W, 3)
    directions = torch.inverse(
        camera_intrinsics) @ points_coord.unsqueeze(0)  # (batch_size, 3, H*W)

    rays_d = camera_transforms[:, :3, :3] @ directions  # (batch_size, 3, H*W)
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d.transpose(1, 2).view(-1, H, W, 3).to(dtype=dtype)

    rays_o = camera_transforms[:, :3, 3].to(dtype=dtype)  # (batch_size, 3)

    return rays_o, rays_d

def get_camera_coordinates_of_lidar_point(
    camera_transforms: torch.tensor,
    camera_intrinsics: torch.tensor,
    ego_transforms: torch.tensor,
    img_shape: tuple[int],
    pts_coords: list[torch.tensor],
):
    """
    Description:
        Get the camera points coordinates in the image space. When nonempty_pillar_indices and sampled_points_indices are given,
        it will return the camera points coordinates for the sampled points. Otherwise, it will return the coordinates of all voxel points.
    Args:
        camera_transforms: Tensor of shape (batch_size, seq_len, num_view, 4, 4)
        camera_intrinsics: Tensor of shape (batch_size, seq_len, num_view, 3, 3)
        patch_size: int - The patch size in the encoder
        pts_coords: list of list of Tensor of shape (batch_size, seq_len, N, 3).
    Return:
        sampled_cam_point_coords: Tensor of shape (batch_size, (lidar_H // ps) * (lidar_W // ps), ps * ps * depth, num_view, 3) when nonempty_pillar_indices and sampled_points_indices are not given. Otherwise, it will return the (x,y,z) coordinates for the sampled points.
        valid_point_mask: Tensor of shape (batch_size, (lidar_H // ps) * (lidar_W // ps), ps * ps * depth) - The mask to indicate the valid points
    """
    # Obtain matrix transformations
    batch_size, seq_len, num_view, _, _ = camera_transforms.shape
    # Must be float32 type for inverse operation
    # [batch_size * seq_len * num_view, 4, 4]
    camera_transforms = camera_transforms.view(-1, 4, 4).float()
    camera_extrinsics = torch.inverse(camera_transforms)
    # [batch_size * seq_len * num_view, 3, 4]
    camera_extrinsics = camera_extrinsics[:, :3, :]
    camera_extrinsics = camera_extrinsics.view(
        batch_size, seq_len, num_view, 3, 4).float()  # [batch_size, seq_len, num_view, 3, 4]
    camera_intrinsics = resize_camera_params(
        camera_intrinsics.view(-1, 3, 3), img_shape)  # [batch_size, num_view, 3, 3]
    camera_intrinsics = camera_intrinsics.view(
        batch_size, seq_len, num_view, 3, 3).float()  # [batch_size, num_view, 3, 3]

    lidar_ego_transforms = ego_transforms[:, :, :1]  # [batch_size, seq_len, 1, 4, 4]
    cam_ego_inverse_transforms = ego_transforms[:, :, 1:].contiguous(
    ).view(-1, 4, 4).float()

    cam_ego_inverse_transforms = torch.inverse(cam_ego_inverse_transforms)
    cam_ego_inverse_transforms = cam_ego_inverse_transforms.view(
        batch_size, seq_len, num_view, 4, 4).float()  # [batch_size, seq_len, num_view, 4, 4]
    cam_pts_coords, cam_pts_coords_depth, valid_point_masks = [], [], []

    # for idx, (pts_coord, camera_intrinsic, camera_extrinsic, lidar_ego_transform, cam_ego_inverse_transform) \
    #     in enumerate(zip(pts_coords, camera_intrinsics, camera_extrinsics, lidar_ego_transforms, cam_ego_inverse_transforms)):
    for i in range(batch_size):
        b_cam_pts_coords, b_cam_pts_coords_depth, b_valid_point_masks = [], [], []
        for j in range(seq_len):
            pts_coord = pts_coords[i][j]
            camera_intrinsic = camera_intrinsics[i][j]
            camera_extrinsic = camera_extrinsics[i][j]
            lidar_ego_transform = lidar_ego_transforms[i][j]
            cam_ego_inverse_transform = cam_ego_inverse_transforms[i][j]
            
            # (1, num_sample_per_pillar, 3)
            pts_coord = pts_coord.unsqueeze(0)

            # Add extra dimension to the sampled points for matrix multiplication
            pts_coord_extra_dim = torch.ones(
                *pts_coord.shape[:-1], 1).to(pts_coord.device)
            # (1, num_pts, 4)
            pts_coord = torch.cat([pts_coord, pts_coord_extra_dim], dim=-1)
            pts_coord = pts_coord.permute(0, 2, 1) # (1, 4, num_pts)
            # Project the points to the world space and the points to the camera ego space
            # (num_view, 4, num_pts)
            # cam_pts_coord = (cam_ego_inverse_transform @ lidar_ego_transform) @ pts_coord
            cam_pts_coord = pts_coord

            # Project the points to the camera space
            # (num_view, 3, 4) @ (num_view, 4, num_pts)
            cam_pts_coord = camera_extrinsic @ cam_pts_coord
            cam_pts_coord = cam_pts_coord.float()
            cam_pts_coord_z = cam_pts_coord[..., -1:, :]
            # (num_view, 3, 3) @ # (num_view, 3, num_pts)
            cam_pts_coord = camera_intrinsic @ (cam_pts_coord / cam_pts_coord_z)
            # Permute the camera points
            cam_pts_coord = cam_pts_coord.transpose(1, 2)  # (num_view, num_pts, 3)
            cam_pts_coord_z = cam_pts_coord_z.transpose(1, 2)  # (num_view, num_pts, 1)

            # get the valid points
            valid_point_mask = (cam_pts_coord_z[..., 0] > 0.) & \
                cam_pts_coord[..., 0].isnan().logical_not() & \
                cam_pts_coord[..., 1].isnan().logical_not() & \
                (cam_pts_coord[..., 0] < img_shape[1]) & \
                (cam_pts_coord[..., 1] < img_shape[0]) & \
                (cam_pts_coord[..., 0] >= 0) & \
                (cam_pts_coord[..., 1] >= 0)
            valid_point_mask = valid_point_mask.contiguous()
            # set the coordinates of invalid points to 0
            cam_pts_coord[..., 0] = torch.where(
                valid_point_mask, cam_pts_coord[..., 0], 0)
            cam_pts_coord[..., 1] = torch.where(
                valid_point_mask, cam_pts_coord[..., 1], 0)
            cam_pts_coord = cam_pts_coord.to(int).contiguous()
            b_cam_pts_coords.append(cam_pts_coord)
            b_cam_pts_coords_depth.append(cam_pts_coord_z)
            b_valid_point_masks.append(valid_point_mask)
        cam_pts_coords.append(b_cam_pts_coords)
        cam_pts_coords_depth.append(b_cam_pts_coords_depth)
        valid_point_masks.append(b_valid_point_masks)
        
    
    return cam_pts_coords, cam_pts_coords_depth, valid_point_masks


def project_pts_to_img(
    imgs: torch.tensor,
    cam_pts_coords: list[torch.tensor],
    cam_pts_coords_depth: list[torch.tensor],
    valid_point_masks: list[torch.tensor],
    img_shape: tuple[int],
    timestamps_idx: list[torch.tensor],
    should_save: bool = True,
    should_overlay: bool = True,
    suffix: str = ""
):
    bs, seq_len, num_view, _, _, _ = imgs.shape
    all_imgs = []
    for i in range(bs):
        bs_imgs = []
        for j in range(seq_len):
            seq_img = []
            max_depth = max([cam_pts_coords_depth[i][j][k].max() for k in range(num_view)])
            for k in range(num_view):
                img_pts_coords = cam_pts_coords[i][j][k]
                img_pts_depth = cam_pts_coords_depth[i][j][k]
                if valid_point_masks is not None:
                    img_pts_masks = valid_point_masks[i][j][k]
                    img_pts_coords = img_pts_coords[img_pts_masks]
                    img_pts_depth = img_pts_depth[img_pts_masks]
                cur_img = imgs[i][j][k] if should_overlay else torch.zeros_like(imgs[i][j][k][:1, ...])
                if should_overlay:
                    cur_img[:, img_pts_coords[:, 1], img_pts_coords[:, 0]] = torch.tensor([[1.], [0.], [0.]]).to(cur_img.device) * \
                        img_pts_depth.expand(-1, 3).transpose(0, 1).to(cur_img.device) / torch.tensor(max_depth).clone().detach().to(cur_img.device)
                else:
                    cur_img[:, img_pts_coords[:, 1], img_pts_coords[:, 0]] = \
                        img_pts_depth.transpose(0, 1).to(cur_img.device)

                seq_img.append(cur_img)
            seq_img = torch.stack(seq_img)
            bs_imgs.append(seq_img)
            if should_save:
                seq_img = seq_img.permute(1, 2, 0, 3).contiguous().view(3, img_shape[0], -1)
                cur_image = transforms.ToPILImage()(seq_img)
                cur_image.save(f"test_img_{i}_{timestamps_idx[j]}{suffix}.png")
        all_imgs.append(torch.stack(bs_imgs))
    all_imgs = torch.stack(all_imgs)
    return all_imgs