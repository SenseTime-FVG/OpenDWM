import os

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
    name='render_utils_cuda',
    sources=[
        os.path.join(parent_dir, path)
        for path in ['../../../../externals/dvgo_cuda/render_utils.cpp', '../../../../externals/dvgo_cuda/render_utils_kernel.cu']],
    verbose=True)


def sample_ray(rays_o, rays_d, near, far, stepsize, xyz_min, xyz_max, voxel_size):
    '''Sample query points on rays.
    All the output points are sorted from near to far.
    Input:
        rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
        near, far:        the near and far distance of the rays.
        stepsize:         the number of voxels of each sample step.
    Output:
        ray_pts:          [M, 3] storing all the sampled points.
        ray_id:           [M]    the index of the ray of each point.
        step_id:          [M]    the i'th step on a ray of each point.
    '''
    # far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
    rays_o = rays_o.contiguous()
    rays_d = rays_d.contiguous()
    stepdist = stepsize  # * voxel_size

    ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
        rays_o, rays_d, xyz_min, xyz_max, near, far, stepdist)
    mask_inbbox = ~mask_outbbox
    ray_pts = ray_pts[mask_inbbox]
    ray_id = ray_id[mask_inbbox]
    step_id = step_id[mask_inbbox]
    return ray_pts, ray_id, step_id


def grid_query(grid, xyz, xyz_min, xyz_max):

    channels = grid.shape[1]
    shape = xyz.shape[:-1]
    xyz = xyz.reshape(1, 1, 1, -1, 3)
    ind_norm = ((xyz - xyz_min) / (xyz_max - xyz_min)) * 2 - 1
    # xyz -> zyx (DHW). It is the rule of grid_sample
    ind_norm = ind_norm.flip((-1,))
    out = F.grid_sample(grid, ind_norm, mode='bilinear', align_corners=True)
    out = out.reshape(channels, -1).T.reshape(*shape, channels)
    if channels == 1:
        out = out.squeeze(-1)
    return out


def dvgo_render(mlp, coarse_mask, rays_o, rays_d, grid, xyz_min, xyz_max, stepsize=0.5, offsets=None, grid_size=None, feat_render=False,
                near=0, far=1e9, hardcode_step=True):
    """
    Args:
        mlp: the MLP to predict alpha, which converts the feature to a single alpha value
        coarse_mask: the mask of the coarse grid to indicate which voxel is non-empty
        rays_o: the origin of the rays
        rays_d: the direction of the rays
        grid: the feature grid
        xyz_min: the minimum value of the grid
        xyz_max: the maximum value of the grid
        stepsize: the step size of the rays
        offsets: the offsets of the rays
        grid_size: the size parameters of the grid. We use the interval to determine the step size of feature grid.
        feat_render: whether to use feature in rendering. If True, it will render feature maps using the grid feature. Otherwise, the alpha will be directly used and the depth value will be outputed.
    """
    # prepare
    N = len(rays_o)
    xyz_min = torch.tensor(xyz_min, dtype=torch.float32, device=rays_o.device)
    xyz_max = torch.tensor(xyz_max, dtype=torch.float32, device=rays_o.device)
    voxel_size = (xyz_max[0]-xyz_min[0])/grid.shape[2]

    # step_size = 0.1 * 8
    if grid_size is not None:
        if hardcode_step:
            step_size_x = grid_size["interval"][0] * 8
            step_size_y = grid_size["interval"][1] * 8
            step_size_z = grid_size["interval"][2] * 4
        else:
            step_size_x = grid_size["interval"][0]
            step_size_y = grid_size["interval"][1]
            step_size_z = grid_size["interval"][2]          # 0.125 * 4
        # assert step_size == 0.15625 * 8         # if you make sure you want try other step_size, you can remove these
        # assert step_size_z == 0.140625 * 4          # 0.125 * 4
    else:
        step_size = 0.15625 * 8
        step_size_z = 0.140625 * 4          # 0.125 * 4
    # step_size = 0.15625

    ray_pts, ray_id, step_id = sample_ray(
        rays_o, rays_d, near, far, stepsize, xyz_min, xyz_max, voxel_size)
    if offsets is not None:
        ray_pts += offsets

    mask = (ray_pts[:, 0] > xyz_min[0]) & (ray_pts[:, 0] < xyz_max[0]) & (ray_pts[:, 1] > xyz_min[1]) & (
        ray_pts[:, 1] < xyz_max[1]) & (ray_pts[:, 2] > xyz_min[2]) & (ray_pts[:, 2] < xyz_max[2])

    ray_pts = ray_pts[mask]
    ray_id = ray_id[mask]
    step_id = step_id[mask]

    if coarse_mask is not None:
        indices_h = torch.trunc(
            (ray_pts[:, 1] - 1e-4 - xyz_min[1]) / step_size_x).long()
        indices_w = torch.trunc(
            (ray_pts[:, 0] - 1e-4 - xyz_min[0]) / step_size_y).long()
        indices_d = torch.trunc(
            (ray_pts[:, 2] - 1e-4 - xyz_min[2]) / step_size_z).long()

        # bool: pts whether in grid (pooled voxels), TODO: 128 x 128 x 16
        pts_mask = coarse_mask[indices_d, indices_h, indices_w].bool()

        # choose grid that is non-empty
        ray_pts = ray_pts[pts_mask]
        ray_id = ray_id[pts_mask]
        step_id = step_id[pts_mask]

    # query
    # alpha_init = 1e-6
    # act_shift = torch.FloatTensor([np.log(1/(1-alpha_init) - 1)])
    # find feature
    alpha_feat = grid_query(grid, ray_pts, xyz_min, xyz_max)
    # alpha = alpha_feat.sigmoid().squeeze(-1)
    alpha = mlp(alpha_feat).sigmoid().squeeze(-1)
    # density = mlp(alpha_feat).squeeze(-1)
    # alpha = Raw2Alpha.apply(density.flatten(), act_shift, stepsize).reshape(density.shape)
    # import ipdb; ipdb.set_trace()

    # # old code
    weights, alphainv_last = Alphas2Weights.apply(alpha.float(), ray_id, N)

    ray_dist = rays_d.norm(dim=-1)[ray_id]
    ray_origin = rays_o[ray_id]
    ray_dist_mask = (ray_dist - (ray_pts - ray_origin).norm(dim=-1)) > near

    # accumulated transmittance
    # weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)

    loss_sdf = torch.zeros((N,), device=rays_o.device)\
        .scatter_add_(
            dim=0, index=ray_id[ray_dist_mask],
            src=weights[ray_dist_mask] ** 2).mean()

    # loss_sdf = (weights[ray_dist_mask] ** 2).sum() / 1000 # * 0.05 # / 1000

    # alpha = torch.where(weight_mask, alpha, torch.zeros_like(alpha))

    # weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)

    # calculate depth
    if feat_render:
        s = alpha_feat
        ray_id = ray_id.unsqueeze(-1).expand(-1, s.shape[-1])
        output = torch.zeros((N, s.shape[-1]), device=rays_o.device).scatter_add_(
            dim=0, index=ray_id, src=weights[..., None] * s)
    else:
        s = step_id * stepsize
        output = torch.zeros((N,), device=rays_o.device)\
            .scatter_add_(dim=0, index=ray_id, src=weights * s)

    return output, loss_sdf, alphainv_last


class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None


class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(
            alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(
                alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
            alpha, weights, T, alphainv_last,
            i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None
