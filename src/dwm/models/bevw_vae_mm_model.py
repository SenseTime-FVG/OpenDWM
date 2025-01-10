import torch.utils
import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusers import AutoencoderKL
import torchvision.transforms.functional

from dwm.models.voxelizer import Voxelizer
from dwm.models.vq_point_cloud import PatchMerging, BasicLayer, get_2d_sincos_pos_embed, VectorQuantizer
from dwm.models.base_vq_models.dvgo_utils import dvgo_render
import timm.models.swin_transformer
import timm.layers
from typing import Union, Optional, List
from diffusers import Transformer2DModel
import pdb


def resize_camera_params(camera_intrinsics: torch.tensor,
                         image_shape: Union[tuple[int], list[int]]) -> torch.tensor:
    """
    Description:
        Resize the camera intrinsics to the new image shape.
    Args:
        camera_intrinsics: (batch_size, view_count, 3, 3)
        image_shape: list or tuple of int with length 2.
    Return:
        camera_intrinsics: (batch_size, view_count, 3, 3) - Resized camera intrinsics
    """
    camera_intrinsics_shape = camera_intrinsics.shape
    camera_intrinsics = camera_intrinsics.view(-1, 3, 3)
    for i in range(camera_intrinsics.shape[0]):
        h_ratio = (image_shape[-2] // 2) / camera_intrinsics[i][1, 2]
        camera_intrinsics[i, :2, :] = camera_intrinsics[i, :2, :] * h_ratio
    camera_intrinsics = camera_intrinsics.view(*camera_intrinsics_shape)
    return camera_intrinsics


def get_rays(camera_transforms: torch.tensor,
             camera_intrinsics: torch.tensor,
             target_size: Union[int, tuple[int]],):
    """
    Description:
        Get the rays of each pixel from the camera extrinsics and intrinsics.
    Args:
        camera_transforms: (N, 4, 4), cam2world
        intrinsics: (N, 3, 3)
        target_size: int or tuple[int]. The height and width of the target size.
    Returns:
        rays_o, rays_d: [N*H*W, 3], where N is the batch size, H and W are the height and width of the target size.
    """
    device = camera_transforms.device

    H, W = (target_size, target_size) if isinstance(
        target_size, int) else target_size

    i, j = torch.meshgrid(torch.linspace(
        0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')  # float

    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5

    zs = torch.ones_like(i)
    points_coord = torch.stack([i, j, zs])  # (H*W, 3)
    directions = torch.inverse(
        camera_intrinsics) @ points_coord.unsqueeze(0)  # (batch_size, 3, H*W)

    rays_d = camera_transforms[:, :3, :3] @ directions  # (batch_size, 3, H*W)
    # [batch_size, 3, H*W)
    rays_o = camera_transforms[:, :3, 3].unsqueeze(-1).expand_as(rays_d)
    rays_o, rays_d = rays_o.permute(0, 2, 1).contiguous(
    ).view(-1, 3), rays_d.permute(0, 2, 1).contiguous().view(-1, 3)  # [batch_size*H*W, 3)
    return rays_o, rays_d


class BEVDecoder(nn.Module):
    def __init__(
        self,
        img_size: Union[int, tuple[int]],
        lidar_size: Union[int, tuple[int]],
        img_latent_size: Union[int, tuple[int]],
        img_decoder: nn.Module,
        patch_size: int = 8,
        feature_depth: int = 40,
        voxel_grid_depth: int = 64,
        embed_dim: int = 512,
        num_heads: int = 16,
        depth: list[int] = [10, 4],
        in_channels: int = 1024,
        upsample_style: str = "conv_transpose",
        use_checkpoint: bool = False,
        # Config to control whether the gt voxel is used in rendering
        use_gt_voxel: bool = True,
        visual_grid_feat_dim: int = 16,
        downsample_size: list[int] = [4, 8, 8],
        grid_size_offset: list[list[int]] = [
            [0, 0, 0],
            [0, 0, 0]
        ],
        bias_init: float = -3,
        # Config to control whether using an additional voxel predictor
        use_voxel_decoder: bool = False,
        # Config to control whether the new type of attention is used
        upcast: bool = False,
        # Config to control the rendering range and stepsize
        render_config: dict = {
            "near": 0,  # The near depth of the rendering range
            "far": 1e9,  # The far depth of the rendering range
            "stepsize": 0.05  # The sampling stepsize of the rendering
        }
    ):
        """
        Args:
            Inputs and Outputs:
                img_size: int or tuple[int]. The height and width of the targeted image.
                lidar_size: int or tuple[int]. The height and width of the targeted lidar point cloud.
                img_latent_size: int or tuple[int]. The height and width of the latent space of the image.
                img_decoder: nn.Module. The image decoder.
                in_channels: int. The input channels of the decoder, which is also the latent dimension.
                patch_size: int. The size of each patch used to decode the lidar voxels.
                feature_depth: int. The depth of the feature grid after unpatchifying.
                voxel_grid_depth: int. The depth of the voxel grid after unpatchifying.
            Rendering:
                grid_size_offset: list[list[int]]. The offset of the xyz grid size. It can enlarge / shrink the rendering range. This offset is added to the grid size.
                use_gt_voxel: bool. Whether to use the gt voxel as coarse mask in rendering. In the previous version of vq_pc, this is enabled. But in the current version, this is disabled since it can lose a lot of details of images.
                downsample_size: list[int]. The downsample voxel size. The downsampling size of visual grid compared with the original voxel grid. The three numbers represent the downsampling size of x, y, z.
                use_voxel_decoder: bool. Whether to use an additional branch to decode lidar voxel directly.
                render_config: dict. The config to control the rendering range and stepsize.
            Others:
                upcast: bool. Whether to use the upcast attention.
                use_checkpoint: bool. Whether to use the checkpointing method to save memory.
                upsample_style: str. The style of the upsample layer.
        """
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        if isinstance(lidar_size, int):
            lidar_size = (lidar_size, lidar_size)
        self.lidar_size = lidar_size
        if isinstance(img_latent_size, int):
            img_latent_size = (img_latent_size, img_latent_size)
        self.render_config = render_config
        self.img_latent_size = img_latent_size
        norm_layer = nn.LayerNorm
        self.patch_size = patch_size
        self.feature_depth = feature_depth
        self.voxel_depth = voxel_grid_depth
        self.use_checkpoint = use_checkpoint
        self.grid_size_offset = grid_size_offset
        self.use_gt_voxel = use_gt_voxel
        self.downsample_size = downsample_size
        # If we use the gt voxel as coarse mask, we need to downsample the voxel grid

        self.get_coarse_mask = nn.MaxPool3d(
            kernel_size=self.downsample_size) if use_gt_voxel else None
        # These flags are used to control the BasicLayer attn
        self.upcast = upcast

        # The embedding layer to embed the lidar features into the latent space
        self.decoder_embed = nn.Linear(
            in_channels, embed_dim, bias=True)
        self.img_decoder = img_decoder
        self.blocks = Transformer2DModel(
            num_attention_heads=num_heads,
            attention_head_dim=embed_dim // num_heads,
            num_layers=depth[0],
            in_channels=embed_dim,
            upcast_attention=upcast
        )

        # Upsample the feature grid
        if upsample_style == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(
                embed_dim, embed_dim // 2, 2, stride=2)
        else:
            self.upsample = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(embed_dim // 4, embed_dim, 1))

        # If we use the voxel decoder, we need to predict the voxel grid
        self.use_voxel_decoder = use_voxel_decoder
        if use_voxel_decoder:
            self.voxel_block = Transformer2DModel(
                num_attention_heads=num_heads,
                attention_head_dim=embed_dim // num_heads,
                num_layers=depth[1],
                in_channels=embed_dim,
                upcast_attention=upcast
            )
            self.voxel_block.gradient_checkpointing = use_checkpoint

            self.voxel_norm = torch.nn.Sequential(
                norm_layer(embed_dim, eps=1e-4), torch.nn.GELU())
            self.voxel_pred = torch.nn.Linear(
                embed_dim, (patch_size)**2 * voxel_grid_depth, bias=True)
            self.apply(self._init_weights)
            torch.nn.init.constant_(self.voxel_pred.bias, bias_init)

        # get the visual and depth prediction
        self.visual_block = Transformer2DModel(
            num_attention_heads=num_heads,
            attention_head_dim=embed_dim // num_heads,
            num_layers=depth[1],
            in_channels=embed_dim,
            upcast_attention=upcast
        )
        self.visual_block.gradient_checkpointing = use_checkpoint
        self.visual_norm = nn.Sequential(
            norm_layer(embed_dim, eps=1e-4), nn.GELU())
        # For each point of visual_pred output, it will be reshaped to a voxel grid with shape
        # (1, 1, feature_depth, visual_grid_feat_dim)
        self.visual_grid_feat_dim = visual_grid_feat_dim
        self.visual_pred = nn.Linear(
            embed_dim, feature_depth * self.visual_grid_feat_dim, bias=True)

        # The density mlp to predict the density of the voxel grid in the rendering
        self.density_mlp = nn.Sequential(
            nn.Linear(self.visual_grid_feat_dim, self.visual_grid_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.visual_grid_feat_dim, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Initialize the weights of the model by following the official JAX ViT
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def ray_render_depth_dvgo(self,
                              features: torch.tensor,
                              points: List[List[torch.tensor]],
                              coarse_mask: torch.tensor = None,
                              offsets: torch.tensor = None,
                              return_alpha_last: bool = False):
        """
        Description:
            Rendering depth from the features.
        Args:
            features: (batch_size, visual_grid_feat_dim, feature_depth, lidar_H, lidar_W)
            points: List[List[torch.tensor]] - The points to be rendered.
            coarse_mask: (batch_size, 1, voxel_grid_depth // downsample_size[0], lidar_H // downsample_size[1], lidar_W // downsample_size[2]) or None.
            offsets: (batch_size, 3) - Rendering center
            return_alpha_last: bool - Whether to return the alpha last of the rendering.
        """
        points = [j for i in points for j in i]
        batch_num = len(points)
        loss_depth = 0.
        loss_sdf = 0.
        rec_points = []
        alphainv_lasts = []

        def soft_l1(pred_depth, gt_depth):
            l1_loss = F.l1_loss(
                pred_depth, gt_depth, reduction='none').flatten()
            top_l1_loss = torch.topk(l1_loss, k=int(
                l1_loss.numel() * 0.95), largest=False)[0].mean()
            return top_l1_loss

        for iter_batch in range(batch_num):
            iter_points = points[iter_batch][:,
                                             :3].contiguous().to(features.device)
            # move origin
            if offsets is not None:
                cur_offsets = offsets[iter_batch].unsqueeze(
                    0).to(iter_points.device)
                rays_o = cur_offsets.repeat_interleave(iter_points.shape[0], 0)
                rays_d = iter_points - cur_offsets
            else:
                rays_o = torch.zeros(iter_points.shape, device=features.device)
                rays_d = iter_points

            gt_depth = torch.norm(rays_d, dim=-1, keepdim=True)
            pred_depth, loss_sdf_i, alphainv_last = dvgo_render(
                self.density_mlp,
                coarse_mask[iter_batch] if coarse_mask is not None else None,
                rays_o, rays_d,
                torch.einsum(
                    'dzyx->dxyz', features[iter_batch].float()).unsqueeze(0),  # pred grids
                self.render_grid_size["min"],
                self.render_grid_size["max"],
                near=self.render_config["near"],
                far=self.render_config["far"],
                stepsize=self.render_config["stepsize"],
                offsets=None, grid_size=self.render_grid_size,
                hardcode_step=False)

            loss_depth = loss_depth + soft_l1(pred_depth, gt_depth.squeeze(-1))
            loss_sdf = loss_sdf + loss_sdf_i
            rec_points.append(
                rays_o + pred_depth.unsqueeze(-1) / gt_depth * rays_d)
            alphainv_lasts.append(alphainv_last)

        # rec_points = torch.stack(rec_points, dim=0)

        if return_alpha_last:
            return loss_depth / len(points), loss_sdf / len(points), rec_points, alphainv_lasts
        else:
            return loss_depth / len(points), loss_sdf / len(points), rec_points

    def ray_render_img_dvgo(self, features, camera_transforms, camera_intrinsic,
                            coarse_mask=None):
        """
        Description:
            Render image latents from features
        Args:
            features: (batch_size, visual_grid_feat_dim, feature_depth, lidar_H, lidar_W)
            camera_transforms: (batch_size, view_count, 4, 4)
            camera_intrinsic: (batch_size, view_count, 3, 3)
            coarse_mask: (batch_size, 1, voxel_grid_depth // downsample_size[0], lidar_H // downsample_size[1], lidar_W // downsample_size[2]) or None.
        """
        batch_num, view_count = camera_transforms.shape[:2]
        pred_img_latents = []
        camera_intrinsic = resize_camera_params(
            camera_intrinsic, self.img_latent_size)

        for iter_batch in range(batch_num):
            # get the ray centers and directions of all cameras in this batch
            rays_o, rays_d = get_rays(camera_transforms=camera_transforms[iter_batch],
                                      camera_intrinsics=camera_intrinsic[iter_batch],
                                      target_size=self.img_latent_size
                                      )  # (view_count * latent_h * latent_w, 3), (view_count * latent_h * latent_w, 3)

            batch_pred_img_latents, _, _ = dvgo_render(
                self.density_mlp,
                coarse_mask[iter_batch] if coarse_mask is not None else None,
                rays_o.float(), rays_d.float(),
                torch.einsum(
                    'dzyx->dxyz', features[iter_batch].float()).unsqueeze(0),  # pred grids
                self.render_grid_size["min"],
                self.render_grid_size["max"],
                near=self.render_config["near"],
                far=self.render_config["far"],
                stepsize=self.render_config["stepsize"],
                offsets=None, grid_size=self.render_grid_size,
                feat_render=True)
            pred_img_latents.append(batch_pred_img_latents)
        pred_img_latents = torch.stack(pred_img_latents, dim=0).contiguous().view(
            batch_num, view_count, *self.img_latent_size, -1)

        return pred_img_latents

    def unpatchify(self, x, p=None):
        if p is None:
            p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        # h, w = self.h, self.w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, w * p))
        return imgs

    def update_model_type(self, model_type: str):
        self.model_type = model_type
        if model_type == "img":
            # If we use the img mode only, we do not need the visual block for lidar
            self.visual_block = None
            self.visual_norm = None
            self.norm_layer = None
            self.visual_grid_feat_dim = None
            self.visual_pred = None
            self.density_mlp = None
            self.blocks = None
            self.upsample = None

    def update_render_grid_size(self, grid_size):
        self.render_grid_size = {
            "min": [i + j for i, j in zip(grid_size["min"], self.grid_size_offset[0])],
            "max": [i + j for i, j in zip(grid_size["max"], self.grid_size_offset[1])],
            # this is only used when coarse mask is used
            "interval": [i * j for i, j in zip(grid_size["interval"], reversed(self.downsample_size))]
        }  # this grid size is for rendering
        # if self.grid_size_offset is not None:
        #     self.grid_size["min"] = [i + j for i, j in zip(self.grid_size["min"], self.grid_size_offset[0])]
        #     self.grid_size["max"] = [i + j for i, j in zip(self.grid_size["max"], self.grid_size_offset[1])]

    def decode_img(self, img_latents: torch.tensor):
        """
        Description:
            Decode the latent features into images
        """
        img_latents_shape = img_latents.shape
        img_latents = img_latents.flatten(0, -2).contiguous()
        img_latents = self.decoder_embed(img_latents).contiguous()
        pred_imgs = self.img_decoder(img_latents.view(
            *img_latents_shape[:-1], -1).contiguous().permute(0, 3, 1, 2).contiguous())

        output = {
            "pred_imgs": pred_imgs
        }
        return output

    def forward(self, x: torch.tensor,
                points: torch.tensor = None,
                voxels: torch.tensor = None,
                camera_intrinsics: torch.tensor = None,
                camera_transforms: torch.tensor = None,
                depth_ray_cast_center: torch.tensor = None):
        """
        Args:
            x: (batch_size, (lidar_H // ps) * (lidar_W // ps), in_channels)
        """
        # embed tokens
        x = self.decoder_embed(x)

        # apply Transformer blocks
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.blocks(x).sample
        x = self.upsample(x)

        # generate features
        visual_feats = self.visual_block(x).sample
        visual_feats = visual_feats.permute(0, 2, 3, 1).contiguous()
        visual_feats = self.visual_norm(visual_feats)
        visual_feats = self.visual_pred(visual_feats)
        visual_feats = self.unpatchify(visual_feats.flatten(
            1, 2), p=self.patch_size // 2).unflatten(1, (self.visual_grid_feat_dim, -1))

        # ray cast the depth and image latents
        pooled_voxels = self.get_coarse_mask(
            voxels) if self.use_gt_voxel else None
        # depth_loss, sdf_loss, lidar_rec = self.ray_render_depth_dvgo(visual_feats, points, pooled_voxels, offsets=depth_ray_cast_center)
        # img_latents = self.ray_render_img_dvgo(visual_feats, camera_transforms, camera_intrinsics, pooled_voxels, offsets=depth_ray_cast_center)
        depth_loss, sdf_loss, lidar_rec = self.ray_render_depth_dvgo(
            visual_feats, points, pooled_voxels, offsets=depth_ray_cast_center)
        img_latents = self.ray_render_img_dvgo(
            visual_feats, camera_transforms, camera_intrinsics, pooled_voxels)

        # decode imgs
        img_latents_shape = img_latents.shape
        pred_imgs = self.img_decoder(
            img_latents.view(-1, *img_latents_shape[2:]).permute(0, 3, 1, 2).contiguous())
        pred_imgs = pred_imgs.view(
            *img_latents_shape[:2], *pred_imgs.shape[1:])
        # reconstruct the voxel grid
        if self.use_voxel_decoder:
            pred_voxel_feat = self.voxel_block(x).sample
            pred_voxel_feat = pred_voxel_feat.permute(0, 2, 3, 1).contiguous()
            pred_voxel_feat = self.voxel_norm(pred_voxel_feat.float())
            pred_voxel_feat = self.voxel_pred(pred_voxel_feat)
            pred_voxel = self.unpatchify(pred_voxel_feat.flatten(1, 2))

        results = {
            "depth_loss": depth_loss,
            "sdf_loss": sdf_loss,
            "lidar_rec": lidar_rec,
            "pred_imgs": pred_imgs,
            "voxels": voxels,
            "pred_voxel": pred_voxel
        }
        return results


class DeformableAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, upcast=True):
        super(DeformableAttention, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Linear layers for query, key, and value projections
        self.query_proj = nn.Linear(in_channels, out_channels)
        self.key_proj = nn.Linear(in_channels, out_channels)
        self.value_proj = nn.Linear(in_channels, out_channels)
        # Multi-head attention module
        self.multihead_attn = nn.MultiheadAttention(
            out_channels, num_heads, batch_first=True)

        # flag to control upcast
        self.upcast = upcast

        # Output projection
        self.output_proj = nn.Linear(out_channels, out_channels)

    def forward(self,
                sampled_img_features: torch.tensor,
                lidar_features: torch.tensor,
                masks: torch.tensor = None):
        """
        Args:
            sampled_img_features: Tensor of shape (N, K, C) - image features treated as values
            lidar_features: Tensor of shape (N, C) - Lidar BEV features treated as queries
        Returns:
            output: Tensor of shape (N, C) - output features after attention
        """
        N, K, C = sampled_img_features.shape
        if self.upcast:
            lidar_features = lidar_features.float()
            sampled_img_features = sampled_img_features.float()

        # Project queries, keys, and values
        queries = self.query_proj(lidar_features)  # Shape (N, out_channels)
        # Shape (N, K, out_channels)
        keys = self.key_proj(sampled_img_features)
        # Shape (N, K, out_channels)
        values = self.value_proj(sampled_img_features)
        # Perform multi-head attention
        output, _ = self.multihead_attn(queries.unsqueeze(
            1), keys, values, key_padding_mask=masks)
        # Concatenate heads and project to output channels
        output = output.view(N, -1)  # Shape (N, out_channels)
        output = self.output_proj(output)  # Final output projection

        return output


class VariationalModel(nn.Module):
    def __init__(self,
                 model_type: str,
                 variational_model_config: dict):
        super(VariationalModel, self).__init__()
        self.model_type = model_type
        assert self.model_type in ["vqvae", "vae"], "Invalid model type"
        if self.model_type == "vqvae":
            self.vector_quantizer = VectorQuantizer(**variational_model_config)
            self.register_buffer(
                "code_age", torch.zeros(self.vector_quantizer.n_e) * 10000)
            self.register_buffer(
                "code_usage", torch.zeros(self.vector_quantizer.n_e))
        elif self.model_type == "vae":

            self.fc_mu = nn.Linear(
                variational_model_config["encoder_out_channels"], variational_model_config["decoder_in_channels"])
            self.fc_var = nn.Linear(
                variational_model_config["encoder_out_channels"], variational_model_config["decoder_in_channels"])

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def forward_vqvae(self, x, return_loss=True):
        z_q, emb_loss, _ = self.vector_quantizer(x)
        return z_q, {"emb_loss": emb_loss} if return_loss else z_q

    def forward_vae(self, x, return_loss=True):
        input_shape = x.shape
        # x = x.flatten(0, -2).contiguous()
        # mu, log_var = x.chunk(2, dim = -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        z = self.reparameterize(mu, log_var)
        kl_loss = self.kl_loss(mu, log_var)
        # z = z.view(*input_shape[:-1], -1).contiguous()
        # mu = mu.view(*input_shape[:-1], -1).contiguous()
        # log_var = log_var.view(*input_shape[:-1], -1).contiguous()
        return z, {"kl_loss": kl_loss, "mu": mu, "log_var": log_var} if return_loss else z

    def forward(self, x, return_loss=True):
        forward_func = getattr(self, f'forward_{self.model_type}')
        return forward_func(x, return_loss)


class VAEBevMultiModality(nn.Module):
    def __init__(self,
                 voxelizer: Voxelizer,
                 lidar_encoder: nn.Module,
                 img_encoder: nn.Module,
                 bev_decoder: BEVDecoder,
                 deformable_transformer: DeformableAttention,
                 variational_model: VariationalModel = None,
                 bias_init: float = -5,
                 num_sample_per_pillar: int = 20,
                 forward_type: str = "normal",
                 pretrained_img_vae: dict = None,
                 enable_img_encoder_embed: bool = False,
                 enable_img_decoder_embed: bool = False,
                 ):
        super(VAEBevMultiModality, self).__init__()
        self.forward_type = forward_type
        self.variational_model = variational_model
        self.img_encoder = img_encoder
        if forward_type == "normal":
            self.img_encoder_out_channels = img_encoder.out_channels
        self.enable_img_encoder_embed = enable_img_encoder_embed
        # Something redundant. Might be deleted later(TODO)
        if forward_type == "normal":
            self.bev_feature_layer = nn.Linear(
                lidar_encoder.out_channels, lidar_encoder.out_channels)
        self.bev_decoder = bev_decoder
        if forward_type == "normal":
            self.img_decoder_in_channels = bev_decoder.img_decoder.in_channels

        if pretrained_img_vae is not None:
            self.img_vae = AutoencoderKL.from_pretrained(
                pretrained_img_vae["model_type"], subfolder=pretrained_img_vae["subfolder"])
            # set up new decoder
            use_checkpoint = self.img_encoder.encoder.gradient_checkpointing
            self.img_encoder = self.img_vae.encoder
            self.img_encoder.gradient_checkpointing = use_checkpoint
            if self.img_vae.quant_conv is not None:
                self.img_encoder = nn.Sequential(
                    self.img_encoder, self.img_vae.quant_conv)
            # set up new encoder
            use_checkpoint = self.bev_decoder.img_decoder.decoder.gradient_checkpointing
            self.bev_decoder.img_decoder = self.img_vae.decoder
            self.bev_decoder.img_decoder.gradient_checkpointing = use_checkpoint
            if self.img_vae.post_quant_conv is not None:
                self.bev_decoder.img_decoder = nn.Sequential(
                    self.img_vae.post_quant_conv, self.bev_decoder.img_decoder, )
            # set up new parameters
            self.img_encoder_out_channels = pretrained_img_vae["encoder_output_channels"]
            self.img_decoder_in_channels = pretrained_img_vae["decoder_input_channels"]
        if enable_img_encoder_embed:
            # Align the output channels of the img encoder with the output channels of the lidar encoder
            self.img_encoder_embed = nn.Linear(
                self.img_encoder_out_channels, lidar_encoder.out_channels)

        self.bev_decoder.enable_img_decoder_embed = enable_img_decoder_embed
        if enable_img_decoder_embed:
            self.bev_decoder.img_decoder_embed = nn.Linear(
                self.bev_decoder.visual_grid_feat_dim, self.img_decoder_in_channels)

        if forward_type == "normal" or forward_type == "lidar":
            self.grid_size = {
                "min": [voxelizer.x_min, voxelizer.y_min, voxelizer.z_min],
                "max": [voxelizer.x_max, voxelizer.y_max, voxelizer.z_max],
                "interval": [voxelizer.step, voxelizer.step, voxelizer.z_step]
            }  # this grid size is for the voxelization
            self.voxelizer = voxelizer
            self.lidar_encoder = lidar_encoder
            self.bev_decoder.update_render_grid_size(self.grid_size)
            self.deformable_transformer = deformable_transformer
        elif forward_type == "img":
            self.bev_decoder.update_model_type("img")
        else:
            raise ValueError(f"Invalid forward type: {forward_type}")

        self.num_sample_per_pillar = num_sample_per_pillar

    def sample_pts_from_voxel(self, voxels: torch.tensor, patch_size: int):
        """
        Args:
            voxels: (batch_size, depth, lidar_H, lidar_W)
            patch_size: int. Patch size in the encoder
        Return:
            nonempty_pillar_indices: (batch_size * lidar_H * lidar_W, )
            sampled_points_indices: (batch_size * lidar_H * lidar_W, num_sample_per_pillar, )
        """
        voxels = voxels.permute(0, 2, 3, 1).contiguous()
        sample_pillar = []

        # concatenate all the points inside the feature pillars
        for i in range(patch_size):
            for j in range(patch_size):
                sample_pillar.append(
                    voxels[:, i::patch_size, j::patch_size, :])
        sample_pillar = torch.cat(sample_pillar, dim=-1)
        sample_pillar_shape = sample_pillar.shape
        # flatten all the pillars
        sample_pillar = sample_pillar.contiguous(
        ).view(-1, sample_pillar_shape[-1])
        # exclude the pillars that do not contain any points
        nonempty_pillar_indices = torch.sum(sample_pillar, dim=-1) > 0
        sample_pillar = sample_pillar[nonempty_pillar_indices]
        # sample points in each pillar
        sampled_points_indices = torch.multinomial(
            sample_pillar, self.num_sample_per_pillar, replacement=True)
        return nonempty_pillar_indices, sampled_points_indices

    @staticmethod
    def get_camera_points_coordinates(camera_transforms: torch.tensor,
                                      camera_intrinsics: torch.tensor,
                                      ego_transforms: torch.tensor,
                                      patch_size: int,
                                      img_shape: tuple[int],
                                      voxel_coords: torch.tensor,
                                      nonempty_pillar_indices: Optional[torch.tensor] = None,
                                      sampled_points_indices: Optional[torch.tensor] = None,
                                      debug_points=None):
        """
        Description:
            Get the camera points coordinates in the image space. When nonempty_pillar_indices and sampled_points_indices are given,
            it will return the camera points coordinates for the sampled points. Otherwise, it will return the coordinates of all voxel points.
        Args:
            camera_transforms: Tensor of shape (batch_size, view_count, 4, 4)
            camera_intrinsics: Tensor of shape (batch_size, view_count, 3, 3)
            patch_size: int - The patch size in the encoder
            voxel_coords: Tensor of shape (lidar_H, lidar_W, lidar_D, 3)
            nonempty_pillar_indices, sampled_points_indices: Tensor of shape - sample_pts_from_voxel() output
        Return:
            sampled_cam_point_coords: Tensor of shape (batch_size, (lidar_H // ps) * (lidar_W // ps), ps * ps * depth, view_count, 3) when nonempty_pillar_indices and sampled_points_indices are not given. Otherwise, it will return the (x,y,z) coordinates for the sampled points.
            valid_point_mask: Tensor of shape (batch_size, (lidar_H // ps) * (lidar_W // ps), ps * ps * depth) - The mask to indicate the valid points
        """
        assert ((nonempty_pillar_indices is None) and (sampled_points_indices is None)) or \
            ((nonempty_pillar_indices is not None) and (sampled_points_indices is not None)), \
            "nonempty_pillar_indices and sampled_points_indices must be both None or both not None"

        # Obtain matrix transformations
        batch_size, view_count, _, _ = camera_transforms.shape
        # Must be float32 type for inverse operation
        # [batch_size * view_count, 4, 4]
        camera_transforms = camera_transforms.view(-1, 4, 4).float()
        camera_extrinsics = torch.inverse(camera_transforms)
        # [batch_size * view_count, 3, 4]
        camera_extrinsics = camera_extrinsics[:, :3, :]
        camera_extrinsics = camera_extrinsics.view(
            batch_size, view_count, 3, 4).float()  # [batch_size, view_count, 3, 4]
        camera_intrinsics = resize_camera_params(
            camera_intrinsics, img_shape)  # [batch_size, view_count, 3, 3]
        camera_intrinsics = camera_intrinsics.view(
            batch_size, view_count, 3, 3).float()  # [batch_size * view_count, 3, 3]

        lidar_ego_transforms = ego_transforms[:, :1]  # [batch_size, 1, 4, 4]
        cam_ego_inverse_transforms = ego_transforms[:, 1:].contiguous(
        ).view(-1, 4, 4).float()

        cam_ego_inverse_transforms = torch.inverse(cam_ego_inverse_transforms)
        cam_ego_inverse_transforms = cam_ego_inverse_transforms.view(
            batch_size, view_count, 4, 4).float()  # [batch_size, view_count, 4, 4]
        sampled_voxel_coords = []
        # Concatenate all the points inside the feature pillars. The points are sampled in a patch_size x patch_size grid
        for i in range(patch_size):
            for j in range(patch_size):
                sampled_voxel_coords.append(
                    voxel_coords[i::patch_size, j::patch_size, ...])
        # [lidar_H // ps, lidar_W // ps, ps * ps * depth, 3]
        sampled_voxel_coords = torch.cat(sampled_voxel_coords, dim=-2)
        if nonempty_pillar_indices is not None:
            # Sample points from voxel grid. Otherwise, use all voxel points
            original_sampled_voxel_shape = sampled_voxel_coords.shape
            flatten_indices = torch.where(nonempty_pillar_indices)[0]
            # batch indices of the sampled points
            batch_indices = flatten_indices // (
                original_sampled_voxel_shape[0] * original_sampled_voxel_shape[1])
            # indices of the sampled points inside the voxel grid
            grid_indices = (flatten_indices % (original_sampled_voxel_shape[0] * original_sampled_voxel_shape[1])).\
                unsqueeze(-1).expand_as(sampled_points_indices)
            sampled_voxel_coords = sampled_voxel_coords.view(
                -1, original_sampled_voxel_shape[-2], 3)
            # (num_non_empty, num_sample_per_pillar, 3)
            sampled_voxel_coords = sampled_voxel_coords[grid_indices,
                                                        sampled_points_indices]
            # select corresponding camera extrinsics and intrinsics
            # (num_non_empty, view_count, 3, 4)
            camera_extrinsics = camera_extrinsics[batch_indices]
            # (num_non_empty, view_count, 3, 3)
            camera_intrinsics = camera_intrinsics[batch_indices]
            # (num_non_empty, 1, 4, 4)
            lidar_ego_transforms = lidar_ego_transforms[batch_indices]
            # (num_non_empty, view_count, 4, 4)
            cam_ego_inverse_transforms = cam_ego_inverse_transforms[batch_indices]
            # Add extra dimension to the sampled points for matrix multiplication
            sampled_voxel_extra_dim = torch.ones(
                *sampled_voxel_coords.shape[:-1], 1).to(sampled_voxel_coords.device)
            # (num_non_empty, num_sample_per_pillar, 4)
            sampled_voxel_coords = torch.cat(
                [sampled_voxel_coords, sampled_voxel_extra_dim], dim=-1)
            sampled_voxel_coords = sampled_voxel_coords.permute(0, 2, 1).unsqueeze(
                1)  # (num_non_empty, 1, 4, num_sample_per_pillar)
            # Project the points to the world space
            # (num_non_empty, 1, 4, num_sample_per_pillar)
            sampled_cam_point_coords = lidar_ego_transforms @ sampled_voxel_coords
            # Project the points to the camera ego space
            # (num_non_empty, view_count, 4, num_sample_per_pillar)
            sampled_cam_point_coords = cam_ego_inverse_transforms @ sampled_cam_point_coords
            # Project the points to the camera space
            # (num_non_empty, view_count, 3, 4) @ (num_non_empty, 1, 4, num_sample_per_pillar)
            sampled_cam_point_coords = camera_extrinsics @ sampled_voxel_coords
            sampled_cam_point_coords = sampled_cam_point_coords.float()
            sampled_cam_point_coords_z = sampled_cam_point_coords[..., -1:, :]
            # sampled_cam_point_coords_z = torch.where(sampled_cam_point_coords_z.abs() < 1e-6, 1e-6 * sampled_cam_point_coords_z.sign(), sampled_cam_point_coords_z) # avoid 0 value in division
            # sampled_cam_point_coords[..., -1:, :] = 1. # set the last dimension to 1
            # (num_non_empty, view_count, 3, 3) @ # (num_non_empty, 1, 3, num_sample_per_pillar)
            sampled_cam_point_coords = camera_intrinsics @ (
                sampled_cam_point_coords / sampled_cam_point_coords_z)
            # Permute the camera points
            sampled_cam_point_coords = sampled_cam_point_coords.permute(
                0, 3, 1, 2)  # (num_non_empty, num_sample_per_pillar, view_count, 3)
            sampled_cam_point_coords_z = sampled_cam_point_coords_z.permute(
                0, 3, 1, 2)  # (num_non_empty, num_sample_per_pillar, view_count, 1)
        else:
            # Flatten the sampled voxel coordinates to (N, 3)
            # The shape is [(lidar_H // ps) * (lidar_W // ps) * ps * ps * depth, 3]
            sampled_voxel_coords_shape = sampled_voxel_coords.shape
            sampled_voxel_coords = sampled_voxel_coords.contiguous().view(-1, 3)

            # Flatten the camera matrices
            # [N, 3, 4] or [batch_size * view_count, 3, 4]
            camera_extrinsics = camera_extrinsics.view(-1, 3, 4)
            # [N, 3, 3] or [batch_size * view_count, 3, 3]
            camera_intrinsics = camera_intrinsics.view(-1, 3, 3)

            # Add extra dimension to the sampled points for matrix multiplication
            sampled_voxel_extra_dim = torch.ones(sampled_voxel_coords.shape[0], 1).to(
                sampled_voxel_coords.device)  # [N, 1]
            sampled_voxel_coords = torch.cat(
                [sampled_voxel_coords, sampled_voxel_extra_dim], dim=-1)  # [N, 4]
            sampled_voxel_coords = sampled_voxel_coords.permute(1, 0)  # [4, N]

            # Project 3d points onto images
            # [batch_size * view_count, 3, 4] @ [4, N]
            sampled_cam_point_coords = camera_extrinsics @ \
                sampled_voxel_coords  # [batch_size * view_count, 3, N]
            sampled_cam_point_coords = sampled_cam_point_coords / \
                sampled_cam_point_coords[:, -1:, :]
            # [batch_size * view_count, 3, N]

            sampled_cam_point_coords = camera_intrinsics @ sampled_cam_point_coords
            sampled_cam_point_coords = sampled_cam_point_coords.view(
                batch_size, view_count, 3, -1)  # [batch_size, view_count, 3, N]
            # reshape the camera points
            sampled_cam_point_coords = sampled_cam_point_coords.permute(
                0, 3, 1, 2)  # [batch_size, N, view_count, 3]
            # [batch_size, (lidar_H // ps) * (lidar_W // ps), ps * ps * depth, view_count, 3]
            sampled_cam_point_coords = sampled_cam_point_coords.view(
                batch_size, -1, sampled_voxel_coords_shape[-2], view_count, 3)
        # get the valid points
        valid_point_mask = (sampled_cam_point_coords_z[..., 0] > 0.) & \
            sampled_cam_point_coords[..., 0].isnan().logical_not() & \
            sampled_cam_point_coords[..., 1].isnan().logical_not() & \
            (sampled_cam_point_coords[..., 0] < img_shape[1]) & \
            (sampled_cam_point_coords[..., 1] < img_shape[0]) & \
            (sampled_cam_point_coords[..., 0] >= 0) & \
            (sampled_cam_point_coords[..., 1] >=
             0)  # TODO: the last four lines are used to filter out the points outside the image
        valid_point_mask = valid_point_mask.contiguous()
        # set the coordinates of invalid points to 0
        sampled_cam_point_coords[..., 0] = torch.where(
            valid_point_mask, sampled_cam_point_coords[..., 0], 0)
        sampled_cam_point_coords[..., 1] = torch.where(
            valid_point_mask, sampled_cam_point_coords[..., 1], 0)
        sampled_cam_point_coords = sampled_cam_point_coords.to(
            int).contiguous()
        # visualize_proj_pts(sampled_cam_point_coords, valid_point_mask, batch_indices)
        # visualize_pc(sampled_voxel_coords[..., :-1, :].permute(0, 1, 3, 2), batch_indices, "test_pc_sample")
        return sampled_cam_point_coords, valid_point_mask

    def encode(self,
               voxels: torch.tensor,
               images: torch.tensor,
               camera_transforms: torch.tensor,
               camera_intrinsics: torch.tensor,
               ego_transforms: torch.tensor):
        """
        Args:
            voxels: (batch_size, depth, lidar_H, lidar_W)
            images: (batch_size, view_count, C, H, W)
            camera_transforms: (batch_size, view_count, 4, 4)
            camera_intrinsics: (batch_size, view_count, 3, 3)
            ego_transforms: (batch_size, view_count + 1, 4, 4)
        Return:
            bev_feats: (batch_size, (lidar_H // ps) * (lidar_W // ps), lidar_encoder.out_channels)
        """
        lidar_feats = self.lidar_encoder(voxels)
        batch_size, lidar_latent_h, lidar_latent_w, _ = lidar_feats.shape
        # (batch_size, (lidar_H // ps) * (lidar_W // ps), lidar_encoder.out_channels)
        lidar_feats = lidar_feats.flatten(1, 2)
        # extract image features
        _, view_count, C, H, W = images.shape
        multiview_imgs = images.view(-1, C, H, W)
        # (batch_size * view_count, img_H // img_ps, img_W // img_ps, img_encoder.out_channels)
        img_feats = self.img_encoder(multiview_imgs)

        # sample points from the voxel
        lidar_patch_size = self.lidar_encoder.downsample_size
        nonempty_pillar_indices, sampled_points_indices = self.\
            sample_pts_from_voxel(
                voxels, lidar_patch_size)  # (batch_size * (lidar_H // ps) * (lidar_W // ps)), (num_noempty_pillar, num_sample_per_pillar)

        # Get voxel volume coordinates
        voxel_coords = self.voxelizer.get_voxel_coordinates().to(
            camera_transforms.device)  # [depth, lidar_H, lidar_W, 3]
        # [lidar_H, lidar_W, lidar_depth, 3]
        voxel_coords = voxel_coords.permute(1, 2, 0, 3)
        # project all voxel points into the camera space
        img_shape = img_feats.shape[-3:-1]
        sampled_cam_point_coords, valid_point_mask = VAEBevMultiModality.get_camera_points_coordinates(
            camera_transforms, camera_intrinsics, ego_transforms, lidar_patch_size, img_shape, voxel_coords,
            nonempty_pillar_indices, sampled_points_indices)  # (num_non_empty, num_sample_per_pillar, view_count, 3)
        # # Obtain coordinates of the sampled points in camera space.
        # # When nonempty_pillar_indices is not provided in get_camera_points_coordinates it will be useful.
        # sampled_cam_point_coords = sampled_cam_point_coords.contiguous().\
        #     view(nonempty_pillar_indices.shape[0], -1, view_count, 3) # [batch_size * (lidar_H // ps) * (lidar_W // ps), ps * ps * depth, view_count, 3]
        # sampled_cam_point_coords = sampled_cam_point_coords[nonempty_pillar_indices] # [num_non_empty, ps * ps * depth, 3]
        # sampled_points_indices = sampled_points_indices[..., None, None].\
        #     expand(-1, -1, *sampled_cam_point_coords.shape[-2:])
        # sampled_cam_point_coords = torch.gather(sampled_cam_point_coords, 1, sampled_points_indices) # (num_noempty_pillar, num_sample_per_pillar, 3)

        # sample lidar features
        # (batch_size, (lidar_H // ps) * (lidar_W // ps), lidar_encoder.out_channels)
        lidar_feats_shape = lidar_feats.shape
        # [batch_size * (lidar_H // ps) * (lidar_W // ps), encoder.out_channels]
        lidar_feats = lidar_feats.view(-1, self.lidar_encoder.out_channels)
        # [num_non_empty, encoder.out_channels]
        sampled_lidar_feats = lidar_feats[nonempty_pillar_indices]

        # obtain the batch indices
        batch_indices = torch.where(nonempty_pillar_indices)[
            0] // (lidar_feats_shape[1])  # [num_non_empty]
        batch_indices = batch_indices[..., None, None].expand_as(
            sampled_cam_point_coords[..., 0])  # [num_non_empty, num_sample_per_pillar, view_count]
        # obtain the view indices
        view_indices = torch.arange(view_count).to(
            int).to(batch_indices.device)  # [view_count]
        view_indices = view_indices[None, None, ...].expand_as(
            sampled_cam_point_coords[..., 0])  # [num_non_empty, num_sample_per_pillar, view_count]
        # obtain the sampled image features
        # (batch_size, view_count, lidar_H // ps, lidar_W // ps, embed_dim)
        sampled_img_feats = img_feats.view(
            batch_size, view_count, *img_feats.shape[1:])
        sampled_cam_point_coords = sampled_cam_point_coords.to(int)
        sampled_img_feats = sampled_img_feats[batch_indices, view_indices,
                                              sampled_cam_point_coords[..., 1], sampled_cam_point_coords[..., 0]]  # [num_non_empty, embed_dim]
        # (num_non_empty, num_sample_per_pillar * view_count, embed_dim)
        sampled_img_feats = sampled_img_feats.view(
            sampled_img_feats.shape[0], -1, sampled_img_feats.shape[-1])
        # (num_non_empty, num_sample_per_pillar * view_count)
        valid_point_mask = valid_point_mask.view(valid_point_mask.shape[0], -1)

        # deformable attention to fuse the features
        fused_feats = self.deformable_transformer(
            sampled_img_feats, sampled_lidar_feats, valid_point_mask)
        # replace the lidar features with the fused_feats
        lidar_feats[nonempty_pillar_indices] = fused_feats.to(
            lidar_feats.dtype)
        bev_feats = self.bev_feature_layer(lidar_feats)
        bev_feats = bev_feats.view(
            batch_size, lidar_latent_h, lidar_latent_w, -1)
        return bev_feats

    def decode(self,
               bev_feats: torch.tensor,
               points: list[list[torch.tensor]],
               voxels: torch.tensor,
               depth_ray_cast_center: torch.tensor,
               camera_intrinsics: torch.tensor,
               camera_transforms: torch.tensor):
        """
        Args:
            bev_feats: (batch_size, (lidar_H // ps) * (lidar_W // ps), lidar_encoder.out_channels)
            points: List[List[(N_x, 3)]] - Lidar points. The first list is the number of batch, the second list id the  number of sequence length, which is always 1 in vae_bev_mm
            voxels: (batch_size, depth, lidar_H, lidar_W) - Voxelized lidar points
            depth_ray_cast_center: - The center of the ray casted depth
            camera_intrinsics: (batch_size, view_count, 3, 3) - Camera intrinsic matrix
            camera_transforms: (batch_size, view_count, 4, 4) - Camera to world transformation matrix
        Return:
            decoder_output: Dict - The output containing the depth loss, voxel / depth reconstruction, and the predicted images
        """
        # deocde the lidar features
        decoder_output = self.bev_decoder(bev_feats, points=points, voxels=voxels,
                                          depth_ray_cast_center=depth_ray_cast_center,
                                          camera_intrinsics=camera_intrinsics,
                                          camera_transforms=camera_transforms)
        return decoder_output

    def forward_normal(self,
                       points: list[list[torch.tensor]],
                       images: torch.tensor,
                       camera_transforms: torch.tensor,
                       camera_intrinsics: torch.tensor,
                       ego_transforms: torch.tensor,
                       depth_ray_cast_center=None) -> dict:
        """
        Description:
            Forward function for the BEV-VAE-MM model.
            It is the normal version of the forward function, which takes both lidar points and images as input, fuses them into a single latent space and decodes them.
        Args:
            points: List[List[(N_x, 3)]] - Lidar points. The first list is the number of batch, the second list id the  number of sequence length, which is always 1 in vae_bev_mm
            images: Tensor of shape (batch_size, view_count, C, H, W) - Multiview images
            camera_intrinsics: (batch_size, view_count, 3, 3) - Camera intrinsic matrix
            camera_transforms: (batch_size, view_count, 4, 4) - Camera to world transformation matrix
            ego_transforms: (batch_size, view_count + 1, 4, 4) - Ego transformation matrix, which are the transformation matrices from the LiDAR coordinate system and the different image coordinate systems to the world coordinate system
            depth_ray_cast_center: - The center of the ray casted depth
        Return:
            results: Dict - The output containing the depth loss, reconstruction, and the predicted images
        """
        # extract fused features

        # (batch_size, depth, lidar_H, lidar_W, embed_dim)
        voxels = self.voxelizer(points)[:, 0]
        bev_feats = self.encode(
            voxels, images, camera_transforms, camera_intrinsics, ego_transforms)
        results = {}
        # Quantize the fused features
        if self.variational_model is not None:
            bev_feats, additional_loss = self.variational_model(bev_feats)
            results.update(additional_loss)
        # decode fused features

        decoder_output = self.decode(
            bev_feats, points, voxels, depth_ray_cast_center, camera_intrinsics, camera_transforms)
        results.update(decoder_output)
        return results

    def encode_imgs(self, images: torch.tensor):
        """
        Description:
            Encode the images into latent features
        """
        img_feats = self.img_encoder(
            images)  # (batch_size * view_count, img_latent_h, img_latent_w, img_encoder.out_channels)
        return img_feats

    def decode_imgs(self, img_latent: torch.tensor):
        """
        Description:
            Decode the latent features into images.
            The image decoder inside the bev_decoder is used for this process.
        Args:
            img_latent: Tensor of shape (batch_size * view_count, img_latent_h, img_latent_w, latent_dim) - Latent features
        Return:
            decoder_output: Dict - The output containing the predicted images
        """
        decoder_output = self.bev_decoder.decode_img(img_latent.contiguous())
        return decoder_output

    def forward_img(self,
                    points: list[list[torch.tensor]] = None,
                    images: torch.tensor = None,
                    camera_transforms: torch.tensor = None,
                    camera_intrinsics: torch.tensor = None,
                    ego_transforms: torch.tensor = None,
                    depth_ray_cast_center=None):
        """
        Description:
            Forward function for the image-only model.
        Args:
            images: Tensor of shape (batch_size, view_count, C, H, W) - Multiview images
        Return:
            results: Dict - The output containing the depth loss, variational loss and the predicted images
        """
        batch_size, view_count, C, H, W = images.shape
        images = images.flatten(0, 1).contiguous()
        # (batch_size * view_count, img_latent_h, img_latent_w, img_encoder.out_channels)
        img_feats = self.encode_imgs(images)
        results = {}
        # Quantize the fused features
        if self.variational_model is not None:
            # (batch_size * view_count, img_latent_h, img_latent_w, latent_dim)
            img_latent, additional_loss = self.variational_model(img_feats)
            results.update(additional_loss)
        # (batch_size * view_count, C, H, W)
        decoder_output = self.decode_imgs(img_latent)
        decoder_output['pred_imgs'] = decoder_output['pred_imgs'].contiguous().view(
            batch_size, view_count, *decoder_output['pred_imgs'].shape[1:])
        results.update(decoder_output)
        return results

    def encode_lidar(self, voxels: torch.tensor):
        """
        Description:
            Encode the voxelized lidar points into latent features.
        Args:
            voxels: (batch_size, depth, lidar_H, lidar_W) - Voxelized lidar points
        Return:
            lidar_feats: Tensor of shape (batch_size, (lidar_H // ps) * (lidar_W // ps), lidar_encoder.out_channels) - Latent features for variational model
        """
        # self.lidar_encoder = self.lidar_encoder.float()
        # (batch_size, (lidar_H // ps) * (lidar_W // ps), lidar_encoder.out_channels)
        lidar_feats = self.lidar_encoder(voxels)
        return lidar_feats

    def decode_lidar(self,
                     lidar_latent: torch.tensor,
                     points: list[list[torch.tensor]],
                     voxels: torch.tensor,
                     depth_ray_cast_center: torch.tensor):
        """
        Description:
            Decode the latent features into lidar points. In this version, the bev_decoder is replaced by the lidar_decoder
        Args:
            lidar_latent: Tensor of shape (batch_size, (lidar_H // ps) * (lidar_W // ps), latent_dim) - Latent features
            points: List[List[(N_x, 3)]] - Lidar points. It is the same as the input points, which is used for the depth loss in rendering.
            depth_ray_cast_center: - The center of the ray casted depth
        Return:
            decoder_output: Dict - The output containing the depth loss, variational loss and voxel / depth reconstruction
        """
        decoder_output = self.bev_decoder(lidar_latent.contiguous(),
                                          points,
                                          voxels,
                                          depth_ray_cast_center,)  # The bev_decoder is replaced by the lidar_decoder
        return decoder_output

    def forward_lidar(self,
                      points: list[list[torch.tensor]] = None,
                      images: torch.tensor = None,
                      camera_transforms: torch.tensor = None,
                      camera_intrinsics: torch.tensor = None,
                      ego_transforms: torch.tensor = None,
                      depth_ray_cast_center=None) -> dict:
        """
        Description:
            Forward function for the lidar-only model.
        Args:
            points: List[List[(N_x, 3)]] - Lidar points. The first list is the number of batch, the second list id the  number of sequence length, which is always 1 in vae_bev_mm
            depth_ray_cast_center: - The center of the ray casted depth
        Return:
            results: Dict - The output containing the depth loss, reconstruction, and the predicted images
        """
        # extract fused features
        # (batch_size, depth, lidar_H, lidar_W, embed_dim)
        voxels = self.voxelizer(points)[:, 0]
        lidar_feats = self.encode_lidar(voxels)

        results = {}
        # Quantize the fused features
        if self.variational_model is not None:
            lidar_feats, additional_loss = self.variational_model(lidar_feats)
            results.update(additional_loss)
        # decode fused features
        decoder_output = self.decode_lidar(
            lidar_feats, points, voxels, depth_ray_cast_center)
        results.update(decoder_output)
        return results

    def forward(self,
                points: list[list[torch.tensor]] = None,
                images: torch.tensor = None,
                camera_transforms: torch.tensor = None,
                camera_intrinsics: torch.tensor = None,
                ego_transforms: torch.tensor = None,
                depth_ray_cast_center=None) -> dict:
        """
        Description:
            Forward function for the BEV-VAE-MM model.
            The forward function is determined by the forward_type attribute.
        """
        forward_func = getattr(self, f'forward_{self.forward_type}')
        return forward_func(
            points=points,
            images=images,
            camera_transforms=camera_transforms,
            camera_intrinsics=camera_intrinsics,
            ego_transforms=ego_transforms,
            depth_ray_cast_center=depth_ray_cast_center)
