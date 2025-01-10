import torch.utils
import torch
import torch.nn.functional as F
import torch.nn as nn
from dwm.models.base_vq_models.dvgo_utils import dvgo_render
from typing import Union, Optional, List
from diffusers import Transformer2DModel
from transformers.models.swinv2.modeling_swinv2 import Swinv2Encoder, Swinv2Embeddings, Swinv2Config

import pdb


class LidarEncoder(nn.Module):
    def __init__(self,
                 img_size: list[int] = [256, 448],
                 depths: list[int] = [4, 8, 4],
                 num_heads: list[int] = [8, 8, 8],
                 window_size: int = 7,
                 patch_size: int = 2,
                 embed_dim: int = 64,
                 num_channels: int = 64,
                 use_checkpoint: bool = False,
                 upcast: bool = False,
                 ):
        """
        Args:
            image_size: list[int]. The size of the image.
            depths: list[int]. The depth of the encoder.
            num_heads: list[int]. The number of heads in the encoder.
            window_size: int. The size of the window.
            patch_size: int. The size of the patch.
            embed_dim: int. The dimension of the embedding. It is the embedding dimension of the first stage. The output dimension of the last stage is embed_dim * patch_size^2
            use_checkpoint: bool. Whether to use checkpoint.
        """
        super().__init__()
        # We adopt swin transformer encoder here
        configuration = Swinv2Config(
            image_size=img_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_channels=num_channels
        )

        self.embeddings = Swinv2Embeddings(configuration)
        self.encoder = Swinv2Encoder(configuration, self.embeddings.patch_grid)
        self.out_channels = embed_dim * 2 ** (len(depths) - 1)
        self.downsample_size = patch_size * 2 ** (len(depths) - 1)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        """
        Args:
            x: (batch_size, D, H, W) - BEV LiDAR
        Return:
            (batch_size, embed_dim * patch_size^2, (H // downsample_size) * (W // downsample_size))
        """
        embedding_output, input_dimensions = self.embeddings(x)
        if not torch.jit.is_scripting() and self.use_checkpoint:
            # use checkpoint mode to save memory
            encoder_outputs = torch.utils.checkpoint.checkpoint(
                self.encoder,
                embedding_output,
                input_dimensions,
                None,
                False,
                True,
                False,
                True,
                use_reentrant=False)
        else:
            encoder_outputs = self.encoder(
                embedding_output,
                input_dimensions,
                output_hidden_states=True,
                return_dict=True,
            )
        return encoder_outputs.reshaped_hidden_states[-1].permute(0, 2, 3, 1).contiguous()


class LidarDecoder(nn.Module):
    def __init__(
        self,
        lidar_size: Union[int, tuple[int]],
        patch_size=2,
        feature_depth=40,
        voxel_depth=64,
        embed_dim=256,
        num_heads=16,
        depth=12,
        in_channels=1024,
        bias_init: float = -3,
        upsample_style="conv_transpose",
        # Config to control whether the gt voxel is used in rendering
        use_gt_voxel: bool = False,
        visual_grid_feat_dim: int = 16,
        downsample_voxel: list[int] = [4, 8, 8],
        grid_size_offset: list[list[int]] = [
            [0, 0, 0],
            [0, 0, 0]
        ],
        use_render_decoder: bool = False,
        use_voxel_decoder: bool = True,
        render_config: bool = None,
        # Config to control whether the new type of attention is used
        upcast=False,
        use_checkpoint=True,
    ):
        """
        Args:
            feature_depth: int. The depth of the feature grid after unpatchifying.
            voxel_grid_dim: int. The dimension of the voxel grid after unpatchifying.
            grid_size_offset: list[list[int]]. The offset of the xyz grid size. It can enlarge / shrink the rendering range.
            use_gt_voxel: bool. Whether to use the gt voxel as coarse mask in rendering.
            downsample_voxel: list[int]. The downsample voxel size. The downsampling size of visual grid. It can be used to adjust the sampling step size.
            upcast: bool. Upcast the dtype to float32 in transformer models
        """
        super().__init__()
        if isinstance(lidar_size, int):
            lidar_size = (lidar_size, lidar_size)
        self.lidar_size = lidar_size

        norm_layer = nn.LayerNorm
        self.feature_depth = feature_depth
        self.voxel_depth = voxel_depth
        self.grid_size_offset = grid_size_offset
        self.latent_h = lidar_size[0] // patch_size
        self.latent_w = lidar_size[1] // patch_size
        self.patch_size = patch_size
        self.use_gt_voxel = use_gt_voxel
        self.downsample_voxel = downsample_voxel
        self.get_coarse_mask = nn.MaxPool3d(
            kernel_size=self.downsample_voxel) if use_gt_voxel else None
        # These flags are used to control the BasicLayer attn
        self.upcast = upcast
        self.decoder_embed = nn.Linear(
            in_channels, embed_dim, bias=True)
        # The positional embedding is included in the diffuser transformer model
        self.blocks = Transformer2DModel(
            num_attention_heads=num_heads,
            attention_head_dim=embed_dim // num_heads,
            num_layers=depth[0],
            in_channels=embed_dim,
            upcast_attention=upcast
        )
        self.blocks.gradient_checkpointing = use_checkpoint

        # Upsampling the feature
        if upsample_style == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(
                embed_dim, embed_dim // 2, 2, stride=2)
        else:
            self.upsample = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(embed_dim // 4, embed_dim, 1))
        # Get the voxel grid prediction
        # This flag is used to control whether to use the voxel decoder. In LidarDecoder, it is set to True.
        self.use_voxel_decoder = use_voxel_decoder
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
            embed_dim, (patch_size)**2 * voxel_depth, bias=True)
        self.apply(self._init_weights)
        torch.nn.init.constant_(self.voxel_pred.bias, bias_init)

        # Get the depth prediction
        self.use_render_decoder = use_render_decoder
        if use_render_decoder:
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
            # (2,2, feature_depth, 16), where 16 is the feature dim
            self.visual_grid_feat_dim = visual_grid_feat_dim
            self.visual_pred = nn.Linear(
                embed_dim, feature_depth * self.visual_grid_feat_dim, bias=True)
            self.density_mlp = nn.Sequential(
                nn.Linear(self.visual_grid_feat_dim,
                          self.visual_grid_feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.visual_grid_feat_dim, 1)
            )
        self.render_config = render_config
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def ray_render_depth_dvgo(self,
                              features, points,
                              coarse_mask=None,
                              offsets=None,
                              return_alpha_last=False):
        """
        Description:
            Rendering depth from the features.
        Args:
            feature: torch.tensor - (N, C, D, H, W). Features for volume rendering
            points: List[torch.tensor]. Sampled points to calculate depth loss
            coarse_mask: torch.tensor - (N, D, H, W). Binary masks to represent the occupancy of each voxel in the voxel grid.
            offsets: torch.tensor - (N, 3). The offset of the rendering center.
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
                self.grid_size["min"],
                self.grid_size["max"],
                near=self.render_config["near"],
                far=self.render_config["far"],
                stepsize=self.render_config["stepsize"],
                offsets=None, grid_size=self.grid_size,
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

    def update_render_grid_size(self, grid_size):
        """
        Description:
            Update the grid size of rendering based on some coefficents
        """
        self.grid_size = {
            "min": grid_size["min"],  # this two are used for ray casting
            "max": grid_size["max"],
            # this is only used when coarse mask is used
            "interval": [i * j for i, j in zip(grid_size["interval"], reversed(self.downsample_voxel))]
        }
        # this two are used for adjusting the rendering range
        self.grid_size["min"] = [
            i + j for i, j in zip(self.grid_size["min"], self.grid_size_offset[0])]
        self.grid_size["max"] = [
            i + j for i, j in zip(self.grid_size["max"], self.grid_size_offset[1])]

    def forward(self, x: torch.tensor,
                points: List[List[torch.tensor]] = None,
                voxels: torch.tensor = None,
                depth_ray_cast_center: torch.tensor = None):
        """
        Args:
            x: tensor - (batch_size, (lidar_H // ps) * (lidar_W // ps), in_channels)
            points: list(list(tensor)). Points to calculate depth loss
            voxels: GT voxels. We use it to match the output of other models
            depth_ray_cast_center: Lidar center for depth rendering
        """
        # embed tokens
        x = self.decoder_embed(x)
        # apply Transformer blocks
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.blocks(x).sample
        x = self.upsample(x)
        results = {}
        # generate features for rendering
        if self.use_render_decoder:
            visual_feats = self.visual_block(x).sample
            visual_feats = visual_feats.permute(0, 2, 3, 1).contiguous()
            # try to avoid nan
            visual_feats = torch.nan_to_num(visual_feats)
            visual_feats = self.visual_norm(visual_feats)
            visual_feats = self.visual_pred(visual_feats)
            visual_feats = self.unpatchify(visual_feats.flatten(
                1, 2), p=1).unflatten(1, (self.visual_grid_feat_dim, -1))
            depth_loss, sdf_loss, lidar_rec = self.ray_render_depth_dvgo(
                visual_feats, points, None, offsets=depth_ray_cast_center)
            results.update({
                "depth_loss": depth_loss,
                "sdf_loss": sdf_loss,
                "lidar_rec": lidar_rec,
            })
        # generate features for voxel prediction
        pred_voxel_feat = self.voxel_block(x).sample
        pred_voxel_feat = pred_voxel_feat.permute(0, 2, 3, 1).contiguous()
        # try to avoid nan
        pred_voxel_feat = torch.nan_to_num(pred_voxel_feat)
        pred_voxel_feat = self.voxel_norm(pred_voxel_feat.float())
        pred_voxel_feat = self.voxel_pred(pred_voxel_feat)
        pred_voxel = self.unpatchify(pred_voxel_feat.flatten(1, 2))
        results.update({
            "voxels": voxels,
            "pred_voxel": pred_voxel,
        })
        return results
