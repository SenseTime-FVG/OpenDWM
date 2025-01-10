import torch.utils
import torch
import torch.nn.functional as F
import torch.nn as nn
import  torchvision.transforms.functional
from dwm.models.voxelizer import Voxelizer
from dwm.models.vq_point_cloud import PatchMerging, BasicLayer, get_2d_sincos_pos_embed, VectorQuantizer
from dwm.models.base_vq_models.dvgo_utils import dvgo_render
import timm.models.swin_transformer
import timm.layers
from typing import Union, Optional

import pdb
# The following functions are used for visualization
def visualize_pc(pc, batch_indices, save_name = "test"):
    if batch_indices is not None:
        batch_indices = (batch_indices == 0).nonzero().squeeze(-1)
        pc = pc[batch_indices]
    pc = pc.contiguous().view(-1, 3)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy())
    o3d.io.write_point_cloud(save_name + ".ply", pcd)
def visualize_proj_pts(proj_pts, pts_valid, batch_indices, image_size = (56, 32), point_size=8, background_color=(0, 0, 0), point_color=(255, 255, 255)):
    from PIL import Image, ImageDraw
    if batch_indices is not None:
        batch_indices = (batch_indices == 0).nonzero().squeeze(-1)
        proj_pts = proj_pts[batch_indices]
        pts_valid = pts_valid[batch_indices]
    proj_pts = proj_pts.permute(2, 0, 1, 3)
    pts_valid = pts_valid.permute(2, 0, 1)
    view_count = proj_pts.shape[0]
    all_imgs = []
    # Draw each point as a square
    for i in range(view_count):
        cur_points = proj_pts[i].view(-1, 3)
        cur_valid = pts_valid[i].view(-1)
        image = Image.new('RGB', image_size, background_color)
        draw = ImageDraw.Draw(image)
        for j in range(len(cur_points)):
            x, y = cur_points[j][0], cur_points[j][1]
            if cur_valid[j] == 1:
                # Calculate the coordinates for the square
                top_left = (x - point_size // 2, y - point_size // 2)
                bottom_right = (x + point_size // 2, y + point_size // 2)
                draw.rectangle([top_left, bottom_right], fill=point_color)
        image.save(f"test_{i}.png")
        all_imgs.append(image)
    preview_imgs = Image.new(
        "RGB", (
            view_count * image_size[0],
         image_size[1]
        ))
    for j, image in enumerate(all_imgs):
        preview_imgs.paste(
            image, (j * image.width, 0))
    preview_imgs.save("test.png")
def visualize_imgs(images):
    from PIL import Image
    batch_imgs = images[0,0].cpu()
    batch_imgs = (batch_imgs)
    view_count = batch_imgs.shape[0]
    preview_imgs = Image.new(
        "RGB", (
            view_count * images.shape[-1],
         images.shape[-2]
        ))
    for j in range(view_count):
        img = torchvision.transforms.functional.to_pil_image(batch_imgs[j])
        preview_imgs.paste(
            img, (j * img.width, 0))
    preview_imgs.save("test_img.png")

def resize_camera_params(camera_intrinsics: torch.tensor,
                         image_shape: Union[tuple[int], list[int]]) -> torch.tensor:
    """
    Args:
        camera_intrinsics: (batch_size, view_count, 3, 3)
        image_shape: list or tuple of int with length 2.
    """
    camera_intrinsics_shape = camera_intrinsics.shape
    camera_intrinsics = camera_intrinsics.view(-1, 3, 3)
    for i in range(camera_intrinsics.shape[0]):
        h_ratio = (image_shape[-2] // 2 ) / camera_intrinsics[i][1,2]
        camera_intrinsics[i, :2, :] = camera_intrinsics[i, :2, :] * h_ratio
    camera_intrinsics = camera_intrinsics.view(*camera_intrinsics_shape)
    return camera_intrinsics

def get_rays(camera_transforms: torch.tensor,
             camera_intrinsics: torch.tensor,
             target_size: Union[int, tuple[int]],):
    ''' get rays
        Args:
            camera_transforms: (N, 4, 4), cam2world
            intrinsics: (N, 3, 3)
        Returns:
            rays_o, rays_d: [N, 3]
            i, j: [N]
    '''
    device = camera_transforms.device

    H, W = (target_size, target_size) if isinstance(target_size, int) else target_size
    i, j = torch.meshgrid(torch.linspace(
        0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')  # float

    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5

    zs = torch.ones_like(i)
    points_coord = torch.stack([i, j, zs]) # (H*W, 3)
    directions = torch.inverse(camera_intrinsics) @ points_coord.unsqueeze(0) # (batch_size, 3, H*W)

    rays_d = camera_transforms[:, :3, :3] @ directions # (batch_size, 3, H*W)
    rays_o = camera_transforms[:, :3, 3].unsqueeze(-1).expand_as(rays_d)  # [batch_size, 3, H*W)
    rays_o, rays_d = rays_o.permute(0, 2, 1).contiguous().view(-1, 3), rays_d.permute(0, 2, 1).contiguous().view(-1, 3) # [batch_size*H*W, 3)
    return rays_o, rays_d

class BasicConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        norm_layer: nn.Module = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: torch.tensor) -> torch.tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out + identity
        out = self.relu(out)

        return out

class ConvDecoder(nn.Module):
    def __init__(
        self, input_size: Union[int, list[int]],
        in_channels: int = 16,
        embed_dim: int = 512,
        depth: list[int] = [4, 6, 4],
        out_channels: int = 3,
        use_checkpoint: bool = False,
    ):
        """
        Description:
            CNN-based Decoder model for 2D input
        Args:
            img_size: int or list[int]. The size of input image.
            patch_size: int. The OVERALL size of the patch. In this model, it uses patch_size // 2 in patch
                        embedding and then merges the neighboring patches into a single patch.
            in_channels: int. The number of input channels.
            embed_dim: int. The dimension of the embedding.
            depth: list[int]. The depth of each block.
            out_channels: int. The dimension of the output.
            use_checkpoint: bool. Whether to use checkpoint.

        """
        super().__init__()

        norm_layer = nn.BatchNorm2d
        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        # input size of the first transformer block, i.e. the size after the patchification
        self.in_h = input_size[0]
        self.in_w = input_size[1]

        self.out_channels = out_channels
        self.use_checkpoint = use_checkpoint

        self.first_conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = embed_dim,
            kernel_size = 1,
            stride = 1
        )
        self.conv_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()
        for d in range(len(depth) - 1):
            cur_conv_blocks = nn.ModuleList([
                BasicConvBlock(
                    in_channels = embed_dim // (2 ** d),
                    out_channels = embed_dim // (2 ** d),
                    norm_layer = norm_layer
                )
             for _ in range(depth[d])
            ])
            cur_conv_blocks = nn.Sequential(*cur_conv_blocks)
            self.conv_blocks.append(cur_conv_blocks)
            self.upsampling_blocks.append(
                nn.ConvTranspose2d(embed_dim // (2 ** d), embed_dim // (2 ** (d + 1)), 2, stride=2)
            )
        final_conv =  nn.ModuleList([
                BasicConvBlock(
                    in_channels = embed_dim // (2 ** (len(depth) - 1)),
                    out_channels = embed_dim // (2 ** (len(depth) - 1)),
                    norm_layer = norm_layer
                )
             for _ in range(depth[-1])
        ])
        self.final_conv = nn.Sequential(*final_conv)
        self.final_pred = nn.Sequential(
            nn.Conv2d(embed_dim // (2 ** (len(depth) - 1)), embed_dim // (2 ** len(depth)), 1, 1, bias=False),
            norm_layer(embed_dim // (2 ** len(depth))),
            nn.Conv2d(embed_dim // (2 ** len(depth)), out_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # nn.init.constant_(self.pre_quant.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x) -> torch.tensor:
        x = self.first_conv(x)
        for (blk, upsamp) in zip(self.conv_blocks, self.upsampling_blocks):
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, use_reentrant=False)
                x = torch.utils.checkpoint.checkpoint(
                    upsamp, x, use_reentrant=False)
            else:
                x = blk(x)
                x = upsamp(x)

        x = self.final_conv(x)
        x = self.final_pred(x)

        return x


class SwimTransformerEncoder(nn.Module):
    def __init__(
        self, img_size: Union[int, list[int]],
        patch_size: int = 8, in_channels: int = 40,
        embed_dim: int = 512, num_heads: int = 16,
        depth: list[int] = [4, 8], out_channels: int = 1024,
        use_checkpoint: bool = False,
        upcast: bool = False,
        normalized_attn: bool = False,
        flatten: bool = True
    ):
        """
        Description:
            Swin-Transformer Encoder model for 2D input
        Args:
            img_size: int or list[int]. The size of input image.
            patch_size: int. The OVERALL size of the patch. In this model, it uses patch_size // 2 in patch
                        embedding and then merges the neighboring patches into a single patch.
            in_channels: int. The number of input channels.
            embed_dim: int. The dimension of the embedding.
            num_heads: int. The number of heads in the multi-head attention.
            depth: list[int]. The depth of each block.
            out_channels: int. The dimension of the codebook.
            use_checkpoint: bool. Whether to use checkpoint.
            upcast: bool. Whether to use upcast in the BasicLayer.
            normalized_attn: bool. Whether to use the new type of attention.
        """
        super().__init__()

        norm_layer = nn.LayerNorm
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        # For patch embedding, we use a patch size of patch_size // (2 ** (len(depth) - 1)) and pass it through the first Transformer Layer.
        # After each Transformer Layer, the size of the feature map is reduced by a factor of 2.
        self.patch_embed = timm.layers.PatchEmbed(
            img_size, patch_size // (2 ** (len(depth) - 1)), in_channels, embed_dim // (2 ** (len(depth) - 1)),
            norm_layer=norm_layer)
        num_patches = self.patch_embed.num_patches

        # input size of the first transformer block, i.e. the size after the patchification
        self.in_h = img_size[0] // patch_size * (2 ** (len(depth) - 1))
        self.in_w = img_size[1] // patch_size * (2 ** (len(depth) - 1))
        # output size of the feature
        self.output_h = img_size[0] // patch_size
        self.output_w = img_size[1] // patch_size
        self.patch_size = patch_size
        # This flag are used to control the BasicLayer attn
        self.upcast = upcast
        self.normalized_attn = normalized_attn
        # This flag determine whether the output results should be flattened
        self.flatten = flatten
        self.out_channels = out_channels

        pos_embed = get_2d_sincos_pos_embed(
            embed_dim // (2 ** (len(depth) - 1)), (self.in_h, self.in_w), cls_token=False)
        self.register_buffer(
            "pos_embed", torch.from_numpy(pos_embed).unsqueeze(0))

        transformer_blocks = nn.ModuleList()
        for d in range(len(depth)):
            transformer_blocks.append(
                BasicLayer(
                    # For the first transformer block, we use embed_dim // 2
                    embed_dim // (2 ** (len(depth) - d - 1)) if d < len(depth) - 1 and len(depth) > 1 else embed_dim,
                    (img_size[0] // patch_size * (2 ** (len(depth) - 1 - d)),
                     img_size[1] // patch_size * (2 ** (len(depth) - 1 - d))),
                    depth[d],
                    num_heads=num_heads,
                    window_size=8 if img_size[0] // patch_size * (2 ** (len(depth) - 1 - d)) % 8 == 0  and \
                                    img_size[1] // patch_size * (2 ** (len(depth) - 1 - d)) % 8 == 0 else 4,
                    # The output of the Transformer block will be downsampled if the block is not the last layer.
                    downsample=PatchMerging if d != len(depth) - 1 and len(depth) > 1 else None,
                    use_checkpoint=use_checkpoint,
                    upcast = upcast,
                    normalized_attn = normalized_attn
                )
            )

        self.blocks = nn.Sequential(*transformer_blocks)

        self.norm = nn.Sequential(norm_layer(embed_dim), nn.GELU())
        self.pre_quant = nn.Linear(embed_dim, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # nn.init.constant_(self.pre_quant.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pre_quant(x)
        if not self.flatten:
            batch_size = x.shape[0]
            x = x.view(batch_size, self.output_h, self.output_w, -1)

        return x


class BEVDecoder(nn.Module):
    def __init__(
        self, img_size: Union[int, tuple[int]],
        lidar_size: Union[int, tuple[int]],
        img_latent_size: Union[int, tuple[int]],
        img_decoder: nn.Module,
        num_patches, patch_size=8, feature_depth=40, voxel_depth=64,
        embed_dim=512, num_heads=16, depth=12, latent_dim=1024, bias_init:float=-3,
        upsample_style="conv_transpose", use_checkpoint=False,
        # Config to control whether the gt voxel is used in rendering
        use_gt_voxel: bool = True,
        visual_grid_feat_dim: int = 16,
        downsample_voxel: list[int] = [4, 8, 8],
        grid_size_offset: list[list[int]] = [
            [0,0,0],
            [0,0,0]
        ],
        # Config to control whether using an additional voxel predictor
        use_voxel_decoder: bool = False,
        # Config to control whether the new type of attention is used
        upcast = False,
        normalized_attn = False
    ):
        """
        Args:
            feature_depth: int. The depth of the feature grid after unpatchifying.
            voxel_grid_dim: int. The dimension of the voxel grid after unpatchifying.
            grid_size_offset: list[list[int]]. The offset of the xyz grid size. It can enlarge / shrink the rendering range.
            use_gt_voxel: bool. Whether to use the gt voxel as coarse mask in rendering.
            downsample_voxel: list[int]. The downsample voxel size. The downsampling size of visual grid. It can be used to adjust the sampling step size.
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
        self.img_latent_size = img_latent_size
        norm_layer = nn.LayerNorm
        self.patch_size = patch_size // 2
        self.feature_depth = feature_depth
        self.voxel_depth = voxel_depth
        self.grid_size_offset = grid_size_offset
        self.latent_h = lidar_size[0] // patch_size
        self.latent_w = lidar_size[1] // patch_size
        self.use_gt_voxel = use_gt_voxel
        self.downsample_voxel = downsample_voxel
        self.get_coarse_mask = nn.MaxPool3d(kernel_size=self.downsample_voxel) if use_gt_voxel else None
        # These flags are used to control the BasicLayer attn
        self.upcast = upcast,
        self.normalized_attn = normalized_attn

        self.num_patches = num_patches
        self.decoder_embed = nn.Linear(
            latent_dim, embed_dim, bias=True)
        # positional embedding block
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.latent_h, self.latent_w), cls_token=False)
        self.register_buffer(
            "pos_embed", torch.from_numpy(pos_embed).unsqueeze(0))
        self.img_decoder = img_decoder
        self.blocks = BasicLayer(
            embed_dim,
            (lidar_size[0] // patch_size, lidar_size[1] // patch_size),
            depth=depth - 2,
            num_heads=num_heads,
            window_size=8,
            use_checkpoint=use_checkpoint,
            upcast = upcast,
            normalized_attn = normalized_attn
        )

        if upsample_style == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(
                embed_dim, embed_dim // 2, 2, stride=2)
        else:
            self.upsample = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(embed_dim // 4, embed_dim, 1))
        self.use_voxel_decoder = use_voxel_decoder
        if use_voxel_decoder:
            self.voxel_block = BasicLayer(
                embed_dim,
                (lidar_size[0] // patch_size * 2, lidar_size[1] // patch_size * 2),
                depth=2,
                num_heads=num_heads,
                window_size=8,
                use_checkpoint=use_checkpoint,
                normalized_attn = normalized_attn
            )
            self.voxel_norm = torch.nn.Sequential(
                norm_layer(embed_dim), torch.nn.GELU())
            self.voxel_pred = torch.nn.Linear(
                embed_dim, (patch_size // 2)**2 * voxel_depth, bias=True)
            self.apply(self._init_weights)
            torch.nn.init.constant_(self.voxel_pred.bias, bias_init)

        # get the visual and depth prediction
        self.visual_block = BasicLayer(
            embed_dim,
            (lidar_size[0] // patch_size * 2, lidar_size[1] // patch_size * 2),
            depth=2,
            num_heads=num_heads,
            window_size=8,
            use_checkpoint=use_checkpoint,
            normalized_attn = normalized_attn,
        )
        self.visual_norm = nn.Sequential(
            norm_layer(embed_dim), nn.GELU())
        # For each point of visual_pred output, it will be reshaped to a voxel grid with shape
        # (2,2, feature_depth, 16), where 16 is the feature dim
        self.visual_grid_feat_dim = visual_grid_feat_dim
        self.visual_pred = nn.Linear(
            embed_dim, 4 * feature_depth * self.visual_grid_feat_dim, bias=True)

        self.density_mlp = nn.Sequential(
            nn.Linear(self.visual_grid_feat_dim, self.visual_grid_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.visual_grid_feat_dim, 1)
        )
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

    def ray_render_depth_dvgo(self, features, points, coarse_mask=None, offsets=None, return_alpha_last=False):
        """
        Description:
            Rendering depth from the features.
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
            iter_points = points[iter_batch][:, :3].contiguous().to(features.device)
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
                self.grid_size["max"], stepsize=0.05,
                offsets=None, grid_size=self.grid_size)

            loss_depth = loss_depth + soft_l1(pred_depth, gt_depth.squeeze(-1))
            loss_sdf = loss_sdf + loss_sdf_i
            rec_points.append(rays_o + pred_depth.unsqueeze(-1) / gt_depth * rays_d)
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
        """
        batch_num, view_count = camera_transforms.shape[:2]
        pred_img_latents = []
        camera_intrinsic = resize_camera_params(camera_intrinsic, self.img_latent_size)

        for iter_batch in range(batch_num):
            # get the ray centers and directions of all cameras in this batch
            rays_o, rays_d = get_rays(camera_transforms = camera_transforms[iter_batch],
                     camera_intrinsics = camera_intrinsic[iter_batch],
                     target_size = self.img_latent_size
                     ) # (view_count * latent_h * latent_w, 3), (view_count * latent_h * latent_w, 3)
            batch_pred_img_latents, _, _ = dvgo_render(
                self.density_mlp,
                coarse_mask[iter_batch] if coarse_mask is not None else None,
                rays_o.float(), rays_d.float(),
                torch.einsum(
                    'dzyx->dxyz', features[iter_batch].float()).unsqueeze(0),  # pred grids
                self.grid_size["min"],
                self.grid_size["max"], stepsize=0.05,
                offsets=None, grid_size=self.grid_size,
                feat_render=True)
            pred_img_latents.append(batch_pred_img_latents)
        pred_img_latents = torch.stack(pred_img_latents, dim = 0).contiguous().view(batch_num, view_count, *self.img_latent_size, -1)
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
    def update_render_grid_size(self, grid_size):
        self.grid_size = {
            "min": grid_size["min"], # this two are used for ray casting
            "max": grid_size["max"],
            "interval": [i * j for i, j in zip(grid_size["interval"], reversed(self.downsample_voxel))] # this is only used when coarse mask is used
        }
        # this two are used for adjusting the rendering range
        self.grid_size["min"] = [i + j for i, j in zip(self.grid_size["min"], self.grid_size_offset[0])]
        self.grid_size["max"] = [i + j for i, j in zip(self.grid_size["max"], self.grid_size_offset[1])]

    def forward(self, x: torch.tensor,
                points: torch.tensor = None,
                voxels: torch.tensor = None,
                camera_intrinsics: torch.tensor = None,
                camera_transforms: torch.tensor = None,
                depth_ray_cast_center: torch.tensor = None):
        """
        Args:
            x: (batch_size, (lidar_H // ps) * (lidar_W // ps), latent_dim)
        """
        batch_size, view_count, _ , _ = camera_transforms.shape
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        x = self.blocks(x)

        N, L, C = x.shape

        hw = int(L ** 0.5)
        x = x.view(N, hw, hw, C).permute(0, 3, 1, 2)
        x = self.upsample(x).permute(0, 2, 3, 1)
        x = x.reshape(N, -1, x.shape[-1])

        # generate features
        visual_feats = self.visual_block(x)
        visual_feats = self.visual_norm(visual_feats)
        visual_feats = self.visual_pred(visual_feats)
        visual_feats = self.unpatchify(visual_feats, p=self.patch_size // 2).unflatten(1, (self.visual_grid_feat_dim, -1))

        # ray cast the depth and image latents
        pooled_voxels = self.get_coarse_mask(voxels) if self.use_gt_voxel else None
        # depth_loss, sdf_loss, lidar_rec = self.ray_render_depth_dvgo(visual_feats, points, pooled_voxels, offsets=depth_ray_cast_center)
        # img_latents = self.ray_render_img_dvgo(visual_feats, camera_transforms, camera_intrinsics, pooled_voxels, offsets=depth_ray_cast_center)
        # TODO: do not use coarse masks in reconstruction
        depth_loss, sdf_loss, lidar_rec = self.ray_render_depth_dvgo(visual_feats, points, pooled_voxels, offsets=depth_ray_cast_center)
        img_latents = self.ray_render_img_dvgo(visual_feats, camera_transforms, camera_intrinsics, pooled_voxels)

        # decode imgs
        img_latents_shape = img_latents.shape
        pred_imgs = self.img_decoder(img_latents.view(-1, *img_latents_shape[2:]).permute(0, 3, 1, 2).contiguous())
        pred_imgs = pred_imgs.view(*img_latents_shape[:2], *pred_imgs.shape[1:])
               # reconstruct the voxel grid
        if self.use_voxel_decoder:
            pred_voxel_feat = self.voxel_block(x)
            pred_voxel_feat = self.voxel_norm(pred_voxel_feat)
            pred_voxel_feat = self.voxel_pred(pred_voxel_feat)
            pred_voxel = self.unpatchify(pred_voxel_feat)
        else:
            # reconstruct the voxel grid from the visual features
            visual_feats_shape = visual_feats.shape
            visual_voxel_feats = visual_feats.contiguous().view(batch_size, self.visual_grid_feat_dim, -1).permute(0, 2, 1).contiguous().view(-1, self.visual_grid_feat_dim)
            visual_voxel_rec = self.density_mlp(visual_voxel_feats).view(batch_size, -1, 1).permute(0, 2, 1).view(batch_size, *visual_feats_shape[2:])
            pred_voxel = F.interpolate(visual_voxel_rec, size = self.lidar_size, mode = "nearest")

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
    def __init__(self, in_channels, out_channels, num_heads, upcast = True):
        super(DeformableAttention, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Linear layers for query, key, and value projections
        self.query_proj = nn.Linear(in_channels, out_channels)
        self.key_proj = nn.Linear(in_channels, out_channels)
        self.value_proj = nn.Linear(in_channels, out_channels)
        # Multi-head attention module
        self.multihead_attn = nn.MultiheadAttention(out_channels, num_heads, batch_first = True)

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
        keys = self.key_proj(sampled_img_features)      # Shape (N, K, out_channels)
        values = self.value_proj(sampled_img_features)  # Shape (N, K, out_channels)
        # Perform multi-head attention
        output, _ = self.multihead_attn(queries.unsqueeze(1), keys, values, key_padding_mask = masks)
        # Concatenate heads and project to output channels
        output = output.view(N, -1)  # Shape (N, out_channels)
        output = self.output_proj(output)  # Final output projection

        return output

class VAEBevMultiModality(nn.Module):
    def __init__(self,
                voxelizer: Voxelizer,
                lidar_encoder: SwimTransformerEncoder,
                img_encoder: SwimTransformerEncoder,
                bev_decoder: BEVDecoder,
                deformable_transformer: DeformableAttention,
                latent_dim: int,
                vector_quantizer: VectorQuantizer = None,
                bias_init: float = -5,
                num_sample_per_pillar: int = 20,
        ):
        super(VAEBevMultiModality, self).__init__()
        self.voxelizer = voxelizer
        self.vector_quantizer = vector_quantizer
        self.lidar_encoder = lidar_encoder
        grid_size = {
            "min": [voxelizer.x_min, voxelizer.y_min, voxelizer.z_min],
            "max": [voxelizer.x_max, voxelizer.y_max, voxelizer.z_max],
            "interval": [voxelizer.step, voxelizer.step, voxelizer.z_step]
        }

        self.img_encoder = img_encoder
        self.bev_decoder = bev_decoder
        self.bev_decoder.update_render_grid_size(grid_size)
        self.bev_feature_layer = nn.Linear(lidar_encoder.out_channels, latent_dim)
        self.deformable_transformer = deformable_transformer
        self.latent_dim = latent_dim # C' in the paper
        self.num_sample_per_pillar = num_sample_per_pillar

        # buffer to store the age and usage of the codebook
        if vector_quantizer is not None:
            self.register_buffer(
                "code_age", torch.zeros(self.vector_quantizer.n_e) * 10000)
            self.register_buffer(
                "code_usage", torch.zeros(self.vector_quantizer.n_e))


    def sample_pts_from_voxel(self, voxels: torch.tensor, patch_size: int):
        """
        Args:
            voxels: (batch_size, depth, lidar_H, lidar_W)
            patch_size: int. Patch size in the encoder
        Return:
            nonempty_pillar_indices: (batch_size * lidar_H * lidar_W, )
            sampled_points_indices: (batch_size * lidar_H * lidar_W, num_sample_per_pillar, )
        """
        voxels = voxels.permute(0, 2, 3, 1)
        sample_pillar = []

        # concatenate all the points inside the feature pillars
        for i in range(patch_size):
            for j in range(patch_size):
                sample_pillar.append(voxels[:, i::patch_size, j::patch_size, :])
        sample_pillar = torch.cat(sample_pillar, dim = -1)
        sample_pillar_shape = sample_pillar.shape
        # flatten all the pillars
        sample_pillar = sample_pillar.contiguous().view(-1, sample_pillar_shape[-1])
        # exclude the pillars that do not contain any points
        nonempty_pillar_indices = torch.sum(sample_pillar, dim = -1) > 0
        sample_pillar = sample_pillar[nonempty_pillar_indices]
        # sample points in each pillar
        sampled_points_indices = torch.multinomial(sample_pillar, self.num_sample_per_pillar, replacement = True)

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
                                      debug_points = None):
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
        camera_transforms = camera_transforms.view(-1, 4, 4).float() # [batch_size * view_count, 4, 4]
        camera_extrinsics = torch.inverse(camera_transforms)
        camera_extrinsics = camera_extrinsics[:, :3, :] # [batch_size * view_count, 3, 4]
        camera_extrinsics = camera_extrinsics.view(batch_size, view_count, 3, 4).float() # [batch_size, view_count, 3, 4]
        camera_intrinsics = resize_camera_params(camera_intrinsics, img_shape) # [batch_size, view_count, 3, 3]
        camera_intrinsics = camera_intrinsics.view(batch_size, view_count, 3, 3).float() # [batch_size * view_count, 3, 3]

        lidar_ego_transforms = ego_transforms[:, :1] # [batch_size, 1, 4, 4]
        cam_ego_inverse_transforms = ego_transforms[:, 1:].contiguous().view(-1, 4, 4).float()
        cam_ego_inverse_transforms = torch.inverse(cam_ego_inverse_transforms)
        cam_ego_inverse_transforms = cam_ego_inverse_transforms.view(batch_size, view_count, 4, 4).float() # [batch_size, view_count, 4, 4]
        sampled_voxel_coords = []
        # Concatenate all the points inside the feature pillars. The points are sampled in a patch_size x patch_size grid
        for i in range(patch_size):
            for j in range(patch_size):
                sampled_voxel_coords.append(voxel_coords[i::patch_size, j::patch_size, ...])
        sampled_voxel_coords = torch.cat(sampled_voxel_coords, dim = -2) # [lidar_H // ps, lidar_W // ps, ps * ps * depth, 3]
        if nonempty_pillar_indices is not None:
            # Sample points from voxel grid. Otherwise, use all voxel points
            original_sampled_voxel_shape = sampled_voxel_coords.shape
            flatten_indices = torch.where(nonempty_pillar_indices)[0]
            # batch indices of the sampled points
            batch_indices = flatten_indices // (original_sampled_voxel_shape[0] * original_sampled_voxel_shape[1])
            # indices of the sampled points inside the voxel grid
            grid_indices = (flatten_indices % (original_sampled_voxel_shape[0] * original_sampled_voxel_shape[1])).\
                unsqueeze(-1).expand_as(sampled_points_indices)
            sampled_voxel_coords = sampled_voxel_coords.view(-1, original_sampled_voxel_shape[-2], 3)
            sampled_voxel_coords = sampled_voxel_coords[grid_indices, sampled_points_indices] # (num_non_empty, num_sample_per_pillar, 3)
            # select corresponding camera extrinsics and intrinsics
            camera_extrinsics = camera_extrinsics[batch_indices] # (num_non_empty, view_count, 3, 4)
            camera_intrinsics = camera_intrinsics[batch_indices] # (num_non_empty, view_count, 3, 3)
            lidar_ego_transforms = lidar_ego_transforms[batch_indices] # (num_non_empty, 1, 4, 4)
            cam_ego_inverse_transforms = cam_ego_inverse_transforms[batch_indices] # (num_non_empty, view_count, 4, 4)
            # Add extra dimension to the sampled points for matrix multiplication
            sampled_voxel_extra_dim = torch.ones(*sampled_voxel_coords.shape[:-1], 1).to(sampled_voxel_coords.device)
            sampled_voxel_coords = torch.cat([sampled_voxel_coords, sampled_voxel_extra_dim], dim = -1) # (num_non_empty, num_sample_per_pillar, 4)
            sampled_voxel_coords = sampled_voxel_coords.permute(0, 2, 1).unsqueeze(1) # (num_non_empty, 1, 4, num_sample_per_pillar)
            # Project the points to the world space
            sampled_cam_point_coords = lidar_ego_transforms @ sampled_voxel_coords # (num_non_empty, 1, 4, num_sample_per_pillar)
            # Project the points to the camera ego space
            sampled_cam_point_coords = cam_ego_inverse_transforms @ sampled_cam_point_coords # (num_non_empty, view_count, 4, num_sample_per_pillar)
            # Project the points to the camera space
            sampled_cam_point_coords = camera_extrinsics @ sampled_voxel_coords # (num_non_empty, view_count, 3, 4) @ (num_non_empty, 1, 4, num_sample_per_pillar)
            sampled_cam_point_coords = sampled_cam_point_coords.float()
            sampled_cam_point_coords_z = sampled_cam_point_coords[..., -1:, :]
            # sampled_cam_point_coords_z = torch.where(sampled_cam_point_coords_z.abs() < 1e-6, 1e-6 * sampled_cam_point_coords_z.sign(), sampled_cam_point_coords_z) # avoid 0 value in division
            # sampled_cam_point_coords[..., -1:, :] = 1. # set the last dimension to 1
            sampled_cam_point_coords = camera_intrinsics @ (sampled_cam_point_coords / sampled_cam_point_coords_z) # (num_non_empty, view_count, 3, 3) @ # (num_non_empty, 1, 3, num_sample_per_pillar)
            # Permute the camera points
            sampled_cam_point_coords = sampled_cam_point_coords.permute(0, 3, 1, 2) # (num_non_empty, num_sample_per_pillar, view_count, 3)
            sampled_cam_point_coords_z = sampled_cam_point_coords_z.permute(0, 3, 1, 2) # (num_non_empty, num_sample_per_pillar, view_count, 1)

        else:
            # Flatten the sampled voxel coordinates to (N, 3)
            # The shape is [(lidar_H // ps) * (lidar_W // ps) * ps * ps * depth, 3]
            sampled_voxel_coords_shape = sampled_voxel_coords.shape
            sampled_voxel_coords = sampled_voxel_coords.contiguous().view(-1, 3)

            # Flatten the camera matrices
            camera_extrinsics = camera_extrinsics.view(-1, 3, 4) # [N, 3, 4] or [batch_size * view_count, 3, 4]
            camera_intrinsics = camera_intrinsics.view(-1, 3, 3) # [N, 3, 3] or [batch_size * view_count, 3, 3]

            # Add extra dimension to the sampled points for matrix multiplication
            sampled_voxel_extra_dim = torch.ones(sampled_voxel_coords.shape[0], 1).to(sampled_voxel_coords.device) # [N, 1]
            sampled_voxel_coords = torch.cat([sampled_voxel_coords, sampled_voxel_extra_dim], dim = -1) # [N, 4]
            sampled_voxel_coords = sampled_voxel_coords.permute(1, 0) # [4, N]

            # Project 3d points onto images
            sampled_cam_point_coords = camera_extrinsics @ sampled_voxel_coords # [batch_size * view_count, 3, 4] @ [4, N]
            sampled_cam_point_coords = sampled_cam_point_coords / sampled_cam_point_coords[:, -1:, :] # [batch_size * view_count, 3, N]
            sampled_cam_point_coords = camera_intrinsics @ sampled_cam_point_coords # [batch_size * view_count, 3, N]
            sampled_cam_point_coords = sampled_cam_point_coords.view(batch_size, view_count, 3, -1) # [batch_size, view_count, 3, N]
            # reshape the camera points
            sampled_cam_point_coords = sampled_cam_point_coords.permute(0, 3, 1, 2) # [batch_size, N, view_count, 3]
            sampled_cam_point_coords = sampled_cam_point_coords.view(batch_size, -1, sampled_voxel_coords_shape[-2], view_count, 3) # [batch_size, (lidar_H // ps) * (lidar_W // ps), ps * ps * depth, view_count, 3]
        # get the valid points
        valid_point_mask = (sampled_cam_point_coords_z[..., 0] > 0.) & \
                           sampled_cam_point_coords[..., 0].isnan().logical_not() & \
                           sampled_cam_point_coords[..., 1].isnan().logical_not() & \
                            (sampled_cam_point_coords[..., 0] < img_shape[1]) & \
                            (sampled_cam_point_coords[..., 1] < img_shape[0]) & \
                            (sampled_cam_point_coords[..., 0] >= 0) & \
                            (sampled_cam_point_coords[..., 1] >= 0) # TODO: the last four lines are used to filter out the points outside the image
        valid_point_mask = valid_point_mask.contiguous()
        # set the coordinates of invalid points to 0
        sampled_cam_point_coords[..., 0] = torch.where(valid_point_mask, sampled_cam_point_coords[..., 0], 0)
        sampled_cam_point_coords[..., 1] = torch.where(valid_point_mask, sampled_cam_point_coords[..., 1], 0)
        sampled_cam_point_coords = sampled_cam_point_coords.to(int).contiguous()
        # visualize_proj_pts(sampled_cam_point_coords, valid_point_mask, batch_indices)
        # visualize_pc(sampled_voxel_coords[..., :-1, :].permute(0, 1, 3, 2), batch_indices, "test_pc_sample")
        return sampled_cam_point_coords, valid_point_mask

    def encode(self,
               voxels: torch.tensor,
               images: torch.tensor,
               camera_transforms: torch.tensor,
               camera_intrinsics: torch.tensor,
               ego_transforms:torch.tensor):
        """
        Args:
            voxels: (batch_size, depth, lidar_H, lidar_W)
            images: (batch_size, view_count, C, H, W)
            camera_transforms: (batch_size, view_count, 4, 4)
            camera_intrinsics: (batch_size, view_count, 3, 3)
            ego_transforms: (batch_size, view_count + 1, 4, 4)
        Return:
            bev_feats: (batch_size, (lidar_H // ps) * (lidar_W // ps), lidar_encoder.out_channels)
            results: Dict - The output may contain the emb_loss if vector_quantizer is used
        """
        lidar_feats = self.lidar_encoder(voxels)  # (batch_size, (lidar_H // ps) * (lidar_W // ps), lidar_encoder.out_channels)
        # extract image features
        batch_size, view_count, C, H, W = images.shape
        multiview_imgs = images.view(-1, C, H, W)
        img_feats = self.img_encoder(multiview_imgs) # (batch_size * view_count, img_H // img_ps, img_W // img_ps, img_encoder.out_channels)

        # sample points from the voxel
        lidar_patch_size = self.lidar_encoder.patch_size
        nonempty_pillar_indices, sampled_points_indices = self.\
            sample_pts_from_voxel(voxels, lidar_patch_size) # (batch_size * (lidar_H // ps) * (lidar_W // ps)), (num_noempty_pillar, num_sample_per_pillar)

        # Get voxel volume coordinates
        voxel_coords = self.voxelizer.get_voxel_coordinates().to(camera_transforms.device) # [depth, lidar_H, lidar_W, 3]
        voxel_coords = voxel_coords.permute(1,2,0,3) # [lidar_H, lidar_W, lidar_depth, 3]
        # project all voxel points into the camera space
        img_shape = img_feats.shape[-3:-1]
        sampled_cam_point_coords, valid_point_mask = VAEBevMultiModality.get_camera_points_coordinates(
            camera_transforms, camera_intrinsics, ego_transforms, lidar_patch_size, img_shape, voxel_coords,
            nonempty_pillar_indices, sampled_points_indices) # (num_non_empty, num_sample_per_pillar, view_count, 3)
        # # Obtain coordinates of the sampled points in camera space.
        # # When nonempty_pillar_indices is not provided in get_camera_points_coordinates it will be useful.
        # sampled_cam_point_coords = sampled_cam_point_coords.contiguous().\
        #     view(nonempty_pillar_indices.shape[0], -1, view_count, 3) # [batch_size * (lidar_H // ps) * (lidar_W // ps), ps * ps * depth, view_count, 3]
        # sampled_cam_point_coords = sampled_cam_point_coords[nonempty_pillar_indices] # [num_non_empty, ps * ps * depth, 3]
        # sampled_points_indices = sampled_points_indices[..., None, None].\
        #     expand(-1, -1, *sampled_cam_point_coords.shape[-2:])
        # sampled_cam_point_coords = torch.gather(sampled_cam_point_coords, 1, sampled_points_indices) # (num_noempty_pillar, num_sample_per_pillar, 3)

        # sample lidar features
        lidar_feats_shape = lidar_feats.shape # (batch_size, (lidar_H // ps) * (lidar_W // ps), lidar_encoder.out_channels)
        lidar_feats = lidar_feats.view(-1, self.lidar_encoder.out_channels) # [batch_size * (lidar_H // ps) * (lidar_W // ps), encoder.out_channels]
        sampled_lidar_feats = lidar_feats[nonempty_pillar_indices] # [num_non_empty, encoder.out_channels]

        # obtain the batch indices
        batch_indices = torch.where(nonempty_pillar_indices)[0] // (lidar_feats_shape[1]) # [num_non_empty]
        batch_indices = batch_indices[..., None, None].expand_as(sampled_cam_point_coords[..., 0]) # [num_non_empty, num_sample_per_pillar, view_count]
        # obtain the view indices
        view_indices = torch.arange(view_count).to(int).to(batch_indices.device) # [view_count]
        view_indices = view_indices[None, None, ...].expand_as(sampled_cam_point_coords[..., 0]) # [num_non_empty, num_sample_per_pillar, view_count]
        # obtain the sampled image features
        sampled_img_feats = img_feats.view(batch_size, view_count, *img_feats.shape[1:]) # (batch_size, view_count, lidar_H // ps, lidar_W // ps, embed_dim)
        sampled_cam_point_coords = sampled_cam_point_coords.to(int)
        sampled_img_feats = sampled_img_feats[batch_indices, view_indices, sampled_cam_point_coords[..., 1], sampled_cam_point_coords[..., 0]] # [num_non_empty, embed_dim]
        sampled_img_feats = sampled_img_feats.view(sampled_img_feats.shape[0], -1, sampled_img_feats.shape[-1]) # (num_non_empty, num_sample_per_pillar * view_count, embed_dim)
        valid_point_mask = valid_point_mask.view(valid_point_mask.shape[0], -1) # (num_non_empty, num_sample_per_pillar * view_count)

        # deformable attention to fuse the features
        fused_feats = self.deformable_transformer(sampled_img_feats, sampled_lidar_feats, valid_point_mask)
        # replace the lidar features with the fused_feats
        lidar_feats[nonempty_pillar_indices] = fused_feats
        bev_feats = self.bev_feature_layer(lidar_feats)
        results = {}
        # Quantize the fused features
        if self.vector_quantizer is not None:
            bev_feats, emb_loss, _ = self.vector_quantizer(
            bev_feats, self.code_age, self.code_usage)
            results["emb_loss"] = emb_loss
        bev_feats = bev_feats.view(*lidar_feats_shape[:-1], -1)
        return bev_feats, results

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
        decoder_output = self.bev_decoder(bev_feats, points = points, voxels = voxels,
                         depth_ray_cast_center = depth_ray_cast_center,
                         camera_intrinsics = camera_intrinsics,
                         camera_transforms = camera_transforms)
        return decoder_output

    def forward(self,
                points: list[list[torch.tensor]],
                images: torch.tensor,
                camera_transforms: torch.tensor,
                camera_intrinsics: torch.tensor,
                ego_transforms:torch.tensor,
                depth_ray_cast_center = None) -> dict:
        """
        Args:
            points: List[List[(N_x, 3)]] - Lidar points. The first list is the number of batch, the second list id the  number of sequence length, which is always 1 in vae_bev_mm
            images: Tensor of shape (batch_size, view_count, C, H, W) - Multiview images
            camera_intrinsics: (batch_size, view_count, 3, 3) - Camera intrinsic matrix
            camera_transforms: (batch_size, view_count, 4, 4) - Camera to world transformation matrix
            ego_transforms: (batch_size, view_count + 1, 4, 4) - Ego transformation matrix
            depth_ray_cast_center: - The center of the ray casted depth
        Return:
            results: Dict - The output containing the depth loss, reconstruction, and the predicted images
        """
        # extract fused features
        voxels = self.voxelizer(points)[:, 0] # (batch_size, depth, lidar_H, lidar_W, embed_dim)
        bev_feats, results = self.encode(voxels, images, camera_transforms, camera_intrinsics, ego_transforms)

        # decode fused features
        decoder_output = self.decode(bev_feats, points, voxels, depth_ray_cast_center, camera_intrinsics, camera_transforms)
        results.update(decoder_output)
        return results
if __name__ == "__main__":
    import os
    import numpy as np
    # "voxelizer": {
    #             "_class_name": "dwm.models.voxelizer.Voxelizer",
    #             "x_min": -50.0,
    #             "x_max": 50.0,
    #             "y_min": -50.0,
    #             "y_max": 50.0,
    #             "z_min": -3.0,
    #             "z_max": 5.0,
    #             "step": 0.15625,
    #             "z_step": 0.125
    #         },
    voxelizer = Voxelizer(
        x_min = -50.0,
        x_max = 50.0,
        y_min = -50.0,
        y_max = 50.0,
        z_min = -3.0,
        z_max = 5.0,
        step = 0.15625,
        z_step = 0.125)

    # "lidar_encoder": {
    #             "_class_name": "dwm.models.base_vq_models.vq_point_cloud.VQEncoder",
    #             "img_size": 640,
    #             "codebook_dim": 1024,
    #             "in_chans": 64,
    #             "embed_dim": 512,
    #             "num_heads": 8,
    #             "use_checkpoint": true
    #         },
    lidar_encoder = SwimTransformerEncoder(
        img_size = [
                    640,
                    640
                ],
        in_channels= 64,
        embed_dim= 512,
        num_heads= 8,
        out_channels= 1024,
        use_checkpoint= True,
        upcast= True,
    )

    # img_encoder
    img_encoder = SwimTransformerEncoder(
        img_size=[256, 448],
        out_channels=1024,
        in_channels=3,
        embed_dim=512,
        num_heads=8,
        use_checkpoint=False,
        flatten=False
    )

    img_decoder = ConvDecoder(
        input_size=[256 // 4, 448 // 4],
        in_channels=16,
        embed_dim=512,
        out_channels=3,
        depth= [3, 3, 2]
    )
    # "lidar_decoder": {
    #             "_class_name": "dwm.models.base_vq_models.vq_point_cloud.VQDecoder",
    #                 "img_size": [
    #                 640,
    #                 640
    #             ],
    #             "num_patches": 6400,
    #             "in_chans": 64,
    #             "embed_dim": 512,
    #             "num_heads": 8,
    #             "codebook_dim": 1024,
    #             "bias_init": -6.9,
    #             "upsample_style": "pixel_shuffle",
    #             "use_checkpoint": true,
    #             "upcast": true
    # }
    # bev decoder
    bev_decoder = BEVDecoder(
        img_size = (256, 448),
        lidar_size = (640, 640),
        img_latent_size = (256 // 4, 448 // 4),
        num_patches = 6400,
        img_decoder = img_decoder,
        feature_depth = 64,
        embed_dim = 256,
        num_heads = 8,
        latent_dim = 4,
        bias_init = -6.9,
        upsample_style =  "pixel_shuffle",
        use_checkpoint = True,
    )


    deformable_attention = DeformableAttention(in_channels=1024, out_channels=1024, num_heads=8)

    model = VAEBevMultiModality(
        voxelizer=voxelizer,
        lidar_encoder=lidar_encoder,
        img_encoder=img_encoder,
        bev_decoder=bev_decoder,
        deformable_transformer=deformable_attention,
        latent_dim=4,
        device='cuda'
    )


    model = model.half().cuda()
    data_root = "/mnt/storage/user/liuyichen/repo/DWM/scripts/lyc_noupload/test_data"
    batch = {}
    batch["lidar_points"] = [[torch.tensor(np.load(os.path.join(data_root, f'test_lidar_points{i}.npy')))] for i in range(2)]
    batch["lidar_transforms"] = torch.tensor(np.load(os.path.join(data_root, 'test_lidar_transforms.npy')))[:2]
    batch["vae_images"] = torch.tensor(np.load(os.path.join(data_root, 'test_vae_images.npy')))[:2]
    batch["camera_transforms"] = torch.tensor(np.load(os.path.join(data_root, 'test_camera_transforms.npy')))[:2]
    batch["camera_intrinsics"] = torch.tensor(np.load(os.path.join(data_root, "test_camera_intrinsics.npy")))[:2]
    batch["ego_transforms"] = torch.tensor(np.load(os.path.join(data_root, "test_ego_transforms.npy")))[:2]

    from dwm.functional import make_homogeneous_vector
    def get_points(batch, device):
            return [
                [
                    (make_homogeneous_vector(
                        p_j.to(device)) @ t_j.permute(1, 0))[:, :3]
                    for p_j, t_j in zip(p_i, t_i.flatten(0, 1))
                ]
                for p_i, t_i in zip(
                    batch["lidar_points"],
                    batch["lidar_transforms"].to(device))
            ]
    points = get_points(
            batch, 'cuda')
    ray_cast_center = [
                1.0,
                0.0,
                2.0
            ]
    batch_size = len(batch["lidar_points"])
    ray_cast_center = torch.tensor([ray_cast_center])\
                    .repeat(batch_size, 1)
    with torch.autocast(device_type="cuda"):
        model(points = points, images = batch['vae_images'],
                camera_transforms = batch["camera_transforms"].cuda(),
                camera_intrinsics = batch["camera_intrinsics"].cuda(),
                depth_ray_cast_center = ray_cast_center)
