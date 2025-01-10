from collections import OrderedDict
from typing import Any, List, Mapping, Optional, Tuple, Union
import torch
from torch import nn
from torch.cuda.amp import autocast
from transformers.pytorch_utils import Conv1D
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
import diffusers

from einops import repeat, rearrange
from dwm.utils.ops.bev_pool_v2.bev_pool import bev_pool_v2

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    omega = omega.to(pos.device)
    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

class SimpleGPT2Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, is_cross_attention=False, layer_idx=None):
        super().__init__()

        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = True
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = False
        self.layer_idx = layer_idx

        self.q_attn = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_attn = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_attn = Conv1D(self.embed_dim, self.embed_dim)

        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        self.is_causal = True

        self.pruned_heads = set()

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # === asserts
        assert attention_mask is None           # for cop4d
        assert head_mask is None           # for cop4d

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(self,
        hidden_states_q, hidden_states_k, hidden_states_v,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        query = self.q_attn(hidden_states_q)
        key = self.k_attn(hidden_states_k)
        value = self.v_attn(hidden_states_v)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)           # residual connection is on the following module

        outputs = (attn_output, )
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class LSSBased2DTO3D(nn.Module):
    def __init__(self, embed_dim=512, out_dim=512, num_depth_adapter=0, adapt_type='avg'):
        super().__init__()
        if num_depth_adapter > 0:
            if adapt_type == 'avg':
                layers = [nn.AvgPool2d(2, 2, 0) for _ in range(num_depth_adapter)]
            elif adapt_type == 'max':
                layers = [nn.MaxPool2d(2, 2, 0) for _ in range(num_depth_adapter)]
            self.depth_adapter =  nn.Sequential(*layers)
        else:
            self.depth_adapter = nn.Identity()
        self.bev_ft_align = nn.Linear(embed_dim, out_dim)

        self.grid_size = None
        self.collapse_z = True

    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        # convert coordinate into the voxel space
        # === old in bev_pool
        # coor = ((coor - self.grid_lower_bound.to(coor)) /
        #         self.grid_interval.to(coor))

        # === new in dwm
        coor = torch.cat([coor, coor.new_zeros(list(coor.shape[:-1]) + [1])], dim=-1)

        # === fin

        coor = coor.long().view(num_points, 3)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, coor, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[0]),
                int(self.grid_size[1])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def forward(self, bev_features, frustum_features,
        frustum_bev_coordinates, last_step_depth_features,
        bev_coordinates=None,
        return_shape='chw', with_softmax=False,
        timesteps=None):
        b, t, bev_h, bev_w, _ = bev_features.shape
        num_d, num_h, num_w = frustum_bev_coordinates.shape[3: 6]

        if self.grid_size is None:
            self.grid_size = [bev_w, bev_h, 1]          # now, ignore z
        frustum_bev_coordinates = (frustum_bev_coordinates + 1) / 2         # [-1, 1] -> [0, 1]
        frustum_bev_coordinates = (frustum_bev_coordinates*frustum_bev_coordinates.new_tensor([bev_w, bev_h])).floor()
        frustum_bev_coordinates = rearrange(frustum_bev_coordinates, "b t v d h w o -> (b t) v d h w o")

        frustum_features = rearrange(frustum_features, "b t v c h w -> (b t) v c h w")

        last_step_depth_features = rearrange(last_step_depth_features, "b t v d h w -> (b t v) d h w")
        last_step_depth_features = self.depth_adapter(last_step_depth_features)
        last_step_depth_features = rearrange(last_step_depth_features, "(b t v) d h w -> (b t) v d h w", b=b, t=t)

        if not with_softmax:
            last_step_depth_features = last_step_depth_features.softmax(-3)

        bev_feat = self.voxel_pooling_v2(frustum_bev_coordinates, last_step_depth_features,
            frustum_features)
        bev_feat = rearrange(bev_feat, "b c h w -> b (h w) c")
        rt = self.bev_ft_align(bev_feat)
        return rt


class LSSBased3DTO2D(nn.Module):
    def __init__(self, embed_dim=512, out_dim=512, num_depth_adapter=0, adapt_type='avg'):
        super().__init__()
        if num_depth_adapter > 0:
            if adapt_type == 'avg':
                layers = [nn.AvgPool2d(2, 2, 0) for _ in range(num_depth_adapter)]
            elif adapt_type == 'max':
                layers = [nn.MaxPool2d(2, 2, 0) for _ in range(num_depth_adapter)]
            self.depth_adapter =  nn.Sequential(*layers)
        else:
            self.depth_adapter = nn.Identity()
        self.bev_ft_align = nn.Conv2d(embed_dim, out_dim, kernel_size=1, padding=0)

    def forward(self, bev_features, frustum_features,
        frustum_bev_coordinates, last_step_depth_features,
        bev_coordinates=None,
        return_shape='chw', with_softmax=False,
        timesteps=None):
        b, t, bev_h, bev_w, _ = bev_features.shape
        num_d, num_h, num_w = frustum_bev_coordinates.shape[3: 6]

        # b,t,v,d,h,w,c
        last_step_depth_features = rearrange(last_step_depth_features, "b t v d h w -> (b t v) d h w")
        last_step_depth_features = self.depth_adapter(last_step_depth_features)
        last_step_depth_features = rearrange(last_step_depth_features, "(b t v) d h w -> b t v d h w", b=b, t=t)

        if not with_softmax:
            last_step_depth_features = last_step_depth_features.softmax(-3)

        # b,t,v,d,h,w,2
        frustum_bev_mask = (frustum_bev_coordinates.abs().amax(dim=-1) < 1.0)* \
            last_step_depth_features.to(dtype=frustum_bev_coordinates.dtype)

        bev_features = rearrange(bev_features, "b t h w c -> (b t) c h w").clone()
        frustum_bev_coordinates = rearrange(frustum_bev_coordinates, "b t v d h w o -> (b t) (d h) (v w) o")
        frustum_bev_mask = rearrange(frustum_bev_mask, "b t v d h w -> (b t) (d h) (v w)")

        # sampled LSS feature shape [B * T, C, D * H, V * W]                
        frustum_bev_features = torch.nn.functional.grid_sample(bev_features, frustum_bev_coordinates, padding_mode="border", align_corners=False)
        frustum_bev_features *= frustum_bev_mask.unsqueeze(1)
        frustum_bev_features = self.bev_ft_align(frustum_bev_features)

        if return_shape == 'chw':
            frustum_bev_features = rearrange(frustum_bev_features, "(b t) c (d h) (v w) -> b t v c h w d", b=b, h=num_h, w=num_w)
        else:
            frustum_bev_features = rearrange(frustum_bev_features,
                "(b t) c (d h) (v w) -> b t v h w c d",
                b=b, h=num_h, w=num_w)
        frustum_bev_features = frustum_bev_features.sum(-1)
        return frustum_bev_features

class PositionBasedCrossAtt(nn.Module):
    def __init__(self, embed_dim=512, out_dim=512, num_depth_adapter=0, num_d=128, adapt_type='conv',
        from_2d_to_3d=False):
        """
        Use position information for 2d-3d interaction, which is set by from_2d_to_3d
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.att = SimpleGPT2Attention(embed_dim, num_attention_heads=4)
        self.bev_ft_align = nn.Linear(embed_dim, out_dim)
        if num_depth_adapter > 0:
            if adapt_type == 'conv':
                layers = [nn.Conv2d(num_d, num_d, kernel_size=2, stride=2, padding=0) for _ in range(num_depth_adapter)]
            elif adapt_type == 'pool':
                layers = [nn.AvgPool2d(2, 2, 0) for _ in range(num_depth_adapter)]
            else:
                raise NotImplementedError
            self.depth_adapter =  nn.Sequential(*layers)
        else:
            self.depth_adapter = nn.Identity()
        self.from_2d_to_3d = from_2d_to_3d

    def pos_2d_to_embed(self, pos):
        assert pos.shape[-1] == 2
        pos_x = get_1d_sincos_pos_embed_from_grid(self.embed_dim//2, pos[..., 0])
        pos_y = get_1d_sincos_pos_embed_from_grid(self.embed_dim//2, pos[..., 1])
        return torch.cat([pos_x, pos_y], dim=-1)

    def forward(self, bev_features, frustum_features,
        frustum_bev_coordinates, last_step_depth_features,
        bev_coordinates=None,
        return_shape='chw', with_softmax=False,
        timesteps=None):
        b, t, bev_h, bev_w, _ = bev_features.shape
        num_d, num_h, num_w = frustum_bev_coordinates.shape[3: 6]
        frustum_bev_coordinates = (frustum_bev_coordinates + 1) / 2         # [-1, 1] -> [0, 1]
        frustum_bev_coordinates = (frustum_bev_coordinates*frustum_bev_coordinates.new_tensor([bev_w, bev_h])).floor()           # xy
        if bev_coordinates is None:
            grid_y, grid_x = torch.meshgrid(torch.arange(bev_h), torch.arange(bev_w), indexing='ij')
            bev_coordinates = torch.stack([grid_x, grid_y], dim=-1)         # h,w,2
            bev_coordinates = self.pos_2d_to_embed(bev_coordinates)
            bev_coordinates = rearrange(bev_coordinates, "(h w) c -> h w c", h=bev_h)
            bev_coordinates = repeat(bev_coordinates, "h w c -> b t h w c", b=b, t=t)
            bev_coordinates = bev_coordinates.to(frustum_bev_coordinates.device)

        # b,t,v,d,h,w,c
        last_step_depth_features = rearrange(last_step_depth_features, "b t v d h w -> (b t v) d h w")
        last_step_depth_features = self.depth_adapter(last_step_depth_features)
        last_step_depth_features = rearrange(last_step_depth_features, "(b t v) d h w -> b t v d h w",b=b, t=t)

        pos_frustum = self.pos_2d_to_embed(frustum_bev_coordinates)
        pos_frustum = pos_frustum.reshape(list(last_step_depth_features.shape) + [self.embed_dim])
        depth_prob = last_step_depth_features.softmax(dim=3)
        pos_frustum = (pos_frustum*depth_prob[..., None]).sum(3)            # b,t,v,h,w,c

        # Align dim
        pos_frustum = rearrange(pos_frustum, "b t v h w c -> (b t) (v h w) c")
        bev_coordinates = rearrange(bev_coordinates, "b t h w c -> (b t) (h w) c")

        if self.from_2d_to_3d:          # 2d to 3d
            frustum_features = rearrange(frustum_features, "b t v c h w -> (b t) (v h w) c")
            rt = self.att(bev_coordinates, frustum_features + pos_frustum, frustum_features)[0]
            rt = self.bev_ft_align(rt)
        else:           # 3d to 2d
            bev_features = rearrange(bev_features, "b t h w c -> (b t) (h w) c")
            # TODO: why should clone it
            rt = self.att(pos_frustum, bev_coordinates + bev_features, bev_features.clone())[0]
            rt = self.bev_ft_align(rt)
            if return_shape == 'chw':
                rt = rearrange(rt, "(b t) (v h w) c -> b t v c h w", b=b, h=num_h, w=num_w)
            else:
                rt = rearrange(rt, "(b t) (v h w) c -> b t v h w c", b=b, h=num_h, w=num_w)
        return rt
        # LSS -> b,c,h,w
        # W Pool -> b,c,w


class PositionBasedCrossAttWithT(nn.Module):
    def __init__(self, embed_dim=512, out_dim=512, num_depth_adapter=0, num_d=128, adapt_type='conv',
        from_2d_to_3d=False):
        """
        Use position information for 2d-3d interaction, which is set by from_2d_to_3d
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.att = SimpleGPT2Attention(embed_dim, num_attention_heads=4)
        self.bev_ft_align = nn.Linear(embed_dim, out_dim)
        self.time_proj = diffusers.models.embeddings.Timesteps(embed_dim, True, 0)
        if num_depth_adapter > 0:
            if adapt_type == 'conv':
                layers = [nn.Conv2d(num_d, num_d, kernel_size=2, stride=2, padding=0) for _ in range(num_depth_adapter)]
            elif adapt_type == 'pool':
                layers = [nn.AvgPool2d(2, 2, 0) for _ in range(num_depth_adapter)]
            else:
                raise NotImplementedError
            self.depth_adapter =  nn.Sequential(*layers)
        else:
            self.depth_adapter = nn.Identity()
        self.from_2d_to_3d = from_2d_to_3d

    def pos_2d_to_embed(self, pos):
        assert pos.shape[-1] == 2
        pos_x = get_1d_sincos_pos_embed_from_grid(self.embed_dim//2, pos[..., 0])
        pos_y = get_1d_sincos_pos_embed_from_grid(self.embed_dim//2, pos[..., 1])
        return torch.cat([pos_x, pos_y], dim=-1)

    def forward(self, bev_features, frustum_features,
        frustum_bev_coordinates, last_step_depth_features,
        bev_coordinates=None,
        return_shape='chw', with_softmax=False,
        timesteps=None):
        b, t, bev_h, bev_w, _ = bev_features.shape
        num_d, num_h, num_w = frustum_bev_coordinates.shape[3: 6]
        frustum_bev_coordinates = (frustum_bev_coordinates + 1) / 2         # [-1, 1] -> [0, 1]
        frustum_bev_coordinates = (frustum_bev_coordinates*frustum_bev_coordinates.new_tensor([bev_w, bev_h])).floor()           # xy
        if bev_coordinates is None:
            grid_y, grid_x = torch.meshgrid(torch.arange(bev_h), torch.arange(bev_w), indexing='ij')
            bev_coordinates = torch.stack([grid_x, grid_y], dim=-1)         # h,w,2
            bev_coordinates = self.pos_2d_to_embed(bev_coordinates)
            bev_coordinates = rearrange(bev_coordinates, "(h w) c -> h w c", h=bev_h)
            bev_coordinates = repeat(bev_coordinates, "h w c -> b t h w c", b=b, t=t)
            bev_coordinates = bev_coordinates.to(frustum_bev_coordinates.device)

        # b,t,v,d,h,w,c
        last_step_depth_features = rearrange(last_step_depth_features, "b t v d h w -> (b t v) d h w")
        last_step_depth_features = self.depth_adapter(last_step_depth_features)
        last_step_depth_features = rearrange(last_step_depth_features, "(b t v) d h w -> b t v d h w",b=b, t=t)

        pos_frustum = self.pos_2d_to_embed(frustum_bev_coordinates)
        pos_frustum = pos_frustum.reshape(list(last_step_depth_features.shape) + [self.embed_dim])
        if not with_softmax:
            depth_prob = last_step_depth_features.softmax(dim=3)
        else:
            raise NotImplementedError
        pos_frustum = (pos_frustum*depth_prob[..., None]).sum(3)            # b,t,v,h,w,c

        # Align dim
        pos_frustum = rearrange(pos_frustum, "b t v h w c -> (b t) (v h w) c")
        bev_coordinates = rearrange(bev_coordinates, "b t h w c -> (b t) (h w) c")

        t_emb = self.time_proj(timesteps)
        t_emb = repeat(t_emb, "b c -> b t k c", t=t, k=1).flatten(0, 1)

        if self.from_2d_to_3d:          # 2d to 3d
            frustum_features = rearrange(frustum_features, "b t v c h w -> (b t) (v h w) c")
            rt = self.att(bev_coordinates, frustum_features + pos_frustum, frustum_features)[0]
            rt = self.bev_ft_align(rt + t_emb)
        else:           # 3d to 2d
            bev_features = rearrange(bev_features, "b t h w c -> (b t) (h w) c")
            # TODO: why should clone it
            rt = self.att(pos_frustum, bev_coordinates + bev_features, bev_features.clone())[0]
            rt = self.bev_ft_align(rt + t_emb)
            if return_shape == 'chw':
                rt = rearrange(rt, "(b t) (v h w) c -> b t v c h w", b=b, h=num_h, w=num_w)
            else:
                rt = rearrange(rt, "(b t) (v h w) c -> b t v h w c", b=b, h=num_h, w=num_w)
        return rt
        # LSS -> b,c,h,w
        # W Pool -> b,c,w


class PositionBasedCrossAttWithTMask(nn.Module):
    def __init__(self, embed_dim=512, out_dim=512, num_depth_adapter=0, num_d=128, adapt_type='conv',
        from_2d_to_3d=False, bev_valid_ratio=None):
        """
        Use position information for 2d-3d interaction, which is set by from_2d_to_3d
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.att = SimpleGPT2Attention(embed_dim, num_attention_heads=4)
        self.bev_ft_align = nn.Linear(embed_dim, out_dim)
        self.time_proj = diffusers.models.embeddings.Timesteps(embed_dim, True, 0)
        if num_depth_adapter > 0:
            if adapt_type == 'conv':
                layers = [nn.Conv2d(num_d, num_d, kernel_size=2, stride=2, padding=0) for _ in range(num_depth_adapter)]
            elif adapt_type == 'pool':
                layers = [nn.AvgPool2d(2, 2, 0) for _ in range(num_depth_adapter)]
            else:
                raise NotImplementedError
            self.depth_adapter =  nn.Sequential(*layers)
        else:
            self.depth_adapter = nn.Identity()
        self.from_2d_to_3d = from_2d_to_3d
        self.bev_valid_ratio = bev_valid_ratio

    def pos_2d_to_embed(self, pos):
        assert pos.shape[-1] == 2
        pos_x = get_1d_sincos_pos_embed_from_grid(self.embed_dim//2, pos[..., 0])
        pos_y = get_1d_sincos_pos_embed_from_grid(self.embed_dim//2, pos[..., 1])
        return torch.cat([pos_x, pos_y], dim=-1)

    def forward(self, bev_features, frustum_features,
        frustum_bev_coordinates, last_step_depth_features,
        bev_coordinates=None,
        return_shape='chw', with_softmax=False,
        timesteps=None):
        b, t, bev_h, bev_w, _ = bev_features.shape
        bev_h_v, bev_w_v = bev_h*self.bev_valid_ratio, bev_w*self.bev_valid_ratio
        num_d, num_h, num_w = frustum_bev_coordinates.shape[3: 6]
        frustum_bev_coordinates = (frustum_bev_coordinates*frustum_bev_coordinates.new_tensor([bev_w, bev_h])).floor()           # xy
        if bev_coordinates is None:
            grid_y, grid_x = torch.meshgrid(torch.arange(bev_h), torch.arange(bev_w), indexing='ij')
            grid_y = grid_y*2 - bev_h
            grid_x = grid_x*2 - bev_w
            bev_coordinates = torch.stack([grid_x, grid_y], dim=-1)         # h,w,2
            bev_coordinates_mask = (bev_coordinates[..., 0].abs() < bev_w_v) &  (bev_coordinates[..., 1].abs() < bev_h_v)
            bev_coordinates_mask = repeat(bev_coordinates_mask, "h w -> b t h w c", b=b, t=t, c=1).to(frustum_bev_coordinates.device)

            bev_coordinates = self.pos_2d_to_embed(bev_coordinates)
            bev_coordinates = rearrange(bev_coordinates, "(h w) c -> h w c", h=bev_h)
            bev_coordinates = repeat(bev_coordinates, "h w c -> b t h w c", b=b, t=t)
            bev_coordinates = bev_coordinates.to(frustum_bev_coordinates.device)

        # b,t,v,d,h,w,c
        last_step_depth_features = rearrange(last_step_depth_features, "b t v d h w -> (b t v) d h w")
        last_step_depth_features = self.depth_adapter(last_step_depth_features)
        last_step_depth_features = rearrange(last_step_depth_features, "(b t v) d h w -> b t v d h w",b=b, t=t)

        pos_frustum = self.pos_2d_to_embed(frustum_bev_coordinates)
        pos_frustum = pos_frustum.reshape(list(last_step_depth_features.shape) + [self.embed_dim])
        if not with_softmax:
            depth_prob = last_step_depth_features.softmax(dim=3)
        else:
            raise NotImplementedError
        pos_frustum = (pos_frustum*depth_prob[..., None]).sum(3)            # b,t,v,h,w,c

        # Align dim
        pos_frustum = rearrange(pos_frustum, "b t v h w c -> (b t) (v h w) c")
        bev_coordinates = rearrange(bev_coordinates, "b t h w c -> (b t) (h w) c")
        bev_coordinates_mask = rearrange(bev_coordinates_mask, "b t h w c -> (b t) (h w) c")

        t_emb = self.time_proj(timesteps)
        t_emb = repeat(t_emb, "b c -> b t k c", t=t, k=1).flatten(0, 1)

        if self.from_2d_to_3d:          # 2d to 3d
            frustum_features = rearrange(frustum_features, "b t v c h w -> (b t) (v h w) c")
            rt = self.att(bev_coordinates, frustum_features + pos_frustum, frustum_features)[0]
            rt = rt*bev_coordinates_mask
            rt = self.bev_ft_align(rt + t_emb)
        else:           # 3d to 2d
            bev_features = rearrange(bev_features, "b t h w c -> (b t) (h w) c")
            # TODO: why should clone it
            rt = self.att(pos_frustum, bev_coordinates + bev_features, bev_features.clone())[0]
            rt = self.bev_ft_align(rt + t_emb)
            if return_shape == 'chw':
                rt = rearrange(rt, "(b t) (v h w) c -> b t v c h w", b=b, h=num_h, w=num_w)
            else:
                rt = rearrange(rt, "(b t) (v h w) c -> b t v h w c", b=b, h=num_h, w=num_w)
        return rt
        # LSS -> b,c,h,w
        # W Pool -> b,c,w