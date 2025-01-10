import dwm.models.base_vq_models.dvgo_utils
import numpy as np
import scipy.cluster.vq
import timm.models.swin_transformer
import timm.layers
import torch
import torch.distributed
import torch.distributed.nn.functional
import torch.nn.functional as F

from timm.models.vision_transformer import Block, Attention
from typing import Optional


class VectorQuantizer(torch.nn.Module):
    def __init__(
        self, n_e, e_dim, beta, cosine_similarity=False, dead_limit=256
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.cosine_similarity = cosine_similarity
        self.dead_limit = dead_limit

        self.embedding = torch.nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.register_buffer("global_iter", torch.zeros(1))
        self.register_buffer("num_iter", torch.zeros(1))
        self.register_buffer("data_initialized", torch.zeros(1))
        self.register_buffer("reservoir", torch.zeros(self.n_e * 10, e_dim))

    def forward(self, z, code_age=None, code_usage=None):
        assert z.shape[-1] == self.e_dim
        z_flattened = z.reshape(-1, self.e_dim)

        if self.cosine_similarity:
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)

        if code_age is not None and code_usage is not None:
            self.update_reservoir(z_flattened, code_age, code_usage)

        if self.cosine_similarity:
            min_encoding_indices = torch\
                .matmul(z_flattened, F.normalize(self.embedding.weight, p=2, dim=-1).T)\
                .max(dim=-1)[1]
        else:
            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

            z_dist = torch.cdist(z_flattened, self.embedding.weight)
            # z_dist = torch.cdist(z_flattened, self.embedding.weight, compute_mode='donot_use_mm_for_euclid_dist')
            min_encoding_indices = torch.argmin(z_dist, dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        if self.cosine_similarity:
            z_q = F.normalize(z_q, p=2, dim=-1)
            z_norm = F.normalize(z, p=2, dim=-1)
            loss = (
                self.beta *
                torch.mean(1 - (z_q.detach() * z_norm).sum(dim=-1)),
                torch.mean(1 - (z_q * z_norm.detach()).sum(dim=-1)),
            )
        else:
            loss = (self.beta * torch.mean((z_q.detach() - z) ** 2),
                    torch.mean((z_q - z.detach()) ** 2))

        # preserve gradients
        z_q = z + (z_q - z).detach()

        if code_age is not None and code_usage is not None:
            code_idx = min_encoding_indices
            if torch.distributed.is_initialized():
                code_idx = torch.cat(
                    torch.distributed.nn.functional.all_gather(code_idx))

            code_age += 1
            code_age[code_idx] = 0
            code_usage.index_add_(0, code_idx, torch.ones_like(
                code_idx, dtype=code_usage.dtype))
        min_encoding_indices = min_encoding_indices.view(z.shape[:-1])
        return z_q, loss, min_encoding_indices

    def update_reservoir(self, z, code_age, code_usage):
        if not (self.embedding.weight.requires_grad and self.training):
            return

        assert z.shape[-1] == self.e_dim
        z_flattened = z.reshape(-1, self.e_dim)

        rp = torch.randperm(z_flattened.size(0))
        if self.data_initialized.item() == 0:
            num_sample: int = self.reservoir.shape[0]  # pylint: disable=access-member-before-definition
            self.reservoir: torch.Tensor = z_flattened[rp[:num_sample]].data
        else:
            num_sample: int = self.reservoir.shape[0] // 100  # pylint: disable=access-member-before-definition
            self.reservoir: torch.Tensor = torch.cat(
                [self.reservoir[num_sample:], z_flattened[rp[:num_sample]].data])

        self.num_iter += 1
        self.global_iter += 1

        # print(self.num_iter.item(), (code_age >= self.dead_limit).sum(), flush=True)

        if self.data_initialized.item() == 0 or ((code_age >= self.dead_limit).sum() / self.n_e) > 0.03:
            if not torch.distributed.is_initialized() or \
                    torch.distributed.get_rank() == 0:
                print('update codebook!')

            self.update_codebook(code_age, code_usage)
            if self.data_initialized.item() == 0:
                self.data_initialized.fill_(1)

            self.num_iter.fill_(0)

    def update_codebook(self, code_age, code_usage):
        if not (self.embedding.weight.requires_grad and self.training):
            return

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            live_code = self.embedding.weight[
                code_age < self.dead_limit].data
            live_code_num = live_code.shape[0]
            if self.cosine_similarity:
                live_code = F.normalize(live_code, p=2, dim=-1)

            all_z = torch.cat([self.reservoir, live_code])
            rp = torch.randperm(all_z.shape[0])
            all_z = all_z[rp]

            init = torch.cat([
                live_code,
                self.reservoir[torch.randperm(self.reservoir.shape[0])[
                    :(self.n_e - live_code_num)]]
            ])
        else:
            all_z = self.reservoir
            rp = torch.randperm(all_z.shape[0])
            all_z = all_z[rp]

            init = self.reservoir[torch.randperm(
                self.reservoir.shape[0])[:self.n_e]]

        if (torch.distributed.is_initialized() and
                torch.distributed.get_rank() == 0) or \
                not torch.distributed.is_initialized():
            init = init.data.cpu().numpy()
            # print(
            #     "running kmeans!!", self.n_e, live_code_num, self.data_initialized.item()
            # )  # data driven initialization for the embeddings
            centroid, assignment = scipy.cluster.vq.kmeans2(
                all_z.float().cpu().numpy(),
                init,
                minit="matrix",
                iter=100,
            )
            z_dist = (
                all_z - torch.from_numpy(centroid[assignment]).to(all_z.device, dtype=all_z.dtype)).norm(dim=1).sum().item()
            self.embedding.weight.data = torch.from_numpy(
                centroid).to(self.embedding.weight.device)
            print("finish kmeans", z_dist)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                print('broadcast!', flush=True)

            torch.distributed.nn.functional.broadcast(
                self.embedding.weight, src=0)

        code_age.fill_(0)
        code_usage.fill_(0)
        self.data_initialized.fill_(1)

    def get_codebook_entry(self, indices, shape=None):
        # shape specifying (batch, height, width, channel)

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        if self.cosine_similarity:
            z_q = F.normalize(z_q, p=2, dim=-1)
        return z_q


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros((1, embed_dim), np.float32), pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(torch.nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        upcast (bool, optional): If True, upcast the input to float32. Default: False 
        normalized_attn (bool, optional): If True, use a special softmax function to normalize the attention and add epsilon to avoid NaN problem. Default: False
    """

    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.,
        proj_drop=0.,
        upcast = False,
        normalized_attn = False,
    ):
        super().__init__()
        self.dim = dim
        self.upcast = upcast
        self.normalized_attn = normalized_attn
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = torch.nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

        timm.layers.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = torch.nn.Softmax(dim=-1) if not normalized_attn else NormSoftmax(dim=-1)

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x, mask = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        if self.upcast:
            x = x.float()
            mask = mask.float() if mask is not None else mask

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NormSoftmax(torch.nn.Module):
    """
    Args:
        dim (int): The channel that will be used to operate softmax.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize the attn by the max value along the feature dim
        x_max = x.max(dim = -1, keepdim = True)[0]
        x_max = torch.nan_to_num(x_max, neginf=0)
        x = x - x_max
        x = x.exp()
        x = x / (x.sum(dim = -1, keepdim = True) + 1e-7)
        return x

class SwinTransformerBlock(torch.nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        upcast (bool, optional): control the upcast in attention. For details, please see WindowAttention module. Default: False 
        normalized_attn (bool, optional): control the normalization of attention. For details, please see WindowAttention module. Default: False
    """

    def __init__(
        self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
        mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
        act_layer=torch.nn.GELU, norm_layer=torch.nn.LayerNorm,
        upcast = False,
        normalized_attn = False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.upcast = upcast
        self.normalized_attn = normalized_attn
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=timm.layers.to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            upcast = upcast,
            normalized_attn = normalized_attn,
        )

        self.drop_path = timm.layers.DropPath(
            drop_path) if drop_path > 0. else torch.nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = timm.layers.Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class CrossAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: torch.nn.Module = torch.nn.LayerNorm,
        cross_attend: bool=True
    ):
        super().__init__(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer
        )
        self.cross_attend = cross_attend
        if cross_attend:
            self.q = torch.nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = torch.nn.Linear(dim, dim * 2, bias=qkv_bias)
            del self.qkv
    
    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        # NOTE default y has same dim with x
        B, N, C = x.shape
        if y is None:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            # print(f'x shape: {x.shape}') # (bsz, 6400, 512)
            # print(f'y shape: {y.shape}') # (bs, 144*2+77=365, 512)
            M = y.shape[1]
            kv = self.kv(y).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# cross attention block
class TransformerBlock(Block):
    def __init__(self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        act_layer: torch.nn.Module = torch.nn.GELU,
        norm_layer: torch.nn.Module = torch.nn.LayerNorm,
        mlp_layer: torch.nn.Module = timm.layers.Mlp
    ):

        super().__init__(
            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, proj_drop=proj_drop,
            attn_drop=attn_drop, init_values=init_values, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer,
            mlp_layer=mlp_layer
        )
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            cross_attend=True
        )

    
    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        if y is not None:
            x = self.norm1(x)
            x = self.attn(x, y)
            x = x + self.drop_path1(self.ls1(x))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x
        else:
            return super.forward(x)


class BasicLayer(torch.nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, dim, input_resolution, depth, num_heads, window_size,
        mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
        norm_layer=torch.nn.LayerNorm, downsample=None, use_checkpoint=False,
        cross_attend = False, upcast = False,
        normalized_attn = False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.upcast = upcast
        self.normalized_attn = normalized_attn

        # build blocks
        self.blocks = torch.nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                norm_layer=norm_layer,
                upcast = upcast,
                normalized_attn = normalized_attn,
            )
            for i in range(depth)
        ])

        self.cross_attend = cross_attend

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, y = None):
        for blk in self.blocks:
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, use_reentrant=False)
            else:
                x = blk(x)
            # (bsz, 6400, 512)
            # point: 640x640 -> 80x80 (real)
            # condition: 160x160 -> 80x80
            # 1024x12x12, 512x24x24 <- 384
            # stage-0(/4), stage-1(/8), stage-2(/16), stage-3(/32)
            #  128,         256,         512,            1024
            # 640x640 -> 40x40x512 -> 80x80x512
            # 640x640 -> 2x80x80x256 -> 80x80x512
            
            if self.cross_attend:
                if y is not None:
                    x = x + y # add condition

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchMerging(torch.nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, output_dim=None, norm_layer=torch.nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim

        if output_dim is None:
            output_dim = 2 * dim

        self.reduction = torch.nn.Linear(4 * dim, output_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class VQEncoder(torch.nn.Module):
    def __init__(
        self, img_size, patch_size=8, in_chans=40, embed_dim=512, num_heads=16,
        depth=12, codebook_dim=1024, use_checkpoint=False,
        upcast = False,
        normalized_attn = False
    ):
        super().__init__()

        norm_layer = torch.nn.LayerNorm
        self.patch_embed = timm.layers.PatchEmbed(
            img_size, patch_size // 2, in_chans, embed_dim // 2,
            norm_layer=norm_layer)
        num_patches = self.patch_embed.num_patches

        self.h = img_size // patch_size * 2
        self.w = img_size // patch_size * 2
        # This flag are used to control the BasicLayer attn
        self.upcast = upcast
        self.normalized_attn = normalized_attn

        pos_embed = get_2d_sincos_pos_embed(
            embed_dim // 2, (self.h, self.w), cls_token=False)
        self.register_buffer(
            "pos_embed", torch.from_numpy(pos_embed).unsqueeze(0))

        self.blocks = torch.nn.Sequential(
            BasicLayer(
                embed_dim // 2,
                (img_size // patch_size * 2, img_size // patch_size * 2),
                4,
                num_heads=num_heads,
                window_size=8,
                downsample=PatchMerging,
                use_checkpoint=use_checkpoint,
                upcast = upcast,
                normalized_attn = normalized_attn
            ),
            BasicLayer(
                embed_dim,
                (img_size // patch_size, img_size // patch_size),
                depth - 4,
                num_heads=num_heads,
                window_size=8,
                downsample=None,
                use_checkpoint=use_checkpoint,
                upcast = upcast,
                normalized_attn = normalized_attn
            ),
        )

        self.blocks = torch.nn.Sequential(*self.blocks)

        self.norm = torch.nn.Sequential(norm_layer(embed_dim), torch.nn.GELU())
        self.pre_quant = torch.nn.Linear(embed_dim, codebook_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # nn.init.constant_(self.pre_quant.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pre_quant(x)

        return x


class VQDecoder(torch.nn.Module):

    def __init__(
        self, img_size, num_patches, patch_size=8, in_chans=40, embed_dim=512,
        num_heads=16, depth=12, codebook_dim=1024, bias_init=-3,
        upsample_style="conv_transpose", use_checkpoint=False,
        # New config to control whether the new type of attention is used
        upcast = False,
        normalized_attn = False
    ):
        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        norm_layer = torch.nn.LayerNorm
        self.patch_size = patch_size // 2
        self.in_chans = in_chans
        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size
        # These flags are used to control the BasicLayer attn
        self.upcast = upcast,
        self.normalized_attn = normalized_attn

        self.num_patches = num_patches
        self.decoder_embed = torch.nn.Linear(
            codebook_dim, embed_dim, bias=True)

        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, (self.h, self.w), cls_token=False)
        self.register_buffer(
            "pos_embed", torch.from_numpy(pos_embed).unsqueeze(0))

        self.blocks = BasicLayer(
            embed_dim,
            (img_size[0] // patch_size, img_size[1] // patch_size),
            depth=depth - 2,
            num_heads=num_heads,
            window_size=8,
            use_checkpoint=use_checkpoint,
            upcast = upcast,
            normalized_attn = normalized_attn
        )

        if upsample_style == "conv_transpose":
            self.upsample = torch.nn.ConvTranspose2d(
                embed_dim, embed_dim // 2, 2, stride=2)
        else:
            self.upsample = torch.nn.Sequential(
                torch.nn.PixelShuffle(2),
                torch.nn.Conv2d(embed_dim // 4, embed_dim // 2, 1))

        self.density_block = BasicLayer(
            embed_dim // 2,
            (img_size[0] // patch_size * 2, img_size[1] // patch_size * 2),
            depth=2,
            num_heads=num_heads,
            window_size=8,
            use_checkpoint=use_checkpoint,
            normalized_attn = normalized_attn
        )
        self.density_norm = torch.nn.Sequential(
            norm_layer(embed_dim // 2), torch.nn.GELU())
        self.density_pred = torch.nn.Linear(
            embed_dim // 2, 4 * in_chans * 16, bias=True)

        self.voxel_block = BasicLayer(
            embed_dim // 2,
            (img_size[0] // patch_size * 2, img_size[1] // patch_size * 2),
            depth=2,
            num_heads=num_heads,
            window_size=8,
            use_checkpoint=use_checkpoint,
            normalized_attn = normalized_attn,
        )
        self.voxel_norm = torch.nn.Sequential(
            norm_layer(embed_dim // 2), torch.nn.GELU())
        self.voxel_pred = torch.nn.Linear(
            embed_dim // 2, (patch_size // 2)**2 * in_chans, bias=True)

        self.apply(self._init_weights)
        torch.nn.init.constant_(self.voxel_pred.bias, bias_init)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

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

    def forward(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        x = self.blocks(x)

        N, L, C = x.shape

        hw = int(L ** 0.5)
        x = x.view(N, hw, hw, C).permute(0, 3, 1, 2)
        x = self.upsample(x).permute(0, 2, 3, 1).reshape(N, -1, C // 2)

        # predictor projection
        density = self.density_block(x)
        density = self.density_norm(density)
        density = self.density_pred(density)
        density = self.unpatchify(density, p=self.patch_size // 2)\
            .unflatten(1, (16, -1))

        voxel = self.voxel_block(x)
        voxel = self.voxel_norm(voxel)
        voxel = self.voxel_pred(voxel)
        voxel = self.unpatchify(voxel)

        return density, voxel


class VQPointCloud(torch.nn.Module):

    @staticmethod
    def soft_l1(pred_depth, gt_depth):
        l1_loss = F.l1_loss(
            pred_depth, gt_depth, reduction='none').flatten()
        top_l1_loss = torch.topk(l1_loss, k=int(
            l1_loss.numel() * 0.95), largest=False)[0].mean()
        return top_l1_loss

    def __init__(
        self, voxelizer, vector_quantizer, lidar_encoder: VQEncoder,
        lidar_decoder: VQDecoder, bias_init: float = -5
    ):
        super().__init__()

        self.voxelizer = voxelizer
        self.vector_quantizer = vector_quantizer
        self.lidar_encoder = lidar_encoder
        self.lidar_decoder = lidar_decoder

        self.grid_size = {
            "min": [voxelizer.x_min, voxelizer.y_min, voxelizer.z_min],
            "max": [voxelizer.x_max, voxelizer.y_max, voxelizer.z_max],
            "interval": [voxelizer.step, voxelizer.step, voxelizer.z_step]
        }
        self.density_mlp = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32, 1)
        )
        torch.nn.init.constant_(self.density_mlp[-1].bias, bias_init)

        self.register_buffer(
            "code_age", torch.zeros(self.vector_quantizer.n_e) * 10000)
        self.register_buffer(
            "code_usage", torch.zeros(self.vector_quantizer.n_e))

    def ray_render_dvgo(
        self, density, points, coarse_mask, offsets=None,
        return_alpha_last=False
    ):
        depth_loss_list = []
        sdf_loss_list = []
        sampled_points = []
        alphainv_lasts = []
        batch_lists = zip(density, points, coarse_mask, offsets)
        for density_i, points_i, mask, offset in batch_lists:
            sampled_points_i = []
            temporal_lists = zip(density_i, points_i, mask)
            for density_j, points_j, mask_j in temporal_lists:
                if offsets is not None:
                    cur_offsets = offset.unsqueeze(0)
                    rays_o = cur_offsets.repeat_interleave(
                        points_j.shape[0], 0)
                    rays_d = points_j - cur_offsets
                else:
                    rays_o = torch.zeros(
                        points_j.shape, device=points_j.device)
                    rays_d = points_j

                gt_depth = torch.norm(rays_d, dim=-1, keepdim=True)
                pred_depth, loss_sdf_i, alphainv_last = dwm.models\
                    .base_vq_models.dvgo_utils.dvgo_render(
                        self.density_mlp, mask_j, rays_o, rays_d,

                        # TODO: simplify permutation
                        torch.einsum(
                            'dzyx->dxyz', density_j.float()).unsqueeze(0),

                        self.grid_size["min"], self.grid_size["max"],
                        stepsize=0.05, grid_size=self.grid_size)

                depth_loss_list.append(
                    VQPointCloud.soft_l1(pred_depth, gt_depth.squeeze(-1)))
                sdf_loss_list.append(loss_sdf_i)
                sampled_points_i.append(
                    rays_o + pred_depth.unsqueeze(-1) / gt_depth * rays_d)
                alphainv_lasts.append(alphainv_last)

            sampled_points.append(sampled_points_i)

        if return_alpha_last:
            return torch.mean(torch.stack(depth_loss_list)), \
                torch.mean(torch.stack(sdf_loss_list)), sampled_points, \
                alphainv_lasts
        else:
            return torch.mean(torch.stack(depth_loss_list)), \
                torch.mean(torch.stack(sdf_loss_list)), sampled_points

    def encode_for_quantized_feature(self, points):
        voxels = self.voxelizer(points)
        features = self.lidar_encoder(voxels.flatten(0, 1))
        quantized_features, _, _ = self.vector_quantizer(features)
        return quantized_features

    def forward(self, points, offsets=None):
        voxels = self.voxelizer(points)
        features = self.lidar_encoder(voxels.flatten(0, 1))
        quantized_features, emb_loss, _ = self.vector_quantizer(
            features, self.code_age, self.code_usage)
        density, lidar_voxel = self.lidar_decoder(quantized_features)
        result = {
            "voxels": voxels,
            "lidar_voxel": lidar_voxel.unflatten(0, voxels.shape[:2]),
            "emb_loss": emb_loss
        }

        if self.density_mlp is not None:
            coarse_mask = F.max_pool3d(voxels, (4, 8, 8))
            depth_loss, sdf_loss, sampled_points = self.ray_render_dvgo(
                density.unflatten(0, voxels.shape[:2]),
                points, coarse_mask, offsets=offsets)

            result["depth_loss"] = depth_loss
            result["sdf_loss"] = sdf_loss
            result["sampled_points"] = sampled_points

        return result
