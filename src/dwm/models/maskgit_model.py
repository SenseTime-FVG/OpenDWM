import torch
import torch.distributed
import torch.distributed.nn.functional
from torch import nn
from dwm.models.vq_point_cloud import (
    get_2d_sincos_pos_embed,
    SwinTransformerBlock
)
from einops import rearrange

from dwm.models.maskgit_base import TemporalTransformerBlock


from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding


class MaskgitTransformer(torch.nn.Module):

    def __init__(
        self, dim, input_resolution, depth, num_heads, window_size,
        mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
        norm_layer=torch.nn.LayerNorm, downsample=None, use_checkpoint=False,
        upcast=False, enable_temporal=False,
        normalized_attn=False,
    ):
        """ 
        A basic MaskGIT layer for one stage. It contains spatial and temporal blocks. 
        Swin Transformer is adopted as the spatial block.
        The temporal blocks are modified from the JointAttention of diffusers.

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
            use_checkpoint (bool): Whether to use gradient checkpointing to save memory. Default: False.
            upcast (bool): Whether to upcast the dtype in the attention. Default: False.
            enable_temporal (bool): Whether to enable temporal blocks. Default: False.
            normalized_attn (bool): Whether to normalize the softmax in the attention by the max value. Default: False.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.upcast = upcast
        self.normalized_attn = normalized_attn
        self.enable_temporal = enable_temporal
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
                upcast=upcast,
                normalized_attn=normalized_attn,
            )
            for i in range(depth)
        ])

        # if enable_temporal, add temporal blocks
        if enable_temporal:
            self.spatial_blocks = self.blocks
            self.rope = RotaryEmbedding(
                dim // num_heads, freqs_for='pixel', learned_freq=True)
            self.temporal_blocks = torch.nn.ModuleList([
                TemporalTransformerBlock(
                    dim=dim,
                    num_attention_heads=num_heads,
                    attention_head_dim=dim // num_heads,
                    context_pre_only=False,
                    qk_norm=None,

                )
                for _ in range(depth)
            ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward_spatial(self, x, context=None):
        for blk in self.blocks:
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, use_reentrant=False)
            else:
                x = blk(x)

            if context is not None:
                if len(context) > 0:
                    x = x + context.pop(0).flatten(
                        -2, -1).permute(0, 2, 1)  # add condition

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def forward_temporal(self, x, context=None, batch_size=None, num_frames=None):
        x = x.unflatten(0, (batch_size, num_frames))  # (B, T, L, C)
        for spatial_blk, temporal_blk in zip(self.spatial_blocks, self.temporal_blocks):
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = rearrange(x, "b t s c -> (b t) s c")
                x = torch.utils.checkpoint.checkpoint(
                    spatial_blk, x, use_reentrant=False)
                x = rearrange(x, "(b t) s c -> (b s) t c",
                              b=batch_size, t=num_frames)
                x = torch.utils.checkpoint.checkpoint(
                    temporal_blk, x, self.rope.rotate_queries_or_keys, use_reentrant=False)
                x = rearrange(x, "(b s) t c -> b t s c",
                              b=batch_size, t=num_frames)
            else:
                x = rearrange(x, "b t s c -> (b t) s c")
                x = spatial_blk(x)
                x = rearrange(x, "(b t) s c -> (b s) t c",
                              b=batch_size, t=num_frames)
                x = temporal_blk(
                    x, rotary_emb=self.rope.rotate_queries_or_keys)
                x = rearrange(x, "(b s) t c -> b t s c",
                              b=batch_size, t=num_frames)
            if context is not None:
                if len(context) > 0:
                    cur_context = context.pop(
                        0).flatten(-2, -1).permute(0, 2, 1)
                    cur_context = rearrange(
                        cur_context, "(b t) s c -> b t s c", b=batch_size, t=num_frames)
                    x = x + cur_context  # Use plus operation to inject condition

        if self.downsample is not None:
            x = self.downsample(x)
        x = x.flatten(0, 1)
        return x

    def forward(self, x, context=None, batch_size=None, num_frames=None):
        if self.enable_temporal:
            return self.forward_temporal(x, context, batch_size=batch_size, num_frames=num_frames)
        else:
            return self.forward_spatial(x, context)


class BidirectionalTransformerWithAdapter(torch.nn.Module):
    def __init__(self,
                 n_e,
                 e_dim,
                 img_size,
                 hidden_dim=512,
                 depth=24,
                 num_heads=16,
                 use_checkpoint=False,
                 use_extra_embedding=False,
                 enable_temporal=False,
                 condition_adapter=None,
                 enable_precomputed_pos_embed=False):
        """
        Args:
            n_e: int. Number of vq codes.
            e_dim: int. Dimension of vq codes.
            img_size: tuple[int]. Image size.
            hidden_dim: int. Hidden dimension of the model.
            depth: int. Depth of the model.
            num_heads: int. Number of attention heads.
            use_checkpoint: bool. Whether to use gradient checkpointing to save memory.
            use_extra_embedding: bool. Whether to use extra embedding for the input indices.
            enable_temporal: bool. Whether to enable temporal blocks.
            condition_adapter: nn.Module. Condition adapter.
            enable_precomputed_pos_embed: bool. Whether to enable precomputed position embedding. In fsdp, it is set to False since the require_grad of all the parameters should be True.
        """
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.decoder_embed = torch.nn.Linear(e_dim, hidden_dim, bias=True)
        self.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, e_dim), requires_grad=True)
        token_size = img_size[0] * img_size[1]
        self.use_extra_embedding = use_extra_embedding

        # This is the extra embedding for the mask token. It is used to replace the embedding of vq code.
        if self.use_extra_embedding:
            self.extra_embedding = nn.Embedding(n_e, e_dim)
        # This is the precomputed position embedding for the mask token. The required_grad is set to False because it is not trainable.
        # If you use fsdp, you can disable this flag.
        if enable_precomputed_pos_embed:
            self.pos_embed = torch.nn.Parameter(torch.zeros(
                1, token_size, hidden_dim), requires_grad=False)
        else:
            self.pos_embed = None
        self.enable_temporal = enable_temporal
        self.blocks = MaskgitTransformer(
            hidden_dim,
            img_size,
            depth,
            num_heads=num_heads,
            window_size=2,
            downsample=None,
            use_checkpoint=use_checkpoint,
            enable_temporal=enable_temporal,
        )
        self.condition_adapter = condition_adapter
        self.norm = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim), torch.nn.GELU())
        self.pred = torch.nn.Linear(hidden_dim, n_e, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.pos_embed is not None:
            pos_embed = get_2d_sincos_pos_embed(
                self.hidden_dim, self.img_size, cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def forward(self,
                x,
                x_id=None,
                context=None,
                batch_size=None, num_frames=None):
        """
        Args:
            x: (B, T, L, C)
            x_id: (B, T, L)
            context: (B, H, W, C)
            batch_size: int
            num_frames: int
        """
        # embed tokens
        if self.use_extra_embedding:
            x = torch.zeros_like(x)
            x_id = x_id.to(torch.long)
            x = torch.where(
                (x_id == -1).unsqueeze(-1).expand(-1, -1, x.shape[2]),
                self.mask_token.expand_as(x),
                self.extra_embedding(x_id.masked_fill(x_id == -1, 0))
            )
        else:
            x = torch.where(
                (x_id == -1).unsqueeze(-1).expand(-1, -1, x.shape[2]),
                self.mask_token.expand_as(x),
                x
            )

        x = self.decoder_embed(x)

        # add pos embed
        if self.pos_embed is None:
            pos_embed = get_2d_sincos_pos_embed(
                self.hidden_dim, self.img_size, cls_token=False)
            pos_embed = torch.from_numpy(
                pos_embed).float().to(x.device).unsqueeze(0)
        else:
            pos_embed = self.pos_embed.unsqueeze(0)
        x = x + pos_embed

        # Apply Transformer blocks. The original shape is the transformer model.
        if self.condition_adapter is not None and context is not None:
            bs = context["context"].shape[0]
            h = w = int(context["context"].shape[1]**0.5)
            context = self.condition_adapter(
                context["context"].view(bs, h, w, -1).permute(0, 3, 1, 2))
            x = self.blocks(
                x, context, batch_size=batch_size, num_frames=num_frames)
        else:
            x = self.blocks(x, batch_size=batch_size, num_frames=num_frames)
        x = self.norm(x)

        # predictor projection
        x = self.pred(x)
        return x


# If you want to put the condition adapter in the pipeline, you can use this class.
class BidirectionalTransformer(torch.nn.Module):
    def __init__(self,
                 n_e,
                 e_dim,
                 img_size,
                 hidden_dim=512,
                 depth=24,
                 num_heads=16,
                 use_checkpoint=False,
                 cross_attend=False,
                 add_cross_proj=True,
                 use_extra_embedding=False,
                 cond_in_channels=512,):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.decoder_embed = torch.nn.Linear(e_dim, hidden_dim, bias=True)
        self.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, e_dim), requires_grad=True)
        token_size = img_size[0] * img_size[1]
        self.use_extra_embedding = use_extra_embedding
        if self.use_extra_embedding:
            self.extra_embedding = nn.Embedding(n_e, e_dim)
        self.pos_embed = torch.nn.Parameter(torch.zeros(
            1, token_size, hidden_dim), requires_grad=False)
        self.blocks = BasicLayer(
            hidden_dim,
            img_size,
            depth,
            num_heads=num_heads,
            window_size=8,
            downsample=None,
            cross_attend=cross_attend,
            use_checkpoint=use_checkpoint,
        )
        self.cross_attend = cross_attend
        self.add_cross_proj = add_cross_proj
        self.norm = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim), torch.nn.GELU())
        self.pred = torch.nn.Linear(hidden_dim, n_e, bias=True)

        if add_cross_proj:
            if not cross_attend:
                raise ValueError(
                    "When add_cross_proj is True, cross_attend must also be True.")
            self.context_proj = torch.nn.Linear(
                cond_in_channels, hidden_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_dim, self.img_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def forward(self,
                x,
                x_id=None,
                action=None, context=None, return_feats=None, attention_mask_temporal=None, feature_collect_range=None):
        # embed tokens
        # TODO hard code
        feature_collect_range = [0, 4]
        if self.use_extra_embedding:
            x = torch.zeros_like(x)
            x_id = x_id.to(torch.long)
            x[x_id == -1] = self.mask_token
            x[x_id != -1] = self.extra_embedding(x_id[x_id != -1])
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        if self.cross_attend:
            if self.add_cross_proj:
                context = self.context_proj(context["context"])
            x = self.blocks(
                x, context)
        else:
            x = self.blocks(x)
        x = self.norm(x)

        # predictor projection
        x = self.pred(x)
        return x
