import numpy as np
import scipy.cluster.vq
import timm.models.swin_transformer
import timm.layers
import torch
import torch.nn as nn
import torch.distributed
import torch.distributed.nn.functional
import torch.nn.functional as F
from dwm.models.vq_point_cloud import BasicLayer, get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid

from transformers.models.bert.modeling_bert import BertLayer
from einops import rearrange, repeat
from transformers import AutoConfig
import diffusers
import diffusers.models.attention
from typing import Tuple, Union, Optional, Dict, Any

class BidirectionalTransformer(torch.nn.Module):
    def __init__(self, n_e, e_dim, img_size, hidden_dim=512, depth=24, num_heads=16, use_checkpoint=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.decoder_embed = torch.nn.Linear(e_dim, hidden_dim, bias=True)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, e_dim), requires_grad=True)
        token_size = img_size**2
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, token_size, hidden_dim), requires_grad=False)
        self.blocks = BasicLayer(
            hidden_dim,
            (img_size, img_size),
            depth,
            num_heads=num_heads,
            window_size=8,
            downsample=None,
            use_checkpoint=use_checkpoint,
        )
        self.norm = torch.nn.Sequential(torch.nn.LayerNorm(hidden_dim), torch.nn.GELU())
        self.pred = torch.nn.Linear(hidden_dim, n_e, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.hidden_dim, (self.img_size, self.img_size), cls_token=False)
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

    def forward(self, x, **kwargs):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # predictor projection
        x = self.pred(x)

        return x


class TemporalBasicTransformerBlockUniMVCausal(
    diffusers.models.attention.TemporalBasicTransformerBlock):

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        num_frames: int,
        causal_mask: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        batch_frames, seq_length, channels = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * seq_length, num_frames, channels)

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)

        if self._chunk_size is not None:
            raise NotImplementedError
        else:
            hidden_states = self.ff_in(hidden_states)

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        causal_mask = causal_mask.repeat(norm_hidden_states.shape[0], 1, 1)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None, attention_mask=causal_mask)
        hidden_states = attn_output + hidden_states

        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self._chunk_size is not None:
            raise NotImplementedError
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states[None, :].reshape(batch_size, seq_length, num_frames, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * num_frames, seq_length, channels)

        return hidden_states


class SpatioTemporalTransformer(BidirectionalTransformer):
    def __init__(self, n_e, e_dim, img_size, hidden_dim=512, depth=24, 
        num_heads=16, frames=12, use_checkpoint=True):
        super().__init__(
                n_e, e_dim, img_size, hidden_dim=hidden_dim, depth=depth, num_heads=num_heads, use_checkpoint=use_checkpoint
            )

        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalBasicTransformerBlockUniMVCausal(
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    hidden_dim // num_heads,
                    cross_attention_dim=None,
                )
                for _ in range(depth//2)
            ]
        )
        self.frames = frames
        self.use_checkpoint = use_checkpoint

    def forward(self, x, **kwargs):
        # x: B*L, N, D -> spatial layers
        # -> B*N, L, D -> temporal layers
        attention_mask_temporal = kwargs['attention_mask_temporal']

        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.pos_embed

        batch_frames, seq_len, _ = x.shape
        batch_size = batch_frames//self.frames

        # for block, temporal_block in zip(self.blocks.blocks, self.temporal_transformer_blocks):
        for bid, block in enumerate(self.blocks.blocks):
            # TODO: support 2 sp + 1 t
            # residual = x
            # x_mix = rearrange(x, '(b f) s c -> (b s) f c', b=batch_size)
            x_mix = x
            # x = block(x)
            if self.training and self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False)
            else:
                x = block(x)

            # === Temporal
            # attention_mask_temporal = attention_mask_temporal[None].repeat(x_mix.shape[0], 1, 1)
            if bid%2 == 1:
                temporal_block = self.temporal_transformer_blocks[bid//2]
                if self.training and self.use_checkpoint:
                    x_mix = torch.utils.checkpoint.checkpoint(
                        temporal_block, x_mix, self.frames, attention_mask_temporal, use_reentrant=False)
                else:
                    x_mix = temporal_block(x_mix, num_frames=self.frames, causal_mask=attention_mask_temporal)
                # x_mix = rearrange(x_mix, '(b s) f c -> (b f) s c', b=batch_size)

            x = 0.5*x + 0.5*x_mix

            # y = rearrange(x, '(b l) d h w -> b l d (h w)', b=batch_size)
            # y = rearrange(y, 'b l d (h w) -> (b h w) l d')
            # y = temporal_block(y)
            # y = rearrange(y, '(b h w) l d -> b (h w) l d', b=batch_size)
            # y = rearrange(y, 'b (h w) l d -> b l d (h w)')
            # y = rearrange(y, 'b l d (h w) -> (b l) d h w')
            # x = residual + x + y

        if self.blocks.downsample is not None:
            x = self.blocks.downsample(x)
        
        x = self.norm(x)

        # predictor projection
        x = self.pred(x)

        return x