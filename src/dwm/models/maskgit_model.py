import numpy as np
import math
import torch
import torch.distributed
import torch.distributed.nn.functional
from torch import nn
from dwm.models.vq_point_cloud import BasicLayer, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
from dwm.models.base_vq_models.gpt2_blocks import GPT2Block
from dwm.models.adapters import ImageAdapter
from transformers.pytorch_utils import Conv1D
from einops import rearrange, repeat

from diffusers.models.transformers.transformer_temporal import TransformerTemporalModel
from diffusers.models.attention import TemporalBasicTransformerBlock

from typing import Optional
from easydict import EasyDict as edict


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
