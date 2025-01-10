import numpy as np
import scipy.cluster.vq
import timm.models.swin_transformer
import timm.layers
import torch
import torch.distributed
import torch.distributed.nn.functional
import torch.nn.functional as F
from dwm.models.vq_point_cloud import BasicLayer, get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid
from dwm.models.vq_point_cloud import PatchMerging


# from transformers.models.bert.modeling_bert import BertLayer
from einops import rearrange, repeat
# from transformers import AutoConfig


from diffusers.models.transformers.transformer_temporal import TransformerTemporalModel
from diffusers.models.attention import TemporalBasicTransformerBlock

from typing import Optional

class TemporalBlock(TransformerTemporalModel):

    def __init__(
        self,
        in_channels,
        num_frames
    ):
        super().__init__(
            in_channels = in_channels,
            out_channels = in_channels
        )
        self.num_frames = num_frames

    def forward(
        self,
        hidden_states,
    ):  
        
        batch_frames, hw, channel= hidden_states.shape
        batch_size = batch_frames // self.num_frames
        num_frames = self.num_frames
        height = width = int(np.sqrt(hw))

        hidden_states = rearrange(hidden_states, 'bf (h w) d -> bf d h w', h=height, w=width)

        # super().forward()
        residual = hidden_states

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4) # (bs, c, l, h, w)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)

        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states
            )
        
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states[None, None, :]
            .reshape(batch_size, height, width, num_frames, channel)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )
        hidden_states = hidden_states.reshape(batch_frames, channel, height, width)

        output = hidden_states + residual

        output = rearrange(output, 'bf d h w -> bf (h w) d', h=height, w=width)

        return output

class TemporalBasicTransformerBlockUniMVCausal(TemporalBasicTransformerBlock):

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

class SpatioTemporalBlock(torch.nn.Module):
    def __init__(
        self, 
        input_channel,
        img_size,
        num_heads=16, 
        num_frames=1,
        window_size=4,
        num_blocks=1,
        with_tempo=False,
        use_checkpoint=True 
    ):

        super().__init__()

        self.num_frames = num_frames
        self.use_checkpoint = use_checkpoint

        self.swin = torch.nn.ModuleList([])
        for _ in range(num_blocks):
            self.swin.append(
                BasicLayer(
                    dim = input_channel,
                    input_resolution = (img_size, img_size),
                    depth = 2, # 
                    num_heads = num_heads,
                    window_size = window_size,
                    use_checkpoint = use_checkpoint,
                )
            )

        self.temporal = torch.nn.ModuleList([])
        for _ in range(num_blocks):
            self.temporal.append(
                TemporalBasicTransformerBlockUniMVCausal(
                    input_channel,
                    input_channel,
                    num_heads,
                    input_channel // num_heads,
                    cross_attention_dim=None,
                )
            )

    def forward(self, x, context=None, tempo_w=0.5, **kwargs): # TODO 
        
        if 'attention_mask_temporal' in kwargs.keys():
            attention_mask_temporal = kwargs['attention_mask_temporal']
        else:
            attention_mask_temporal = torch.eye(self.num_frames, dtype=x.dtype, device=x.device)

        for swin, temporal in zip(self.swin, self.temporal):

            if self.training and self.use_checkpoint:
                if context is not None:
                    x_s = torch.utils.checkpoint.checkpoint(
                        swin, x, context, use_reentrant=False
                    )
                else:
                    x_s = torch.utils.checkpoint.checkpoint(
                        swin, x, use_reentrant=False
                    )
            else:
                if context is not None:
                    x_s = swin(x, context)
                else:
                    x_s = swin(x)

            if self.training and self.use_checkpoint:
                x_t = torch.utils.checkpoint.checkpoint(
                    temporal, x, self.num_frames, attention_mask_temporal, use_reentrant=False)
            else:
                x_t = temporal(x, num_frames=self.num_frames, causal_mask=attention_mask_temporal)

        x = (1-tempo_w) * x_s + tempo_w * x_t

        return x


class BidirectionalTransformer(torch.nn.Module):
    def __init__(self, n_e, e_dim, img_size, hidden_dim=512, depth=24, num_heads=16, use_checkpoint=False, cross_attend=False, add_cross_proj=False, cross_proj_dim=4096):
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
            cross_attend=cross_attend,
            use_checkpoint=use_checkpoint,
        )
        
        self.cross_attend = cross_attend
        self.add_cross_proj = add_cross_proj
        self.norm = torch.nn.Sequential(torch.nn.LayerNorm(hidden_dim), torch.nn.GELU())
        self.pred = torch.nn.Linear(hidden_dim, n_e, bias=True)

        if add_cross_proj:
            if not cross_attend:
                raise ValueError("When add_cross_proj is True, cross_attend must also be True.")
            self.context_proj = torch.nn.Linear(cross_proj_dim, hidden_dim, bias=True)

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

    def forward(self, x, context=None):
        # embed tokens
        x = self.decoder_embed(x) 

        # add pos embed
        x = x + self.pos_embed 

        # apply Transformer blocks
        if self.cross_attend:
            if self.add_cross_proj:
                context = self.context_proj(context)
            x = self.blocks(x, context)
        else:
            x = self.blocks(x)
        x = self.norm(x)

        # predictor projection
        x = self.pred(x)

        return x


class UnetTransformer(torch.nn.Module):
    
    def __init__(
        self,
        n_e = 2048, 
        e_dim = 1024, 
        img_size = [128, 64],
        block_out_channels = [256, 384], 
        mid_block_channels = 512, 
        num_down_block = [2, 2],
        num_up_block = [1, 2],
        num_heads = 16,
        num_frames = 12,
        window_size = 8,
        cross_attend = True,
        use_checkpoint = True
        
    ):
        super().__init__()
        self.n_e = n_e
        self.num_frames = num_frames
        self.img_size = img_size[0]

        hidden_dim = block_out_channels[0]
        self.hidden_dim = hidden_dim
        self.cross_attend = cross_attend

        self.block_out_channels = block_out_channels

        self.decoder_embed = torch.nn.Linear(e_dim, hidden_dim, bias=True)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, e_dim), requires_grad=True)
        self.spatial_embed = torch.nn.Parameter(
            torch.zeros(1, img_size[0]**2, hidden_dim), requires_grad=False
        )
        self.temporal_embed = torch.nn.Parameter(
            torch.zeros(1, num_frames, hidden_dim), requires_grad=False
        )

        # down
        self.down_spatio_temporal = torch.nn.ModuleList([])
        self.downsample = torch.nn.ModuleList([])
        self.down_linear = torch.nn.ModuleList([])
        
        for i, _ in enumerate(block_out_channels[:-1]):

            self.down_spatio_temporal.append(
                SpatioTemporalBlock(
                    input_channel = block_out_channels[i],
                    img_size = img_size[i], 
                    num_heads = num_heads,
                    window_size = window_size,
                    num_blocks = num_down_block[i],
                    use_checkpoint = use_checkpoint
                )
            )

            self.downsample.append(
                PatchMerging(
                    input_resolution = (img_size[i], img_size[i]),
                    dim = block_out_channels[i],
                    output_dim = block_out_channels[i+1]
                )
            )

        # mid

        self.mid_spatio_temporal = SpatioTemporalBlock(
            input_channel = mid_block_channels,
            img_size = img_size[-1]//2, 
            num_heads = num_heads,
            window_size = window_size,
            num_blocks = 1
        )

        self.up_spatio_temporal = torch.nn.ModuleList([])
        self.upsample = torch.nn.ModuleList([])
        
        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_img_size = list(reversed(img_size))
        
        for i, _ in enumerate(reversed_block_out_channels[:-1]):
            
            self.upsample.append(
                torch.nn.ConvTranspose2d(
                    in_channels = reversed_block_out_channels[i], 
                    out_channels = reversed_block_out_channels[i+1], 
                    kernel_size = 2, 
                    stride = 2
                )
            )
            for _ in range(num_up_block[i]):
                self.up_spatio_temporal.append(
                    SpatioTemporalBlock(
                        input_channel = reversed_block_out_channels[i+1],
                        img_size = reversed_img_size[i], 
                        num_heads = num_heads,
                        window_size = window_size,
                        num_blocks = num_up_block[i]
                    )    
            )

        self.norm = torch.nn.Sequential(torch.nn.LayerNorm(hidden_dim), torch.nn.GELU())
        self.pred = torch.nn.Linear(hidden_dim, n_e, bias=True)
        self.initialize_weights()
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        spatial_embed = get_2d_sincos_pos_embed(self.hidden_dim, (self.img_size, self.img_size), cls_token=False)
        self.spatial_embed.data.copy_(torch.from_numpy(spatial_embed).float())

        t_list = torch.arange(self.num_frames)
        t_list = t_list.view(self.num_frames, )
        temporal_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_dim, t_list)
        self.temporal_embed.data.copy_(torch.from_numpy(temporal_embed).float())

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

    def forward(self, x, context):

        # (bs*l, h*w, c)
        # for mini: 4, 6400, 1024
        x = self.decoder_embed(x)

        bf, hw, d = x.shape
        bs = bf // self.num_frames

        x = x + self.spatial_embed
        x = rearrange(x, '(b l) hw d -> (b hw) l d', b=bs)
        x = x + self.temporal_embed
        x = rearrange(x, '(b hw) l d -> (b l) hw d', b=bs)

        # down
        res = ()
        for i in range(len(self.block_out_channels)-1):
            res += (x, )
            if self.cross_attend:
                x = self.down_spatio_temporal[i](x, context)
            else:
                x = self.down_spatio_temporal[i](x)
            
            x = self.downsample[i](x)

        # mid
        if self.cross_attend:
            x = self.mid_spatio_temporal(x, context)
        else:
            x = self.mid_spatio_temporal(x)
        
        # up
        reversed_block_out_channels = list(reversed(self.block_out_channels))
        for i in range(len(reversed_block_out_channels)-1):

            h = w = int(np.sqrt(x.shape[1]))
            x = rearrange(x, "b (h w) d -> b d h w ", h=h, w=w)
            x = self.upsample[i](x)
            x = rearrange(x, "b d h w -> b (h w) d")
            x += res[-i-1]

            if self.cross_attend:
                x = self.up_spatio_temporal[i](x, context)
            else:
                x = self.up_spatio_temporal[i](x)
    
        x = self.norm(x)
        x = self.pred(x)

        return x


        


