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
        #self.reduction = torch.nn.Linear(4 * dim, 2 * dim, bias=False)
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
        attention_mask: torch.FloatTensor,
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
        attention_mask = attention_mask.repeat(norm_hidden_states.shape[0], 1, 1)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None, attention_mask=attention_mask)
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
        window_size=4,
        num_blocks=1,
        use_checkpoint=True,
        dtype_t='diffusers',
        cross_attend = False,
        enable_temporal = False
    ):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.enable_temporal = enable_temporal
        self.swin = torch.nn.ModuleList([])
        for _ in range(num_blocks):
            self.swin.append(
                BasicLayer(
                    dim = input_channel,
                    input_resolution = (img_size, img_size),
                    depth = 2, # 
                    num_heads = num_heads,
                    window_size = window_size,
                    cross_attend = cross_attend, 
                    use_checkpoint = use_checkpoint,
                )
            )
            
        # TODO None list
        if enable_temporal:
            self.temporal = torch.nn.ModuleList([])
            for _ in range(num_blocks):
                if dtype_t == 'diffusers':
                    self.temporal.append(
                        TemporalBasicTransformerBlockUniMVCausal(
                            input_channel,
                            input_channel,
                            num_heads,
                            input_channel // num_heads,         # inner is heads*dim_head, which is keep same
                            cross_attention_dim=None,
                        )
                    )
                elif dtype_t == 'gpt2':
                    self.temporal.append(
                        GPT2Block(
                            config=edict(           # From configuration_gpt2.py
                                dict(
                                    hidden_size=input_channel,
                                    num_attention_heads=num_heads,
                                    scale_attn_weights=True,
                                    scale_attn_by_inverse_layer_idx=False,
                                    reorder_and_upcast_attn=False,
                                    add_cross_attention=False,
                                    activation_function='gelu_new',
                                    attn_pdrop=0.1,
                                    resid_pdrop=0.1,            # !!! diff1: used in gpt2, which is different from its counterpart in diffusers (0),
                                    n_inner=None,               # !!! diff2: 4*hidden will be used, which is single layer in diffusers
                                    layer_norm_epsilon=1e-5,
                                )
                            )
                        )
                    )
                else:
                    raise NotImplementedError
        else:
            self.temporal = [None for _ in range(num_blocks)]

    def forward(self, x, context=None, tempo_w=0.5, **kwargs): # TODO 
        
        # if 'attention_mask_temporal' in kwargs.keys():
        attention_mask_temporal = kwargs['attention_mask_temporal']
        num_frames = kwargs['num_frames']
        
        # else:
        #     attention_mask_temporal = torch.eye(self.num_frames, dtype=x.dtype, device=x.device)
        for swin, temporal in zip(self.swin, self.temporal):
            if self.training and self.use_checkpoint:
                if context is not None:
                    x = torch.utils.checkpoint.checkpoint(
                        swin, x, context, use_reentrant=False
                    )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        swin, x, use_reentrant=False
                    )
            else:
                if context is not None:
                    x = swin(x, context)
                else:
                    x = swin(x)

            if temporal is not None: 
                if self.training and self.use_checkpoint:           # residual to all
                    x = torch.utils.checkpoint.checkpoint(
                        temporal, x, num_frames, attention_mask_temporal, use_reentrant=False)
                else:
                    x = temporal(x, num_frames=num_frames, attention_mask=attention_mask_temporal)
            
            # x = (1-tempo_w) * x_s + tempo_w * x_t

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

    def forward(self, x, action=None, context=None, return_feats=None, attention_mask_temporal = None, feature_collect_range=None):
        # embed tokens
        # TODO hard code
        feature_collect_range = [0, 4]

        x = self.decoder_embed(x) 

        # add pos embed
        x = x + self.pos_embed 

        # apply Transformer blocks
        if self.cross_attend:
            if self.add_cross_proj:
                context = self.context_proj(context["context"])
            x, features = self.blocks(
                x, context, feature_collect_range=feature_collect_range)
        else:
            x, features = self.blocks(x, feature_collect_range=feature_collect_range)
        x = self.norm(x)

        # predictor projection
        x = self.pred(x)
        return x, features

class LevelMerging(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super().__init__()
        # as in VQ Decode, which follow this, but is not consistent with the paper (with layernorm)
        self.upsample = torch.nn.ConvTranspose2d(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = kernel_size, 
                stride = stride
            )
        self.norm = torch.nn.LayerNorm(out_channels*2)
        # as in paper, no bias
        self.linear = torch.nn.Linear(out_channels*2, out_channels, bias=False)

    def forward(self, x, x_low):
        x = self.upsample(x)

        # TODO: residual is after upsample ???
        x = rearrange(x, "b d h w -> b (h w) d")
        # x += res[-i-1]
        residual = x_low
        x_cat = torch.cat([x, x_low], dim=-1)
        x_cat = self.norm(x_cat)
        x_cat = self.linear(x_cat)

        return residual + x_cat



class UnetTransformerAlignV2(torch.nn.Module):
    
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
        sequence_length = 12,
        window_size = 8,
        use_action = False, 
        enable_temporal = False, 
        cross_attend = False,
        act_type = "copliot4D", 
        use_checkpoint = True,
        # from gpt2 default config
        initializer_range=0.02,
        initializer_type='cop4d',
        dtype_t='diffusers',
        cond_inject='Merge',
        cond_in_channels = 512,
        cond_is_downblocks = [False, True, True],
        cond_num_res_blocks = 2,
        cond_downscale_factor = 1
    ):
        super().__init__()
        self.n_e = n_e
        self.num_frames = sequence_length
        self.img_size = img_size[0]
        self.initializer_range = initializer_range
        self.initializer_type = initializer_type

        hidden_dim = block_out_channels[0]
        self.hidden_dim = hidden_dim
        self.use_action = use_action
        self.act_type = act_type
        self.cross_attend = cross_attend
        self.enable_temporal = enable_temporal
        self.cond_inject = cond_inject # or Adapter

        self.block_out_channels = block_out_channels

        self.decoder_embed = torch.nn.Linear(e_dim, hidden_dim, bias=True)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, e_dim), requires_grad=True)
        spatial_embed = get_2d_sincos_pos_embed(
            self.hidden_dim, (self.img_size, self.img_size),
            cls_token=False)
        self.register_buffer(
            "spatial_embed", torch.from_numpy(spatial_embed).unsqueeze(0))

        if self.cross_attend:
            if self.cond_inject == "Merge":
                self.context_proj = torch.nn.Linear(512 * 2, block_out_channels[0], bias=True)
                nn.init.zeros_(self.context_proj.weight)
                nn.init.zeros_(self.context_proj.bias)
            elif self.cond_inject == "Adapter":
                self.context_adapter = ImageAdapter(
                    in_channels = cond_in_channels,
                    channels = self.block_out_channels,
                    is_downblocks = cond_is_downblocks,
                    num_res_blocks = cond_num_res_blocks,
                    downscale_factor = cond_downscale_factor
                )

        if self.enable_temporal: 
            self.temporal_embed = torch.nn.Parameter(
                torch.zeros(1, self.num_frames, hidden_dim), requires_grad=True
            )

        # down
        self.down_spatio_temporal = torch.nn.ModuleList([])
        self.downsample = torch.nn.ModuleList([])
        self.down_linear = torch.nn.ModuleList([])

        if self.cond_inject == 'Merge':
            self.downsample_context = torch.nn.ModuleList([])
        
        for i, _ in enumerate(block_out_channels[:-1]):

            self.down_spatio_temporal.append(
                SpatioTemporalBlock(
                    input_channel = block_out_channels[i],
                    img_size = img_size[i], 
                    num_heads = num_heads,
                    window_size = window_size,
                    num_blocks = num_down_block[i],
                    use_checkpoint = use_checkpoint,
                    dtype_t = dtype_t,
                    cross_attend = self.cross_attend, 
                    enable_temporal = self.enable_temporal
                )
            )

            self.downsample.append(
                PatchMerging(
                    input_resolution = (img_size[i], img_size[i]),
                    dim = block_out_channels[i],
                    output_dim = block_out_channels[i+1]
                )
            )

            if self.cond_inject == 'Merge':
                self.downsample_context.append(
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
            num_blocks = 1,
            dtype_t=dtype_t,
            cross_attend = self.cross_attend, 
            enable_temporal = self.enable_temporal
        )

        self.up_spatio_temporal = torch.nn.ModuleList([])
        self.upsample = torch.nn.ModuleList([])

        if self.cond_inject == 'Merge':
            self.upsample_context = torch.nn.ModuleList([])
        
        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_img_size = list(reversed(img_size))
        
        for i, _ in enumerate(reversed_block_out_channels[:-1]):
            
            self.upsample.append(
                LevelMerging(
                    in_channels = reversed_block_out_channels[i], 
                    out_channels = reversed_block_out_channels[i+1], 
                    kernel_size = 2, 
                    stride = 2
                )
            )

            if self.cond_inject == "Merge":
                self.upsample_context.append(
                LevelMerging(
                    in_channels = reversed_block_out_channels[i], 
                    out_channels = reversed_block_out_channels[i+1], 
                    kernel_size = 2, 
                    stride = 2
                    )
                )

            self.up_spatio_temporal.append(
                SpatioTemporalBlock(
                    input_channel = reversed_block_out_channels[i+1],
                    img_size = reversed_img_size[i], 
                    num_heads = num_heads,
                    window_size = window_size,
                    num_blocks = num_up_block[i],
                    dtype_t=dtype_t,
                    cross_attend = self.cross_attend, 
                    enable_temporal = self.enable_temporal
                )
            )

        # action network: dwm or mlp
        self.act_type = act_type 
        if self.use_action == False:
            self.act_type = None
            
        # if self.act_type == "dwm":
        # # define the action in a time step as (∆x, ∆y), 
        # # which represents the movement of ego location to the next time step.
        # # We use an MLP to map the action into a d-dimension embedding a ∈ R^2×d
        #     self.action_embedding = torch.nn.Linear(1, 512, bias=True)

        if self.act_type == "copliot4D" or self.act_type == "copliot4D_v2" or self.act_type == "copliot4D_v3" or self.act_type == "dwm":
            self.action_embedding = torch.nn.ModuleList([])
        # The ego vehicle poses, which are the actions of the ego vehicle. 
        # Those 4 × 4 matrices are flattened into a 16-dimensional vector, which then goes through Linear → LayerNorm → Linear, 
        # and added to all feature map locations of corresponding temporal frames;
            if self.act_type == "dwm":
                c_in = 3
            else:
                c_in = 16
            for i in range(len(block_out_channels)):
                self.action_embedding.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(c_in, block_out_channels[i]),
                        torch.nn.LayerNorm(block_out_channels[i]),
                        torch.nn.Linear(block_out_channels[i], block_out_channels[i])
                    )
                )

        self.n_layer = (1 + sum(num_down_block) + sum(num_up_block))*3
                
        self.align = torch.nn.Linear(hidden_dim, hidden_dim*4, bias=False)
        self.norm = torch.nn.LayerNorm(hidden_dim*4)
        self.pred = torch.nn.Linear(hidden_dim*4, n_e, bias=False)
        self.initialize_weights()
    
    def initialize_weights(self):
        # initialization
        t_list = torch.arange(self.num_frames)
        t_list = t_list.view(self.num_frames, )
        temporal_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_dim, t_list)
        if self.enable_temporal:
            self.temporal_embed.data.copy_(torch.from_numpy(temporal_embed).float())

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize the weights.
        Copy from gpt2 in transformers
        """
        if isinstance(module, (nn.Linear, Conv1D)):         # nn.Conv2d in patch_embed (vq encoder), but not use here
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            if self.initializer_type == 'cop4d':            # 1/sqrt(3N)
                self.initializer_range = 1 / math.sqrt(3*module.weight.shape[0])
            else:
                assert self.initializer_type == 'gpt2'          # 0.02

            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            # !!! new in cop4d
            # to_out.0.weight from diffusers temporal block
            # c_proj.weight from gpt2 temporal block (TODO)
            # attn.proj.weight from swin spatial block
            if name in ["to_out.0.weight", "c_proj.weight", "attn.proj.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.initializer_range / math.sqrt(2 * self.n_layer)))

    # def _init_weights(self, m):
    #     if isinstance(m, torch.nn.Linear):
    #         # we use xavier_uniform following official JAX ViT:
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if isinstance(m, torch.nn.Linear) and m.bias is not None:
    #             torch.nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, torch.nn.LayerNorm):
    #         torch.nn.init.constant_(m.bias, 0)
    #         torch.nn.init.constant_(m.weight, 1.0)

    def _forward_prepare(self, x, context=None, action=None, attention_mask_temporal=None):
        assert (context is None) ^ (self.cross_attend)
        num_frames = self.num_frames

        # (bs*l, h*w, c)
        # for mini: 4, 6400, 1024
        x = self.decoder_embed(x)

        bf, hw, d = x.shape
        bs = bf // num_frames
        if self.enable_temporal:
            cur_temporal_embed = self.temporal_embed

        # spatio & tempo embedding
        x = x + self.spatial_embed

        if self.enable_temporal:
            x = rearrange(x, '(b l) hw d -> (b hw) l d', b=bs)
            x = x + cur_temporal_embed[:, :num_frames]
            x = rearrange(x, '(b hw) l d -> (b l) hw d', b=bs)
       
        if self.use_action: 
            action = action.flatten(-2, -1)
        else:
            action = None
        
        # down
        if self.cross_attend:
            context_list = None
            if self.cond_inject == "Merge":
                context = self.context_proj(context)
            elif self.cond_inject == "Adapter":
                raise NotImplementedError("Not support now...")
        return x, context, action

    def _forward_down(self, x, i, context, action, attention_mask_temporal, num_frames):
        bs = x.shape[0]
        assert (context is None) ^ (self.cross_attend)
        if self.use_action:
            if self.act_type == "copliot4D" or self.act_type == "copliot4D_v2" or self.act_type == "copliot4D_v3" or self.act_type == "dwm":
                action = rearrange(action, '(b t) c -> b t c', b=bs)[:, :num_frames].flatten(0, 1)
                act_embed = self.action_embedding[i](action)
                act_embed = repeat(act_embed, "b d -> b l d", l=x.shape[1])
                x += act_embed
            else:
                raise NotImplementedError

        x = self.down_spatio_temporal[i](x, context, attention_mask_temporal=attention_mask_temporal, num_frames=num_frames)
        
        x = self.downsample[i](x)
        if self.cross_attend:
            if self.cond_inject == "Merge":
                context = self.downsample_context[i](context)
        return x, context

    def _forward_mid(self, x, i, context, action, attention_mask_temporal, num_frames):
        bs = x.shape[0]
        assert (context is None) ^ (self.cross_attend)

        if self.use_action:
            if self.act_type == "copliot4D" or self.act_type == "copliot4D_v2" or self.act_type == "copliot4D_v3" or self.act_type == "dwm":
                action = rearrange(action, '(b t) c -> b t c', b=bs)[:, :num_frames].flatten(0, 1)
                act_embed = self.action_embedding[-1](action)
                act_embed = repeat(act_embed, "b d -> b l d", l=x.shape[1])
                x += act_embed
            else:
                raise NotImplementedError

        x = self.mid_spatio_temporal(x, context, attention_mask_temporal=attention_mask_temporal, num_frames=num_frames)

        return x, context

    def _forward_up(self, x, i, context, res, res_context, attention_mask_temporal, num_frames):
        h = w = int(np.sqrt(x.shape[1]))
        x = rearrange(x, "b (h w) d -> b d h w ", h=h, w=w)
        x = self.upsample[i](x, res[-i-1])

        if self.cross_attend:
            if self.cond_inject == "Merge":
                context = rearrange(context, "b (h w) d -> b d h w ", h=h, w=w)
                context = self.upsample_context[i](context, res_context[-i-1])
            elif self.cond_inject == "Adapter":
                raise NotImplementedError("Not support now")

        if self.cross_attend:
            x = self.up_spatio_temporal[i](x, context, attention_mask_temporal=attention_mask_temporal, num_frames=num_frames)
        else:
            x = self.up_spatio_temporal[i](x, attention_mask_temporal=attention_mask_temporal, num_frames=num_frames)
        return x, context

    def forward_post(self, x):
        x = self.align(x)
        x = self.norm(x)
        x = self.pred(x)

        return x
        
    def forward(self, x, context=None, action=None, return_feats=False, attention_mask_temporal=None):
        # Note: no condition is only supported in forward_up/mid/down

        num_frames = self.num_frames

        # (bs*l, h*w, c)
        # for mini: 4, 6400, 1024
        x = self.decoder_embed(x)

        bf, hw, d = x.shape
        bs = bf // num_frames
        mid_rts = {
            'features': [],
            'encoder_features': []
        }

        if self.enable_temporal:
            cur_temporal_embed = self.temporal_embed

        # spatio & tempo embedding
        x = x + self.spatial_embed

        if self.enable_temporal:
            x = rearrange(x, '(b l) hw d -> (b hw) l d', b=bs)
            x = x + cur_temporal_embed[:, :num_frames]
            x = rearrange(x, '(b hw) l d -> (b l) hw d', b=bs)
       
        if self.use_action: 
            action = action.flatten(-2, -1)
        
        res = ()
        # down
        if context is not None:
            if self.cond_inject == "Merge":
                context = self.context_proj(context)
                res_context = ()
            elif self.cond_inject == "Adapter":
                h = w = math.sqrt(context.shape[1])
                context = rearrange(context, 'b (h w) c -> b c h w', h=int(h), w=int(w))
                context_list = self.context_adapter(context)

        for i in range(len(self.block_out_channels)-1):
            res += (x, )

            if context is not None:
                if self.cond_inject == "Merge":
                    res_context += (context, )
                elif self.cond_inject == "Adapter":
                    context = context_list[i]
                    context = rearrange(context, 'b c h w -> b (h w) c')

            # === cache encoder features
            mid_rts['encoder_features'].append(x)
            if self.cross_attend:
                if self.use_action:
                    if self.act_type == "copliot4D" or self.act_type == "copliot4D_v2" or self.act_type == "copliot4D_v3" or self.act_type == "dwm":
                        action = rearrange(action, '(b t) c -> b t c', b=bs)[:, :num_frames].flatten(0, 1)
                        act_embed = self.action_embedding[i](action)
                        act_embed = repeat(act_embed, "b d -> b l d", l=x.shape[1])
                        x += act_embed
                    else:
                        raise NotImplementedError

                x = self.down_spatio_temporal[i](x, context, attention_mask_temporal=attention_mask_temporal, num_frames=num_frames)
            else:
                if self.use_action:
                    if self.act_type == "copliot4D" or self.act_type == "copliot4D_v2" or self.act_type == "copliot4D_v3" or self.act_type == "dwm":
                        action = rearrange(action, '(b t) c -> b t c', b=bs)[:, :num_frames].flatten(0, 1)
                        act_embed = self.action_embedding[i](action)
                        act_embed = repeat(act_embed, "b d -> b l d", l=x.shape[1])
                        x += act_embed
                    else:
                        raise NotImplementedError

                x = self.down_spatio_temporal[i](x, attention_mask_temporal=attention_mask_temporal, num_frames=num_frames)
            
            x = self.downsample[i](x)
            if context is not None and self.cond_inject == "Merge":
                context = self.downsample_context[i](context)

            # === cache mid_rts
            # mid_rts['act_embed'].append(act_embed)

        # mid
        if context is not None and self.cond_inject == "Adapter":
            context = context_list[-1]
            context = rearrange(context, 'b c h w -> b (h w) c')

        if self.cross_attend:
            if self.use_action:
                if self.act_type == "copliot4D" or self.act_type == "copliot4D_v2" or self.act_type == "copliot4D_v3" or self.act_type == "dwm":
                    action = rearrange(action, '(b t) c -> b t c', b=bs)[:, :num_frames].flatten(0, 1)
                    act_embed = self.action_embedding[-1](action)
                    act_embed = repeat(act_embed, "b d -> b l d", l=x.shape[1])
                    x += act_embed
                else:
                    raise NotImplementedError

            x = self.mid_spatio_temporal(x, context, attention_mask_temporal=attention_mask_temporal, num_frames=num_frames)
        else:
            if self.use_action:
                if self.act_type == "copliot4D" or self.act_type == "copliot4D_v2" or self.act_type == "copliot4D_v3" or self.act_type == "dwm":
                    action = rearrange(action, '(b t) c -> b t c', b=bs)[:, :num_frames].flatten(0, 1)
                    act_embed = self.action_embedding[-1](action)
                    act_embed = repeat(act_embed, "b d -> b l d", l=x.shape[1])
                    x += act_embed
                else:
                    raise NotImplementedError

            x = self.mid_spatio_temporal(x, attention_mask_temporal=attention_mask_temporal, num_frames=num_frames)
        # === cache mid_rts
        # mid_rts['act_embed'].append(act_embed)
        mid_rts['features'].append(x)
        mid_rts['encoder_features'].append(x)        
        # up
        reversed_block_out_channels = list(reversed(self.block_out_channels))
        for i in range(len(reversed_block_out_channels)-1):

            h = w = int(np.sqrt(x.shape[1]))
            x = rearrange(x, "b (h w) d -> b d h w ", h=h, w=w)
            x = self.upsample[i](x, res[-i-1])

            if context is not None:
                if self.cond_inject == "Merge":
                    context = rearrange(context, "b (h w) d -> b d h w ", h=h, w=w)
                    context = self.upsample_context[i](context, res_context[-i-1])
                elif self.cond_inject == "Adapter":
                    context = context_list[1-i]
                    context = rearrange(context, 'b c h w -> b (h w) c')
            # x = rearrange(x, "b d h w -> b (h w) d")
            # x += res[-i-1]

            if self.cross_attend:
                x = self.up_spatio_temporal[i](x, context, attention_mask_temporal=attention_mask_temporal, num_frames=num_frames)
            else:
                x = self.up_spatio_temporal[i](x, attention_mask_temporal=attention_mask_temporal, num_frames=num_frames)
            
            # === cache mid_rts
            mid_rts['features'].append(x)
    
        x = self.align(x)
        x = self.norm(x)
        x = self.pred(x)

        if return_feats:
            return x, mid_rts
        else:
            return x