import diffusers
import diffusers.models.unets.unet_3d_blocks
import dwm.models.crossview_temporal
import dwm.models.adapters
import dwm.models.depth_net
import re
import torch


class UNetMidBlockCrossviewTemporal(
    diffusers.models.unets.unet_3d_blocks.UNetMidBlockSpatioTemporal
):
    def __init__(
        self, in_channels: int, temb_channels: int,
        enable_crossview: bool = True, enable_temporal: bool = True,
        enable_rowwise_crossview: bool = False,
        enable_rowwise_temporal: bool = False, num_layers: int = 1,
        transformer_layers_per_block=1, resnet_eps: float = 1e-5,
        num_attention_heads: int = 1, cross_attention_dim: int = 1280,
        merge_factor: float = 0.5
    ):
        super().__init__(
            in_channels, temb_channels, num_layers,
            transformer_layers_per_block, num_attention_heads,
            cross_attention_dim)

        # support for variable transformer layers per block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [
                transformer_layers_per_block] * num_layers

        # there is always at least one resnet
        resnets = [
            dwm.models.crossview_temporal.ResBlock(
                in_channels, in_channels, temb_channels=temb_channels,
                eps=resnet_eps, enable_temporal=enable_temporal,
                merge_factor=merge_factor)
        ]
        attentions = []

        for i in range(num_layers):
            resnets.append(
                dwm.models.crossview_temporal.ResBlock(
                    in_channels, in_channels, temb_channels=temb_channels,
                    eps=resnet_eps, enable_temporal=enable_temporal,
                    merge_factor=merge_factor))
            attentions.append(
                dwm.models.crossview_temporal.TransformerModel(
                    num_attention_heads, in_channels // num_attention_heads,
                    in_channels=in_channels, enable_crossview=enable_crossview,
                    enable_temporal=enable_temporal,
                    enable_rowwise_crossview=enable_rowwise_crossview,
                    enable_rowwise_temporal=enable_rowwise_temporal,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    merge_factor=merge_factor))

        self.resnets = torch.nn.ModuleList(resnets)
        self.attentions = torch.nn.ModuleList(attentions)

    def forward(
        self, hidden_states: torch.Tensor, temb=None,
        encoder_hidden_states=None, disable_crossview=None,
        disable_temporal=None, crossview_attention_mask=None
    ):
        hidden_states = self.resnets[0](
            hidden_states, temb, disable_temporal=disable_temporal)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states, encoder_hidden_states=encoder_hidden_states,
                disable_crossview=disable_crossview,
                disable_temporal=disable_temporal,
                crossview_attention_mask=crossview_attention_mask,
                return_dict=False)[0]
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    resnet, hidden_states, temb, disable_temporal,
                    use_reentrant=False)
            else:
                hidden_states = resnet(
                    hidden_states, temb, disable_temporal=disable_temporal)

        return hidden_states


class DownBlockCrossviewTemporal(
    diffusers.models.unets.unet_3d_blocks.DownBlockSpatioTemporal
):
    def __init__(
        self, in_channels: int, out_channels: int, temb_channels: int,
        enable_temporal: bool = True, num_layers: int = 1,
        resnet_eps: float = 1e-5, add_downsample: bool = True,
        merge_factor: float = 0.5
    ):
        super().__init__(
            in_channels, out_channels, temb_channels, num_layers,
            add_downsample)

        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                dwm.models.crossview_temporal.ResBlock(
                    in_channels, out_channels, temb_channels=temb_channels,
                    eps=resnet_eps, enable_temporal=enable_temporal,
                    merge_factor=merge_factor))

        self.resnets = torch.nn.ModuleList(resnets)

    def forward(
        self, hidden_states: torch.Tensor, temb=None, disable_temporal=None
    ):
        output_states = ()
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    resnet, hidden_states, temb, disable_temporal,
                    use_reentrant=False)
            else:
                hidden_states = resnet(
                    hidden_states, temb, disable_temporal=disable_temporal)

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states.flatten(0, 2))\
                    .unflatten(0, tuple(hidden_states.shape[:3]))

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnDownBlockCrossviewTemporal(
    diffusers.models.unets.unet_3d_blocks.CrossAttnDownBlockSpatioTemporal
):
    def __init__(
        self, in_channels: int, out_channels: int, temb_channels: int,
        enable_crossview: bool = True, enable_temporal: bool = True,
        enable_rowwise_crossview: bool = False,
        enable_rowwise_temporal: bool = False, num_layers: int = 1,
        transformer_layers_per_block=1, resnet_eps: float = 1e-5,
        num_attention_heads: int = 1, cross_attention_dim: int = 1280,
        add_downsample: bool = True, merge_factor: float = 0.5
    ):
        super().__init__(
            in_channels, out_channels, temb_channels, num_layers,
            transformer_layers_per_block, num_attention_heads,
            cross_attention_dim, add_downsample)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = \
                [transformer_layers_per_block] * num_layers

        resnets = []
        attentions = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                dwm.models.crossview_temporal.ResBlock(
                    in_channels, out_channels, temb_channels=temb_channels,
                    eps=resnet_eps, enable_temporal=enable_temporal,
                    merge_factor=merge_factor))
            attentions.append(
                dwm.models.crossview_temporal.TransformerModel(
                    num_attention_heads, out_channels // num_attention_heads,
                    in_channels=out_channels,
                    enable_crossview=enable_crossview,
                    enable_temporal=enable_temporal,
                    enable_rowwise_crossview=enable_rowwise_crossview,
                    enable_rowwise_temporal=enable_rowwise_temporal,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    merge_factor=merge_factor))

        self.resnets = torch.nn.ModuleList(resnets)
        self.attentions = torch.nn.ModuleList(attentions)

    def forward(
        self, hidden_states: torch.Tensor, temb=None,
        encoder_hidden_states=None, disable_crossview=None,
        disable_temporal=None, crossview_attention_mask=None
    ):
        output_states = ()
        blocks = list(zip(self.resnets, self.attentions))
        for resnet, attn in blocks:
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    resnet, hidden_states, temb, disable_temporal,
                    use_reentrant=False)
            else:
                hidden_states = resnet(
                    hidden_states, temb, disable_temporal=disable_temporal)

            hidden_states = attn(
                hidden_states, encoder_hidden_states=encoder_hidden_states,
                disable_crossview=disable_crossview,
                disable_temporal=disable_temporal,
                crossview_attention_mask=crossview_attention_mask,
                return_dict=False)[0]

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states.flatten(0, 2))\
                    .unflatten(0, tuple(hidden_states.shape[:3]))

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class UpBlockCrossviewTemporal(
    diffusers.models.unets.unet_3d_blocks.UpBlockSpatioTemporal
):
    def __init__(
        self, in_channels: int, prev_output_channel: int, out_channels: int,
        temb_channels: int, resolution_idx=None, enable_temporal: bool = True,
        num_layers: int = 1, resnet_eps: float = 1e-5,
        add_upsample: bool = True, merge_factor: float = 0.5
    ):
        super().__init__(
            in_channels, prev_output_channel, out_channels, temb_channels,
            resolution_idx, num_layers, resnet_eps, add_upsample)

        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (
                i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(
                dwm.models.crossview_temporal.ResBlock(
                    resnet_in_channels + res_skip_channels, out_channels,
                    temb_channels=temb_channels, eps=resnet_eps,
                    enable_temporal=enable_temporal,
                    merge_factor=merge_factor))

        self.resnets = torch.nn.ModuleList(resnets)

    def forward(
        self, hidden_states: torch.Tensor, res_hidden_states_tuple: tuple,
        temb=None, disable_temporal=None
    ):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat(
                [hidden_states, res_hidden_states], dim=-3)

            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    resnet, hidden_states, temb, disable_temporal,
                    use_reentrant=False)
            else:
                hidden_states = resnet(
                    hidden_states, temb, disable_temporal=disable_temporal)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states.flatten(0, 2))\
                    .unflatten(0, tuple(hidden_states.shape[:3]))

        return hidden_states


class CrossAttnUpBlockCrossviewTemporal(
    diffusers.models.unets.unet_3d_blocks.CrossAttnUpBlockSpatioTemporal
):
    def __init__(
        self, in_channels: int, out_channels: int, prev_output_channel: int,
        temb_channels: int, resolution_idx=None, enable_crossview: bool = True,
        enable_temporal: bool = True, enable_rowwise_crossview: bool = False,
        enable_rowwise_temporal: bool = False, num_layers: int = 1,
        transformer_layers_per_block=1, resnet_eps: float = 1e-5,
        num_attention_heads: int = 1, cross_attention_dim: int = 1280,
        add_upsample: bool = True, merge_factor: float = 0.5
    ):
        super().__init__(
            in_channels, out_channels, prev_output_channel, temb_channels,
            resolution_idx, num_layers, transformer_layers_per_block,
            resnet_eps, num_attention_heads, cross_attention_dim, add_upsample)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = \
                [transformer_layers_per_block] * num_layers

        resnets = []
        attentions = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (
                i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(
                dwm.models.crossview_temporal.ResBlock(
                    resnet_in_channels + res_skip_channels, out_channels,
                    temb_channels=temb_channels, eps=resnet_eps,
                    enable_temporal=enable_temporal,
                    merge_factor=merge_factor))
            attentions.append(
                dwm.models.crossview_temporal.TransformerModel(
                    num_attention_heads, out_channels // num_attention_heads,
                    in_channels=out_channels,
                    enable_crossview=enable_crossview,
                    enable_temporal=enable_temporal,
                    enable_rowwise_crossview=enable_rowwise_crossview,
                    enable_rowwise_temporal=enable_rowwise_temporal,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    merge_factor=merge_factor))

        self.resnets = torch.nn.ModuleList(resnets)
        self.attentions = torch.nn.ModuleList(attentions)

    def forward(
        self, hidden_states: torch.Tensor, res_hidden_states_tuple: tuple,
        temb=None, encoder_hidden_states=None, disable_crossview=None,
        disable_temporal=None, crossview_attention_mask=None
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat(
                [hidden_states, res_hidden_states], dim=-3)

            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    resnet, hidden_states, temb, disable_temporal,
                    use_reentrant=False)
            else:
                hidden_states = resnet(
                    hidden_states, temb, disable_temporal=disable_temporal)

            hidden_states = attn(
                hidden_states, encoder_hidden_states=encoder_hidden_states,
                disable_crossview=disable_crossview,
                disable_temporal=disable_temporal,
                crossview_attention_mask=crossview_attention_mask,
                return_dict=False)[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states.flatten(0, 2))\
                    .unflatten(0, tuple(hidden_states.shape[:3]))

        return hidden_states


class UNetCrossviewTemporalConditionModel(
    diffusers.UNetSpatioTemporalConditionModel
):
    @staticmethod
    def try_to_convert_state_dict(state_dict: dict):
        sd21_resnet_pattern = re.compile(r"resnets.(\d+).conv")
        is_sd21_checkpoint = len([
            i for i in state_dict.keys()
            if sd21_resnet_pattern.search(i) is not None
        ]) > 0
        if is_sd21_checkpoint:
            pattern = re.compile(r"resnets.(\d+)")
            replace = r"resnets.\1.spatial_res_block"
            return {
                (pattern.sub(replace, k) if "resnets" in k else k): v
                for k, v in state_dict.items()
            }
        else:
            return state_dict

    @diffusers.configuration_utils.register_to_config
    def __init__(
        self, sample_size=None, in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: tuple = (
            "CrossAttnDownBlockCrossviewTemporal",
            "CrossAttnDownBlockCrossviewTemporal",
            "CrossAttnDownBlockCrossviewTemporal",
            "DownBlockCrossviewTemporal",
        ),
        up_block_types: tuple = (
            "UpBlockCrossviewTemporal",
            "CrossAttnUpBlockCrossviewTemporal",
            "CrossAttnUpBlockCrossviewTemporal",
            "CrossAttnUpBlockCrossviewTemporal",
        ),
        block_out_channels: tuple = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim=768,
        layers_per_block=2,
        norm_eps: float = 1e-5, cross_attention_dim: int = 1024,
        transformer_layers_per_block=1,
        num_attention_heads=(5, 10, 20, 20),
        merge_factor: float = 0.5, enable_crossview: bool = True,
        enable_temporal: bool = True,
        enable_rowwise_crossview: bool = False,
        enable_rowwise_temporal: bool = False,
        condition_image_adapter_config=None,
        depth_net_config=None,
        depth_frustum_range=None,
        enforce_align_projection=None
    ):
        super().__init__(
            sample_size, in_channels, out_channels, tuple([
                i.replace("CrossviewTemporal", "SpatioTemporal")
                for i in down_block_types
            ]),
            tuple([
                i.replace("CrossviewTemporal", "SpatioTemporal")
                for i in up_block_types
            ]), block_out_channels, addition_time_embed_dim,
            768 if projection_class_embeddings_input_dim is None
            else projection_class_embeddings_input_dim, layers_per_block,
            cross_attention_dim, transformer_layers_per_block,
            num_attention_heads)

        time_embed_dim = block_out_channels[0] * 4
        self.down_blocks = torch.nn.ModuleList([])
        self.up_blocks = torch.nn.ModuleList([])

        # remove the unused part for better compatibility with DDP model
        # wrapper
        if projection_class_embeddings_input_dim is None:
            self.add_embedding = None

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * \
                len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [
                transformer_layers_per_block] * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type == "CrossAttnDownBlockCrossviewTemporal":
                down_block = CrossAttnDownBlockCrossviewTemporal(
                    in_channels=input_channel, out_channels=output_channel,
                    temb_channels=blocks_time_embed_dim,
                    enable_crossview=enable_crossview,
                    enable_temporal=enable_temporal,
                    enable_rowwise_crossview=enable_rowwise_crossview,
                    enable_rowwise_temporal=enable_rowwise_temporal,
                    num_layers=layers_per_block[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    resnet_eps=norm_eps,
                    num_attention_heads=num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim,
                    add_downsample=not is_final_block,
                    merge_factor=merge_factor)
            elif down_block_type == "DownBlockCrossviewTemporal":
                down_block = DownBlockCrossviewTemporal(
                    in_channels=input_channel, out_channels=output_channel,
                    temb_channels=blocks_time_embed_dim,
                    enable_temporal=enable_temporal,
                    num_layers=layers_per_block[i], resnet_eps=norm_eps,
                    add_downsample=not is_final_block,
                    merge_factor=merge_factor)
            else:
                down_block = diffusers.models.unets.unet_3d_blocks.get_down_block(
                    down_block_type, num_layers=layers_per_block[i],
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    in_channels=input_channel, out_channels=output_channel,
                    temb_channels=blocks_time_embed_dim,
                    add_downsample=not is_final_block, resnet_eps=norm_eps,
                    cross_attention_dim=cross_attention_dim,
                    num_attention_heads=num_attention_heads[i],
                    resnet_act_fn="silu")

            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockCrossviewTemporal(
            block_out_channels[-1], temb_channels=blocks_time_embed_dim,
            enable_crossview=enable_crossview, enable_temporal=enable_temporal,
            enable_rowwise_crossview=enable_rowwise_crossview,
            enable_rowwise_temporal=enable_rowwise_temporal,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            resnet_eps=norm_eps, num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim, merge_factor=merge_factor)

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_transformer_layers_per_block = list(
            reversed(transformer_layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            if up_block_type == "CrossAttnUpBlockCrossviewTemporal":
                up_block = CrossAttnUpBlockCrossviewTemporal(
                    in_channels=input_channel, out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=blocks_time_embed_dim, resolution_idx=i,
                    enable_crossview=enable_crossview,
                    enable_temporal=enable_temporal,
                    enable_rowwise_crossview=enable_rowwise_crossview,
                    enable_rowwise_temporal=enable_rowwise_temporal,
                    num_layers=reversed_layers_per_block[i] + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                    resnet_eps=norm_eps,
                    num_attention_heads=reversed_num_attention_heads[i],
                    cross_attention_dim=cross_attention_dim,
                    add_upsample=add_upsample, merge_factor=merge_factor)
            elif up_block_type == "UpBlockCrossviewTemporal":
                up_block = UpBlockCrossviewTemporal(
                    in_channels=input_channel,
                    prev_output_channel=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=blocks_time_embed_dim, resolution_idx=i,
                    enable_temporal=enable_temporal,
                    num_layers=reversed_layers_per_block[i] + 1,
                    resnet_eps=norm_eps, add_upsample=add_upsample,
                    merge_factor=merge_factor)
            else:
                up_block = diffusers.models.unets.unet_3d_blocks.get_up_block(
                    up_block_type, num_layers=reversed_layers_per_block[i] + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                    in_channels=input_channel, out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=blocks_time_embed_dim,
                    add_upsample=add_upsample, resnet_eps=norm_eps,
                    resolution_idx=i,
                    cross_attention_dim=cross_attention_dim,
                    num_attention_heads=reversed_num_attention_heads[i],
                    resnet_act_fn="silu")

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # Image Adapter
        if condition_image_adapter_config is not None:
            self.condition_image_adapter = dwm.models.adapters.ImageAdapter(
                **condition_image_adapter_config)
        else:
            self.condition_image_adapter = None

        # Depth Net
        self.depth_frustum_range = depth_frustum_range
        self.depth_net = None if depth_net_config is None else \
            dwm.models.depth_net.DepthNet(**depth_net_config)
        self.depth_decoder = None

        if enforce_align_projection is not None:
            self._build_extra_projection(enforce_align_projection)
        else:
            self.align_projection = None

        # TODO: remove debug code
        self.align_projection_debug_print = True

    def _build_extra_projection(self, enforce_align_projection):
        # TODO: support align different module
        modules = {}
        if 'text' in enforce_align_projection:
            norm_layer = torch.nn.LayerNorm(
                enforce_align_projection['text'][1]) \
                if enforce_align_projection['with_norm'] \
                else torch.nn.Identity()
            modules['text'] = torch.nn.Sequential(
                torch.nn.Linear(
                    enforce_align_projection['text'][0],
                    enforce_align_projection['text'][1]), norm_layer)
        if 'dwm_action' in enforce_align_projection:
            norm_layer = torch.nn.LayerNorm(
                enforce_align_projection['dwm_action'][1]) \
                if enforce_align_projection['with_norm'] \
                else torch.nn.Identity()
            modules['dwm_action'] = torch.nn.Sequential(
                torch.nn.Linear(
                    enforce_align_projection['dwm_action'][0],
                    enforce_align_projection['dwm_action'][1]), norm_layer)

        # For cross_att embedding
        self.cross_att_embedding = None

        self.align_projection = torch.nn.ModuleDict(modules)

    def _merge_encoder_hidden_states(self, encoder_hidden_states, dtype):
        rt = []
        # Forward MLP
        for k, v in encoder_hidden_states.items():
            if k in self.align_projection:
                # If dict, preprocess
                if isinstance(v, dict):
                    ft = v['ft']
                    mask = v['mask']
                    nviews = v['nviews']
                    do_cfg = v['do_classifier_free_guidance']
                    ft = self.align_projection[k](ft.to(dtype))
                    if mask is not None:           # This has to be zero
                        for i in range(mask.shape[0]):
                            if not mask[i]:
                                ft[i] = 0
                    if do_cfg:
                        ft = torch.cat(
                            [torch.zeros_like(ft), ft])

                    rt.append(ft)
                else:
                    rt.append(self.align_projection[k](v.to(dtype)))
                if self.align_projection_debug_print:
                    print(f"=== Align with {k}: ", rt[-1].shape)
            else:
                rt.append(v.to(dtype))
                if self.align_projection_debug_print:
                    print(f"=== Not Align with {k}: ", rt[-1].shape)
        self.align_projection_debug_print = False
        if self.cross_att_embedding is not None:
            rt = torch.cat(rt, dim=3)
            rt += self.cross_att_embedding
            return rt
        else:
            return torch.cat(rt, dim=3)

    def forward(
        self, sample: torch.Tensor, timesteps, frustum_bev_residuals=None,
        encoder_hidden_states=None, condition_image_tensor=None,
        disable_crossview=None, disable_temporal=None,
        crossview_attention_mask=None, camera_intrinsics=None,
        camera_transforms=None, added_time_ids=None,
        camera_intrinsics_norm=None, camera2referego= None,
        return_dict=False  # TODO: support view modeling 
    ):
        """The forward method.

        Args:
            sample (`torch.Tensor`): The noisy input tensor with the following
                shape `(batch_size, sequence_length, view_count, in_channels,
                height, width)`.
            timesteps (`torch.Tensor`): The number of timesteps to denoise an
                input following the shape `(batch_size, sequence_length,
                view_count)`.
            encoder_hidden_states (`torch.Tensor`): The encoder hidden states
                with shape `(batch_size, sequence_length, view_count,
                token_count, cross_attention_dim)`.
            condition_image_tensor (`torch.Tensor`): The condition image tensor
                with shape `(batch_size, sequence_length, view_count,
                condition_channels, condition_height, condition_width)`.
            disable_crossview (`torch.BoolTensor`, *optional*, default to None):
                The flags in the shape of `(batch_size, )` to disable the
                cross-view attension result if the item is True.
            disable_temporal (`torch.BoolTensor`, *optional*, default to None):
                The flags in the shape of `(batch_size, )` to disable the
                temporal attension result if the item is True.
            added_time_ids: (`torch.Tensor`): The additional time ids with
                shape `(batch_size, sequence_length, view_count,
                additional_id_count)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.

        Returns:
            A `tuple` is returned where the first element is the sample tensor.
        """
        # If the input does not have the required dimensions, add the missing dimensions
        should_add_dim = len(sample.shape) < 6
        if should_add_dim:
            sample = sample.unsqueeze(2)
            timesteps = timesteps.unsqueeze(2)
            if condition_image_tensor is not None:
                condition_image_tensor = condition_image_tensor.unsqueeze(2)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(2)
            if disable_temporal is not None:
                disable_temporal = disable_temporal.unsqueeze(2)
            
            

        batch_size, sequence_length, view_count, _, height, width = \
            sample.shape

        # Extra-1. merge
        if isinstance(encoder_hidden_states, dict):
            encoder_hidden_states = self._merge_encoder_hidden_states(
                encoder_hidden_states, sample.dtype)
        # 1. time embeddings
        t_emb = self.time_proj(timesteps.flatten()).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb).unflatten(0, timesteps.shape[:3])
        if added_time_ids is not None:
            t_aug_emb = self.add_time_proj(added_time_ids.flatten())\
                .to(dtype=sample.dtype)
            aug_emb = self.add_embedding(
                t_aug_emb.view(batch_size * sequence_length * view_count, -1))
            emb += aug_emb.view(batch_size, sequence_length, view_count, -1)

        # 2. pre-process

        condition_residuals = None if \
            self.condition_image_adapter is None or \
            condition_image_tensor is None else \
            self.condition_image_adapter(condition_image_tensor)

        sample = self.conv_in(sample.flatten(0, 2))\
            .unflatten(0, tuple(sample.shape[:3]))
        depth_net_input_list = [sample]

        if condition_residuals is not None and len(condition_residuals) > 0:
            sample = sample + condition_residuals.pop(0)

        if frustum_bev_residuals is not None:
            sample = sample + frustum_bev_residuals.pop(0)

        maskgit_input_list_down = []
        maskgit_input_list_down.append(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and \
                    downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    sample, emb, encoder_hidden_states=encoder_hidden_states,
                    disable_crossview=disable_crossview,
                    disable_temporal=disable_temporal,
                    crossview_attention_mask=crossview_attention_mask)
            else:
                sample, res_samples = downsample_block(
                    sample, emb, disable_temporal=disable_temporal)

            depth_net_input_list.append(sample)

            if condition_residuals is not None and len(condition_residuals) > 0:
                sample = sample + condition_residuals.pop(0)
                res_samples = res_samples[:-1] + (sample,)

            if frustum_bev_residuals is not None:
                sample = sample + frustum_bev_residuals.pop(0)
                res_samples = res_samples[:-1] + (sample,)

            down_block_res_samples += res_samples

            if len(maskgit_input_list_down) < 3:
                maskgit_input_list_down.append(sample)

        # depth estimation
        if self.depth_net is not None and camera_intrinsics is not None and \
                camera_transforms is not None:
            depth_features = self.depth_net(
                torch.cat([
                    torch.nn.functional
                    .interpolate(i.flatten(0, 2), (height, width))
                    .view(
                        batch_size, sequence_length, view_count, -1, height,
                        width)
                    for i in depth_net_input_list
                ], -3),
                torch.cat([
                    camera_intrinsics.flatten(-2), camera_transforms.flatten(-2)
                ], -1).unsqueeze(-1).unsqueeze(-1))
        else:
            depth_features = None

        # 4. mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states,
            disable_crossview=disable_crossview,
            disable_temporal=disable_temporal,
            crossview_attention_mask=crossview_attention_mask)

        maskgit_input_list_up = []

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(
                upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    sample, res_samples, emb,
                    encoder_hidden_states=encoder_hidden_states,
                    disable_crossview=disable_crossview,
                    disable_temporal=disable_temporal,
                    crossview_attention_mask=crossview_attention_mask)
            else:
                sample = upsample_block(
                    sample, res_samples, emb,
                    disable_temporal=disable_temporal)

            if i != 0:
                maskgit_input_list_up.append(sample)

        # 6. post-process
        sample = self.conv_norm_out(sample.flatten(0, 2))
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        sample = sample.view(
            batch_size, sequence_length, view_count, *sample.shape[1:])

        if should_add_dim:
            sample = sample.squeeze(2)
        result = (sample,)
        if return_dict:
            return {
                "noise_pred": sample,
            }
        if depth_features is not None:
            result = result + (depth_features,)

        if len(maskgit_input_list_up) > 0 and len(maskgit_input_list_down) > 0:
            print(maskgit_input_list_up[0].shape)
            return result, maskgit_input_list_up, maskgit_input_list_down

        return result
