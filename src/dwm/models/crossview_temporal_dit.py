from typing import Optional, Union
import einops
import diffusers
import torch
import math

import dwm.models.adapters
from dwm.models.crossview_temporal import VTSelfAttentionBlock, AlphaBlender, Mixer


class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_octaves=8, start_octave=0):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave

    def forward(self, coords):
        batch_size, num_points, dim = coords.shape

        octaves = torch.arange(
            self.start_octave, self.start_octave + self.num_octaves)
        octaves = octaves.float().to(coords)
        multipliers = 2**octaves * math.pi
        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers

        sines = torch.sin(scaled_coords).reshape(
            batch_size, num_points, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(
            batch_size, num_points, dim * self.num_octaves)

        result = torch.cat((sines, cosines), -1)
        return result


class RayEncoder(torch.nn.Module):
    def __init__(self, pos_octaves=8, pos_start_octave=0,
                 ray_octaves=4, ray_start_octave=0,
                 cond_proj_dim=72, in_channels=1536):
        super().__init__()
        self.pos_encoding = PositionalEncoding(
            num_octaves=pos_octaves, start_octave=pos_start_octave)
        self.ray_encoding = PositionalEncoding(
            num_octaves=ray_octaves, start_octave=ray_start_octave)

        self.proj = torch.nn.Linear(cond_proj_dim, in_channels, bias=False)

    def forward(self, pos, rays):

        batchsize, height, width, dims = rays.shape
        pos_enc = self.pos_encoding(pos.unsqueeze(1))
        pos_enc = pos_enc.view(batchsize, 1, 1, pos_enc.shape[-1])
        pos_enc = pos_enc.repeat(1, height, width, 1)

        rays = rays.flatten(1, 2)
        ray_enc = self.ray_encoding(rays)
        ray_enc = ray_enc.view(batchsize, height, width, ray_enc.shape[-1])
        x = torch.cat((pos_enc, ray_enc), -1)

        return self.proj(x)


def get_rays(
        camera_intrinsics: torch.tensor,
        camera_transforms: torch.tensor,
        target_size: Union[int, tuple[int]],):
    ''' get rays
        Args:
            camera_transforms: (B*T*V, 4, 4), cam2world
            intrinsics: (B*T*V, 3, 3)
        Returns:
            rays_o: [B*T*V, 3]
            rays_d: [B*T*V, H, W, 3]
    '''
    device = camera_transforms.device
    dtype = camera_transforms.dtype
    camera_transforms = camera_transforms.to(dtype=torch.float32)
    camera_intrinsics = camera_intrinsics.to(dtype=torch.float32)

    H, W = (target_size, target_size) if isinstance(
        target_size, int) else target_size
    i, j = torch.meshgrid(torch.linspace(
        0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')

    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5

    zs = torch.ones_like(i)
    points_coord = torch.stack([i, j, zs])  # (H*W, 3)
    directions = torch.inverse(
        camera_intrinsics) @ points_coord.unsqueeze(0)  # (batch_size, 3, H*W)

    rays_d = camera_transforms[:, :3, :3] @ directions  # (batch_size, 3, H*W)
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d.transpose(1, 2).view(-1, H, W, 3).to(dtype=dtype)

    rays_o = camera_transforms[:, :3, 3].to(dtype=dtype)  # (batch_size, 3)

    return rays_o, rays_d


class DiTCrossviewTemporalConditionModel(diffusers.SD3Transformer2DModel):
    @diffusers.configuration_utils.register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        projection_class_embeddings_input_dim: int = None,
        condition_image_adapter_config: Optional[dict] = None,
        enable_crossview: bool = False,
        enable_temporal: bool = False,
        crossview_attention_type: str = None,
        temporal_attention_type: str = None,
        merge_factor: float = 2, merge_strategy="learned_with_images",
        crossview_block_layers: Optional[dict] = None,
        temporal_block_layers: Optional[dict] = None,
        crossview_gradient_checkpointing: bool = False,
        temporal_gradient_checkpointing: bool = False,
        mixer_type: str = "AlphaBlender",
        perspective_modeling_type: str = "",
        disable_view_emb_on_temporal_module: bool = False,
        qk_norm_on_additional_modules=None,
        mask_module=None,
        **kwargs
    ):
        super().__init__(
            patch_size=patch_size,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            **kwargs
        )
        self.crossview_gradient_checkpointing = crossview_gradient_checkpointing
        self.temporal_gradient_checkpointing = temporal_gradient_checkpointing
        self.disable_view_emb_on_temporal_module = disable_view_emb_on_temporal_module

        # image condition adapter
        if condition_image_adapter_config is not None:
            self.condition_image_adapter = \
                dwm.models.adapters.ImageAdapter(
                    **condition_image_adapter_config
                )
        else:
            self.condition_image_adapter = None

        # views and frames index embedding
        inner_dim = attention_head_dim * num_attention_heads
        self.index_proj = diffusers.models.embeddings.Timesteps(
            inner_dim, True, 0)

        self.perspective_modeling_type = perspective_modeling_type
        if perspective_modeling_type == "explicit":
            # Explicit Perspective Modeling
            self.rayencoder = RayEncoder(
                cond_proj_dim=72, in_channels=self.inner_dim)   
        elif perspective_modeling_type == "implicit":
            # Implicit Perspective Modeling
            self.view_cam_proj = diffusers.models.embeddings.Timesteps(
                num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.view_embedding = diffusers.models.embeddings.TimestepEmbedding(
                in_channels=projection_class_embeddings_input_dim, 
                time_embed_dim=self.inner_dim)  

        self.enable_crossview = enable_crossview
        self.crossview_attention_type = crossview_attention_type
        self.crossview_block_layers = crossview_block_layers
        if enable_crossview:
            self.view_pos_embeds = torch.nn.ModuleList([
                diffusers.models.embeddings.TimestepEmbedding(
                    inner_dim, inner_dim * 4, out_dim=inner_dim)
                for _ in range(len(crossview_block_layers))
            ])

            self.crossview_transformer_blocks = torch.nn.ModuleList([
                VTSelfAttentionBlock(
                    inner_dim, inner_dim, num_attention_heads,
                    attention_head_dim, qk_norm=qk_norm_on_additional_modules)
                for _ in range(len(crossview_block_layers))
            ])

            self.view_mixers = torch.nn.ModuleList([
                AlphaBlender(merge_factor, merge_strategy=merge_strategy)
                if mixer_type == "AlphaBlender" else
                Mixer(channel=inner_dim)
                for _ in range(len(crossview_block_layers))
            ])

        self.enable_temporal = enable_temporal
        self.temporal_attention_type = temporal_attention_type
        self.temporal_block_layers = temporal_block_layers
        if enable_temporal:
            self.time_pos_embeds = torch.nn.ModuleList([
                diffusers.models.embeddings.TimestepEmbedding(
                    inner_dim, inner_dim * 4, out_dim=inner_dim)
                for _ in range(len(temporal_block_layers))
            ])

            self.temporal_transformer_blocks = torch.nn.ModuleList([
                VTSelfAttentionBlock(
                    inner_dim, inner_dim, num_attention_heads,
                    attention_head_dim, qk_norm=qk_norm_on_additional_modules)
                for _ in range(len(temporal_block_layers))
            ])

            self.time_mixers = torch.nn.ModuleList([
                AlphaBlender(merge_factor, merge_strategy=merge_strategy)
                if mixer_type == "AlphaBlender" else
                Mixer(channel=inner_dim)
                for _ in range(len(temporal_block_layers))
            ])

        # Depth Net 
        self.depth_net = None   # TODO: support joint training of image and lidar

        # Mask Reconstruction
        self.mask_module = mask_module

    def forward_crossview_block_and_mix_result(
        self, crossview_block: torch.nn.Module, mixer, hidden_states: torch.Tensor,
        view_emb: torch.Tensor, batch_size: int, sequence_length: int,
        view_count: int, width: int, height: int, disable_crossview: torch.BoolTensor,
        crossview_attention_mask, crossview_attention_index
    ):
        crossview_hidden_states = hidden_states + \
            view_emb      # [b*T*V, h*w, c]
        if self.crossview_attention_type == "fuse":
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(bt v) hw c -> bt v hw c", v=view_count)
            crossview_attention_index = crossview_attention_index.unsqueeze(1).unsqueeze(
                -1).unsqueeze(-1).expand(-1, sequence_length, -1,
                                         crossview_hidden_states.shape[-2],
                                         crossview_hidden_states.shape[-1]).flatten(0, 1)
            crossview_hidden_states = torch.gather(
                crossview_hidden_states, 1, crossview_attention_index)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(b t) (v n) hw c -> (b v) (t n hw) c",
                v=view_count, n=3, t=sequence_length)
            crossview_hidden_states = crossview_block(
                crossview_hidden_states,
                self_attention_mask=crossview_attention_mask)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(b v) (t n hw) c -> (b t v) n hw c",
                v=view_count, t=sequence_length, n=3)
            crossview_hidden_states = \
                crossview_hidden_states[:, 1, :, :]

        elif self.crossview_attention_type == "adj_fuse":
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(b t v) hw c -> b t v hw c",
                v=view_count, t=sequence_length)

            adj_frame_index = torch.cat(
                (torch.arange(view_count).unsqueeze(0),
                 torch.arange(view_count).unsqueeze(0)))
            adj_frame_index[0, 1:] -= 1
            adj_frame_index = adj_frame_index.t().flatten()
            adj_frame_index = adj_frame_index.unsqueeze(0).unsqueeze(-1).unsqueeze(
                -1).unsqueeze(-1).expand(batch_size, -1, view_count,
                                         crossview_hidden_states.shape[-2],
                                         crossview_hidden_states.shape[-1]).to(
                                             crossview_attention_index)
            crossview_hidden_states = torch.gather(
                crossview_hidden_states, 1, adj_frame_index)

            crossview_attention_index = crossview_attention_index.unsqueeze(1).unsqueeze(
                -1).unsqueeze(-1).expand(-1, sequence_length*2, -1,
                                         crossview_hidden_states.shape[-2],
                                         crossview_hidden_states.shape[-1])
            crossview_hidden_states = torch.gather(
                crossview_hidden_states, 2, crossview_attention_index)

            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "b (t l) (v n) hw c -> (b t v) (l n hw) c",
                v=view_count, t=sequence_length, n=3, l=2)
            crossview_hidden_states = crossview_block(
                crossview_hidden_states,
                self_attention_mask=crossview_attention_mask)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(b t v) (l n hw) c -> (b t v) l n hw c",
                v=view_count, t=sequence_length, n=3, l=2)
            crossview_hidden_states = \
                crossview_hidden_states[:, 1, 1, :, :]

        elif self.crossview_attention_type == "full":
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(bt v) (h w) c -> bt (h v w) c",
                v=view_count, w=width)
            crossview_hidden_states = crossview_block(
                crossview_hidden_states,
                self_attention_mask=crossview_attention_mask)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "bt (h v w) c -> (bt v) (h w) c",
                v=view_count, w=width)

        elif self.crossview_attention_type == "rowwise":
            if crossview_attention_mask is not None:
                crossview_attention_mask = crossview_attention_mask\
                    .repeat_interleave(width, 2)\
                    .repeat_interleave(width, 1)\
                    .repeat_interleave(sequence_length*height, 0)

            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(bt v) (h w) c -> (bt h) (v w) c",
                w=width, v=view_count)
            crossview_hidden_states = crossview_block(
                crossview_hidden_states,
                self_attention_mask=crossview_attention_mask)
            crossview_hidden_states = einops.rearrange(
                crossview_hidden_states, "(bt h) (v w) c -> (bt v) (h w) c",
                bt=batch_size*sequence_length, v=view_count)

        else:
            raise f"Not support {self.crossview_attention_type}"

        return mixer(
            hidden_states.view(
                batch_size, sequence_length * view_count,
                *hidden_states.shape[1:]),
            crossview_hidden_states.view(
                batch_size, sequence_length * view_count,
                *crossview_hidden_states.shape[1:]),
            image_only_indicator=disable_crossview).flatten(0, 1)

    def forward_temporal_block_and_mix_result(
        self, temporal_block: torch.nn.Module, mixer, hidden_states: torch.Tensor,
        sequence_emb: torch.Tensor, batch_size: int, sequence_length: int,
        view_count: int, width: int, disable_temporal: torch.BoolTensor
    ):
        temporal_hidden_states = hidden_states + sequence_emb
        if self.temporal_attention_type == "full":
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b t v) hw c -> (b v) (t hw) c",
                b=batch_size, t=sequence_length)
            temporal_hidden_states = temporal_block(
                temporal_hidden_states)
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b v) (t hw) c -> (b t v) hw c",
                b=batch_size, t=sequence_length)
        elif self.temporal_attention_type == "rowwise":
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b t v) (h w) c -> (b v h) (t w) c",
                b=batch_size, v=view_count, w=width)
            temporal_hidden_states = temporal_block(
                temporal_hidden_states)
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b v h) (t w) c -> (b t v) (h w) c",
                b=batch_size, v=view_count, w=width)
        else:
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b t v) hw c -> (b v hw) t c",
                b=batch_size, t=sequence_length)
            temporal_hidden_states = temporal_block(
                temporal_hidden_states)
            temporal_hidden_states = einops.rearrange(
                temporal_hidden_states, "(b v hw) t c -> (b t v) hw c",
                b=batch_size, v=view_count, t=sequence_length)

        return mixer(
            hidden_states.view(
                batch_size, sequence_length * view_count,
                *hidden_states.shape[1:]),
            temporal_hidden_states.view(
                batch_size, sequence_length * view_count,
                *temporal_hidden_states.shape[1:]),
            image_only_indicator=disable_temporal).flatten(0, 1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.LongTensor = None,
        frustum_bev_residuals: torch.Tensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        condition_image_tensor: torch.Tensor = None,
        disable_crossview: torch.BoolTensor = None,
        disable_temporal: torch.BoolTensor = None,
        crossview_attention_mask: torch.Tensor = None,
        crossview_attention_index: torch.Tensor = None,
        camera_intrinsics: torch.Tensor = None,
        camera_transforms: torch.Tensor = None,   
        camera_intrinsics_norm: torch.Tensor = None,
        camera2referego: torch.Tensor = None,
        added_time_ids: torch.Tensor = None,
        noise: torch.Tensor = None,
        return_dict: bool = False
    ):
        should_add_dim = len(sample.shape) < 6
        if should_add_dim:
            sample = sample.unsqueeze(2)
            timestep = timestep.unsqueeze(2)
            if condition_image_tensor is not None:
                condition_image_tensor = condition_image_tensor.unsqueeze(2)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(2)
            if disable_temporal is not None:
                disable_temporal = disable_temporal.unsqueeze(2)
            if pooled_projections is not None:
                pooled_projections = pooled_projections.unsqueeze(2)

        hidden_states = sample
        batch_size, sequence_length, view_count, _, height, width = \
            hidden_states.shape

        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        self.view_count = view_count
        self.width = width

        hidden_states = hidden_states.flatten(0, 2)     # [b, 16, 32, 56]
        pooled_projections = pooled_projections.flatten(0, 2)
        encoder_hidden_states = encoder_hidden_states.flatten(0, 2)


        hidden_states = self.pos_embed(hidden_states)   # [b, 448, 1536]    
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states)    # [b, 154, 1536]
        if self.mask_module is not None and noise is not None:
            y_t = encoder_hidden_states
            y_t = einops.rearrange(
                encoder_hidden_states, 
                "(b t v) s c -> b v t s c", t=sequence_length, v=view_count)
            y_lens = [y_t.shape[1]] * y_t.shape[0]

        temb = self.time_text_embed(timestep.flatten(), pooled_projections)

        view_cam_emb = 0
        if self.perspective_modeling_type == "implicit":
            view_emb = self.view_cam_proj(added_time_ids.flatten())\
                .to(dtype=hidden_states.dtype)
            view_cam_emb = self.view_embedding(
                view_emb.view(batch_size * sequence_length * view_count, -1))\
                .unsqueeze(1)
        elif self.perspective_modeling_type == "explicit":
            camera_intrinsics_norm = camera_intrinsics_norm.clone()
            camera_intrinsics_norm[..., 0, 0] = \
                camera_intrinsics_norm[..., 0, 0] * width
            camera_intrinsics_norm[..., 1, 1] = \
                camera_intrinsics_norm[..., 1, 1] * height
            camera_intrinsics_norm[..., 0, 2] = \
                camera_intrinsics_norm[..., 0, 2] * width
            camera_intrinsics_norm[..., 1, 2] = \
                camera_intrinsics_norm[..., 1, 2] * height

            rays_o, rays_d = get_rays(
                camera_intrinsics_norm.flatten(0, 2),
                camera2referego.flatten(0, 2),
                (height, width)
            )
            raymap = self.rayencoder(rays_o, rays_d)
            view_cam_emb = raymap.flatten(1, 2)

        condition_residuals = None if \
            self.condition_image_adapter is None or \
            condition_image_tensor is None else \
            self.condition_image_adapter(condition_image_tensor)

        if self.mask_module is not None and noise is not None:
            hidden_states = einops.rearrange(hidden_states, "(b t v) hw c -> (b v) t hw c", 
                t=sequence_length, v=view_count)
            noise = einops.rearrange(noise, "b t v c h w -> (b v) c t h w")
            hidden_states, mask_metas, condition_residuals = self.mask_module.random_masking(
                hidden_states, noise, height, width, timestep, condition_residuals=condition_residuals)
            hidden_states = einops.rearrange(hidden_states, "(b v) t hw c -> (b t v) hw c", 
                t=sequence_length, v=view_count)
            ori_width = width
            width = int(width*(1-self.mask_module.mask_ratio))

        for i, block in enumerate(self.transformer_blocks):
            if self.mask_module is not None and noise is not None and\
                self.mask_module.is_first_decoder_layer(
                    i, len(self.transformer_blocks)):
                hidden_states = einops.rearrange(hidden_states, 
                    "(b t v) hw c -> (b v t) hw c", t=sequence_length, v=view_count)
                temb_v_first = einops.rearrange(temb, 
                    "(b t v) c -> (b v t) c", t=sequence_length, v=view_count)
                hidden_states = self.mask_module.mask_reconstruction(
                    hidden_states, mask_metas, ori_shape=(batch_size*view_count, 
                    sequence_length, height, ori_width), y_t=y_t, y_lens=y_lens, 
                    temb=temb_v_first)
                hidden_states = einops.rearrange(hidden_states, 
                    "(b v t) hw c -> (b t v) hw c", t=sequence_length, v=view_count)
                width = int(width/(1-self.mask_module.mask_ratio))

            if condition_residuals is not None and len(condition_residuals) > 0:
                hidden_states = hidden_states + \
                    condition_residuals.pop(0).flatten(0, 2)\
                    .flatten(2).permute(0, 2, 1)    

            # text-spatio
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                encoder_hidden_states, hidden_states = \
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        use_reentrant=False
                    )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb
                )

            if encoder_hidden_states is not None:
                self.encoder_hidden_states_var = torch.var(encoder_hidden_states).item()

            # temporal
            if self.enable_temporal and i in self.temporal_block_layers:
                sequence_emb = torch\
                    .arange(sequence_length, device=hidden_states.device)\
                    .unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, view_count)
                sequence_emb = self.index_proj(sequence_emb.flatten())\
                    .to(dtype=hidden_states.dtype)
                sequence_emb = self.time_pos_embeds[
                    self.temporal_block_layers.index(i)](sequence_emb).unsqueeze(1)

                if self.enable_crossview and not self.disable_view_emb_on_temporal_module:
                    sequence_emb = sequence_emb + view_cam_emb

                if self.training and self.temporal_gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        self.forward_temporal_block_and_mix_result,
                        self.temporal_transformer_blocks[
                            self.temporal_block_layers.index(i)],
                        self.time_mixers[self.temporal_block_layers.index(i)],
                        hidden_states, sequence_emb,
                        batch_size, sequence_length, view_count, width,
                        disable_temporal, use_reentrant=False)
                else:
                    hidden_states = self.forward_temporal_block_and_mix_result(
                        self.temporal_transformer_blocks[
                            self.temporal_block_layers.index(i)],
                        self.time_mixers[self.temporal_block_layers.index(i)],
                        hidden_states, sequence_emb,
                        batch_size, sequence_length, view_count, width,
                        disable_temporal)

            # cross-view
            if self.enable_crossview and i in self.crossview_block_layers:
                view_emb = torch\
                    .arange(view_count, device=hidden_states.device)\
                    .unsqueeze(0).unsqueeze(0)\
                    .repeat(batch_size, sequence_length, 1)
                view_emb = self.index_proj(view_emb.flatten())\
                    .to(dtype=hidden_states.dtype)
                view_emb = self.view_pos_embeds[
                    self.crossview_block_layers.index(i)](view_emb).unsqueeze(1)

                view_emb = view_emb + view_cam_emb

                if self.training and self.crossview_gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        self.forward_crossview_block_and_mix_result,
                        self.crossview_transformer_blocks[
                            self.crossview_block_layers.index(i)],
                        self.view_mixers[self.crossview_block_layers.index(i)]
                        if self.view_mixers is not None else None,
                        hidden_states, view_emb, batch_size,
                        sequence_length, view_count, width, height, disable_crossview,
                        crossview_attention_mask,
                        crossview_attention_index,
                        use_reentrant=False)
                else:
                    hidden_states = self.forward_crossview_block_and_mix_result(
                        self.crossview_transformer_blocks[
                            self.crossview_block_layers.index(i)],
                        self.view_mixers[self.crossview_block_layers.index(
                            i)]
                        if self.view_mixers is not None else None,
                        hidden_states, view_emb,
                        batch_size, sequence_length, view_count, width, height,
                        disable_crossview, crossview_attention_mask,
                        crossview_attention_index)

        # debug code
        self.hidden_states_var = hidden_states.var().item()
        if self.enable_temporal:
            self.temporal_embedding_var = sequence_emb.var().item()

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                batch_size, sequence_length, view_count,
                self.out_channels,
                height * patch_size,
                width * patch_size,
            )
        )

        result = [output,]
        if should_add_dim:
            output = output.squeeze(2)
        if return_dict:
            return {
                "noise_pred": output,
            }
        return result, _, _
