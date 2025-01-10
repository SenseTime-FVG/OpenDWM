# from image_model import UNetCrossviewTemporalConditionModel
# from lidar_model import UnetTransformerAlignV2
from transformers.models.x_clip.modeling_x_clip import x_clip_loss
import dwm.functional
from einops import rearrange, repeat
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
import torch
import torch.nn.functional as F

class IAModule(torch.nn.Module):
    def __init__(self, ia_type, 
        ia_module, bevformer_fix_ratio=False,           # bevformer
        depth_frustum_range=None
        ) -> None:
        super().__init__()
        self.ia_module = ia_module
        self.ia_type = ia_type
        if ia_type == 'bevformer':
            self.bevformer_fix_ratio = bevformer_fix_ratio
            self.bev_proj = torch.nn.Linear(self.ia_module.embed_dims, 512)
        else:
            pass
        self.depth_frustum_range = depth_frustum_range

    def _bevformer(self, ia_context, ia_list,
        lidar2img, img_shape, can_bus):
        ctsd_features_list = [ia_context[v] for v in ia_list]

        ctsd_features_list_new = []

        for feat in ctsd_features_list:
            # new_order = [3,5,4,0,1,2] # point -> bevformer
            # new_order = [4,5,3,1,2,0] # ctsd -> bevformer # TODO need check the order
            new_order = [1,2,0,4,5,3]
            feat = torch.index_select(feat, dim=2, index=torch.tensor(new_order).to(feat.device))
            feat = feat[:, 0].flatten(0, 1)
            ctsd_features_list_new.append(feat)

        bev_meta = {
            'lidar2img': lidar2img,
            'img_shape': img_shape,
            'can_bus': can_bus
        }

        bev_embed = self.ia_module(
            ctsd_features_list_new, 
            bev_meta,
            prev_bev=None,
            img_feats_resnet=None
        )

        learnable_bev_features = self.bev_proj(bev_embed)

        if self.ia_module.type == 'base' or self.ia_module.type == 'unet' or self.ia_module.type == 'unet-down' or self.ia_module.type == 'unet-down-resnet':
            h = 200
        elif self.ia_module.type == 'tiny':
            h = 50

        maskgit_feature = rearrange(learnable_bev_features, 'b (h w) d -> b d h w', h=h)
        maskgit_feature = F.interpolate(maskgit_feature,
                            size=(80, 80), # (80,80) xiaodong (128, 128) UnetMaskgit
                            mode='bilinear',
                            align_corners=False)
        maskgit_feature = rearrange(maskgit_feature, 'b d h w -> b (h w) d')
        
        return maskgit_feature

    def _cross_modal_proj(self, sample, bev_features, 
        last_step_depth_features, 
        bev_from_frustum_transform, 
        pos_based_module, 
        timesteps=None, 
        interaction_condition_mask=None):
        """
        params:
            sample: 2d feature
            bev_features: 3d feature
            last_step_depth_features: depth prob
        """
        assert self.depth_frustum_range is not None and \
            last_step_depth_features is not None and \
            bev_from_frustum_transform is not None
        batch_size, sequence_length, view_count, _, height, width = \
            sample.shape
        # if '3d_bev_align' in self.align_projection:
        #     bev_features = self.align_projection['3d_bev_align'](bev_features)
        frustum = dwm.functional\
            .create_frustum(
                self.depth_frustum_range, height, width, sample.device)\
            .unsqueeze(0)
        frustum_bev_coordinates = \
            (bev_from_frustum_transform @ frustum.flatten(-3))\
            .transpose(-2, -1)[..., :2]

        frustum_bev_coordinates = frustum_bev_coordinates\
            .view(
                batch_size, sequence_length, view_count,
                *frustum.shape[-3:], -1)

        # sampled LSS feature shape [B * T, C, D * H, V * W]  
        cross_modal_features = pos_based_module(
            bev_features, sample, frustum_bev_coordinates, 
            last_step_depth_features=last_step_depth_features,
            timesteps=timesteps
        )
        if interaction_condition_mask is not None:
            if cross_modal_features.ndim == 3:          # 2d->3d
                interaction_condition_mask = interaction_condition_mask[:, None, None, None].repeat(1, sequence_length, 1, 1).flatten(0, 1)
            else:           # 3d->2d
                interaction_condition_mask = interaction_condition_mask.reshape(
                    list(interaction_condition_mask.shape[:1]) + [1]*(cross_modal_features.ndim-1))
            cross_modal_features = cross_modal_features*interaction_condition_mask

        return cross_modal_features

    def forward(self, x, ia_context, ia_list, metas, target='2d'):
        assert target in ['2d', '3d']
        if self.ia_type == 'bevformer':
            assert target == '3d'
            return self._bevformer(ia_context, ia_list, **metas)
        else:
            residual = 0
            for i, vid in enumerate(ia_list):
                x_2d = x if target == '2d' else ia_context[vid]
                x_3d = x if target == '3d' else ia_context[vid]
                cross_ft = self._cross_modal_proj(x_2d, x_3d, 
                    pos_based_module=self.ia_module[i], **metas)

                residual += cross_ft           # residual
            return residual


class BlockWiseHoloDrive(torch.nn.Module):

    def __init__(
        self, ctsd, maskgit, ia_modules, 
        forward_depth_recipe, forward_2d_3d_recipe,
        bevformer_id=None, bevformer_fix_ratio=False,
    ):
        super().__init__()

        self.ctsd = ctsd
        self.maskgit = maskgit
        if ia_modules is None:
            self.ia_modules = None
        else:
            for k in ia_modules:
                ia_modules[k] = torch.nn.ModuleList(ia_modules[k])
                ia_modules[k] = IAModule(k, ia_modules[k], 
                    bevformer_fix_ratio=bevformer_fix_ratio,
                    depth_frustum_range=self.ctsd.depth_frustum_range)
            self.ia_modules = torch.nn.ModuleDict(ia_modules)
        self.forward_depth_recipe = forward_depth_recipe
        self.forward_2d_3d_recipe = forward_2d_3d_recipe
        if bevformer_id is not None:
            self.bevformer = self.ia_modules[bevformer_id]          # for load & judge in pipeline
        else:
            self.bevformer = None
        self.lss = None
        # TODO: support ia_modules set trainable

        if self.maskgit is not None and self.ia_modules is not None:            # for compatible with bevformer
            self.maskgit_cond_proj = torch.nn.Linear(4096, 512)

    def _get_safety_auto_wrap_policy(self):
        def custom_auto_wrap_policy(
            module: torch.nn.Module,
            recurse: bool,
            nonwrapped_numel: int,
            ) -> bool:
            if recurse:
                return True         # iter all modules
            if any([x in module.__class__.__name__ for x in [
                'CrossAttnDownBlockCrossviewTemporal',
                'DownBlockCrossviewTemporal'
            ]]):
                print(list(module.named_children()))
                print("Set to FSDP: {}".format(type(module)))
                return True
            else:
                return False
        return custom_auto_wrap_policy

    def _get_ignored_module(self):
        return set([self.maskgit, self.ctsd.depth_net])

    def forward(
        self, 
        sample, code,
        timesteps,
        ctsd_conditions, maskgit_conditions,
        maskgit_action, attention_mask_temporal,
        ctsd_features= None, maskgit_features= None, # lidar2img_features
        last_step_depth_features=None, # NOTE = last_step_depth_features
        bev_from_frustum_transform=None,
        bev_specification=None,
        return_ctsd_feature=True,
        return_maskgit_feature=True,
        lidar2img=None,
        img_shape=None,
        can_bus=None,
        interaction_condition_mask=None,
        forward_depth=False,
    ):
        ia_context = dict()
        if forward_depth:
            forward_2d, forward_3d = True, False
        else:
            forward_2d, forward_3d = self.ctsd is not None, self.maskgit is not None
        # == prepare ==
        # a. 3D
        if forward_3d:
            sequence_length = self.maskgit.num_frames
            x, action, context = code, maskgit_action, maskgit_conditions["context"]
            if self.maskgit.cross_attend:
                context = self.maskgit_cond_proj(context)
            else:
                assert context is None
            forward_depth_features = None
            if bev_specification is not None:
                bev_shape = bev_specification['shape']
            res_3d, res_3d_context = (), ()
        # b. 2D
        if forward_2d:
            batch_size, sequence_length, view_count, _, height, width = \
                sample.shape
            condition_residuals = []
            depth_net_input_list = []
            # Extra-1. merge
            encoder_hidden_states = ctsd_conditions.get('encoder_hidden_states', None) 
            if isinstance(encoder_hidden_states, dict):
                encoder_hidden_states = self.ctsd._merge_encoder_hidden_states(
                    encoder_hidden_states, sample.dtype)
            # 1. 2d time embeddings
            t_emb = self.ctsd.time_proj(timesteps.flatten())\
                .to(dtype=sample.dtype)
            emb = self.ctsd.time_embedding(t_emb)\
                .unflatten(0, timesteps.shape[:3])
            added_time_ids = ctsd_conditions.get('added_time_ids', None)
            condition_image_tensor = ctsd_conditions.get('condition_image_tensor', None)
            disable_crossview = ctsd_conditions.get('disable_crossview', None) 
            disable_temporal = ctsd_conditions.get('disable_temporal', None) 
            if added_time_ids is not None:
                t_aug_emb = self.ctsd.add_time_proj(added_time_ids.flatten())\
                    .to(dtype=sample.dtype)
                aug_emb = self.ctsd.add_embedding(
                    t_aug_emb.view(batch_size * sequence_length * view_count, -1))
                emb += aug_emb.view(batch_size, sequence_length, view_count, -1)

            # 2. 2d pre-process
            condition_residuals = None if \
                self.ctsd.condition_image_adapter is None or \
                condition_image_tensor is None else \
                self.ctsd.condition_image_adapter(condition_image_tensor)

            sample = self.ctsd.conv_in(sample.flatten(0, 2))\
                .view(batch_size, sequence_length, view_count, -1, height, width)
            depth_net_input_list = [sample]
            if condition_residuals is not None and len(condition_residuals) > 0:
                sample = sample + condition_residuals.pop(0)
            down_block_res_samples = (sample,)

            # 3. ia pre-interactions
            metas_bevformer = dict(lidar2img=lidar2img, img_shape=img_shape, can_bus=can_bus)
            metas_general_lss = dict(last_step_depth_features=last_step_depth_features, 
                bev_from_frustum_transform=bev_from_frustum_transform, timesteps=timesteps,
                interaction_condition_mask=interaction_condition_mask)
        # == prepare ==
        forward_recipe = self.forward_depth_recipe if forward_depth else self.forward_2d_3d_recipe
        for rid, meta in enumerate(forward_recipe):
            mtype, ptype, lid, ia_id, ia_list = meta.split('_')
            if lid != 'none':
                lid = int(lid)
            if mtype == '2d':
                ia_list = list( map(lambda x: f'3d-{x}', ia_list.split(',')))
                # 2d is shared for spatial and temporal, so directly write here
                if ptype == 'd':
                    downsample_block = self.ctsd.down_blocks[lid]
                    if hasattr(downsample_block, "has_cross_attention") and \
                            downsample_block.has_cross_attention:
                        sample, res_samples = downsample_block(
                            sample, emb, encoder_hidden_states=encoder_hidden_states,
                            disable_crossview=disable_crossview,
                            disable_temporal=disable_temporal)
                    else:
                        sample, res_samples = downsample_block(
                            sample, emb, disable_temporal=disable_temporal)

                    depth_net_input_list.append(sample)

                    if condition_residuals is not None and len(condition_residuals) > 0:
                        sample = sample + condition_residuals.pop(0)
                        # res_samples = res_samples[:-1] + (sample,)
                elif ptype == 'mid':
                    sample = self.ctsd.mid_block(
                        sample, emb, encoder_hidden_states=encoder_hidden_states,
                        disable_crossview=disable_crossview,
                        disable_temporal=disable_temporal)
                elif ptype == 'u':
                    upsample_block = self.ctsd.up_blocks[lid]
                    res_samples = down_block_res_samples[-len(upsample_block.resnets):]
                    down_block_res_samples = down_block_res_samples[: -len(
                        upsample_block.resnets)]

                    if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                        sample = upsample_block(
                            sample, res_samples, emb,
                            encoder_hidden_states=encoder_hidden_states,
                            disable_crossview=disable_crossview,
                            disable_temporal=disable_temporal)
                    else:
                        sample = upsample_block(
                            sample, res_samples, emb,
                            disable_temporal=disable_temporal)
                elif ptype == 'depth':
                    if self.ctsd.depth_net is not None and ctsd_conditions['camera_intrinsics'] is not None and \
                            ctsd_conditions['camera_transforms'] is not None:
                        forward_depth_features = self.ctsd.depth_net(
                            torch.cat([
                                torch.nn.functional
                                .interpolate(i.flatten(0, 2), (height, width))
                                .view(
                                    batch_size, sequence_length, view_count, -1, height,
                                    width)
                                for i in depth_net_input_list
                            ], -3),
                            torch.cat([
                                ctsd_conditions['camera_intrinsics'].flatten(-2), ctsd_conditions['camera_transforms'].flatten(-2)
                            ], -1).unsqueeze(-1).unsqueeze(-1))
                elif ptype == 'none':
                    pass
                else:
                    raise NotImplementedError(f"Not support {ptype}.")

                # TODO: support various orders
                if self.ia_modules is not None:
                    ia_context[f'{mtype}-{ptype}-{lid}'] = sample
                # now, interaction last
                if ia_id != 'none':
                    sample += self.ia_modules[ia_id](sample, ia_context, ia_list, metas_general_lss, '2d')

                if ptype == 'd':
                    res_samples = res_samples[:-1] + (sample,)
                    down_block_res_samples += res_samples
            else:
                ia_list = list(map(lambda x: f'2d-{x}', ia_list.split(',')))
                # 3d is more complex, so forward one layer is implemented innerly
                if ptype == 'd':
                    res_3d += (x, )
                    res_3d_context += (context, )
                    x, context = self.maskgit._forward_down(x, lid, context, action, 
                        attention_mask_temporal, sequence_length)
                elif ptype == 'mid':
                    x, context = self.maskgit._forward_mid(x, lid, context, action, 
                        attention_mask_temporal, sequence_length)
                elif ptype == 'u':
                    x, context = self.maskgit._forward_up(x, lid, context, res_3d, res_3d_context,
                        attention_mask_temporal, sequence_length)
                elif ptype == 'none':           # for interaction
                    pass
                elif ptype == 'pre':            # for support bevformer
                    x, context, action = self.maskgit._forward_prepare(x, context, action)
                else:
                    raise NotImplementedError(f"Not support {ptype}.")

                # TODO: support various orders
                if self.ia_modules is not None:
                    y2x = bev_shape[0]/bev_shape[1]
                    size_x = int((x.shape[1]*y2x)**0.5)
                    x_expand = x.reshape(batch_size, sequence_length, size_x, int(size_x*y2x), x.shape[-1])
                    ia_context[f'{mtype}-{ptype}-{lid}'] = x_expand
                # now, interaction last
                if ia_id != 'none':
                    if self.ia_modules[ia_id].ia_type == 'bevformer':
                        ia_ft = self.ia_modules[ia_id](x, ia_context, ia_list,
                            metas_bevformer, '3d')
                        context = context + ia_ft
                    else:
                        x += self.ia_modules[ia_id](x_expand, ia_context, ia_list,
                            metas_general_lss, '3d')

        if forward_depth:
            return dict(
                depth=forward_depth_features
            )
        else:
            result = dict()
            # == 2d post-process ==
            if forward_2d:
                sample = self.ctsd.conv_norm_out(sample.flatten(0, 2))
                sample = self.ctsd.conv_act(sample)
                if self.ctsd.depth_decoder is not None:
                    raise NotImplementedError

                sample = self.ctsd.conv_out(sample)
                sample = sample.view(
                    batch_size, sequence_length, view_count, *sample.shape[1:])
                result["depth"] = forward_depth_features
                result["image"] = sample
            # == 3d post-process ==
            if forward_3d:
                x = self.maskgit.forward_post(x)
                result["lidar"] = x
            return result