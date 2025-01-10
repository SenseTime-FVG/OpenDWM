# from image_model import UNetCrossviewTemporalConditionModel
# from lidar_model import UnetTransformerAlignV2
import dwm.functional
from einops import rearrange, repeat
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
import torch
import torch.nn.functional as F

class JointModel(torch.nn.Module):

    def __init__(
        self, ctsd, maskgit, bevformer, lss, bevformer_fix_ratio=False, _2d_to_3d=False
    ):
        super().__init__()

        self.ctsd = ctsd
        self.maskgit = maskgit
        self.bevformer = bevformer
        self.lss = lss
        self.bevformer_fix_ratio = bevformer_fix_ratio
        self._2d_to_3d = _2d_to_3d

        if self.bevformer is not None:
            self.bev_proj = torch.nn.Linear(self.bevformer.embed_dims, 512)

        if self.maskgit is not None:
            self.maskgit_cond_proj = torch.nn.Linear(4096, 512)

    def get_ctsd_features_from_maskgit(
        self, sample_shape, bev_features, bev_from_frustum_transform, last_step_depth_features, bev_specification
    ):     
        
        batch_size, sequence_length, view_count, channel, height, width = sample_shape
        frustum = dwm.functional.create_frustum(
                self.ctsd.depth_frustum_range, height, width, self.ctsd.device
            ).unsqueeze(0)
        
        frustum_bev_coordinates = (
            bev_from_frustum_transform @ frustum.flatten(-3)
        ).transpose(-2, -1)[..., :2]
        
        frustum_bev_coordinates = frustum_bev_coordinates\
            .reshape(
                batch_size, sequence_length, view_count,
                *frustum.shape[-3:], -1)\
            .permute(0, 1, 3, 4, 2, 5, 6).flatten(0, 1).flatten(1, 2)\
            .flatten(2, 3)

        frustum_bev_mask = \
            (frustum_bev_coordinates[..., :].abs().amax(dim=-1) < 1.0) * \
            last_step_depth_features\
            .reshape(
                batch_size * sequence_length, view_count, -1, height, width)\
            .permute(0, 2, 3, 1, 4).flatten(1, 2).flatten(2, 3)\
            .to(dtype=frustum_bev_coordinates.dtype)

        bev_features = bev_features.reshape(
            -1, sequence_length, *bev_specification["shape"],
            bev_features.shape[-1])
        
        frustum_bev_features = torch.nn.functional.grid_sample(
            bev_features.flatten(0, 1).permute(0, 3, 1, 2),
            frustum_bev_coordinates, padding_mode="border",
            align_corners=False)
        frustum_bev_features *= frustum_bev_mask.unsqueeze(1)

        # reshape back to [B, T, V, C, H, W] (reduce dimension D)
        frustum_bev_features = frustum_bev_features\
            .reshape(*frustum_bev_features.shape[:2], -1, height, view_count,width)\
            .sum(dim=2).permute(0, 3, 1, 2, 4)
        frustum_bev_features = frustum_bev_features.reshape(
            batch_size, sequence_length, *frustum_bev_features.shape[1:])

        return self.lss(frustum_bev_features)
 
        
    def get_maskgit_features_from_ctsd(
        self, lidar2img, img_shape, can_bus, ctsd_features_list
    ):

        ctsd_features_list_new = []

        for feat in ctsd_features_list:
            # new_order = [3,5,4,0,1,2] # point -> bevformer
            # new_order = [4,5,3,1,2,0] # ctsd -> bevformer # TODO need check the order
            new_order = [1,2,0,4,5,3]
            feat = torch.index_select(feat, dim=2, index=torch.tensor(new_order).to(self.ctsd.device))
            feat = feat[:, 0].flatten(0, 1)
            ctsd_features_list_new.append(feat)

        bev_meta = {
            'lidar2img': lidar2img,
            'img_shape': img_shape,
            'can_bus': can_bus
        }

        bev_embed = self.bevformer(
            ctsd_features_list_new, 
            bev_meta,
            prev_bev=None,
            img_feats_resnet=None
        )

        learnable_bev_features = self.bev_proj(bev_embed)

        if self.bevformer.type == 'base' or self.bevformer.type == 'unet' or self.bevformer.type == 'unet-down' or self.bevformer.type == 'unet-down-resnet':
            h = 200
        elif self.bevformer.type == 'tiny':
            h = 50

        maskgit_feature = rearrange(learnable_bev_features, 'b (h w) d -> b d h w', h=h)
        maskgit_feature = F.interpolate(maskgit_feature,
                            size=(80, 80), # (80,80) xiaodong (128, 128) UnetMaskgit
                            mode='bilinear',
                            align_corners=False)
        maskgit_feature = rearrange(maskgit_feature, 'b d h w -> b (h w) d')
        
        return maskgit_feature


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
        interaction_condition_mask=None
    ):
        pred = {}
        if self.ctsd is not None:
            if self.lss is not None and \
                self.ctsd.depth_frustum_range is not None and \
                maskgit_features is not None and \
                last_step_depth_features is not None and \
                bev_from_frustum_transform is not None:

                maskgit_features = maskgit_features["encoder_features"][1]

                frustum_bev_residuals = self.get_ctsd_features_from_maskgit(
                    sample.shape,
                    maskgit_features, 
                    bev_from_frustum_transform,
                    last_step_depth_features,
                    bev_specification
                )
                if interaction_condition_mask is not None:
                    frustum_bev_residuals = [
                        i * interaction_condition_mask.view(
                            *interaction_condition_mask.shape[:1],
                            *[1 for _ in i.shape[1:]])
                        for i in frustum_bev_residuals
                    ]

            else:
                frustum_bev_residuals = None

        if maskgit_conditions is not None:
            if maskgit_conditions["context"] is not None:
                maskgit_conditions = maskgit_conditions["context"] # (bs, 128*128, 4096)
                maskgit_conditions = self.maskgit_cond_proj(maskgit_conditions) 
                # tmp_maskgit_conditions = torch.randn(maskgit_conditions["context"].shape[0], 16384, maskgit_conditions["context"].shape[2]).to(maskgit_conditions["context"].device)
                # maskgit_conditions["context"] = tmp_maskgit_conditions
                if ctsd_features is not None and self.bevformer is not None:
                    maskgit_cond_from_ctsd = self.get_maskgit_features_from_ctsd(
                        lidar2img, img_shape, can_bus, ctsd_features)
                    if interaction_condition_mask is not None:
                        maskgit_cond_from_ctsd = maskgit_cond_from_ctsd * \
                        interaction_condition_mask.view(
                            *interaction_condition_mask.shape[:1],
                            *[1 for _ in maskgit_cond_from_ctsd.shape[1:]])

                    # maskgit_conditions = torch.cat([maskgit_conditions, maskgit_cond_from_ctsd], dim=-1) # 128*128 4608 = 4096 + 512
                    maskgit_conditions = maskgit_conditions + maskgit_cond_from_ctsd
                # else:
                #     zero_maskgit_cond_from_ctsd = torch.randn(maskgit_conditions["context"].shape[0], maskgit_conditions["context"].shape[1], 512).to(maskgit_conditions["context"].device)
                #     maskgit_conditions = torch.cat([maskgit_conditions, zero_maskgit_cond_from_ctsd], dim=-1)
            else:
                maskgit_conditions = None
        
        # get depth
        if self.ctsd is not None:
            if not self._2d_to_3d or ctsd_features is None:
                ctsd_pred, ctsd_up_feats, ctsd_down_feats = self.ctsd(
                    sample, timesteps, frustum_bev_residuals, **ctsd_conditions)

                pred["image"] = ctsd_pred[0]
                if not self._2d_to_3d and len(ctsd_pred) > 1:
                    pred["depth"] = ctsd_pred[1]

                if return_ctsd_feature:
                    pred["ctsd_up_feats"] = ctsd_up_feats
                    pred["ctsd_down_feats"] = ctsd_down_feats

        if self.maskgit is not None:
            if not self._2d_to_3d or ctsd_features is not None:
                pred["lidar"], pred["maskgit_feature"] = self.maskgit(code, 
                    action = maskgit_action, 
                    context = maskgit_conditions,
                    return_feats = return_maskgit_feature, 
                    attention_mask_temporal = attention_mask_temporal)

        return pred