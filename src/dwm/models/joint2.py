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

    def _build_extra_projection(self, enforce_align_projection):
        # TODO: support align different module
        modules = dict()

        # For PositionBasedCrossAtt
        if 'pos_based_3d_to_2d' in enforce_align_projection:
            metas = enforce_align_projection['pos_based_3d_to_2d']
            if isinstance(metas, (list, tuple)):
                self.pos_based_3d_to_2d = nn.ModuleList(
                    [self._parse_mm_utils(v) for v in metas]
                )
            else:
                self.pos_based_3d_to_2d = self._parse_mm_utils(metas)

        if 'pos_based_2d_to_3d' in enforce_align_projection:
            metas = enforce_align_projection['pos_based_2d_to_3d']
            if isinstance(metas, (list, tuple)):
                self.pos_based_2d_to_3d = nn.ModuleList(
                    [self._parse_mm_utils(v) for v in metas]
                )
            else:
                self.pos_based_2d_to_3d = self._parse_mm_utils(metas)

    def _parse_mm_utils(self, v):
        if 'type' in v:
            ctype = v.pop('type')
            if ctype == 'lss':
                cmodule = LSSBased3DTO2D
            elif ctype == 'lss_r':
                cmodule = LSSBased2DTO3D
            elif ctype == 'att_t':
                cmodule = PositionBasedCrossAttWithT
            elif ctype == 'att_t_mask':
                cmodule = PositionBasedCrossAttWithTMask
        else:
            cmodule = PositionBasedCrossAtt
        return cmodule(**v)

    def get_ctsd_features_from_maskgit(
        self, sample, bev_features, bev_from_frustum_transform, last_step_depth_features, bev_specification,
        timesteps=None
    ):     
        
        batch_size, sequence_length, view_count, channel, height, width = sample.shape

        bev_features = bev_features.reshape(
            -1, sequence_length, *bev_specification["shape"],
            bev_features.shape[-1])

        if timesteps is None:
            timesteps = torch.tensor(
                [0], dtype=sample.dtype, device=sample.device)
            timesteps = timesteps.expand(batch_size)
        
        cm_features = self._cross_modal_proj(
            sample, last_step_depth_features, bev_from_frustum_transform, bev_features,
            self.pos_based_3d_to_2d, timesteps, interaction_condition_mask=None
        )
        return self.lss(cm_features)

    def _cross_modal_proj(self, sample, last_step_depth_features, 
        bev_from_frustum_transform, bev_features, 
        pos_based_3d_to_2d, timesteps=None, interaction_condition_mask=None):
        assert self.ctsd.depth_frustum_range is not None and \
            last_step_depth_features is not None and \
            bev_from_frustum_transform is not None
        batch_size, sequence_length, view_count, _, height, width = \
            sample.shape

        # ctsd not support now
        # if '3d_bev_align' in self.align_projection:
        #     bev_features = self.align_projection['3d_bev_align'](bev_features)
        frustum = dwm.functional\
            .create_frustum(
                self.ctsd.depth_frustum_range, height, width, self.ctsd.device)\
            .unsqueeze(0)
        frustum_bev_coordinates = \
            (bev_from_frustum_transform @ frustum.flatten(-3))\
            .transpose(-2, -1)[..., :2]

        # reshape to [B * T, D * H, V * W, 2]
        # [B T V D H W 2]
        frustum_bev_coordinates = frustum_bev_coordinates\
            .view(
                batch_size, sequence_length, view_count,
                *frustum.shape[-3:], -1)

        # sampled LSS feature shape [B * T, C, D * H, V * W]                
        frustum_bev_features = pos_based_3d_to_2d(
            bev_features, sample, frustum_bev_coordinates, 
            last_step_depth_features=last_step_depth_features,
            timesteps=timesteps
        )
        if interaction_condition_mask is not None:
            if frustum_bev_features.ndim == 3:
                interaction_condition_mask = interaction_condition_mask[:, None, None, None].repeat(1, sequence_length, 1, 1).flatten(0, 1)
            else:
                interaction_condition_mask = interaction_condition_mask.reshape(
                    list(interaction_condition_mask.shape[:1]) + [1]*(frustum_bev_features.ndim-1))
            frustum_bev_features = frustum_bev_features*interaction_condition_mask

        return frustum_bev_residuals
 
        
    def get_maskgit_features_from_ctsd(
        self, batch, ctsd_features_list
    ):
        
        multi_view_images = batch["bev_images"]
        img_h, img_w = multi_view_images.shape[-2:]

        ctsd_features_list_new = []

        for feat in ctsd_features_list:
            # new_order = [3,5,4,0,1,2] # point -> bevformer
            # new_order = [4,5,3,1,2,0] # ctsd -> bevformer # TODO need check the order
            new_order = [1,2,0,4,5,3]
            feat = torch.index_select(feat, dim=2, index=torch.tensor(new_order).to(self.ctsd.device))
            feat = feat[:, 0].flatten(0, 1)
            ctsd_features_list_new.append(feat)
        
        camera_intrinsics = batch["camera_intrinsics"] #(bsz, t, 6, 3, 3)
        image_size = batch["image_size"] #(bsz, t, 6, 2)
        batch_size, sequence_length, view_count = image_size.shape[:3]

        intrinsics = dwm.functional.make_homogeneous_matrix(camera_intrinsics).to(self.ctsd.device)
        with torch.cuda.amp.autocast(enabled=False):
            camera_from_lidar = torch.linalg.solve(
                batch["ego_transforms"][:, :, 1:] @ batch["camera_transforms"], 
                batch["ego_transforms"][:, :, :1] @ batch["lidar_transforms"]).to(self.ctsd.device)
        
        # NOTE align with bevformer
        if img_w == 1600:
            scale_ratio = 1
        elif img_w == 800:
            scale_ratio = 0.5
        elif img_w == 384:
            scale_ratio = 0.24
        scale_factor = np.eye(4)
        if not self.bevformer_fix_ratio:
            scale_factor[0, 0] *= img_w / image_size[0,0,0,0]
            scale_factor[1, 1] *= img_h / image_size[0,0,0,1]
        else:
            scale_factor[0, 0] *= scale_ratio
            scale_factor[1, 1] *= scale_ratio
        scale_factor = torch.from_numpy(scale_factor).to(self.ctsd.device)
        scale_factor = scale_factor.repeat(batch_size, sequence_length, view_count, 1, 1).to(intrinsics.dtype)
        # scale_factor = scale_factor.to(self.ctsd.device)
        
        lidar2img = scale_factor @ intrinsics @ camera_from_lidar
        lidar2img = lidar2img[:, 0].to(self.ctsd.device)

        rotation = batch["bev_rotation"]
        translation = torch.tensor([[0.,0.,0.]])

        rotations = []
        for rot in rotation:
            rot = Quaternion(rot.cpu().numpy().tolist())
            rotations.append(rot)

        pos = batch["ego_pos"][:, 0]
        orient = batch["ego_orient"][:, 0]

        ego_accel = batch["ego_accel"][:, 0]
        ego_rotation_rate = batch["ego_rotation_rate"][:, 0]
        ego_vel = batch["ego_vel"][:, 0]

        last_pos = pos[:, 0]
        last_orient = orient[:, 0]

        pad = torch.zeros_like(last_pos)[:, :2]

        can_bus = torch.cat([last_pos, last_orient, ego_accel, ego_rotation_rate, ego_vel, pad], dim=-1) # bsz 18
        can_bus[:,:3] = translation

        tmp_array = np.array([0, 0, 0, 0])
        patch_angles = []
        rotats = []
        for rot in rotations:
            # tmp_array[:4] = rot
            tmp_tensor = torch.tensor([rot.w, rot.w, rot.w, rot.w])
            rotats.append(tmp_tensor)
            patch_angle = quaternion_yaw(rot) / np.pi * 180
            if patch_angle < 0:
                patch_angle += 360
            patch_angles.append(torch.tensor(patch_angle))

        patch_angle = torch.tensor([patch_angles])
        rotation = torch.stack(rotats, dim=0)
        can_bus[:, 3:7] = rotation

        can_bus[:, -2] = patch_angle / 180 * np.pi 
        can_bus[:, -1] = 0 
        can_bus = can_bus.to(self.ctsd.device)

        if self.bevformer_fix_ratio:
            if img_w == 1600:
                img_shape = (1600, 928, 3)
            elif img_w == 800:
                img_shape = (800, 480, 3)
            elif img_w == 384:
                img_shape = (384, 192, 3)   #(384, 224, 3)
            elif img_w == 448:
                img_shape = (448, 256, 3)
        else:
            img_shape = (img_w, img_h, 3)

        # TODO should debug
        # TODO hard-code for cfg
        if batch_size != lidar2img.shape[0]:
            lidar2img = torch.cat([lidar2img, lidar2img])
            can_bus = torch.cat([can_bus, can_bus])

        bev_meta = {
            'lidar2img': lidar2img,
            'img_shape': img_shape, # 384, 192, 3
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
        timesteps, batch, 
        ctsd_conditions, maskgit_conditions,
        maskgit_action, attention_mask_temporal,
        ctsd_features= None, maskgit_features= None, # lidar2img_features
        last_step_depth_features=None, # NOTE = last_step_depth_features
        bev_from_frustum_transform=None,
        bev_specification=None,
        return_ctsd_feature=True,
        return_maskgit_feature=True,
    ):
        pred = {}
        if self.ctsd is not None:
            batch_size, sequence_length, view_count, channel, height, width = sample.shape

            if self.lss is not None and \
                self.ctsd.depth_frustum_range is not None and \
                maskgit_features is not None and \
                last_step_depth_features is not None and \
                bev_from_frustum_transform is not None:

                maskgit_features = maskgit_features["encoder_features"][0]

                frustum_bev_residuals = self.get_ctsd_features_from_maskgit(
                    sample,
                    maskgit_features, 
                    bev_from_frustum_transform,
                    last_step_depth_features,
                    bev_specification
                )

            else:
                frustum_bev_residuals = None

        # Done lss: lidar feature to img
        if maskgit_conditions is not None:
            maskgit_conditions = maskgit_conditions["context"] # (bs, 128*128, 4096)
            maskgit_conditions = self.maskgit_cond_proj(maskgit_conditions) 
            # tmp_maskgit_conditions = torch.randn(maskgit_conditions["context"].shape[0], 16384, maskgit_conditions["context"].shape[2]).to(maskgit_conditions["context"].device)
            # maskgit_conditions["context"] = tmp_maskgit_conditions
            if ctsd_features is not None and self.bevformer is not None:
                maskgit_cond_from_ctsd = self.get_maskgit_features_from_ctsd(batch, ctsd_features)
                # maskgit_conditions = torch.cat([maskgit_conditions, maskgit_cond_from_ctsd], dim=-1) # 128*128 4608 = 4096 + 512
                maskgit_conditions = maskgit_conditions + maskgit_cond_from_ctsd
            # else:
            #     zero_maskgit_cond_from_ctsd = torch.randn(maskgit_conditions["context"].shape[0], maskgit_conditions["context"].shape[1], 512).to(maskgit_conditions["context"].device)
            #     maskgit_conditions = torch.cat([maskgit_conditions, zero_maskgit_cond_from_ctsd], dim=-1)
    
        # get depth
        if self.ctsd is not None:
            ctsd_pred, ctsd_up_feats, ctsd_down_feats = self.ctsd(
                sample, timesteps, frustum_bev_residuals, **ctsd_conditions)

            pred["image"] = ctsd_pred[0]
            if not self._2d_to_3d and len(ctsd_pred) > 1:
                pred["depth"] = ctsd_pred[1]

            if return_ctsd_feature:
                pred["ctsd_up_feats"] = ctsd_up_feats
                pred["ctsd_down_feats"] = ctsd_down_feats

        if self.maskgit is not None:
            pred["lidar"], pred["maskgit_feature"] = self.maskgit(code, 
                action = maskgit_action, 
                context = maskgit_conditions,
                return_feats = return_maskgit_feature, 
                attention_mask_temporal = attention_mask_temporal)

        return pred
