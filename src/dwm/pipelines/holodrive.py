import imageio
import contextlib
import copy
from collections import defaultdict
import diffusers
import diffusers.image_processor
import torchvision.transforms.functional
import dwm.common
import dwm.functional
import dwm.models.base_vq_models.dvgo_utils
import dwm.pipelines.ctsd
import dwm.utils.preview
from dwm.utils.metrics_copilot4d import compute_chamfer_distance, compute_chamfer_distance_inner, compute_ray_errors
from dwm.utils.metrics import evaluate_svd, evaluate_svd_3d
from einops import rearrange
import math
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
import os
import pickle
import re
import safetensors.torch
import time
import timm
import warnings
import torch
import torch.cuda.amp
import torch.distributed.fsdp
import torch.distributed.fsdp.sharded_grad_scaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.utils.tensorboard
import torchvision
import transformers
from typing import Optional, Union
import torch.nn.functional as F
import pdb



def convnext_forward_features(clx, x):
    x = clx.stem(x)
    for (idx, stage) in enumerate(clx.stages):
        x = stage(x)
        if idx == 1:  # /8
            break
    # x = clx.stages(x)
    # x = clx.norm_pre(x)
    # 80*80 feature map
    # R50, last stage
    return x


def count(idx):
    unique_elements, counts = torch.unique(idx, return_counts=True)
    sorted_indices = torch.argsort(counts, descending=True)
    sorted_elements = unique_elements[sorted_indices]
    sorted_counts = counts[sorted_indices]
    return sorted_elements, sorted_counts


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def draw_bev_lidar_from_list(voxels_list, pth):
    with imageio.get_writer(pth, fps=2) as video_writer:
        for voxels in voxels_list:
            if isinstance(voxels, np.ndarray):
                voxels = torch.from_numpy(voxels)
            image = (voxels.max(dim=0)[0][:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255).astype(np.uint8)
            video_writer.append_data(image)


def gamma_func(mode="cosine"):
    if mode == "linear":
        return lambda r: 1 - r
    elif mode == "cosine":
        return lambda r: torch.cos(r * math.pi / 2)
    elif mode == "square":
        return lambda r: 1 - r**2
    elif mode == "cubic":
        return lambda r: 1 - r**3
    else:
        raise NotImplementedError


def _make_eye_mask(
    input_ids_shape, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len),
                      torch.finfo(dtype).min, device=device)
    mask *= (1 - torch.eye(tgt_len, dtype=dtype, device=device))

    assert past_key_values_length == 0          # only square
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _make_causal_mask(
    input_ids_shape, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


class DWM():

    @staticmethod
    def load_state(path: str):
        if path.endswith(".safetensors"):
            state = safetensors.torch.load_file(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu")

        return state

    @staticmethod
    def get_ctsd_conditions(
        sd_name, text_encoder, tokenizer, common_config: dict,
        dynamic_cfg: dict, batch: dict, device, dtype,
        text_condition_mask=None, _3dbox_condition_mask=None,
        hdmap_condition_mask=None, segmentation_condition_mask=None,
        action_condition_mask=None, do_classifier_free_guidance: bool = False
    ):
        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        if do_classifier_free_guidance:
            batch_size *= 2

        condition_embedding_dict = dict()
        condition_image_list = []

        # text prompt
        if text_encoder is not None:
            flattened_clip_text = []
            parsed_shape = []
            dwm.pipelines.ctsd.CrossviewTemporalSD.flatten_clip_text(
                batch["clip_text"], flattened_clip_text, parsed_shape,
                text_condition_mask=text_condition_mask,
                do_classifier_free_guidance=do_classifier_free_guidance)

            pooled_text_embeddings = None
            if sd_name == "SD2":
                text_inputs = tokenizer(
                    flattened_clip_text, padding="max_length",
                    max_length=tokenizer.model_max_length, truncation=True,
                    return_tensors="pt")
                text_embeddings = text_encoder(
                    text_inputs.input_ids.to(device))[0]
                if len(parsed_shape) == 1:
                    # all times and views share the same text prompt
                    text_embeddings = text_embeddings.unsqueeze(1)\
                        .unsqueeze(1)\
                        .repeat(1, sequence_length, view_count, 1, 1)
                else:
                    # all times and views use different text prompts
                    text_embeddings = text_embeddings.unflatten(
                        0, parsed_shape)

                condition_embedding_dict['text'] = text_embeddings

            elif sd_name == "SD3":
                clip_tokenizers = tokenizer[:2]
                clip_text_encoders = text_encoder[:2]

                clip_text_embeddings_list = []
                clip_pooled_text_embeddings_list = []
                for clip_tokenizer, clip_text_encoder in zip(
                    clip_tokenizers, clip_text_encoders
                ):
                    text_embeddings, pooled_text_embeddings = dwm.pipelines.ctsd\
                        .CrossviewTemporalSD.sd3_encode_prompt_with_clip(
                            clip_text_encoder, clip_tokenizer, common_config,
                            flattened_clip_text, clip_text_encoder.device)
                    clip_text_embeddings_list.append(text_embeddings)
                    clip_pooled_text_embeddings_list.append(
                        pooled_text_embeddings)

                clip_text_embeddings = torch.cat(
                    clip_text_embeddings_list, dim=-1)
                pooled_text_embeddings = torch.cat(
                    clip_pooled_text_embeddings_list, dim=-1)

                t5_prompt_embed = dwm.pipelines.ctsd.CrossviewTemporalSD\
                    .sd3_encode_prompt_with_t5(
                        text_encoder[-1], tokenizer[-1], common_config,
                        prompt=flattened_clip_text, device=device)

                clip_text_embeddings = torch.nn.functional.pad(
                    clip_text_embeddings,
                    (0, t5_prompt_embed.shape[-1] -
                     clip_text_embeddings.shape[-1]),
                )
                text_embeddings = torch.cat(
                    [clip_text_embeddings, t5_prompt_embed], dim=-2)

                if len(parsed_shape) == 1:
                    # all times and views share the same text prompt
                    text_embeddings = text_embeddings.unsqueeze(1)\
                        .unsqueeze(1)\
                        .repeat(1, sequence_length, view_count, 1, 1)\
                        .to(dtype=dtype)
                    pooled_text_embeddings = pooled_text_embeddings\
                        .unsqueeze(1).unsqueeze(1)\
                        .repeat(1, sequence_length, view_count, 1)\
                        .to(dtype=dtype)
                else:
                    # all times and views use different text prompts
                    text_embeddings = text_embeddings\
                        .unflatten(0, parsed_shape).to(dtype=dtype)
                    pooled_text_embeddings = pooled_text_embeddings\
                        .unflatten(0, parsed_shape).to(dtype=dtype)

                condition_embedding_dict['text'] = text_embeddings

        # layout condition (uses the 1st frame condition)
        condition_on_all_frames = common_config.get(
            "condition_on_all_frames", False)
        if "3dbox_images" in batch:
            if condition_on_all_frames:
                _3dbox_images = batch["3dbox_images"].to(device)
            else:
                # the other frames need to be padded with zeros?
                _3dbox_images = batch["3dbox_images"][:, :1].to(device)

            if _3dbox_condition_mask is not None:
                _3dbox_images[
                    _3dbox_condition_mask.logical_not().to(device)] = 0

            if do_classifier_free_guidance:
                if common_config.get("with_layout_cfg", True):
                    _3dbox_images = torch.cat(
                        [torch.zeros_like(_3dbox_images), _3dbox_images])
                else:
                    _3dbox_images = torch.cat(
                        [_3dbox_images, _3dbox_images])

            condition_image_list.append(_3dbox_images)

        if "hdmap_images" in batch:
            if condition_on_all_frames:
                hdmap_images = batch["hdmap_images"].to(device)
            else:
                hdmap_images = batch["hdmap_images"][:, :1].to(device)

            if hdmap_condition_mask is not None:
                hdmap_images[
                    hdmap_condition_mask.logical_not().to(device)] = 0

            if do_classifier_free_guidance:
                if common_config.get("with_layout_cfg", True):
                    hdmap_images = torch.cat(
                        [torch.zeros_like(hdmap_images), hdmap_images])
                else:
                    hdmap_images = torch.cat(
                        [hdmap_images, hdmap_images])

            condition_image_list.append(hdmap_images)

        if "segmentation_images" in batch:
            if condition_on_all_frames:
                segm_images = batch["segmentation_images"].to(device)
            else:
                segm_images = batch["segmentation_images"][:, :1].to(device)

            if segmentation_condition_mask is not None:
                segm_images[
                    segmentation_condition_mask.logical_not().to(device)] = 0

            if do_classifier_free_guidance:
                segm_images = torch.cat(
                    [torch.zeros_like(segm_images), segm_images])

            condition_image_list.append(segm_images)

        if common_config['cond_with_action']:
            # for cond with zero input (cxt, layout...), following info is not needed
            # for cond with zero output, following info is required
            # assert "added_time_ids" not in common_config
            # infer without these infos
            act_type = common_config.get('ctsd_act_type', 'dwm_xyz')
            if act_type == 'dwm_xyz' or act_type == 'dwm_xy':
                num_act = 3 if act_type == 'dwm_xyz' else 2
                pre_batch_size = batch_size // 2 if do_classifier_free_guidance else batch_size
                action = batch["ego_transforms"].new_zeros(
                    (pre_batch_size, sequence_length, view_count, num_act)).to(device)
                ego_transforms = batch['ego_transforms'][:, :, :view_count]

                for bid in range(pre_batch_size):
                    for fid in range(sequence_length-1):
                        for vid in range(view_count):
                            tfm_inverse = torch.inverse(
                                ego_transforms[bid, fid+1, vid])
                            diff_pose = tfm_inverse @ ego_transforms[bid, fid, vid]

                            action[bid, fid,
                                   vid] = diff_pose[0:num_act, -1].to(device)
                if dynamic_cfg.get('action_jitter', None) is not None:
                    action_jitter = dynamic_cfg['action_jitter']
                    print(
                        f"With action jitter: {action_jitter}, Before: {action}")
                    action = action * action.new_tensor(action_jitter)
                    print(f"After: {action}")
                action = action[..., None]          # b, t, v, 3, 1

                condition_embedding_dict["dwm_action"] = dict(
                    ft=action, mask=action_condition_mask,
                    do_classifier_free_guidance=do_classifier_free_guidance, nviews=view_count)
            else:
                raise NotImplementedError

        if len(condition_embedding_dict) > 0:
            if common_config["cat_condition"]:
                encoder_hidden_states = torch.cat(list(condition_embedding_dict.values()), dim=-2)\
                    .to(dtype=dtype)
            else:
                encoder_hidden_states = condition_embedding_dict
        else:
            encoder_hidden_states = None

        if len(condition_image_list) > 0:
            condition_image_tensor = torch.cat(condition_image_list, -3)
        else:
            condition_image_tensor = None

        # SVD numeric condition
        if "added_time_ids" in common_config:
            if common_config["added_time_ids"] == "fps_camera_transforms":
                assert "fps" in batch and "camera_intrinsics" in batch and \
                    "camera_transforms" in batch and \
                    "camera_intrinsic_embedding_indices" in common_config and \
                    "camera_intrinsic_denom_embedding_indices" in common_config and \
                    "camera_transform_embedding_indices" in common_config
                added_time_ids = torch.cat([
                    batch["fps"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    .repeat(1, sequence_length, view_count, 1),
                    batch["camera_intrinsics"].flatten(-2, -1)[
                        ...,
                        common_config["camera_intrinsic_embedding_indices"]
                    ] / batch["image_size"][
                        ...,
                        common_config[
                            "camera_intrinsic_denom_embedding_indices"
                        ]
                    ],
                    batch["camera_transforms"].flatten(-2, -1)[
                        ...,
                        common_config["camera_transform_embedding_indices"]
                    ]
                ], -1)
                if do_classifier_free_guidance:
                    added_time_ids = torch.cat(
                        [added_time_ids, added_time_ids], 0)

                added_time_ids = added_time_ids.to(device)
            else:
                added_time_ids = None

        # depth net input
        has_depth_input = "camera_intrinsics" in batch and \
            "camera_transforms" in batch
        if has_depth_input:
            camera_intrinsics = batch["camera_intrinsics"].to(device)
            camera_transforms = batch["camera_transforms"].to(device)
            if do_classifier_free_guidance:
                camera_intrinsics = torch.cat(
                    [camera_intrinsics, camera_intrinsics])
                camera_transforms = torch.cat(
                    [camera_transforms, camera_transforms])

        result = {
            "encoder_hidden_states": encoder_hidden_states,
            "condition_image_tensor": condition_image_tensor,

            "disable_crossview": torch.tensor(
                [common_config["disable_crossview"]],
                device=device).repeat(batch_size)
            if "disable_crossview" in common_config else None,

            "disable_temporal": torch.tensor(
                [common_config["disable_temporal"]],
                device=device).repeat(batch_size)
            if "disable_temporal" in common_config else None,

            "crossview_attention_mask": (
                torch.cat([batch["crossview_mask"], batch["crossview_mask"]])
                    if do_classifier_free_guidance else batch["crossview_mask"]
            ).to(device)
            if "crossview_mask" in batch else None,

            "camera_intrinsics": camera_intrinsics if has_depth_input else
            None,
            "camera_transforms": camera_transforms if has_depth_input else
            None,

            "added_time_ids": added_time_ids
            if "added_time_ids" in common_config else None
        }
        if sd_name == "SD3":
            result["pooled_projections"] = pooled_text_embeddings

        return result

    @staticmethod
    def preprocess_points(batch, device):
        mhv = dwm.functional.make_homogeneous_vector
        return [
            [
                (mhv(p_j.to(device)) @ t_j.permute(1, 0))[:, :3]
                for p_j, t_j in zip(p_i, t_i.flatten(0, 1))
            ]
            for p_i, t_i in zip(
                batch["lidar_points"], batch["lidar_transforms"].to(device))
        ]

    @staticmethod
    def postprocess_points(batch, ego_space_points):
        return [
            [
                (
                    dwm.functional.make_homogeneous_vector(p_j.cpu()) @
                    torch.linalg.inv(t_j).permute(1, 0)
                )[:, :3]
                for p_j, t_j in zip(p_i, t_i.flatten(0, 1))
            ]
            for p_i, t_i in zip(
                ego_space_points, batch["lidar_transforms"])
        ]

    @staticmethod
    def get_maskgit_conditions(
        bev_layout_encoder, common_config: dict, batch: dict, device, dtype,
        _3dbox_condition_mask: Optional[torch.Tensor] = None,
        hdmap_condition_mask: Optional[torch.Tensor] = None,
        do_classifier_free_guidance: bool = False
    ):
        condition_embedding_list = []

        with torch.no_grad():
            # layout condition
            if bev_layout_encoder is not None:
                if "3dbox_bev_images" in batch:
                    # _3dbox_images = torch.flip(
                    #     batch["3dbox_bev_images"], dims=[3])

                    _3dbox_images = batch["3dbox_bev_images"]
                    # _3dbox_images = torch.rot90(_3dbox_images, -1, (3, 4))

                    _3dbox_images = _3dbox_images.to(device).flatten(0, 1)
                    if _3dbox_condition_mask is not None:
                        for i in range(_3dbox_condition_mask.shape[0]):
                            if not _3dbox_condition_mask[i]:
                                _3dbox_images[i] = 0

                    if do_classifier_free_guidance:
                        _3dbox_images = torch.cat(
                            [torch.zeros_like(_3dbox_images), _3dbox_images])
                    condition_embedding_list.append(
                        bev_layout_encoder.forward_features(_3dbox_images)
                        .flatten(2, 3).permute(0, 2, 1))

                if "hdmap_bev_images" in batch:
                    # hdmap_images = torch.flip(
                    #     batch["hdmap_bev_images"], dims=[3])

                    # hdmap_images = torch.rot90(hdmap_images, -1, (3, 4))
                    hdmap_images = batch["hdmap_bev_images"]
                    hdmap_images = hdmap_images.to(device).flatten(0, 1)
                    if hdmap_condition_mask is not None:
                        for i in range(hdmap_condition_mask.shape[0]):
                            if not hdmap_condition_mask[i]:
                                hdmap_images[i] = 0

                    if do_classifier_free_guidance:
                        hdmap_images = torch.cat(
                            [torch.zeros_like(hdmap_images), hdmap_images])

                    condition_embedding_list.append(
                        bev_layout_encoder.forward_features(hdmap_images)
                        .flatten(2, 3).permute(0, 2, 1))
        if len(condition_embedding_list) > 0:
            # [batch_size, token_count, embedding_feature_size]
            encoder_hidden_states = torch.cat(condition_embedding_list, 2)\
                .to(dtype=dtype)
        else:
            encoder_hidden_states = None

        result = {
            "context": encoder_hidden_states
        }
        if "feature_collect_range" in common_config:
            result["feature_collect_range"] = \
                common_config["feature_collect_range"]

        return result

    @staticmethod
    def get_bevformer_input(
        common_config: dict, batch: dict, device,
        do_classifier_free_guidance: bool = False
    ):
        multi_view_images = batch["bev_images"]
        img_h, img_w = multi_view_images.shape[-2:]

        camera_intrinsics = batch["camera_intrinsics"]  # (bsz, t, 6, 3, 3)
        image_size = batch["image_size"]  # (bsz, t, 6, 2)
        batch_size, sequence_length, view_count = image_size.shape[:3]

        intrinsics = dwm.functional.make_homogeneous_matrix(
            camera_intrinsics).to(device)
        camera_from_ego = torch.linalg.solve(
            batch["ego_transforms"][:, :, 1:] @ batch["camera_transforms"],
            batch["ego_transforms"][:, :, :1]).to(device)

        # NOTE align with bevformer
        if img_w == 1600:
            scale_ratio = 1
        elif img_w == 800:
            scale_ratio = 0.5
        elif img_w == 384:
            scale_ratio = 0.24
        scale_factor = np.eye(4)
        if common_config.get("bevformer_fix_ratio", False):
            scale_factor[0, 0] *= scale_ratio
            scale_factor[1, 1] *= scale_ratio
        else:
            scale_factor[0, 0] *= img_w / image_size[0, 0, 0, 0]
            scale_factor[1, 1] *= img_h / image_size[0, 0, 0, 1]

        scale_factor = torch.from_numpy(scale_factor).to(device)
        scale_factor = scale_factor.repeat(
            batch_size, sequence_length, view_count, 1, 1).to(intrinsics.dtype)

        ego2img = scale_factor @ intrinsics @ camera_from_ego
        ego2img = ego2img[:, 0].to(device)

        rotation = batch["bev_rotation"]
        translation = torch.tensor([[0., 0., 0.]])

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

        can_bus = torch.cat([
            last_pos, last_orient, ego_accel, ego_rotation_rate, ego_vel, pad
        ], dim=-1)  # bsz 18
        can_bus[:, :3] = translation

        patch_angles = []
        rotats = []
        for rot in rotations:
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
        can_bus = can_bus.to(device)

        if common_config.get("bevformer_fix_ratio", False):
            if img_w == 1600:
                img_shape = (1600, 928, 3)
            elif img_w == 800:
                img_shape = (800, 480, 3)
            elif img_w == 384:
                img_shape = (384, 192, 3)  # (384, 224, 3)
            elif img_w == 448:
                img_shape = (448, 256, 3)

        else:
            img_shape = (img_w, img_h, 3)

        if do_classifier_free_guidance:
            ego2img = torch.cat([ego2img, ego2img])
            can_bus = torch.cat([can_bus, can_bus])

        return {
            "lidar2img": ego2img,
            "img_shape": img_shape,
            "can_bus": can_bus
        }

    @staticmethod
    def get_projected_point_depth(
        batch: dict, depth_frustum_range: list, height: int, width: int
    ):
        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]

        normalized_intrinsics = dwm.functional.make_homogeneous_matrix(
            dwm.functional.normalize_intrinsic_transform(
                batch["image_size"], batch["camera_intrinsics"]))
        camera_from_lidar = torch.linalg.solve(
            batch["ego_transforms"][:, :, 1:] @ batch["camera_transforms"],
            batch["ego_transforms"][:, :, :1] @ batch["lidar_transforms"])
        frustum_from_lidar = normalized_intrinsics @ camera_from_lidar

        scale = bias = torch.tensor([[[0.5 * width, 0.5 * height]]])
        image_tensor = torch.zeros(
            (batch_size, sequence_length, view_count, height * width))
        for i in range(batch_size):
            for j in range(sequence_length):
                points = dwm.functional.make_homogeneous_vector(
                    batch["lidar_points"][i][j][:, :3]).t()
                projected_points = frustum_from_lidar[i, j] @ points
                p = (projected_points[:, 0:2] / projected_points[:, 2:3])\
                    .transpose(-2, -1)
                fd = 1 - (projected_points[:, 2] - depth_frustum_range[0]) / \
                    (depth_frustum_range[1] - depth_frustum_range[0])
                mask = torch.logical_and(
                    p.abs().amax(dim=-1) < 0.999,
                    torch.logical_and(fd >= 0, fd < 1))
                for k in range(view_count):
                    p2 = p[k, mask[k]]
                    p2 = (p2 * scale + bias).floor().long()
                    indices = p2[..., 0] + p2[..., 1] * width
                    image_tensor[i, j, k, indices] = fd[k, mask[k]]

        return image_tensor.view(
            batch_size, sequence_length, view_count, height, width)

    @staticmethod
    def get_bev_from_frustum_transform(
        batch,
        bev_specification,
        device
    ):

        camera_from_ego = torch.linalg.solve(
            batch["ego_transforms"][:, :, 1:] @ batch["camera_transforms"],
            batch["ego_transforms"][:, :, :1])
        ego_from_bev = torch.tensor([
            [bev_specification["range"][0], 0, 0, 0],
            [0, bev_specification["range"][1], 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]],
            dtype=torch.float32
        )

        normalized_intrinsics = dwm.functional.make_homogeneous_matrix(
            dwm.functional.normalize_intrinsic_transform(
                batch["image_size"], batch["camera_intrinsics"])
        )

        frustum_from_ego = normalized_intrinsics @ camera_from_ego
        ego_from_bev = ego_from_bev.view(1, 1, 1, *ego_from_bev.shape)

        bev_from_frustum_transform = torch.linalg.inv(
            frustum_from_ego @ ego_from_bev
        ).to(device)

        return bev_from_frustum_transform

    @staticmethod
    def get_action(
        batch, act_type, device,
    ):
        batch_size, num_frames = batch["ego_transforms"].shape[0], batch["ego_transforms"].shape[1]

        if act_type == "copliot4D":

            action = torch.zeros_like(batch["ego_transforms"]).to(device)

            for bid in range(batch_size):
                for fid in range(num_frames-1):
                    diff_pose = torch.matmul(
                        batch["ego_transforms"][bid, fid+1], torch.inverse(batch["ego_transforms"][bid, fid]))
                    action[bid, fid] = diff_pose

        elif act_type == "copliot4D_v2":
            action = torch.zeros_like(batch["ego_transforms"]).to(device)
            ego_transforms = batch['ego_transforms']
            lidar_transforms = batch['lidar_transforms']

            for bid in range(batch_size):
                for fid in range(num_frames-1):
                    tfm_inverse = torch.inverse(
                        ego_transforms[bid, fid+1] @ lidar_transforms[bid, fid+1])
                    diff_pose = tfm_inverse @ ego_transforms[bid,
                                                             fid] @ lidar_transforms[bid, fid]
                    action[bid, fid] = diff_pose

        elif act_type == "copliot4D_v3":
            reference_idx = num_frames // 2 - 1
            action = torch.zeros_like(batch["ego_transforms"]).to(device)
            ego_transforms = batch['ego_transforms']
            lidar_transforms = batch['lidar_transforms']

            for bid in range(batch_size):
                for fid in range(num_frames-1):
                    tfm_inverse = torch.inverse(
                        ego_transforms[bid, reference_idx] @ lidar_transforms[bid, reference_idx])
                    diff_pose = tfm_inverse @ ego_transforms[bid,
                                                             fid] @ lidar_transforms[bid, fid]
                    action[bid, fid] = diff_pose

        elif act_type == "dwm":

            # action = torch.zeros_like(batch_size, num_frames, 2).to(device)

            # for bid in range(batch_size):
            #     for fid in range(num_frames-1):
            #         diff_xy = torch.matmul(batch["ego_transforms"][bid, fid+1], torch.inverse(batch["ego_transforms"][bid, fid]))[0:2, 3]
            #         action[bid, fid] = diff_xy
            action = batch["ego_transforms"].new_zeros(
                (batch_size, num_frames, 3)).to(device)
            ego_transforms = batch['ego_transforms']            # b/t/k
            lidar_transforms = batch['lidar_transforms']        # b/t

            for bid in range(batch_size):
                for fid in range(num_frames-1):
                    tfm_inverse = torch.inverse(
                        ego_transforms[bid, fid+1] @ lidar_transforms[bid, fid+1])
                    diff_pose = tfm_inverse @ ego_transforms[bid,
                                                             fid] @ lidar_transforms[bid, fid]

                    action[bid, fid] = diff_pose[0:3, -1]
            action = action[..., None]

        action = rearrange(action, "b l d m n -> (b l) d m n")

        return action

    @staticmethod
    def fill_svd_mask(input_cfg, latent, infer):
        ori_values = latent.new_zeros(latent.shape)
        mask = latent.new_zeros(list(latent.shape[:-3]) + [1] + list(latent.shape[-2:]))

        ori_values[:, :input_cfg['num_init_frames']] = latent[:, :input_cfg['num_init_frames']]
        mask[:, :input_cfg['num_init_frames']] = 1
        sum_v = ori_values.abs().sum(list(range(1, ori_values.ndim)), keepdim=True)
        mask *= (sum_v > 0)

        return ori_values, mask

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, generator=self.generator)  # noise in [0, 1]
        noise = noise.to(x.device)

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def _data_process(self, batch, **kwargs):
        # gpu dataprocessor
        lidar_points = None
        if self.common_config.get('gpu_roi_filter', False):
            lidar_points = copy.deepcopy(batch["lidar_points_raw"])
            assert lidar_points is not None         # avoid warning
            range_x, range_y, range_z = self.common_config['roi_range']
            for bid in range(len(lidar_points)):
                for fid in range(len(lidar_points[0])):
                    mask = (lidar_points[bid][fid][:, 0] >= range_x[0]) & (lidar_points[bid][fid][:, 1] >= range_y[0]) & \
                        (lidar_points[bid][fid][:, 2] >= range_z[0]) & (lidar_points[bid][fid][:, 0] <= range_x[1]) & \
                        (lidar_points[bid][fid][:, 1] <= range_y[1]) & (
                            lidar_points[bid][fid][:, 2] <= range_z[1])
                    lidar_points[bid][fid] = lidar_points[bid][fid][mask]
        assert lidar_points is not None
        if lidar_points is not None:
            batch["lidar_points"] = lidar_points
        return batch

    def mask_code(self, code, mask_token, cond=None, mask_ratio=None, eta=20, with_noise=False):
        if with_noise:
            # equal to align_dual_rate = True in old pipeline
            return self._mask_noise_code(code, mask_token, cond, mask_ratio, eta)
        else:
            return self._mask_code(code, mask_token, cond, mask_ratio)

    def _mask_code(self, code, mask_token, cond=None, mask_ratio=None):
        # code -> 16, 6400, 1024; mask_token -> 1, 1, 1024
        if mask_ratio == None:
            mask_ratio = self.gamma(torch.rand((1,), generator=self.generator))

        # masking: length -> length * mask_ratio
        # x -> 16, 6400*mask_ratio, 1024; mask -> 16, 6400
        # ids_restore -> 16, 6400; ids_keep -> 16, 6400*mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(code, mask_ratio)
        # append mask tokens to sequence

        # after cat -> 16,6400,1024
        # mask_token -> 1,1,1024; requires_grad
        mask_tokens = mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)  # no cls token
        # after gather -> 16,6400,1024
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        if cond is not None:
            x = torch.cat([x, cond], dim=-1)

        return x, mask, ids_restore

    def _mask_noise_code(self, code, mask_token, cond=None, mask_ratio=None, eta=20,):
        if mask_ratio == None:
            mask_ratio = self.gamma(np.random.uniform())

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(code, mask_ratio)
        # append mask tokens to sequence

        mask_tokens = mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        # NOTE add noise token to unmaksed tokens
        N, Lx, D = x.shape
        noise_ratio = mask_ratio * eta * 0.01
        noise_len = int(torch.round(Lx * noise_ratio))

        # random choose noise_len tokens from codebook
        codebook_size = self.vq_point_cloud.vector_quantizer.n_e
        noise_indices = np.random.choice(
            codebook_size, (N, noise_len), replace=True)

        noise_indices = torch.tensor(noise_indices).to(x.device)

        noisy_token = self.vq_point_cloud.vector_quantizer.get_codebook_entry(
            noise_indices)  # N, noise_len, d

        replaced_x = x.clone()
        new_mask = mask.clone()
        for i in range(N):
            selected_indices = torch.randperm(Lx)[:noise_len]
            replaced_x[i, selected_indices, :] = noisy_token[i]
            replace_ids_keep = ids_keep[i, selected_indices]
            new_mask[i, replace_ids_keep] = 1

        x = torch.cat([replaced_x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        if cond is not None:
            x = torch.cat([x, cond], dim=-1)

        return x, new_mask, ids_restore

    def process_inference_input(self, task_code, x, num_frames, code, ctype='pre', 
        skip_past=0):
        assert task_code == 0
        num_frames_past = self.common_config['unet_input_cfg']['num_init_frames']
        num_frames_future = num_frames - num_frames_past
        if ctype == 'pre' or ctype == 'fin':
            code = rearrange(code, "(b f) ... -> b f ...", f=num_frames)
            x = rearrange(x, "(b f) ... -> b f ...", f=num_frames_future)
            batch_size = code.shape[0]
            x_past = code[:, :num_frames_past]
            if skip_past > 0:
                x_past = x_past[:, skip_past:]
            x = torch.cat([x_past, x], dim=1).flatten(0, 1)
        elif ctype == 'after':
            x = rearrange(x, "(b f) ... -> b f ...", f=num_frames)
            x = x[:, num_frames_past:].flatten(0, 1)
        else:
            raise NotImplementedError
        return x

    def multitask_init_code(self, batch_size, num_frames, task_code=0):
        unet_input_cfg = self.common_config.get("unet_input_cfg", {})
        if unet_input_cfg.get("num_init_frames", None) is None:
            # Without reference frames
            return self.wm.maskgit.mask_token.repeat(
            1, self.wm.maskgit.img_size**2, 1)

        assert task_code == 0            # Note: for avoid config confusion
        num_frames_past = self.common_config['unet_input_cfg']['num_init_frames']
        num_frames_future = num_frames - num_frames_past
        x_future = self.wm.maskgit.mask_token[:, None].repeat(batch_size, num_frames_future, 
            self.wm.maskgit.img_size**2, 1).flatten(0, 1)
        return x_future

    def multitask_mask_code(self, num_frames, code, task_code=0, mask_ratio=None):
        """
        1. sample a task
        """
        unet_input_cfg = self.common_config.get("unet_input_cfg", {})
        if unet_input_cfg.get("num_init_frames", None) is None:
            return self.mask_code(
                code, self.wm.maskgit.mask_token,
                mask_ratio=mask_ratio, with_noise=self.use_discrete_diffusion
            )

        use_discrete_diffusion = self.use_discrete_diffusion
        assert use_discrete_diffusion and task_code == 0            # Note: for avoid config confusion
        num_frames_past = self.common_config['unet_input_cfg']['num_init_frames']
        num_frames_future = num_frames - num_frames_past

        if task_code == 0:
            code = rearrange(code, "(b f) ... -> b f ...", f=num_frames)
            batch_size = code.shape[0]
            past = code[:, :num_frames_past].flatten(0, 1)
            future = code[:, num_frames_past:].flatten(0, 1)
            x_past, mask_past, ids_restore_past = self.mask_code(past, 
                self.wm.maskgit.mask_token, mask_ratio=0, with_noise=False)
            x_future, mask_future, ids_restore_future = self.mask_code(future, 
                self.wm.maskgit.mask_token, mask_ratio=mask_ratio, with_noise=use_discrete_diffusion)

            # merge
            x = torch.cat([rearrange(x_past, "(b f) ... -> b f ...", b=batch_size), 
                rearrange(x_future, "(b f) ... -> b f ...", b=batch_size)], dim=1).flatten(0, 1)
            mask = torch.cat([rearrange(mask_past, "(b f) ... -> b f ...", b=batch_size), 
                rearrange(mask_future, "(b f) ... -> b f ...", b=batch_size)], dim=1).flatten(0, 1)
            ids_restore = torch.cat([rearrange(ids_restore_past, "(b f) ... -> b f ...", b=batch_size), 
                rearrange(ids_restore_future, "(b f) ... -> b f ...", b=batch_size)], dim=1).flatten(0, 1)
        else:
            x, mask, ids_restore = self.mask_code(code, 
                self.wm.maskgit.mask_token, mask_ratio=mask_ratio, with_noise=use_discrete_diffusion)
        return x, mask, ids_restore

    def action_tying(self, ctsd_conditions, bev_infer_metas):
        old_ft = ctsd_conditions['encoder_hidden_states']['dwm_action']['ft']
        action_embed = bev_infer_metas['action_embed']
        action_embed = torch.cat([v[:, 0:1] for v in action_embed], dim=-1)
        action_embed = rearrange(
            action_embed, "(b f) k c -> b f k s c", b=old_ft.shape[0], s=1)
        action_embed = action_embed.repeat(1, 1, old_ft.shape[2], 1, 1)

    def gen_unet_input(self, noise, latent, infer, cxt_condition_mask=None, rescale=False):
        if self.common_config['unet_input_type'] == 'sd':
            return noise
        elif self.common_config['unet_input_type'] == 'svd':
            if rescale:
                latent = latent * self.vae.config.scaling_factor
            if cxt_condition_mask is not None:
                for i in range(cxt_condition_mask.shape[0]):
                    if not cxt_condition_mask[i]:
                        latent[i] = 0

            if infer and self.dynamic_cfg.get("infer_without_cxt", False):
                print("===Set cxt to zero")
                latent *= 0

            extra_latents, mask = DWM.fill_svd_mask(
                self.common_config['unet_input_cfg'], latent, infer)
            noise = torch.cat([noise, extra_latents, mask], dim=-3)
            return noise
        elif self.common_config['unet_input_type'] == 'vista':
            num_max_vista_frames = 4
            probs = [2**i for i in range(num_max_vista_frames)]
            probs = [i/sum(probs) for i in probs]
            random_size = random.choices(
                list(range(num_max_vista_frames)), weights=probs, k=1)[0]

            extra_latents, mask = DWM.fill_svd_mask(
                dict(num_init_frames=random_size), latent, infer)
        else:
            raise NotImplementedError

    def _encode_vae_image(self, image: torch.Tensor, do_classifier_free_guidance):
        _, num_frames, nviews = image.shape[:3]
        image = rearrange(image, "b f k c h w -> (b f k) c h w")
        image = image.to(device=self.device)
        image_latents = self.vae.encode(image).latent_dist.mode()
        image_latents = rearrange(
            image_latents, "(b f k) c h w -> b f k c h w", f=num_frames, k=nviews)

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([negative_image_latents, image_latents])

        return image_latents

    def _get_attention_mask_temporal(self, task_code, code, batch_size=None):
        if code.ndim == 3:
            sequence_length = code.shape[0] // batch_size
        else:
            batch_size, sequence_length = code.shape[0], code.shape[1]
        if task_code in [0, 1]:
            attention_mask_temporal = _make_causal_mask((batch_size, sequence_length), dtype=code.dtype, device=code.device)[0, 0]
        else:
            attention_mask_temporal = _make_eye_mask((batch_size, sequence_length), dtype=code.dtype, device=code.device)[0, 0]
        return attention_mask_temporal

    def __init__(
        self, output_path, config: dict, device, common_config: dict,
        training_config: dict, inference_config: dict, vq_point_cloud,
        wm, pretrained_model_name_or_path: str, wm_ckpt_path=None,
        ctsd_ckpt_path=None, maskgit_ckpt_path=None, bevformer_ckpt_path=None,
        vq_point_cloud_ckpt_path=None, vq_blank_code_path=None,
        load_state_args: dict = {}, unet_load_state_args: dict = {},
        bev_layout_encoder_config={}, metrics: dict = {}, resume_from = None
    ):
        self.should_save = not torch.distributed.is_initialized() or \
            torch.distributed.get_rank() == 0
        self.config = config
        self.device = device
        self.common_config = common_config
        self.training_config = training_config
        self.inference_config = inference_config
        self.dynamic_cfg = dict()  # TODO dynamic_cfg for temporal 3d
        self.joint_forward_policy = self.common_config.get(
            'joint_forward_policy', 'cascade')
        assert self.joint_forward_policy in ['cascade', 'separate']

        self.use_discrete_diffusion = self.common_config.get(
            "use_discrete_diffusion", False)
        unet_input_cfg = self.common_config.get("unet_input_cfg", {})
        self.use_init_frames = unet_input_cfg.get("num_init_frames", None) is not None

        if wm.ctsd is None or "unet" in type(wm.ctsd).__name__.lower():
            self.sd_name = "SD2"
        elif "dit" in type(wm.ctsd).__name__.lower():
            self.sd_name = "SD3"
            assert self.common_config.get("distribution_framework") == "fsdp"
        else:
            raise f"ctsd does not support {type(wm.ctsd).__name__}"

        if self.should_save:
            print(f"use_discrete_diffusion is {self.use_discrete_diffusion}")

        # Note: now, only cond_with_action decide whether use action (previous with two many dependence)
        self.use_action = self.common_config.get("cond_with_action", False)

        self.generator = torch.Generator()
        if "generator_seed" in self.config:
            self.generator.manual_seed(self.config["generator_seed"])
        else:
            self.generator.seed()

        # tokenizer & text encoder
        if (
            wm.ctsd is not None and
            isinstance(wm.ctsd, diffusers.UNetSpatioTemporalConditionModel)
        ):
            self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
                pretrained_model_name_or_path, subfolder="tokenizer")
            self.text_encoder = transformers.CLIPTextModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder",
                variant="fp16", torch_dtype=torch.float16)
            self.text_encoder.requires_grad_(False)
            self.text_encoder.to(self.device)
        elif (
            wm.ctsd is not None and
            isinstance(wm.ctsd, diffusers.SD3Transformer2DModel)
        ):
            tokenizer_one = transformers.CLIPTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer",
            )
            tokenizer_two = transformers.CLIPTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer_2",
            )
            tokenizer_three = transformers.T5TokenizerFast.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer_3"
            )
            self.tokenizer = [tokenizer_one, tokenizer_two, tokenizer_three]

            text_encoder_one = transformers.CLIPTextModelWithProjection.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=torch.float16,
            )
            text_encoder_two = transformers.CLIPTextModelWithProjection.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="text_encoder_2",
                torch_dtype=torch.float16,
            )
            text_encoder_one.requires_grad_(False)
            text_encoder_two.requires_grad_(False)
            text_encoder_one.to(device)
            text_encoder_two.to(device)
            text_encoder_one.eval()
            text_encoder_two.eval()
            text_encoder_three = transformers.T5EncoderModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="text_encoder_3",
                torch_dtype=torch.float16,
            )
            text_encoder_three.requires_grad_(False)
            text_encoder_three.to(dtype=torch.float16)
            text_encoder_three = FSDP(
                text_encoder_three,
                device_id=torch.cuda.current_device(),
                **self.common_config.get("t5_fsdp_wrapper_settings", {})
            )
            self.text_encoder = [text_encoder_one,
                                 text_encoder_two, text_encoder_three]
        elif wm.ctsd is not None:
            raise Exception("Unsupported diffusion model type.")

        # vae & image processor
        self.vae = diffusers.AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae")
        self.vae.requires_grad_(False)
        self.vae.to(self.device)
        self.image_processor = diffusers.image_processor.VaeImageProcessor(
            vae_scale_factor=2 ** (len(self.vae.config.block_out_channels) - 1))

        # bev layout encoder
        if bev_layout_encoder_config is not None:
            if "name" in bev_layout_encoder_config and bev_layout_encoder_config["name"] == "adapter":
                self.bev_layout_encoder = dwm.models.adapters.ImageAdapter(
                    channels = [512, 1024],
                    is_downblocks = [True, True],
                    num_res_blocks = 4, 
                    downscale_factor = 2,
                )
                self.bev_layout_encoder.to(device)
            else:
                from timm.models.resnet import Bottleneck
                self.bev_layout_encoder = timm.create_model(
                    'resnet50', pretrained=True, block=Bottleneck, layers=[3, 4, 6, 3],
                    num_classes=0, in_chans=3, output_stride=8, zero_init_last=False,
                    global_pool=None)
                self.bev_layout_encoder.requires_grad_(False)
                self.bev_layout_encoder.to(device)
        else:
            self.bev_layout_encoder = None

        # vq
        self.vq_point_cloud = vq_point_cloud
        if self.vq_point_cloud is not None:
            self.vq_point_cloud.requires_grad_(False)
            self.vq_point_cloud.to(device)

            if vq_point_cloud_ckpt_path is not None:
                vq_state_dict = DWM.load_state(vq_point_cloud_ckpt_path)
                if "state_dict" in vq_state_dict.keys():  # jingcheng's vq
                    vq_state_dict = vq_state_dict["state_dict"]

                if "vector_quantizer.reservoir" in vq_state_dict:
                    vq_state_dict.pop("vector_quantizer.reservoir")

                self.vq_point_cloud.load_state_dict(
                    vq_state_dict, strict=False)

        # scheduler
        if self.sd_name == "SD2":
            self.train_scheduler = diffusers.DDPMScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder="scheduler")

            test_scheduler_type = dwm.common.get_class(
                self.inference_config.get("scheduler", "diffusers.DDIMScheduler"))
            self.test_scheduler = test_scheduler_type.from_pretrained(
                pretrained_model_name_or_path, subfolder="scheduler")
        elif self.sd_name == "SD3":
            self.train_scheduler = (
                diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
                    pretrained_model_name_or_path, subfolder="scheduler"
                )
            )
            self.test_scheduler = (
                diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
                    pretrained_model_name_or_path, subfolder="scheduler"
                )
            )

        self.wm_wrapper = self.wm = wm
        distribution_framework = self.common_config.get(
            "distribution_framework", "ddp")
        if (
            not torch.distributed.is_initialized() or
            distribution_framework == "ddp"
        ):
            self.wm.to(self.device)
        elif (
            distribution_framework == "fsdp" and
            "fsdp_ignored_module_pattern" in common_config
        ):
            pattern = re.compile(common_config["fsdp_ignored_module_pattern"])
            for name, module in self.wm.named_modules():
                if pattern.match(name) is not None:
                    module.to(self.device)

        # load_state
        if resume_from is not None:
            wm_ckpt = DWM.load_state(
                os.path.join(
                    output_path, "checkpoints", "{}.pth".format(resume_from)))
            self.wm.load_state_dict(wm_ckpt)

        else:
            if wm_ckpt_path is not None:
                wm_ckpt = DWM.load_state(wm_ckpt_path)
                missing_keys, unexpected_keys = self.wm.load_state_dict(wm_ckpt, **load_state_args)
                if self.should_save:
                    print(f'load wm from {wm_ckpt_path}')
                    if self.common_config.get("print_load_state_info", False):
                        print(f"wm missing keys: {missing_keys}")
                        print(f"wm unexpected keys: {unexpected_keys}")

            if ctsd_ckpt_path is not None:
                ctsd_state_dict = DWM.load_state(ctsd_ckpt_path)
                if self.sd_name == "SD2":
                    if self.common_config.get("transform_wm_to_ctsd", False):
                        ctsd_state_dict = {
                            k.replace("ctsd.", ""): v
                            for k, v in ctsd_state_dict.items()
                        }

                    missing_keys, unexpected_keys = self.wm.ctsd.load_state_dict_from_sd(
                        ctsd_state_dict, **unet_load_state_args)
                    if (
                        self.should_save and
                        self.common_config.get("print_load_state_info", False)
                    ):
                        print(f"ctsd missing keys: {missing_keys}")
                        print(f"ctsd unexpected keys: {unexpected_keys}")
                elif self.sd_name == "SD3":
                    self.wm.ctsd.load_state_dict(ctsd_state_dict, strict=False)

            if maskgit_ckpt_path is not None:
                self.wm.maskgit.load_state_dict(
                    DWM.load_state(maskgit_ckpt_path))
                if self.should_save:
                    print(f"load maskgit from {maskgit_ckpt_path}")

            # NOTE add bevformer ckpt
            if bevformer_ckpt_path is not None:
                state_dict = DWM.load_state(bevformer_ckpt_path)["state_dict"]
                vit_state_dict = {}
                for key, value in state_dict.items():
                    # if 'embedding' in key:
                    #     print(key)
                    if 'pts_bbox_head.transformer' in key:
                        new_key = key.replace(
                            'pts_bbox_head.transformer', 'transformer')
                        vit_state_dict[new_key] = value
                    elif 'pts_bbox_head' in key:
                        new_key = key.replace('pts_bbox_head.', '')
                        vit_state_dict[new_key] = value
                    # NOTE if ignore img_neck due to mismatch FPN network shape
                    elif 'img_neck' in key:  # NOTE need check here # TODO
                        continue
                    else:
                        vit_state_dict[key] = value

                missing_keys, unexpected_keys = self.wm.bevformer.load_state_dict(
                    vit_state_dict, strict=False)
                if self.should_save:
                    print(f'loaded bevformer from {bevformer_ckpt_path}')

        if "freezing_pattern" in training_config:
            pattern = re.compile(training_config["freezing_pattern"])
            frozen_module_count = 0
            for name, module in self.wm.named_modules():
                if pattern.match(name) is not None:
                    module.requires_grad_(False)
                    frozen_module_count += 1
                    if self.should_save:
                        print("{} is frozen.".format(name))

            if self.should_save:
                print("{} modules are frozen.".format(frozen_module_count))

        if self.wm.ctsd is not None:
            self.wm.ctsd.enable_gradient_checkpointing()

        if self.wm.maskgit is not None and self.use_action:
            self.act_type = self.wm.maskgit.act_type
        else:
            self.act_type = None

        self.gamma = gamma_func("cosine")
        self.iter = 0
        self.T = self.inference_config["inference_steps"]

        if vq_blank_code_path is not None:
            with open(vq_blank_code_path, "rb") as f:
                self.BLANK_CODE = pickle.load(f)
        else:
            self.BLANK_CODE = None

        # setup training parts
        self.loss_report_list = []
        self.step_duration = 0

        self.distribution_framework = self.common_config.get(
            "distribution_framework", "ddp")
        if self.training_config.get("enable_grad_scaler", False):
            if (
                not torch.distributed.is_initialized() or
                self.distribution_framework == "ddp"
            ):
                self.grad_scaler = torch.cuda.amp.GradScaler()
            elif self.distribution_framework == "fsdp":
                self.grad_scaler = torch.distributed.fsdp.sharded_grad_scaler\
                    .ShardedGradScaler()

        if torch.distributed.is_initialized():
            if self.distribution_framework == "ddp":
                self.wm_wrapper = torch.nn.parallel.DistributedDataParallel(
                    self.wm, device_ids=[int(os.environ["LOCAL_RANK"])],
                    **self.common_config["ddp_wrapper_settings"])
            elif self.distribution_framework == "fsdp":
                if "fsdp_ignored_module_pattern" in common_config:
                    pattern = re.compile(
                        common_config["fsdp_ignored_module_pattern"])
                    ignored_named_modules = [
                        (name, module)
                        for name, module in self.wm.named_modules()
                        if pattern.match(name) is not None
                    ]
                    ignored_modules = [i[1] for i in ignored_named_modules]
                    if self.should_save:
                        print(
                            "{} modules are ignored by FSDP."
                            .format(len(ignored_named_modules)))
                        print(
                            "These ignored modules are {}."
                            .format([i[0] for i in ignored_named_modules]))
                else:
                    ignored_modules = None

                self.wm_wrapper = FSDP(
                    self.wm, device_id=torch.cuda.current_device(),
                    ignored_modules=ignored_modules,
                    **self.common_config["ddp_wrapper_settings"])
            else:
                raise Exception(
                    "Unknown data parallel framework {}."
                    .format(self.distribution_framework))

        if self.should_save and output_path is not None:
            self.summary = torch.utils.tensorboard.SummaryWriter(
                os.path.join(output_path, "log"))

        if "lr_decay_pattern" in training_config:
            p_lr_decay, p_other = [], []
            pattern = re.compile(training_config["lr_decay_pattern"])
            frozen_module_count = 0
            for name, params in self.wm.named_parameters():
                if pattern.match(name) is not None:
                    p_lr_decay.append(params)
                else:
                    p_other.append(params)

            if self.should_save:
                print("Lr decay rate: {} and {}".format(
                    len(p_lr_decay), len(p_other)))
            params_to_optimize = [
                {'params': p_lr_decay, 'lr': config["optimizer"]["lr"]*0.1},
                {'params': p_other}]
        else:
            params_to_optimize = self.wm_wrapper.parameters()

        self.optimizer = dwm.common.create_instance_from_config(
            config["optimizer"], params=params_to_optimize
        ) if "optimizer" in config else None

        if resume_from is not None:
            optimizer_state_path = os.path.join(
                output_path, "optimizer", "{}.pth".format(resume_from))
            optimizer_state = torch.load(
                optimizer_state_path, map_location="cpu")
            if (
                torch.distributed.is_initialized() and
                self.distribution_framework == "fsdp"
            ):
                optimizer_state = FSDP.optim_state_dict_to_load(
                    self.wm_wrapper, self.optimizer, optimizer_state)

            self.optimizer.load_state_dict(optimizer_state)

        self.lr_scheduler = dwm.common.create_instance_from_config(
            config["lr_scheduler"], optimizer=self.optimizer) \
            if "lr_scheduler" in config else None

        self.metrics = metrics
        for i in self.metrics.values():
            i.to(self.device)

    def get_loss_coef(self, name, timestep):
        ignore_coef = 1
        if "ignore_list_of_non_init_step" in self.training_config:
            ignore_coef = 0 \
                if name in self.training_config["ignore_list_of_non_init_step"] and timestep < 970 \
                else 1

        loss_coef = 1
        if "loss_coef_dict" in self.training_config:
            loss_coef = self.training_config["loss_coef_dict"].get(name, 1.0)

        return ignore_coef * loss_coef

    def save_checkpoint(self, output_path: str, steps: int):
        if (
            torch.distributed.is_initialized() and
            self.distribution_framework == "fsdp"
        ):
            sdc = torch.distributed.fsdp.FullStateDictConfig(rank0_only=True)
            osdc = torch.distributed.fsdp.FullOptimStateDictConfig(
                rank0_only=True)
            with FSDP.state_dict_type(
                self.wm_wrapper,
                torch.distributed.fsdp.StateDictType.FULL_STATE_DICT,
                state_dict_config=sdc, optim_state_dict_config=osdc
            ):
                wm_state_dict = self.wm_wrapper.state_dict()
                optimizer_state_dict = FSDP.optim_state_dict(
                    self.wm_wrapper, self.optimizer)

        elif self.should_save:
            wm_state_dict = self.wm.state_dict()
            optimizer_state_dict = self.optimizer.state_dict()

        if self.should_save:
            os.makedirs(
                os.path.join(output_path, "checkpoints"), exist_ok=True)
            torch.save(
                wm_state_dict,
                os.path.join(
                    output_path, "checkpoints", "{}.pth".format(steps)))

            os.makedirs(os.path.join(output_path, "optimizer"), exist_ok=True)
            torch.save(
                optimizer_state_dict,
                os.path.join(output_path, "optimizer", "{}.pth".format(steps)))

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def log(self, global_step: int, log_steps: int):
        if self.should_save:
            if len(self.loss_report_list) > 0 and \
                    isinstance(self.loss_report_list[0], dict):
                loss_values = {
                    i: sum([j[i] for j in self.loss_report_list]) /
                    len(self.loss_report_list)
                    for i in self.loss_report_list[0].keys()
                }
                loss_message = ", ".join([
                    "{}: {:.4f}".format(k, v) for k, v in loss_values.items()
                ])
                print(
                    "Step {} ({:.1f} s/step), {}".format(
                        global_step, self.step_duration / log_steps,
                        loss_message))
                for key, value in loss_values.items():
                    self.summary.add_scalar(
                        "train/{}".format(key), value, global_step)

            else:
                loss_value = sum(self.loss_report_list) / \
                    len(self.loss_report_list)
                print(
                    "Step {} ({:.1f} s/step), loss: {:.4f}".format(
                        global_step, self.step_duration / log_steps,
                        loss_value))
                self.summary.add_scalar("train/Loss", loss_value, global_step)

        self.loss_report_list.clear()
        self.step_duration = 0

    def get_autocast_context(self):
        if "autocast" in self.common_config:
            return torch.autocast(**self.common_config["autocast"])
        else:
            return contextlib.nullcontext()

    def train_step(self, batch: dict, global_step: int):
        self.wm_wrapper.train()

        t0 = time.time()

        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        image_tensor = self.image_processor.preprocess(
            batch["vae_images"].flatten(0, 2).to(self.device))

        shift_factor = self.vae.config.shift_factor\
            if self.vae.config.shift_factor is not None else 0
        latents = dwm.functional.memory_efficient_split_call(
            self.vae, image_tensor,
            lambda block, tensor: (
                block.encode(tensor).latent_dist.sample() - shift_factor
            ) * block.config.scaling_factor,
            self.common_config.get("memory_efficient_batch", -1))
        latents = latents.unflatten(0, batch["vae_images"].shape[:3])
        noise = torch.randn(
            latents.shape, generator=self.generator).to(self.device)

        if self.sd_name == "SD2":
            timesteps = torch.randint(
                0, self.train_scheduler.config.num_train_timesteps,
                (1,), generator=self.generator).repeat(batch_size).to(self.device)
            noisy_latents = self.train_scheduler.add_noise(
                latents, noise, timesteps)
            if self.train_scheduler.config.prediction_type == "epsilon":
                image_target = noise
            elif self.train_scheduler.config.prediction_type == "v_prediction":
                image_target = self.train_scheduler.get_velocity(
                    latents, noise, timesteps)
            else:
                raise Exception("Unknown training target of the UNet.")

        elif self.sd_name == "SD3":
            u = dwm.pipelines.ctsd.CrossviewTemporalSD\
                .sd3_compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal",
                    batch_size=latents.shape[0], logit_mean=0.0, logit_std=1.0,
                    mode_scale=1.29)
            indices = (
                u * self.train_scheduler.config.num_train_timesteps
            ).long()
            timesteps = self.train_scheduler.timesteps[indices].to(self.device)

            # Add noise according to flow matching.
            sigmas = dwm.pipelines.ctsd.CrossviewTemporalSD.sd3_get_sigmas(
                self.train_scheduler, timesteps, n_dim=latents.ndim,
                dtype=latents.dtype, device=latents.device)
            noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
            image_target = latents

        timesteps = timesteps.unsqueeze(-1).unsqueeze(-1)\
            .repeat(1, sequence_length, view_count)

        # 2D conditions
        text_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("text_prompt_condition_ratio", 1.0))\
            .tolist()
        _3dbox_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("3dbox_condition_ratio", 1.0))\
            .to(self.device)
        hdmap_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("hdmap_condition_ratio", 1.0))\
            .to(self.device)
        segmentation_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("segmentation_condition_ratio", 1.0))\
            .to(self.device)
        action_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("action_condition_ratio", 1.0))\
            .to(self.device)
        cxt_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("cxt_condition_ratio", 1.0))\
            .to(self.device)
        interaction_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("interaction_condition_ratio", 1.0))\
            .to(self.device)
        stage1_loss_keep_flag = (
            torch.rand((), generator=self.generator) <
            self.training_config.get("stage1_loss_keep_ratio", 1.0))\
            .to(dtype=torch.float32, device=self.device)

        # 3D Lidar (run with fp32)
        if self.vq_point_cloud is not None:
            voxels = self.vq_point_cloud.voxelizer(
                DWM.preprocess_points(batch, self.device))
            lidar_features = self.vq_point_cloud.lidar_encoder(
                voxels.flatten(0, 1))
            lidar_quantized_features, _, lidar_target_code = \
                self.vq_point_cloud.vector_quantizer(lidar_features)

            if self.common_config.get("use_bev_vq_feature", False):
                maskgit_feature_from_vq = lidar_quantized_features
            else:
                maskgit_feature_from_vq = None

        with self.get_autocast_context():

            if self.wm.ctsd is not None:
                ctsd_conditions = DWM.get_ctsd_conditions(
                    self.sd_name, self.text_encoder, self.tokenizer,
                    self.common_config, self.dynamic_cfg, batch, self.device,
                    torch.float32, text_condition_mask, _3dbox_condition_mask,
                    hdmap_condition_mask, segmentation_condition_mask,
                    action_condition_mask)
            else:
                ctsd_conditions = None

            # for svd style mask input
            if self.common_config.get('unet_input_type', None) is not None:
                noisy_latents = self.gen_unet_input(
                    noisy_latents, latents, infer=False, cxt_condition_mask=cxt_condition_mask)

            if self.common_config.get('action_tying', False):
                ctsd_conditions = self.action_tying(
                    ctsd_conditions, bev_infer_metas)

            if self.wm.maskgit is not None:
                # NOTE mask_r shoud align with timesteps in ctsd
                scale_r = (999 - timesteps.flatten()[0]) * 0.001
                mask_ratio = self.gamma(scale_r)

                task_prob = self.common_config.get('task_prob', [1.0, 0.0, 0.0])
                temporal_task_code = np.random.choice(3, size=1, p=task_prob)[0]
                pdb.set_trace()
                lidar_quantized_features, mask, ids_restore = self.multitask_mask_code(
                    sequence_length, lidar_quantized_features,
                    task_code=temporal_task_code, mask_ratio=mask_ratio)

                if self.use_action:
                    action = DWM.get_action(batch, self.act_type, self.device)
                else:
                    action = None

                # === Cop4D mask strategy
                # TODO: sync task, now, only prediction task in cop4d is used, for other tasks is more difficult to merge
                attention_mask_temporal = self._get_attention_mask_temporal(
                    temporal_task_code, lidar_quantized_features, batch_size)

                maskgit_conditions = DWM.get_maskgit_conditions(
                    self.bev_layout_encoder, self.common_config, batch, self.device,
                    torch.float32, _3dbox_condition_mask, hdmap_condition_mask,
                )
            else:
                lidar_quantized_features = None
                action = None
                attention_mask_temporal = None
                maskgit_conditions = None
            # forward
            # 1st w/o feature

            loss_dict = {}
            extra_cfg = {}
            if self.joint_forward_policy == 'separate':
                extra_cfg["forward_depth"] = True

            pred_1 = self.wm_wrapper(
                noisy_latents, lidar_quantized_features,
                timesteps,
                ctsd_conditions, maskgit_conditions,
                action[:, 0, :] if action is not None else None,
                attention_mask_temporal,
                ctsd_features=None, maskgit_features=None,
                last_step_depth_features=None,
                bev_from_frustum_transform=None,
                return_ctsd_feature=True,
                return_maskgit_feature=True,
                **extra_cfg
            )

        # TODO task_code

        if self.wm.ctsd is not None and "image" in pred_1 and \
            self.training_config.get("train_ctsd_down", True) and \
                not self.training_config.get("mg_loss_only", False):
            if self.sd_name == "SD3":
                pred_1["image"] = pred_1["image"] * (-sigmas) + noisy_latents
            loss_dict["sd1"] = torch.nn.functional.mse_loss(
                pred_1["image"].float(), image_target.float(), reduction="mean"
            ) * self.get_loss_coef("sd1", timesteps[0]) * stage1_loss_keep_flag

        if "lidar" in pred_1:
            loss_dict["mg1"] = (torch.nn.functional.cross_entropy(
                pred_1["lidar"].flatten(0, 1), lidar_target_code, reduction="none", label_smoothing=0.1
            ) * mask.flatten(0, 1)).sum() / mask.sum() * \
                self.get_loss_coef("mg1", timesteps[0]) * stage1_loss_keep_flag

        if "depth" in pred_1:
            loss_dict["d1"] = dwm.pipelines.ctsd.CrossviewTemporalSD\
                .make_depth_loss(
                    batch_size, sequence_length, view_count,
                    self.wm.ctsd.depth_frustum_range, batch, pred_1["depth"],
                    self.get_loss_coef("d1", timesteps[0]),
                    self.common_config["point_count_limit_per_view"],
                    self.common_config["point_bundle_size"],
                    self.device
                ) * stage1_loss_keep_flag

        should_optimize = \
            "gradient_accumulation_steps" not in self.training_config or \
            (global_step + 1) % \
            self.training_config["gradient_accumulation_steps"] == 0
        loss_report = dict()
        if self.joint_forward_policy == 'separate':
            loss = torch.stack([i[1] for i in loss_dict.items()]).sum()
            loss_report["loss1"] = loss.item()
            if len(loss_dict.items()) > 1:
                for i in loss_dict.items():
                    loss_report[i[0]] = i[1].item()

            if self.config["device"] == "cuda":
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if should_optimize:
                if "max_norm_for_grad_clip" in self.training_config:
                    if self.training_config.get("enable_grad_scaler", False):
                        self.grad_scaler.unscale_(self.optimizer)

                    if (
                        torch.distributed.is_initialized() and
                        self.distribution_framework == "fsdp"
                    ):
                        self.wm_wrapper.clip_grad_norm_(
                            self.training_config["max_norm_for_grad_clip"])
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.wm.parameters(),
                            self.training_config["max_norm_for_grad_clip"])

                if self.training_config.get("enable_grad_scaler", False):
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

        if self.training_config.get("train_feature_exchange", False):

            if self.joint_forward_policy == 'separate':
                ctsd_feature = pred_1["ctsd_down_feats"].detach(
                ) if "ctsd_down_feats" in pred_1 else None
                maskgit_feature = pred_1["maskgit_feature"].detach(
                ) if "maskgit_feature" in pred_1 else None
                depth = pred_1["depth"].detach() if "depth" in pred_1 else None
            elif self.joint_forward_policy == 'cascade':
                ctsd_feature = pred_1["ctsd_down_feats"] if "ctsd_down_feats" in pred_1 else None
                maskgit_feature = pred_1["maskgit_feature"] if "maskgit_feature" in pred_1 else None
                depth = pred_1["depth"] if "depth" in pred_1 else None
            else:
                raise NotImplementedError

            bev_from_frustum_transform = DWM.get_bev_from_frustum_transform(
                batch,
                self.common_config["bev_specification"],
                self.device
            )

            if self.wm.bevformer is not None:
                bevformer_input = DWM.get_bevformer_input(
                    self.common_config, batch, self.device)
            else:
                bevformer_input = {}

            if self.common_config.get("use_bev_vq_feature", False):
                maskgit_feature = maskgit_feature_from_vq

            if self.joint_forward_policy == 'separate':
                extra_cfg["forward_depth"] = False
                loss_dict = {}

            with self.get_autocast_context():
                # 2nd w feature
                pred_2 = self.wm_wrapper(
                    noisy_latents, lidar_quantized_features,
                    timesteps,
                    ctsd_conditions, maskgit_conditions,
                    action[:, 0, :] if action is not None else None,
                    attention_mask_temporal,
                    ctsd_features=ctsd_feature, maskgit_features=maskgit_feature,  # code,
                    last_step_depth_features=depth,
                    bev_from_frustum_transform=bev_from_frustum_transform,
                    bev_specification=self.common_config["bev_specification"],
                    interaction_condition_mask=interaction_condition_mask,
                    **extra_cfg,
                    **bevformer_input
                )

            if "image" in pred_2:
                if self.sd_name == "SD3":
                    pred_2["image"] = pred_1["image"] * \
                        (-sigmas) + noisy_latents
                loss_dict["sd2"] = torch.nn.functional.mse_loss(
                    pred_2["image"].float(), image_target.float(), reduction="mean"
                ) * self.get_loss_coef("sd2", timesteps[0])

            if "lidar" in pred_2:
                loss_dict["mg2"] = (torch.nn.functional.cross_entropy(
                    pred_2["lidar"].flatten(0, 1), lidar_target_code, reduction="none", label_smoothing=0.1
                ) * mask.flatten(0, 1)).sum() / mask.sum() * \
                    self.get_loss_coef("mg2", timesteps[0])

            forward2_with_depth_loss = self.training_config.get(
                "forward2_with_depth_loss", False)
            if not self.training_config.get("2d-to-3d", True):
                warnings.warn(
                    "2d-to-3d is not recommend to set, please use forward2_with_depth_loss")
                forward2_with_depth_loss = True

            if forward2_with_depth_loss:
                assert self.joint_forward_policy != 'separate'
                loss_dict["d2"] = dwm.pipelines.ctsd.CrossviewTemporalSD\
                    .make_depth_loss(
                        batch_size, sequence_length, view_count,
                        self.wm.ctsd.depth_frustum_range, batch,
                        pred_2["depth"],
                        self.get_loss_coef("d2", timesteps[0]),
                        self.common_config["point_count_limit_per_view"],
                        self.common_config["point_bundle_size"],
                        self.device)

        loss = torch.stack([i[1] for i in loss_dict.items()]).sum()
        loss_report["loss2"] = loss.item()
        if len(loss_dict.items()) > 1:
            for i in loss_dict.items():
                loss_report[i[0]] = i[1].item()

        self.loss_report_list.append(loss_report)

        if self.training_config.get("enable_grad_scaler", False):
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        if should_optimize:
            if "max_norm_for_grad_clip" in self.training_config:
                if self.training_config.get("enable_grad_scaler", False):
                    self.grad_scaler.unscale_(self.optimizer)

                if (
                    torch.distributed.is_initialized() and
                    self.distribution_framework == "fsdp"
                ):
                    self.wm_wrapper.clip_grad_norm_(
                        self.training_config["max_norm_for_grad_clip"])
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.wm.parameters(),
                        self.training_config["max_norm_for_grad_clip"])

            if self.training_config.get("enable_grad_scaler", False):
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.step_duration += time.time() - t0

    @staticmethod
    def voxels2points(vq_point_cloud, voxels):
        interval = torch.tensor([vq_point_cloud.grid_size["interval"]])
        min = torch.tensor([vq_point_cloud.grid_size["min"]])
        return [
            [
                torch.nonzero(v_j).flip(-1).cpu() * interval + min
                for v_j in v_i
            ]
            for v_i in voxels
        ]

    def inference_pipeline(self, batch, output_type):
        self.wm_wrapper.eval()

        do_classifier_free_guidance = "guidance_scale" in self.inference_config
        guidance_scale = self.inference_config.get("guidance_scale", 1)

        infer_feature_exchange = self.inference_config.get(
            "infer_feature_exchange", True)

        # preview_depth or not
        preview_depth = False
        if self.wm.ctsd is not None:
            preview_depth = self.wm.ctsd.depth_net is not None and \
                "camera_intrinsics" in batch and "camera_transforms" in batch
            depth_result = []

        shift_factor = self.vae.config.shift_factor\
            if self.vae.config.shift_factor is not None else 0
        self.test_scheduler.set_timesteps(
            self.inference_config["inference_steps"], self.device)

        # ctsd
        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]

        # parse cfg
        if isinstance(guidance_scale, dict):
            min_guidance_scale = guidance_scale["min_guidance_scale"]
            max_guidance_scale = guidance_scale["max_guidance_scale"]

            guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, sequence_length).unsqueeze(0)
        else:
            guidance_scale = torch.ones(sequence_length).unsqueeze(0) * guidance_scale
        guidance_scale = guidance_scale.repeat(batch_size, 1)

        if self.wm.ctsd is not None:
            ctsd_latents_shape = tuple(batch["vae_images"].shape[:3]) + (
                self.vae.config.latent_channels,
                batch["vae_images"].shape[-2] //
                (2 ** (len(self.vae.config.down_block_types) - 1)),
                batch["vae_images"].shape[-1] //
                (2 ** (len(self.vae.config.down_block_types) - 1))
            )

            ctsd_latents = torch\
                .randn(ctsd_latents_shape, generator=self.generator)\
                .to(self.device) * getattr(self.test_scheduler, "init_noise_sigma", 1)

            ctsd_conditions = DWM.get_ctsd_conditions(
                self.sd_name, self.text_encoder, self.tokenizer,
                self.common_config, self.dynamic_cfg, batch, self.device,
                torch.float32,
                do_classifier_free_guidance=do_classifier_free_guidance)
        else:
            ctsd_latents = None
            ctsd_conditions = None

        last_ctsd_features = None

        # maskgit input & conditions & output_cache
        choice_temperature = self.inference_config.get(
            "choice_temperature", 2.0)

        if self.common_config.get("use_bev_vq_feature", False) or self.use_init_frames:
            gt_points = DWM.preprocess_points(batch, self.device)
            voxels = self.vq_point_cloud.voxelizer(gt_points)
            voxels = voxels.flatten(0, 1)

            lidar_feats = self.vq_point_cloud.lidar_encoder(voxels)
            lidar_quantized_features, _, _ = \
                self.vq_point_cloud.vector_quantizer(lidar_feats)

            maskgit_feature_from_vq = (
                torch.cat([lidar_quantized_features, lidar_quantized_features])
                if do_classifier_free_guidance
                else lidar_quantized_features
            )

        else:
            lidar_quantized_features = None

        if self.wm.maskgit is not None:

            temporal_task_code = 0           # TODO: now, only support temporal_task_code = 0

            x = self.multitask_init_code(batch_size, sequence_length, task_code=temporal_task_code)
            code_idx = torch.ones((x.shape[0], x.shape[1]), dtype=torch.int64, device=x.device) * -1
            num_unknown_code = (code_idx == -1).sum(dim=-1)
            if self.use_discrete_diffusion:
                num_unknown_code = torch.tensor(
                    [self.wm.maskgit.img_size**2]).to(x.device)

            maskgit_conditions = DWM.get_maskgit_conditions(
                self.bev_layout_encoder,
                self.common_config, batch, self.device,
                torch.float32,
                do_classifier_free_guidance=do_classifier_free_guidance
            )

            if lidar_quantized_features is not None:
                attention_mask_temporal = self._get_attention_mask_temporal(
                    temporal_task_code, lidar_quantized_features, batch_size)
            else:
                attention_mask_temporal = None
        else:
            x = None
            maskgit_conditions = None
            attention_mask_temporal = None

        if self.wm.bevformer is not None:
            bevformer_input = DWM.get_bevformer_input(
                self.common_config, batch, self.device,
                do_classifier_free_guidance=do_classifier_free_guidance)
        else:
            bevformer_input = {}

        if "bev_specification" in self.common_config:
            bev_specification = self.common_config["bev_specification"]
            bev_from_frustum_transform = DWM.get_bev_from_frustum_transform(
                batch,
                bev_specification,
                self.device
            )
            if do_classifier_free_guidance:
                bev_from_frustum_transform = torch.cat([
                    bev_from_frustum_transform, bev_from_frustum_transform
                ])
        else:
            bev_specification = None
            bev_from_frustum_transform = None

        last_maskgit_features = None
        last_depth_features = None

        if self.use_action:
            action = DWM.get_action(batch, self.act_type, self.device)
        else:
            action = None

        if action is not None:
            action_zero = torch.zeros(action[:, 0, :].shape).to(self.device)
            actions_input = torch.cat([action[:, 0, :], action_zero])
        else:
            actions_input = None

        assert len(self.test_scheduler.timesteps) == len(range(self.T))

        # TODO first to get ctsd_features
        if self.training_config.get("2d-to-3d", False):
            assert self.common_config.get('unet_input_type', None) is None          # not compatible with autoregressive
            t1 = self.test_scheduler.timesteps[0]
            if self.wm.ctsd is not None:
                # start with noise
                ctsd_latents_input = self.test_scheduler.scale_model_input(
                    ctsd_latents.repeat(2, 1, 1, 1, 1, 1) if
                    do_classifier_free_guidance else ctsd_latents,
                    t1).to(dtype=torch.float32)
            else:
                ctsd_latents_input = None
            with self.get_autocast_context():
                pred = self.wm_wrapper(
                    ctsd_latents_input,
                    (torch.cat(
                        [x, x]) if do_classifier_free_guidance else x) if x is not None else None,
                    t1.repeat(
                        batch["vae_images"].shape[0] *
                        (2 if do_classifier_free_guidance else 1)),
                    ctsd_conditions, maskgit_conditions,
                    actions_input,
                    attention_mask_temporal,
                    ctsd_features=None,
                    maskgit_features=None,
                    last_step_depth_features=None,
                    bev_from_frustum_transform=bev_from_frustum_transform,
                    bev_specification=bev_specification,
                    return_ctsd_feature=True,
                    return_maskgit_feature=True,
                    **bevformer_input
                )
                if "guidance_scale" in self.inference_config:
                    if "image" in pred:
                        ctsd_pred_uncond, ctsd_pred_cond = pred["image"].chunk(
                            2)
                        ctsd_pred = ctsd_pred_uncond + guidance_scale * \
                            (ctsd_pred_cond - ctsd_pred_uncond)

                last_ctsd_features = pred["ctsd_down_feats"]

                ctsd_latents = self.test_scheduler.step(
                    ctsd_pred, t1, ctsd_latents).prev_sample

        zero_tensor = torch.tensor(
            [0], device=self.test_scheduler.timesteps.device)
        unet_timesteps = torch.cat([self.test_scheduler.timesteps[1:], zero_tensor]) \
            if self.training_config.get("2d-to-3d", False) else self.test_scheduler.timesteps
        if self.common_config.get('unet_input_type', None) is not None:
            vae_images = batch["vae_images"].flatten(0, 2)
            vae_images = self.image_processor.preprocess(vae_images.to(self.device))  # -1~1
            vae_images = rearrange(vae_images, "(b f k) c h w -> b f k c h w", f=sequence_length, k=view_count) 
            image_latents = self._encode_vae_image(
                vae_images, do_classifier_free_guidance=do_classifier_free_guidance)

        for t1, t2 in zip(unet_timesteps, range(self.T)):

            timesteps = t1.unsqueeze(-1).unsqueeze(-1)\
                .repeat(*batch["vae_images"].shape[:3])
            if self.wm.ctsd is not None:

                if isinstance(
                    self.wm.ctsd, diffusers.UNetSpatioTemporalConditionModel
                ):
                    ctsd_latents_input = self.test_scheduler.scale_model_input(
                        ctsd_latents, t1).to(dtype=torch.float32)

                if do_classifier_free_guidance:
                    ctsd_latents_input = torch.cat(
                        [ctsd_latents_input, ctsd_latents_input])
                    timesteps = torch.cat([timesteps, timesteps])

                # svd style autoregressive policy
                cxt_condition_mask = torch.cat([
                    ctsd_latents_input.new_zeros(batch_size),
                    ctsd_latents_input.new_ones(batch_size)
                ])
                if self.common_config.get('unet_input_type', None) is not None:
                    ctsd_latents_input = self.gen_unet_input(
                        ctsd_latents_input, image_latents, infer=True,
                        cxt_condition_mask=cxt_condition_mask, rescale=True)

            else:
                ctsd_latents_input = None

            extra_cfg = dict()

            if self.use_init_frames:
                x = self.process_inference_input(
                    temporal_task_code, x, sequence_length,
                    lidar_quantized_features, ctype='pre')

            # 1. prepare depth
            with self.get_autocast_context():
                if self.joint_forward_policy == 'separate':
                    assert not infer_feature_exchange
                    extra_cfg["forward_depth"] = True
                    last_depth_features = self.wm_wrapper(
                        ctsd_latents_input,
                        (torch.cat([x, x])
                         if do_classifier_free_guidance else x)
                        if x is not None else None,
                        timesteps,
                        ctsd_conditions, maskgit_conditions,
                        actions_input,
                        attention_mask_temporal,
                        **extra_cfg,
                    )['depth']

                    extra_cfg["forward_depth"] = False

                pred = self.wm_wrapper(
                    ctsd_latents_input,
                    (torch.cat([x, x]) if do_classifier_free_guidance else x)
                    if x is not None else None,
                    timesteps,
                    ctsd_conditions, maskgit_conditions,
                    actions_input,
                    attention_mask_temporal,
                    ctsd_features=last_ctsd_features,
                    maskgit_features=last_maskgit_features,
                    last_step_depth_features=last_depth_features,
                    bev_from_frustum_transform=bev_from_frustum_transform,
                    bev_specification=bev_specification,
                    return_ctsd_feature=True,
                    return_maskgit_feature=True,
                    **extra_cfg,
                    **bevformer_input
                )

            if "guidance_scale" in self.inference_config:

                if "image" in pred:
                    ctsd_pred_uncond, ctsd_pred_cond = pred["image"].chunk(2)
                    guidance_scale = guidance_scale.to(
                        ctsd_pred_cond.device, ctsd_pred_cond.dtype)
                    ctsd_pred = ctsd_pred_uncond + _append_dims(guidance_scale, ctsd_pred_cond.ndim) * (
                        ctsd_pred_cond - ctsd_pred_uncond)

                if "lidar" in pred:
                    maskgit_pred_uncond, maskgit_pred_cond = pred["lidar"].chunk(
                        2)
                    guidance_scale = guidance_scale.to(maskgit_pred_cond.device, maskgit_pred_cond.dtype)
                    maskgit_pred = maskgit_pred_uncond + _append_dims(
                        guidance_scale, maskgit_pred_cond.ndim+1).flatten(0, 1) * \
                        (maskgit_pred_cond - maskgit_pred_uncond)

            else:
                if "image" in pred:
                    ctsd_pred = pred["image"]

                if "lidar" in pred:
                    maskgit_pred = pred["lidar"]

            if self.wm.maskgit is not None:
                if self.use_init_frames:
                    maskgit_pred = self.process_inference_input(
                        temporal_task_code, maskgit_pred, sequence_length,
                        lidar_quantized_features, ctype='after')

                if t2 < 10:
                    maskgit_pred[..., self.BLANK_CODE] = -10000

                sample_ids = torch.distributions.Categorical(
                    logits=maskgit_pred).sample()
                prob = torch.softmax(maskgit_pred, dim=-1)
                prob = torch.gather(
                    prob, -1, sample_ids.unsqueeze(-1)).squeeze(-1)
                sample_ids[code_idx != -1] = code_idx[code_idx != -1]

                if not self.use_discrete_diffusion:
                    prob[code_idx != -1] = 1e3
                else:
                    if self.common_config.get('discrete_diffusion_max_exist', False):
                        prob[code_idx != -1] = 1e3
                    if self.common_config.get('update_unmask_pos', False):
                        raise NotImplementedError

                ratio = (999 - t1) * 0.001
                mask_ratio = self.gamma(ratio)

                mask_len = num_unknown_code * mask_ratio  # all code len
                mask_len = torch.minimum(mask_len, num_unknown_code - 1)
                mask_len = mask_len.clamp(min=1).long()

                temperature = choice_temperature * (1.0 - ratio)
                gumbels = -torch.empty_like(
                    prob, memory_format=torch.legacy_contiguous_format).exponential_().log()
                confidence = prob.log() + temperature * gumbels

                cutoff = torch.sort(confidence, dim=-1)[0][
                    torch.arange(
                        mask_len.shape[0], device=mask_len.device), mask_len
                ].unsqueeze(1)

                mask = confidence < cutoff

                x = self.vq_point_cloud.vector_quantizer.get_codebook_entry(
                    sample_ids)

                code_idx = sample_ids.clone()

                if t2 + 1 < self.T:
                    code_idx[mask] = -1
                    x[mask] = self.wm.maskgit.mask_token

            if self.wm.ctsd is not None:
                ctsd_latents = self.test_scheduler.step(
                    ctsd_pred, t1, ctsd_latents).prev_sample

            # prev feature for next step
            if infer_feature_exchange:
                last_ctsd_features = pred["ctsd_down_feats"] if "ctsd_down_feats" in pred else None
                last_maskgit_features = pred["maskgit_feature"] if not self.training_config.get(
                    "2d-to-3d", False) else None
                last_depth_features = pred["depth"] if "depth" in pred else None
                if self.common_config.get("use_bev_vq_feature", False):
                    last_maskgit_features = maskgit_feature_from_vq

            else:
                last_ctsd_features = None
                last_maskgit_features = None
                # last_depth_features = None

            # update the depth visualization
            if preview_depth and ("depth" in pred or last_depth_features is not None):
                depth_features = pred["depth"] if pred["depth"] is not None else last_depth_features
                depth_features = depth_features.chunk(2)[1] \
                    if do_classifier_free_guidance else depth_features

                noisy_image_tensor = dwm.functional.memory_efficient_split_call(
                    self.vae,
                    ctsd_latents.flatten(0, 2).to(dtype=self.vae.dtype),
                    lambda block, tensor: block.decode(
                        tensor / block.config.scaling_factor + shift_factor,
                        return_dict=False
                    )[0],
                    self.common_config.get("memory_efficient_batch", -1))
                noisy_images = self.image_processor.postprocess(
                    noisy_image_tensor, output_type="pt")

                depth_images = (
                    1 - depth_features.argmax(-3) / depth_features.shape[-3]
                ).flatten(0, 2).unsqueeze(1)
                depth_images = torch.nn.functional.interpolate(
                    depth_images, noisy_images.shape[-2:])
                depth_images = depth_images\
                    .unflatten(0, ctsd_latents.shape[:3])\
                    .repeat_interleave(3, dim=-3).permute(3, 0, 1, 4, 2, 5)
                noisy_images = noisy_images\
                    .unflatten(0, ctsd_latents.shape[:3])\
                    .permute(3, 0, 1, 4, 2, 5)
                depth_result.append(
                    torch.cat([noisy_images, depth_images], -3).flatten(-2)
                    .flatten(-4, -2))

        if self.wm.ctsd is not None:
            image_tensor = dwm.functional.memory_efficient_split_call(
                self.vae, ctsd_latents.flatten(0, 2).to(dtype=self.vae.dtype),
                lambda block, tensor: block.decode(
                    tensor / block.config.scaling_factor + shift_factor,
                    return_dict=False
                )[0],
                self.common_config.get("memory_efficient_batch", -1))

        result = {}
        if self.wm.ctsd is not None:
            result["images"] = self.image_processor.postprocess(
                image_tensor, output_type=output_type)

            if preview_depth and "depth" in pred:
                result["depth"] = depth_result
                result["depth_features"] = depth_features

        if self.wm.maskgit is not None:
            if self.use_init_frames:
                x = self.process_inference_input(
                    temporal_task_code, x, sequence_length,
                    lidar_quantized_features, ctype='fin')

            result["points"] = x
            _, lidar_voxel = self.vq_point_cloud.lidar_decoder(x)
            generated_ego_space_points = DWM.voxels2points(
                self.vq_point_cloud,
                dwm.functional.gumbel_sigmoid(
                    lidar_voxel.unflatten(0, batch["pts"].shape[:2]),
                    hard=True, generator=self.generator))
            result["raw_points"] = DWM.postprocess_points(
                batch, generated_ego_space_points)

            # ==== Eval Point Cloud
            if self.inference_config.get('cop4d_visual_point_cloud', False):
                lidar_density, lidar_voxel = self.vq_point_cloud.lidar_decoder(x)
                generated_sample_v = dwm.functional.gumbel_sigmoid(
                    lidar_voxel, hard=True, generator=self.generator)
                result['voxel_sequence'] = generated_sample_v
                result['voxel_sequence_gt'] = voxels

            if self.inference_config.get('cop4d_eval_point_cloud', False):
                pooled_voxels = F.max_pool3d(generated_sample_v, (4, 8, 8))
                pooled_voxels = pooled_voxels.unflatten(0, batch["pts"].shape[:2])
                _, _, rec_points, alphainv_lasts = self.vq_point_cloud.ray_render_dvgo(
                    lidar_density.unflatten(0, batch["pts"].shape[:2]), gt_points,
                    pooled_voxels,
                    offsets=torch.tensor(
                        [self.common_config["ray_cast_center"]],
                        device=self.device
                    ).repeat(batch_size, 1), return_alpha_last=True)
                rec_points = [j for i in rec_points for j in i]
                if self.inference_config["filter_out_roi"]:
                    for idx in range(len(alphainv_lasts)):
                        rec_points[idx] = rec_points[idx][alphainv_lasts[idx]<=0.5]
                result['geneted_points_r'] = [pt.detach().cpu().numpy() for pt in rec_points]
                gt_points = sum(gt_points, [])
                result['gt_points'] = [pt.detach().cpu().numpy() for pt in gt_points]

        return result

    @torch.no_grad()
    def preview_pipeline(
        self, batch: dict, output_path: str, global_step: int
    ):
        sequence_length = batch["pts"].shape[1]
        pipeline_output = self.inference_pipeline(batch, "pt")

        os.makedirs(os.path.join(output_path, "preview"), exist_ok=True)

        if self.should_save and "points" in pipeline_output:
            gt_voxels = self.vq_point_cloud.voxelizer(
                DWM.preprocess_points(batch, self.device))
            _, lidar_voxel = self.vq_point_cloud.lidar_decoder(
                pipeline_output["points"])
            generated_sample_v = dwm.functional.gumbel_sigmoid(
                lidar_voxel.unflatten(0, gt_voxels.shape[:2]), hard=True,
                generator=self.generator)

            preview_tensor = dwm.utils.preview.make_lidar_preview_tensor(
                gt_voxels, generated_sample_v, batch, self.inference_config)
            if sequence_length == 1:
                image_output_path = os.path.join(
                    output_path, "preview", "{}_lidar.png".format(global_step))
                torchvision.transforms.functional.to_pil_image(preview_tensor)\
                    .save(image_output_path)
            else:
                video_output_path = os.path.join(
                    output_path, "preview", "{}_lidar.mp4".format(global_step))
                dwm.utils.preview.save_tensor_to_video(
                    video_output_path, "libx264", batch["fps"][0].item(),
                    preview_tensor)

        if self.should_save and "images" in pipeline_output:
            preview_tensor = dwm.utils.preview.make_ctsd_preview_tensor(
                pipeline_output["images"], batch, self.inference_config)
            if sequence_length == 1:
                image_output_path = os.path.join(
                    output_path, "preview", "{}.png".format(global_step))
                torchvision.transforms.functional.to_pil_image(preview_tensor)\
                    .save(image_output_path)
            else:
                video_output_path = os.path.join(
                    output_path, "preview", "{}.mp4".format(global_step))
                dwm.utils.preview.save_tensor_to_video(
                    video_output_path, "libx264", batch["fps"][0].item(),
                    preview_tensor)

            preview_depth = (
                self.wm.ctsd.depth_net is not None or
                self.wm.ctsd.depth_decoder is not None) and \
                "camera_intrinsics" in batch and "camera_transforms" in batch
            if preview_depth and "depth" in pipeline_output:
                os.makedirs(os.path.join(
                    output_path, "preview_depth"), exist_ok=True)
                video_output_path = os.path.join(
                    output_path, "preview_depth", "{}.mp4".format(global_step))
                dwm.utils.preview.save_tensor_to_video(
                    video_output_path, "libx264", 5, pipeline_output["depth"])

    @torch.no_grad()
    def evaluate_pipeline(
        self, global_step: int, dataset_length: int,
        validation_dataloader: torch.utils.data.DataLoader,
        validation_datasampler=None
    ):
        optional_metric_type = self.inference_config.get('optional_metric_type', None)
        if optional_metric_type in ['FVD', 'FVD_FID', 'ALL']:
            if self.inference_config.get('cop4d_eval_point_cloud', False) and optional_metric_type == 'ALL':
                cop4d_metrics = defaultdict(lambda: 0)
            else:
                cop4d_metrics = None
            evaluate_svd(
                optional_metric_type, self, self.should_save, global_step,
                dataset_length, validation_dataloader, validation_datasampler,
                cop4d_metrics=cop4d_metrics)
            return          # now, multiple metrics is not supported
        elif optional_metric_type in ['COP4D']:
            cop4d_metrics = defaultdict(lambda: 0)
            evaluate_svd_3d(
                optional_metric_type, self, self.should_save, global_step,
                dataset_length, validation_dataloader, validation_datasampler,
                cop4d_metrics=cop4d_metrics)
            return          # now, multiple metrics is not supported

        # avg_chamfer_scores = []
        metrics = defaultdict(lambda: 0)

        if torch.distributed.is_initialized():
            validation_datasampler.set_epoch(0)

        world_size = torch.distributed.get_world_size() \
            if torch.distributed.is_initialized() else 1
        iteration_count = \
            self.inference_config["evaluation_item_count"] // world_size \
            if "evaluation_item_count" in self.inference_config else None
        for (idx, batch) in enumerate(validation_dataloader):
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            batch_size = batch["fps"].shape[0]
            if (
                iteration_count is not None and
                idx * batch_size >= iteration_count
            ):
                break

            if self.inference_config.get("ar_infer", False):
                pass  # TODO: the auto-regressively inference for long video
            else:
                pipeline_output = self.inference_pipeline(batch, "pt")

            if "fid" in self.metrics:
                self.metrics["fid"].update(
                    batch["vae_images"].flatten(0, 2).to(self.device),
                    real=True)
                self.metrics["fid"].update(
                    pipeline_output["images"], real=False)

            if "fvd" in self.metrics:
                visible_frame = self.common_config.get("visible_frame", 0)
                self.metrics["fvd"].update(
                    rearrange(
                        batch["vae_images"][:, visible_frame:].to(self.device),
                        "b t k c h w -> (b k) t c h w"),
                    real=True)
                self.metrics["fvd"].update(
                    rearrange(
                        pipeline_output["images"],
                        "(b t k) c h w -> (b k) t c h w", b=batch_size,
                        k=batch["vae_images"].shape[2])[:, visible_frame:],
                    real=False)

            if "rmse" in self.metrics:
                preds_target_generator = dwm.pipelines.ctsd\
                    .CrossviewTemporalSD.enum_depth_preds_and_targets(
                        *batch["vae_images"].shape[:3],
                        self.wm.ctsd.depth_frustum_range, batch,
                        pipeline_output["depth_features"],
                        self.common_config["point_count_limit_per_view"],
                        self.common_config["point_bundle_size"], self.device)
                for preds, target in preds_target_generator:
                    self.metrics["rmse"].update(preds, target)

            # lidar
            metrics["count"] += 1
            if self.wm.maskgit is not None:
                batch_size = len(batch["lidar_points"])
                # The evaluation only support batch size == 1
                points = DWM.preprocess_points(batch, self.device)

                lidar_density, lidar_voxel = self.vq_point_cloud.lidar_decoder(
                    pipeline_output["points"])
                lidar_voxel = lidar_voxel.unflatten(0, batch["pts"].shape[:2])
                generated_sample_v = dwm.functional.gumbel_sigmoid(
                    lidar_voxel, hard=True, generator=self.generator)
                pooled_voxels = F.max_pool3d(generated_sample_v, (4, 8, 8))

                _, _, lidar_recs, alphainv_lasts = self.vq_point_cloud.ray_render_dvgo(
                    lidar_density.unflatten(0, batch["pts"].shape[:2]),
                    points, pooled_voxels,
                    offsets=torch.tensor(
                        [self.common_config["ray_cast_center"]],
                        device=self.device
                    ).repeat(batch_size, 1), return_alpha_last=True)

                lidar_recs = [j for i in lidar_recs for j in i]
                for idx in range(len(alphainv_lasts)):
                    lidar_recs[idx] = lidar_recs[idx][alphainv_lasts[idx] <= 0.5]

                lidar_recs = [pt.detach().cpu() for pt in lidar_recs]
                lidar_recs = lidar_recs[0]

                gt_points = points
                gt_points = [j.detach().cpu() for i in gt_points for j in i][0]

                chamfer_score = compute_chamfer_distance(
                    lidar_recs, gt_points, self.device)
                metrics["chamfer_distance"] += chamfer_score.item()

                chamfer_score_inner_40 = compute_chamfer_distance_inner(
                    lidar_recs, gt_points, self.device, pc_range=[-40.0, -40.0, -3, 40.0, 40.0, 5], )
                metrics["chamfer_distance_40"] += chamfer_score_inner_40.item()

                chamfer_score_inner_30 = compute_chamfer_distance_inner(
                    lidar_recs, gt_points, self.device, pc_range=[-30.0, -30.0, -3, 30.0, 30.0, 5], )
                metrics["chamfer_distance_30"] += chamfer_score_inner_30.item()

                # TODO: ray cast center as origin
                origin = torch.tensor(self.common_config["ray_cast_center"])
                l1_error, absrel_error, l1_error_med, absrel_error_med = compute_ray_errors(
                    lidar_recs, gt_points, origin, self.device, pc_range=[-50.0, -50.0, -3, 50.0, 50.0, 5], )
                metrics["l1_error_mean"] += l1_error
                metrics["absrel_error_mean"] += absrel_error
                metrics["l1_error_median"] += l1_error_med
                metrics["absrel_error_median"] += absrel_error_med

                l1_error, absrel_error, l1_error_med, absrel_error_med = compute_ray_errors(
                    lidar_recs, gt_points, origin, self.device, pc_range=[-40.0, -40.0, -3, 40.0, 40.0, 5], )
                metrics["l1_error_mean_40"] += l1_error
                metrics["absrel_error_mean_40"] += absrel_error
                metrics["l1_error_median_40"] += l1_error_med
                metrics["absrel_error_median_40"] += absrel_error_med

                l1_error, absrel_error, l1_error_med, absrel_error_med = compute_ray_errors(
                    lidar_recs, gt_points, origin, self.device, pc_range=[-30.0, -30.0, -3, 30.0, 30.0, 5], )
                metrics["l1_error_mean_30"] += l1_error
                metrics["absrel_error_mean_30"] += absrel_error
                metrics["l1_error_median_30"] += l1_error_med
                metrics["absrel_error_median_30"] += absrel_error_med

        metrics = dict(metrics)
        if torch.distributed.is_initialized():
            all_metrics = [None for _ in range(
                torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(all_metrics, metrics)

        else:
            # all_avg_chamfer_scores = avg_chamfer_scores
            all_metrics = [metrics]

        merge_metrics = defaultdict(lambda: 0)
        for mt in all_metrics:
            for k, v in mt.items():
                merge_metrics[k] += v

        count = merge_metrics.pop('count')
        for k, v in merge_metrics.items():
            v_avg = v / count
        # chamfer = np.array(all_avg_chamfer_scores).mean()
            if self.should_save:
                print(f"{k}: {v_avg:.5f} with count {count}")

                self.summary.add_scalar(f"evaluation/{k}", v_avg, global_step)

        text = "Step {},".format(global_step)
        for k, metric in self.metrics.items():
            value = metric.compute()
            metric.reset()
            text += " {}: {:.3f}".format(k, value)
            if self.should_save:
                self.summary.add_scalar(
                    "evaluation/{}".format(k), value, global_step)

        if self.should_save:
            print(text)
