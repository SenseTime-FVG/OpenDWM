import argparse
import dwm.common
import json
import os
import torch
import random
import torch.nn.functional as F
import copy
import numpy as np
import imageio
from easydict import EasyDict as edict
from collections import defaultdict
from einops import rearrange
from torchmetrics.image.fid import FrechetInceptionDistance
from dwm.utils.fvd_utils.fvd_tats import get_logits, frechet_distance
from dwm.utils.fvd_utils.pytorch_i3d import InceptionI3d

import open3d as o3d
from dwm.utils.metrics_copilot4d import (
    compute_chamfer_distance, compute_chamfer_distance_inner, compute_ray_errors)
from dwm.utils.metrics_base import DWMFVD

def is_main_process():
    if not torch.distributed.is_initialized():
        return True
    else:
        return torch.distributed.get_rank() == 0

def get_world_size():
    if not torch.distributed.is_initialized():
        return True
    else:
        return torch.distributed.get_world_size()

def draw_bev_lidar_from_list(voxels_list, pth):
    with imageio.get_writer(pth, fps=2) as video_writer:
        for voxels in voxels_list:
            if isinstance(voxels, np.ndarray):
                voxels = torch.from_numpy(voxels)
            image = (voxels.max(dim=0)[0][:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255).astype(np.uint8)
            video_writer.append_data(image)


def evaluate_cop4d(pipeline, result, metrics, num_frames=None):
    num_frames_past = pipeline.common_config['unet_input_cfg']['num_init_frames']

    gt_points_all = result['gt_points']
    geneted_points_all = result['geneted_points_r']
    skip_l1_align = pipeline.inference_config.get("skip_l1_align", False)
    offsets = None                      # TODO

    th_num_frames, th_frames_past = num_frames, num_frames_past
    gt_points_all = list(map(lambda x: x[1], filter(lambda x: (x[0]%th_num_frames) >= th_frames_past, enumerate(gt_points_all))))
    geneted_points_all = list(map(lambda x: x[1], filter(lambda x: (x[0]%th_num_frames) >= th_frames_past, enumerate(geneted_points_all))))

    for fid in range(len(gt_points_all)):
        gt_points = gt_points_all[fid]
        geneted_points_v = geneted_points_all[fid]

        # gt
        point_cloud_gt = o3d.geometry.PointCloud()
        point_cloud_gt.points = o3d.utility.Vector3dVector(gt_points)

        # output_origin = None # torch.zeros_like(output_origin)
        gt_pcd = torch.from_numpy(np.array(point_cloud_gt.points))             # n,3
        # voxel
        point_cloud_v = o3d.geometry.PointCloud()
        point_cloud_v.points = o3d.utility.Vector3dVector(geneted_points_v)

        pred_pcd = torch.from_numpy(np.array(point_cloud_v.points))           # n,3
        if offsets is not None:
            assert False
        else:
            origin = pred_pcd.new_zeros(pred_pcd.shape[-1])           # in batch, fid -> 3

        pc_range = pipeline.inference_config["pc_range"]
        metrics["count"] += 1
        metrics["chamfer_distance"] += compute_chamfer_distance(pred_pcd, gt_pcd, pipeline.device).item()
        metrics["chamfer_distance_inner"] += compute_chamfer_distance_inner(pred_pcd, gt_pcd, pipeline.device, pc_range=pc_range).item()
        # The dtype of origin code is not very clear
        l1_error, absrel_error, l1_error_med, absrel_error_med = compute_ray_errors(pred_pcd, gt_pcd, origin, pipeline.device, pipeline=pipeline, pc_range=pc_range, skip_l1_align=skip_l1_align)
        metrics["l1_error_mean"] += l1_error
        metrics["absrel_error_mean"] += absrel_error
        metrics["l1_error_med"] += l1_error_med
        metrics["absrel_error_med"] += absrel_error_med
        # print(metrics)

@torch.no_grad()
def evaluate_svd(metric_type, pipeline, should_save: bool, global_step: int, dataset_length: int,
    validation_dataloader: torch.utils.data.DataLoader,
    validation_datasampler=None,
    cop4d_metrics=None):
    """
    params in inference_config:
        1. from_nviews
    """
    # eval FVD_FID
    assert metric_type in ['FID', 'FVD', 'OLD_FID', 'FVD_FID', 'ALL']
    metric_tfm = lambda x: rearrange(x, "b f k c h w -> (b k) f c h w")
    metric = DWMFVD(device="cuda")

    from_nviews = pipeline.inference_config.get("from_nviews", True)
    num_videos = 0

    if should_save:
        from dwm.utils.visualizer import SimpleVisualizer
        visualizer_save_name = pipeline.inference_config['visualizer_cfg'].pop("visualizer_save_name", 'vis_data')
        visualizer = SimpleVisualizer(os.path.join(pipeline.output_path, visualizer_save_name, 'vis_image'), **pipeline.inference_config['visualizer_cfg'])
    else:
        visualizer = None
    
    if metric_type == 'FID':
        metric = FrechetInceptionDistance(normalize=True, sync_on_compute=True).to("cuda")
        dwm_test_img_num = 10000 // get_world_size()
        dwm_test_vid_num = 2048 // get_world_size()
        if from_nviews:
            dwm_test_vid_num //= 6          # nviews
    elif metric_type == 'FVD' or metric_type == 'FVD_FID' or metric_type == 'ALL':
        metric = DWMFVD(device="cuda")
        dwm_test_vid_num = 900 // get_world_size()
        dwm_test_img_num = dwm_test_vid_num
        if from_nviews:
            dwm_test_vid_num //= 6          # nviews
    elif metric_type == 'OLD_FID':
        metric = FrechetInceptionDistance(normalize=True, sync_on_compute=True).to("cuda")
        dwm_test_vid_num = dataset_length // get_world_size()
        dwm_test_img_num = None

    real_images = []
    fake_images = []

    if should_save:
        print(f"======Test {dwm_test_vid_num} per process...")
    if pipeline.ddp:
        validation_datasampler.set_epoch(0)
    skip_ar_for_3d = True
    infer_layout_mode = pipeline.inference_config.get("infer_layout_mode", "first")
    assert infer_layout_mode in ["first", "all", "none"]

    for n, data in enumerate(validation_dataloader):
        if num_videos + len(data["vae_images"]) > dwm_test_vid_num:
            continue
        num_videos += len(data["vae_images"])
        if should_save:
            print(f"======Test {num_videos} of {dwm_test_vid_num}")
        # Real
        # if from_nviews:
        batch_size = data['vae_images'].shape[0]
        nviews = data['vae_images'].shape[2]
        real_images.append(metric_tfm(data["vae_images"]))
        # else:
        #     real_images.append(metric_tfm(data["vae_images"]))
        # Fake
        with torch.no_grad():
            if pipeline.inference_config['ar_infer']:
                """
                Now, only support scene_description, as this is clip consistent
                """
                num_frames, cond_frames = pipeline.inference_config['single_step_frames'], pipeline.common_config['unet_input_cfg']['num_init_frames']
                ar_infer_steps = (data['vae_images'].shape[1]-cond_frames)//(num_frames-cond_frames)
                if should_save:
                    print("===AR infer steps: ", ar_infer_steps)
                
                cur_input = dict(
                    vae_images=copy.deepcopy(data['vae_images'][:, :num_frames]),
                    fps=data['fps'],
                    pts=data['pts'][:, :num_frames]
                    )
                if "convnext_images" in data:
                    cur_input["convnext_images"] = copy.deepcopy(data['convnext_images'][:, :num_frames])
                if "lidar_points_raw" in data:
                    cur_input["lidar_points_raw"] = copy.deepcopy([v[:num_frames] for v in data['lidar_points_raw']])
                elif "lidar_points" in data:
                    cur_input["lidar_points"] = copy.deepcopy([v[:num_frames] for v in data['lidar_points']])
                if "ego_steering" in data:
                    cur_input["ego_steering"] = copy.deepcopy(data['ego_steering'][:, :num_frames])
                    cur_input["ego_speed"] = copy.deepcopy(data['ego_speed'][:, :num_frames])
                    
                cur_input['3dbox_images'] = copy.deepcopy(data['3dbox_images'][:, :num_frames])
                cur_input['hdmap_images'] = copy.deepcopy(data['hdmap_images'][:, :num_frames])
                if infer_layout_mode == "none":
                    print("Init generation without layout")
                    cur_input['3dbox_images'] *= 0
                    cur_input['hdmap_images'] *= 0
                for k in ['ego_transforms', 'lidar_transforms', 'camera_transforms', 'image_size', 'camera_intrinsics']:
                    if k in data:
                        cur_input[k] = data[k][:, :num_frames]
                if isinstance(data['clip_text'][0], (tuple, list)):          # for unshared text
                    cur_input['clip_text'] = [data['clip_text'][i][:num_frames] for i in range(len(data['clip_text']))]
                else:
                    cur_input['clip_text'] = data['clip_text']

                cur_pos = 0
                cur_gen_rts = []
                cur_gen_pcs, cur_gt_pcs = [], []
                for cid in range(ar_infer_steps):
                    if pipeline.inference_config.get("auto_infer_cxt", False):
                        pipeline.dynamic_cfg['infer_without_cxt'] = (cid == 0)
                    if pipeline.inference_config.get("auto_skip_interaction", False):
                        pipeline.dynamic_cfg['infer_without_interaction'] = (cid > 0)
                    if pipeline.inference_config.get("auto_skip_vis", False):
                        pipeline.dynamic_cfg['infer_without_vis'] = (cid > 0)
                    pipeline_output = pipeline.inference_pipeline(cur_input, "pt")            # btchw, 0-1
                    
                    if isinstance(pipeline_output, dict):
                        fake_image = pipeline_output['images']
                    else:
                        fake_image = pipeline_output[0]

                    if cop4d_metrics is not None:
                        if skip_ar_for_3d and cid > 0:
                            pass
                        else:
                            evaluate_cop4d(pipeline, pipeline_output, cop4d_metrics, num_frames=num_frames)

                    fake_image = rearrange(fake_image, "(b f k) c h w -> b f k c h w", f=num_frames, k=nviews)
                    # ===Transform to target distribution
                    # ======Convnext
                    if pipeline.inference_config['cxt_transform'] == 'convnext':
                        target_shape = [1 for _ in range(fake_image.ndim - 3)] + [-1, 1, 1]
                        mean = fake_image.new_tensor([0.485, 0.456, 0.406]).view(target_shape)
                        std = fake_image.new_tensor([0.229, 0.224, 0.225]).view(target_shape)
                        fake_image_tfm = (fake_image - mean) / std
                    else:
                        fake_image_tfm = fake_image

                    # ===Update cond info
                    # if from_nviews:
                    #     nviews = data['vae_images'].shape[2]
                        # fake_image_tfm = rearrange(fake_image_tfm, "b f c h (k w) -> b f k c h w", k=nviews)
                    if pipeline.inference_config['cxt_transform'] == 'convnext':
                        cur_input['convnext_images'][:, :cond_frames] = fake_image_tfm[:, -cond_frames:]
                    else:
                        cur_input['vae_images'][:, :cond_frames] = fake_image_tfm[:, -cond_frames:]
                    
                    # ===Update gt info
                    cur_pos += (num_frames-cond_frames)
                    if should_save:
                        print("===New start", cur_pos)
                    if metric_type == 'ALL' and (not pipeline.dynamic_cfg.get("infer_without_interaction", False)):
                        if "lidar_points_raw" in data and ar_infer_steps > 1:
                            # Not support lidar_points now
                            assert batch_size == 1, "Not support batch_size > 1"
                            geneted_points_r = pipeline_output['geneted_points_r']
                            cur_input["lidar_points_raw"] = copy.deepcopy([v[cur_pos:cur_pos+num_frames] for v in data['lidar_points_raw']])
                            cur_input["lidar_points_raw"][0][:cond_frames] = [torch.from_numpy(v) for v in geneted_points_r[-cond_frames:]]
                        elif "lidar_points" in data and ar_infer_steps > 1:
                            # Not support lidar_points now
                            assert batch_size == 1, "Not support batch_size > 1"
                            geneted_points_r = pipeline_output['geneted_points_r']
                            cur_input["lidar_points"] = copy.deepcopy([v[cur_pos:cur_pos+num_frames] for v in data['lidar_points']])
                            cur_input["lidar_points"][0][:cond_frames] = [torch.from_numpy(v) for v in geneted_points_r[-cond_frames:]]
                    if "ego_steering" in data:
                        cur_input['ego_steering'] = data['ego_steering'][:, cur_pos:cur_pos+num_frames]
                        cur_input['ego_speed'] = data['ego_speed'][:, cur_pos:cur_pos+num_frames]
                    for k in ['ego_transforms', 'lidar_transforms', 'camera_transforms', 'image_size', 'camera_intrinsics']:
                        if k in data:
                            cur_input[k] = data[k][:, cur_pos:cur_pos+num_frames]
                    if isinstance(data['clip_text'][0], (tuple, list)):          # for unshared text
                        cur_input['clip_text'] = [data['clip_text'][i][cur_pos:cur_pos+num_frames] for i in range(len(data['clip_text']))]
                    else:
                        cur_input['clip_text'] = data['clip_text']
                    if infer_layout_mode == "all":
                        cur_input['3dbox_images'] = copy.deepcopy(data['3dbox_images'][:, cur_pos:cur_pos+num_frames])
                        cur_input['hdmap_images'] = copy.deepcopy(data['hdmap_images'][:, cur_pos:cur_pos+num_frames])
                    else:
                        print("Generation without layout")
                        cur_input['3dbox_images'] *= 0
                        cur_input['hdmap_images'] *= 0

                    if cid == 0:
                        cur_gen_rts.append(fake_image)
                    else:
                        cur_gen_rts.append(fake_image[:, cond_frames:])

                    # Visualize PC
                    if metric_type == 'ALL' and (not pipeline.dynamic_cfg.get("infer_without_vis", False)):
                        assert batch_size == 1
                        if should_save:
                            generated_sample_v = pipeline_output['voxel_sequence']
                            voxels = pipeline_output['voxel_sequence_gt']
                            if cid == 0:
                                cur_gen_pcs.append(generated_sample_v)
                                cur_gt_pcs.append(voxels)
                            else:
                                cur_gen_pcs.append(generated_sample_v[cond_frames:])
                                cur_gt_pcs.append(voxels[cond_frames:])
                            if cid == ar_infer_steps -1:
                                folder_name = os.path.join(visualizer.pth, "preview", f'FVD_{global_step}')
                                os.makedirs(folder_name, exist_ok=True)
                                cur_gen_pcs = torch.cat(cur_gen_pcs, dim=0)
                                cur_gt_pcs = torch.cat(cur_gt_pcs, dim=0)

                                draw_bev_lidar_from_list(cur_gen_pcs, f"{folder_name}/{n:03d}_generated_v.mp4")
                                draw_bev_lidar_from_list(cur_gt_pcs, f"{folder_name}/{n:03d}_voxel_gt.mp4")

                cur_gen_rts = torch.cat(cur_gen_rts, dim=1)

                # if from_nviews:
                #     nviews = data['vae_images'].shape[2]
                #     vis_images = rearrange(cur_gen_rts, "b f c h (k w) -> b f k c h w", k=nviews)
                #     cur_gen_rts = rearrange(cur_gen_rts, "b f c h (k w) -> (b k) f c h w", k=nviews)
                # else:
                #     nviews = 1
                vis_images = cur_gen_rts            # b f k c h w
                fake_images.append(metric_tfm(cur_gen_rts))
                if should_save:
                    visualizer.uni_preview(data, vis_images, global_step, vis=False)
                    visualizer.draw(global_step, etype=metric_type, inner_step=n)
                    visualizer.clear()
                torch.cuda.empty_cache()
            else:
                pipeline_output = pipeline.inference_pipeline(data, "pt")
                if isinstance(pipeline_output, dict):
                    fake_image = pipeline_output['images']
                else:
                    fake_image = pipeline_output[0]

                if cop4d_metrics is not None:
                    evaluate_cop4d(pipeline, pipeline_output, cop4d_metrics)

                # if from_nviews:
                #     nviews = data['vae_images'].shape[2]
                #     # for compute metrics, k is moved to batch, for visualize, k is moved to independent dimension
                #     vis_images = rearrange(fake_image['samples'], "b f c h (k w) -> b f k c h w", k=nviews)
                #     fake_image['samples'] = rearrange(fake_image['samples'], "b f c h (k w) -> (b k) f c h w", k=nviews)
                # else:
                #     nviews = 1
                #     vis_images = fake_image['samples']
                fake_images.append(metric_tfm(fake_image))
                if should_save:
                    visualizer.uni_preview(data, fake_image, global_step, vis=False)
                    visualizer.draw(global_step, etype=metric_type, inner_step=n)
                    visualizer.clear()
                torch.cuda.empty_cache()
    
    if metric_type == 'ALL':
        assert cop4d_metrics is not None
        metrics = dict(cop4d_metrics)
        if pipeline.ddp:
            all_metrics = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(all_metrics, metrics)
        else:
            all_metrics = [metrics]
        merge_metrics = defaultdict(lambda: 0)
        for mt in all_metrics:
            for k, v in mt.items():
                merge_metrics[k] += v
        
        count = merge_metrics.pop('count')
        for k, v in merge_metrics.items():
            v_avg = v / count
            if should_save:
                pipeline.summary.add_scalar("metrics/{}".format(k), v_avg, global_step)
                print(f'{k}: {v_avg} with count {count}\n')

    real_images = torch.cat(real_images)
    fake_images = torch.cat(fake_images)
    if should_save:
        print("======With Max 16: ", real_images.shape, fake_images.shape)
    if real_images.ndim == 5 and real_images.shape[1] > 16:
        st = random.randint(0, real_images.shape[1]-16)
        real_images = real_images[:, st: st+16]
        fake_images = fake_images[:, st: st+16]
    if should_save:
        print("======Num of imgs: ", real_images.shape, fake_images.shape)
    if dwm_test_img_num is None:
        assert len(real_images) == len(fake_images)
        dwm_test_img_num = len(real_images)
    ids = random.sample(list(range(len(real_images))), k=min(dwm_test_img_num, len(real_images)))
    if should_save:
        print("======Ids: ", ids, len(ids))
    real_images = real_images[ids]
    fake_images = fake_images[ids]

    metric.update(real_images.to("cuda"), real=True)
    metric.update(fake_images.to("cuda"), real=False)
    metric._should_unsync = False
    fid_value = float(metric.compute())
    if should_save:
        print(f"{metric_type}: {fid_value}, {metric.real_features_num_samples}, {metric.fake_features_num_samples}")
        pipeline.summary.add_scalar(metric_type, fid_value, global_step)
        
    if metric_type == 'FVD_FID' or metric_type == 'ALL':
        torch.cuda.empty_cache()
        metric2 = FrechetInceptionDistance(normalize=True, sync_on_compute=True).to("cuda")
        real_images = real_images.flatten(0, 1)
        fake_images = fake_images.flatten(0, 1)

        # ===sample for 10k fid
        ids = random.sample(list(range(len(real_images))), k=min(10000 // get_world_size(), len(real_images)))
        if should_save:
            print("======Ids for FID: ", ids, len(ids))
        real_images = real_images[ids]
        fake_images = fake_images[ids]

        metric2.update(real_images.to("cuda"), real=True)
        metric2.update(fake_images.to("cuda"), real=False)
        metric2._should_unsync = False
        fid_value = float(metric2.compute())
        if should_save:
            print(f"{metric_type} value 2: {fid_value}, {metric2.real_features_num_samples}, {metric2.fake_features_num_samples}")
            pipeline.summary.add_scalar(f"{metric_type}_2", fid_value, global_step)

    # if should_save:
    #     visualizer.draw(global_step, etype=metric_type)
    # visualizer.clear()
    torch.distributed.barrier()

@torch.no_grad()
def evaluate_svd_3d(metric_type, pipeline, should_save: bool, global_step: int, dataset_length: int,
    validation_dataloader: torch.utils.data.DataLoader,
    validation_datasampler=None,
    cop4d_metrics=None):
    """
    params in inference_config:
        1. from_nviews
    """
    # eval FVD_FID
    assert metric_type in ['COP4D']
    metric_tfm = lambda x: rearrange(x, "b f k c h w -> (b k) f c h w")
    metric = DWMFVD(device="cuda")

    from_nviews = pipeline.inference_config.get("from_nviews", True)
    num_videos = 0

    if should_save:
        from dwm.utils.visualizer import SimpleVisualizer
        visualizer_save_name = pipeline.inference_config['visualizer_cfg'].pop("visualizer_save_name", 'vis_data')
        visualizer = SimpleVisualizer(os.path.join(pipeline.output_path, visualizer_save_name, 'vis_image'), **pipeline.inference_config['visualizer_cfg'])
    else:
        visualizer = None
    
    dwm_test_vid_num = dataset_length // get_world_size()
    dwm_test_img_num = None

    real_images = []
    fake_images = []

    if should_save:
        print(f"======Test {dwm_test_vid_num} per process...")
    if pipeline.ddp:
        validation_datasampler.set_epoch(0)
    skip_ar_for_3d = True
    infer_layout_mode = pipeline.inference_config.get("infer_layout_mode", "first")
    assert infer_layout_mode in ["first", "all", "none"]

    for n, data in enumerate(validation_dataloader):
        if num_videos + len(data["lidar_points"]) > dwm_test_vid_num:
            continue
        num_videos += len(data["lidar_points"])
        if should_save:
            print(f"======Test {num_videos} of {dwm_test_vid_num}")

        batch_size = len(data["lidar_points"])
        with torch.no_grad():
            """
            Now, only support scene_description, as this is clip consistent
            """
            num_frames, cond_frames = pipeline.inference_config['single_step_frames'], pipeline.common_config['unet_input_cfg']['num_init_frames']
            ar_infer_steps = 1
            if should_save:
                print("===AR infer steps: ", ar_infer_steps)
            
            cur_input = dict(
                vae_images=copy.deepcopy(data['vae_images'][:, :num_frames]),
                fps=data['fps'],
                pts=data['pts'][:, :num_frames]
                )
            if "convnext_images" in data:
                cur_input["convnext_images"] = copy.deepcopy(data['convnext_images'][:, :num_frames])
            if "lidar_points_raw" in data:
                cur_input["lidar_points_raw"] = copy.deepcopy([v[:num_frames] for v in data['lidar_points_raw']])
            elif "lidar_points" in data:
                cur_input["lidar_points"] = copy.deepcopy([v[:num_frames] for v in data['lidar_points']])
            if "ego_steering" in data:
                cur_input["ego_steering"] = copy.deepcopy(data['ego_steering'][:, :num_frames])
                cur_input["ego_speed"] = copy.deepcopy(data['ego_speed'][:, :num_frames])
                
            for k in ['ego_transforms', 'lidar_transforms', 'camera_transforms', 'image_size', 'camera_intrinsics']:
                if k in data:
                    cur_input[k] = data[k][:, :num_frames]
            if isinstance(data['clip_text'][0], (tuple, list)):          # for unshared text
                cur_input['clip_text'] = [data['clip_text'][i][:num_frames] for i in range(len(data['clip_text']))]
            else:
                cur_input['clip_text'] = data['clip_text']

            cur_pos = 0
            cur_gen_rts = []
            cur_gen_pcs, cur_gt_pcs = [], []
            for cid in range(ar_infer_steps):
                if pipeline.inference_config.get("auto_infer_cxt", False):
                    pipeline.dynamic_cfg['infer_without_cxt'] = (cid == 0)
                if pipeline.inference_config.get("auto_skip_interaction", False):
                    pipeline.dynamic_cfg['infer_without_interaction'] = (cid > 0)
                if pipeline.inference_config.get("auto_skip_vis", False):
                    pipeline.dynamic_cfg['infer_without_vis'] = (cid > 0)
                pipeline_output = pipeline.inference_pipeline(cur_input, "pt")            # btchw, 0-1

                if cop4d_metrics is not None:
                    if skip_ar_for_3d and cid > 0:
                        pass
                    else:
                        evaluate_cop4d(pipeline, pipeline_output, cop4d_metrics, num_frames=num_frames)

                # Visualize PC
                if metric_type == 'COP4D' and (not pipeline.dynamic_cfg.get("infer_without_vis", False)):
                    assert batch_size == 1
                    if should_save:
                        generated_sample_v = pipeline_output['voxel_sequence']
                        voxels = pipeline_output['voxel_sequence_gt']
                        if cid == 0:
                            cur_gen_pcs.append(generated_sample_v)
                            cur_gt_pcs.append(voxels)
                        else:
                            cur_gen_pcs.append(generated_sample_v[cond_frames:])
                            cur_gt_pcs.append(voxels[cond_frames:])
                        if cid == ar_infer_steps -1:
                            folder_name = os.path.join(visualizer.pth, "preview", f'FVD_{global_step}')
                            os.makedirs(folder_name, exist_ok=True)
                            cur_gen_pcs = torch.cat(cur_gen_pcs, dim=0)
                            cur_gt_pcs = torch.cat(cur_gt_pcs, dim=0)

                            draw_bev_lidar_from_list(cur_gen_pcs, f"{folder_name}/{n:03d}_generated_v.mp4")
                            draw_bev_lidar_from_list(cur_gt_pcs, f"{folder_name}/{n:03d}_voxel_gt.mp4")
    
    if metric_type == 'COP4D':
        assert cop4d_metrics is not None
        metrics = dict(cop4d_metrics)
        if pipeline.ddp:
            all_metrics = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(all_metrics, metrics)
        else:
            all_metrics = [metrics]
        merge_metrics = defaultdict(lambda: 0)
        for mt in all_metrics:
            for k, v in mt.items():
                merge_metrics[k] += v
        
        count = merge_metrics.pop('count')
        for k, v in merge_metrics.items():
            v_avg = v / count
            if should_save:
                pipeline.summary.add_scalar("metrics/{}".format(k), v_avg, global_step)
                print(f'{k}: {v_avg} with count {count}\n')

    torch.distributed.barrier()
