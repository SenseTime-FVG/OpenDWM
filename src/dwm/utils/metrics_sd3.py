import os
import torch
import random
from einops import rearrange
from torchmetrics.image.fid import FrechetInceptionDistance

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
    
@torch.no_grad()
def evaluate_sd3(metric_type, pipeline, should_save: bool, global_step: int, dataset_length: int,
    validation_dataloader: torch.utils.data.DataLoader,
    validation_datasampler=None):

    # eval FVD_FID
    assert metric_type in ['FID', 'FVD', 'FVD_FID']
    metric_tfm = lambda x: rearrange(x, "b f k c h w -> (b k) f c h w")
    metric = DWMFVD(device="cuda")

    from_nviews = pipeline.inference_config.get("from_nviews", True)
    num_videos = 0

    if should_save and pipeline.inference_config['use_visualizer']:
        from dwm.utils.visualizer import SimpleVisualizer
        visualizer_save_name = pipeline.inference_config['visualizer_cfg'].pop("visualizer_save_name", 'vis_data')
        os.makedirs(os.path.join(pipeline.output_path, visualizer_save_name, 'vis_image'), exist_ok=True)
        visualizer = SimpleVisualizer(os.path.join(pipeline.output_path, visualizer_save_name, 'vis_image'), **pipeline.inference_config['visualizer_cfg'])
    else:
        visualizer = None
    
    if metric_type == 'FID':
        metric = FrechetInceptionDistance(normalize=True, sync_on_compute=True).to("cuda")
        dwm_test_img_num = 10000 // get_world_size()
        dwm_test_vid_num = 2048 // get_world_size()
        if from_nviews:
            dwm_test_vid_num //= 6          # nviews
    elif metric_type == 'FVD' or metric_type == 'FVD_FID':
        metric = DWMFVD(device="cuda")
        dwm_test_vid_num = 900 // get_world_size()
        dwm_test_img_num = dwm_test_vid_num
        if from_nviews:
            dwm_test_vid_num //= 6          # nviews

    real_images = []
    fake_images = []

    if should_save:
        print(f"======Test {dwm_test_vid_num} per process...")
    if pipeline.ddp:
        validation_datasampler.set_epoch(0)

    for n, data in enumerate(validation_dataloader):
        if pipeline.ddp:
            torch.distributed.barrier()

        if num_videos + len(data["vae_images"]) > dwm_test_vid_num:
            break
        num_videos += len(data["vae_images"])
        if should_save:
            print(f"======Test {num_videos} of {dwm_test_vid_num}")
      
        batch_size = data['vae_images'].shape[0]
        nviews = data['vae_images'].shape[2]
        real_images.append(metric_tfm(data["vae_images"][:, pipeline.common_config["visible_frame"]:, ...]))

        pipeline_output = pipeline.inference_pipeline(data, "pt")
        fake_image = pipeline_output['images']
        fake_image = rearrange(fake_image, "(b f k) c h w -> b f k c h w", b=batch_size, k=nviews)
        
        if should_save and visualizer is not None:
            visualizer.uni_preview(data, fake_image, global_step, vis=False)
            visualizer.draw(global_step, etype=metric_type, inner_step=n)
            visualizer.clear()
        torch.cuda.empty_cache()

        fake_image = fake_image[:, pipeline.common_config["visible_frame"]:, ...]
        fake_images.append(metric_tfm(fake_image))
    
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

    if pipeline.ddp:
        torch.distributed.barrier()
