import os
import re
from typing import Optional

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from collections import defaultdict

class SimpleVisualizer:
    """
    A Simple Visualizer, it can accept generated results with shape [b,c,h,w] and [b,t,c,h,w]
    """
    def __init__(self, pth, grid_shape=(3, 2), with_3dbox=False, 
        with_hdmap=False, with_ori=False, save_type='mp4', rescale=False, fps=10):
        self.max_imgs = grid_shape[0]*grid_shape[1]
        self.grid_shape = grid_shape
        self.videos = defaultdict(list)
        self.pth = pth
        self.with_3dbox = with_3dbox
        self.with_hdmap = with_hdmap
        self.with_ori = with_ori
        self.save_type = save_type
        self.rescale = rescale
        self.pcs = None
        self.fps = fps

    def save_videos_grid(self, videos: torch.Tensor,
                     path: str,):
        # visualizer.save_videos_grid(fake_image.transpose(1, 2), f'/mnt/afs/user/nijingcheng/workspace/codes/DWM/work_dirs/video_long_gen/nusc_front_loadpt_svd_ddim_dwm_text_cxt_skip_t_cross_bs32_a1_tonly/vis_data/tmp/{kkk}.gif')
        videos = rearrange(videos, 'b c t h w -> t b c h w')
        outputs = []
        fps = self.fps
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=self.grid_shape[0])
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            if self.rescale:
                x = (x + 1.0) / 2.0  # -1,1 -> 0,1
            x = (x * 255).numpy().astype(np.uint8)
            outputs.append(x)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        # imageio v3 doesn't support fps
        if self.save_type == 'gif':
            if imageio.__version__ < '2.28.0':
                imageio.mimsave(path, outputs, fps=fps)
            else:
                imageio.mimsave(path, outputs, duration=1000 * 1 / fps, loop=10)
        else:
            path = path.replace('.gif', '.mp4')
            with imageio.get_writer(path, fps=fps) as video_writer:
                for image in outputs:
                    video_writer.append_data(image)

    def clear(self):
        self.videos = defaultdict(list)

    def _process(self, x: torch.FloatTensor, imagenet_reverse=False, from_nviews=False):
        if from_nviews:
            x = rearrange(x, 'b f k c h w -> b f c h (k w)')
        if x.ndim == 4:
            x = x[:, None]          # btchw
        if imagenet_reverse:            # btchw
            target_shape = [1 for _ in range(x.ndim - 3)] + [-1, 1, 1]
            mean = x.new_tensor([0.485, 0.456, 0.406]).view(target_shape)
            std = x.new_tensor([0.229, 0.224, 0.225]).view(target_shape)
            x = x*std + mean            # 0-1
        x = x.transpose(1, 2)           # btchw -> bcthw
        return x

    def update(self, x: torch.FloatTensor, data, from_nviews=False):
        if len(self.videos['fake']) < self.max_imgs:
            self.videos['fake'].append(self._process(x, from_nviews=from_nviews))
            if self.with_ori:
                self.videos['real'].append(self._process(data['vae_images'], from_nviews=from_nviews))
            if self.with_3dbox:
                self.videos['bbox'].append(self._process(data['3dbox_images'], imagenet_reverse=True, from_nviews=from_nviews))
            if self.with_hdmap:
                self.videos['hdmap'].append(self._process(data['hdmap_images'], imagenet_reverse=True, from_nviews=from_nviews))

    def draw(self, steps, etype='FID', inner_step=None):
        if inner_step is None:
            inner_step = steps
        for k, v in self.videos.items():
            videos = torch.cat(self.videos[k], dim=0)[:self.max_imgs]
            print("===Save video's shape: ", videos.shape)
            self.save_videos_grid(videos, os.path.join(self.pth, f'{steps:06d}_{etype}', f'{k}_{inner_step}.gif'))

        if self.pcs is None:
            pass

    def uni_preview(self, batch, output_images, global_step, inner_step=None, vis=True):
        preview_batch = dict()
        if "3dbox_images" in batch:
            preview_batch['3dbox_images'] = rearrange(batch["3dbox_images"], "b f k c h w -> b f c h (k w)").cpu()
        if "hdmap_images" in batch:
            preview_batch['hdmap_images'] = rearrange(batch["hdmap_images"], "b f k c h w -> b f c h (k w)").cpu()
        if "vae_images" in batch:
            preview_batch['vae_images'] = rearrange(batch["vae_images"], "b f k c h w -> b f c h (k w)").cpu()

        pred_images = rearrange(output_images, "b f k c h w -> b f c h (k w)").cpu()
        self.update(pred_images, preview_batch)
        if vis:
            self.draw(global_step, inner_step=inner_step)
            self.clear()