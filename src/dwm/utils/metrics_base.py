import argparse
import dwm.common
import json
import os
import torch
import random
import torch.nn.functional as F
import copy
import numpy as np
from easydict import EasyDict as edict
from collections import defaultdict
from einops import rearrange
from torchmetrics.image.fid import FrechetInceptionDistance
from dwm.utils.fvd_utils.fvd_tats import get_logits, frechet_distance
from dwm.utils.fvd_utils.pytorch_i3d import InceptionI3d

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


class DWMFVD:
    def __init__(self, 
        i3d_path = '/mnt/afs/user/wuzehuan/Downloads/models/inception_3d/i3d_pretrained_400.pt',
        device='cuda', max_batch_size=8):
        # TODO: support all gather from multi-gpus
        self.device = device
        # ====== prepare i3d ======
        i3d = InceptionI3d(400, in_channels=3).to(self.device)
        i3d.load_state_dict(torch.load(i3d_path, map_location='cpu'))
        i3d.to(device)
        i3d.eval()
        self.inception = i3d
        self.real_features = []
        self.fake_features = []
        self.max_batch_size = max_batch_size
        self.target_resolution = (224, 224)         # this is fixed in fvd
        self.i3d_min = 10

    def update(self, images, real=True):
        """
        Notes:
            1. transform is recommended to be vae transform, this align the gt/infer images
        """
        target_resolution = (self.i3d_min, ) + self.target_resolution
        for start in range(0, len(images), self.max_batch_size):
            cur = images[start: start+self.max_batch_size]
            video_length = cur.shape[1]
            cur = rearrange(cur, 'b f c h w -> (b f) c h w')
            resized_videos = F.interpolate(cur, size=self.target_resolution, mode='bilinear', align_corners=False)
            resized_videos = rearrange(resized_videos, '(b f) c h w -> b c f h w', f=video_length)
            if resized_videos.shape[2] < self.i3d_min:
                # I3D要求输入维度大于等于10，否则无法池化
                print(f"Current frame number < {self.i3d_min}, and we pad it to {self.i3d_min}.")
                resized_videos = F.interpolate(resized_videos, size=target_resolution, mode='trilinear', align_corners=False)
                print(f"Current shape {resized_videos.shape}.")
            resized_videos = 2. * resized_videos - 1 # [-1, 1]
            feat = get_logits(self.inception, resized_videos, self.device)
            if real == True:
                self.real_features.append(feat)
            else:
                self.fake_features.append(feat)

    def compute(self):
        real = torch.cat(self.real_features, dim=0)
        fake = torch.cat(self.fake_features, dim=0)
        world_size = get_world_size()
        if world_size > 1:
            print("===Shape before gather: ", real.shape, fake.shape)
            all_real_features = real.new_zeros(
                (len(real)*world_size, ) + real.shape[1:])
            all_fake_features = fake.new_zeros(
                (len(fake)*world_size, ) + fake.shape[1:])
            torch.distributed.all_gather_into_tensor(
                all_real_features, real)
            torch.distributed.all_gather_into_tensor(
                all_fake_features, fake)
            real, fake = all_real_features, all_fake_features
        print("===Shape to compute FVD: ", real.shape, fake.shape)
        self.real_features_num_samples = len(real)
        self.fake_features_num_samples = len(fake)
        return frechet_distance(real, fake)