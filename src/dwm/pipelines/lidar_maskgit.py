import dwm.common
import os
import cv2
import numpy as np
import safetensors.torch
import time
import torch
import torch.cuda.amp
import torch.nn.functional as F
import contextlib
from PIL import Image
import torchvision
import torch.utils.tensorboard
import wandb
import open3d as o3d
from dwm.pipelines.holodrive import DWM
from dwm.pipelines.bevw_vae import BEVWorldVAE
from tqdm import tqdm
from einops import rearrange
from collections import defaultdict
import imageio
import pickle
import math
import copy
import re
from easydict import EasyDict as edict
from typing import List, Optional
import pdb

from dwm.models.voxelizer import Voxelizer
from dwm.utils.metrics_copilot4d import (
    compute_chamfer_distance, compute_chamfer_distance_inner, compute_ray_errors, clamp,
    point_cloud_to_histogram, jsd_2d, compute_mmd
)

def gamma_func(self, mode="cosine"):
    if mode == "linear":
        return lambda r: 1 - r
    elif mode == "cosine":
        return lambda r: torch.cos(torch.tensor(r) * math.pi / 2)
    elif mode == "square":
        return lambda r: 1 - r**2
    elif mode == "cubic":
        return lambda r: 1 - r**3
    else:
        raise NotImplementedError

def _sample_logistic(shape, out=None, generator=None):
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape, generator=generator)
    return torch.log(U) - torch.log(1 - U)


def _sigmoid_sample(logits, tau=1, generator=None):
    """
    Implementation of Bernouilli reparametrization based on Maddison et al. 2017
    """
    dims = logits.dim()
    logistic_noise = _sample_logistic(logits.size(), out=logits.data.new(), generator=generator)
    y = logits + logistic_noise
    return torch.sigmoid(y / tau)


def gumbel_sigmoid(logits, tau=1, hard=False, generator=None):

    gumbel_sigmoid_coeff = 1.0
    y_soft = _sigmoid_sample(logits * gumbel_sigmoid_coeff, tau=tau, generator=generator)
    if hard:
        y_hard = torch.where(y_soft > 0.5, torch.ones_like(y_soft), torch.zeros_like(y_soft))
        y = y_hard.data - y_soft.data + y_soft
    else:
        y = y_soft
    return y

def draw_bev_lidar(voxels, pth):
    cv2.imwrite(
        pth,
        voxels[0].max(dim=0)[0][:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255,
    )

def draw_bev_lidar_from_list(voxels_list, pth):
    with imageio.get_writer(pth, fps=2) as video_writer:
        for voxels in voxels_list:
            if isinstance(voxels, np.ndarray):
                voxels = torch.from_numpy(voxels)
            image = (voxels.max(dim=0)[0][:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255).astype(np.uint8)
            video_writer.append_data(image)

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

def _make_eye_mask(
    input_ids_shape, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask *= (1 - torch.eye(tgt_len, dtype=dtype, device=device))

    assert past_key_values_length == 0          # only square
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

class MaskGITPipeline(torch.nn.Module):
    @staticmethod
    def load_state(path: str):
        if path.endswith(".safetensors"):
            state = safetensors.torch.load_file(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu")
        return state

    def __init__(
        self, output_path: str, config: dict,
        device, vq_point_cloud, bi_directional_Transformer,
        bev_layout_encoder,
        vq_point_cloud_ckpt_path: str = None,
        vq_blank_code_path: str = None,
        bi_directional_Transformer_checkpoint_path: str = None,
        metrics: dict=dict(),
        training_config: dict=dict(),
        inference_config: dict=dict(),
        common_config: dict=dict(),
        resume_from = None
        ):
        r"""
        Args:
            training_config (`dict`):
                training related parameters, e.g. dropout
            inference_config (`dict`):
                inference related parameters, e.g. cfg
            common_config (`dict`):
                config for model, used for both train/val
        TODO:
            1. add action (move cond (dropout) to training_config)
            2. unet no 32, action may be not right
        Notes:
            1. unet mid-block is half of the img_size[-1]

        !!!
        !!!: pc_range in metrics is fixed, which should be adjusted correspondingly
        """
        super().__init__()
        self.ddp = torch.distributed.is_initialized()
        self.should_save = not torch.distributed.is_initialized() or \
            torch.distributed.get_rank() == 0
        config = edict(config)
        self.config = config
        self.device = device
        self.generator = torch.Generator()
        self.common_config = common_config
        self.training_config = training_config
        self.inference_config = inference_config
        if "generator_seed" in config:
            self.generator.manual_seed(config["generator_seed"])
        else:
            self.generator.seed()

        self.vq_point_cloud = vq_point_cloud
        self.vq_point_cloud.to(self.device)
        if vq_point_cloud_ckpt_path is not None:
            state_dict = MaskGITPipeline.load_state(vq_point_cloud_ckpt_path)
            if 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']
            missing_keys, unexpected_keys = self.vq_point_cloud.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print("Missing keys in state dict:", missing_keys)
            if unexpected_keys:
                print("Unexpected keys in state dict:", unexpected_keys)
        self.vq_point_cloud.eval()

        self.bev_layout_encoder_wrapper = self.bev_layout_encoder = bev_layout_encoder
        self.bev_layout_encoder.to(self.device)

        self.bi_directional_Transformer_wrapper = self.bi_directional_Transformer = bi_directional_Transformer
        self.bi_directional_Transformer.to(self.device)

        if self.training_config.get('set_vq_no_grad', False):
            print("Set vq no grad")         # no influence, only for test security
            self.vq_point_cloud.requires_grad_(False)

        # self.vq_point_cloud.vector_quantizer.embedding
        if self.training_config.get('weight_tying', False):
            print("Set weight tying!!!")
            self.bi_directional_Transformer.pred.requires_grad_(False)
            self.bi_directional_Transformer.pred.weight.copy_(self.vq_point_cloud.vector_quantizer.embedding.weight)

        if resume_from is not None:
            model_state_dict = MaskGITPipeline.load_state(
                os.path.join(
                    output_path, "checkpoints", "{}.pth".format(resume_from)))
            self.bi_directional_Transformer.load_state_dict(model_state_dict["bi_directional_Transformer"])
            self.bev_layout_encoder.load_state_dict(model_state_dict["bev_layout_encoder"])

        if bi_directional_Transformer_checkpoint_path is not None:
            if self.should_save:
                print(f"===Load from {bi_directional_Transformer_checkpoint_path}...")
            self.bi_directional_Transformer.load_state_dict(
                MaskGITPipeline.load_state(bi_directional_Transformer_checkpoint_path)["bi_directional_Transformer"])
            self.bev_layout_encoder.load_state_dict(
                MaskGITPipeline.load_state(bi_directional_Transformer_checkpoint_path)["bev_layout_encoder"])

        self.gamma = gamma_func("cosine")
        self.iter = 0
        self.T = self.inference_config.get("sample_steps", 30)
        self.BLANK_CODE = None
        self.grad_scaler = torch.cuda.amp.GradScaler() \
            if ("autocast" in self.common_config) else None

        # setup training parts
        self.loss_list = []
        self.step_duration = 0
        self.metrics = metrics

        if self.ddp:
            find_unused_parameters=self.training_config.get("find_unused_parameters", False)
            self.bev_layout_encoder_wrapper = torch.nn.parallel.DistributedDataParallel(
                self.bev_layout_encoder,
                device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=find_unused_parameters)
            self.bi_directional_Transformer_wrapper = torch.nn.parallel.DistributedDataParallel(
                self.bi_directional_Transformer,
                device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=find_unused_parameters)

        if self.should_save:
            self.summary = torch.utils.tensorboard.SummaryWriter(
                os.path.join(output_path, "log"))

        # change decay by name
        if len(self.training_config.get('to_skip_decay', [])) > 0:
            to_skip_decay = self.training_config.get('to_skip_decay', [])
            params1, params2 = [], []
            for name, params in self.bi_directional_Transformer_wrapper.named_parameters():
                flag = False
                for n in to_skip_decay:
                    if re.fullmatch(n, name):
                        flag = True
                # TODO: layernorm
                if flag:
                    params1.append(params)
                    if self.should_save:
                        print("{} without weight decay.".format(name))
                else:
                    params2.append(params)
            self.optimizer = dwm.common.create_instance_from_config(
                config["optimizer"],
                params=[
                    {'params': params1, 'weight_decay': 0},
                    {'params': params2}         # use default
                ])
        else:
            self.optimizer = dwm.common.create_instance_from_config(
                config["optimizer"],
                params=self.bi_directional_Transformer_wrapper.parameters())

        self.lr, self.grad_norm = 0, 0
        if self.training_config.get('warmup_iters', None) is not None:
            from torch.optim.lr_scheduler import LinearLR
            total_iters = self.training_config['warmup_iters']
            self.warmup_scheduler = LinearLR(self.optimizer, start_factor=0.001, total_iters=total_iters)
            self.total_iters = total_iters
        if "lr_scheduler" in config:
            self.lr_scheduler = dwm.common.create_instance_from_config(
                config["lr_scheduler"], optimizer=self.optimizer)
        else:
            self.lr_scheduler = None

        if resume_from is not None:
            if os.path.exists(os.path.join(output_path, "optimizers")):
                optimizer_state_dict = torch.load(
                    os.path.join(
                        output_path, "optimizers",
                        "{}.pth".format(resume_from)))
                self.optimizer.load_state_dict(optimizer_state_dict)
            if os.path.exists(os.path.join(output_path, "schedulers")):
                if self.lr_scheduler is not None:
                    scheduler_state_dict = torch.load(
                    os.path.join(
                        output_path, "schedulers",
                        "{}.pth".format(resume_from)))
                    self.lr_scheduler.load_state_dict(scheduler_state_dict)
        self.output_path = output_path
        # === Load BLANK CODE
        with open(vq_blank_code_path, 'rb') as f:
            blank_code = pickle.load(f)
            self.BLANK_CODE = blank_code
            print("=== Load BLANK CODE: ", blank_code)

    @staticmethod
    def get_action(
        batch, act_type, device,
    ):
        # TODO: if is_multimodal, then with extra dim
        batch_size, num_frames = batch["ego_transforms"].shape[0], batch["ego_transforms"].shape[1]

        if act_type == "copliot4D":

            action = torch.zeros_like(batch["ego_transforms"]).to(device)

            for bid in range(batch_size):
                for fid in range(num_frames-1):
                    diff_pose = torch.matmul(batch["ego_transforms"][bid, fid+1], torch.inverse(batch["ego_transforms"][bid, fid]))
                    action[bid, fid] = diff_pose

        elif act_type == "copliot4D_v2":
            action = torch.zeros_like(batch["ego_transforms"]).to(device)
            ego_transforms = batch['ego_transforms']
            lidar_transforms = batch['lidar_transforms']

            for bid in range(batch_size):
                for fid in range(num_frames-1):
                    tfm_inverse = torch.inverse(ego_transforms[bid, fid+1] @ lidar_transforms[bid, fid+1])
                    diff_pose = tfm_inverse @ ego_transforms[bid, fid] @ lidar_transforms[bid, fid]
                    action[bid, fid] = diff_pose

        elif act_type == "copliot4D_v3":
            reference_idx = num_frames // 2 - 1
            action = torch.zeros_like(batch["ego_transforms"]).to(device)
            ego_transforms = batch['ego_transforms']
            lidar_transforms = batch['lidar_transforms']

            for bid in range(batch_size):
                for fid in range(num_frames-1):
                    tfm_inverse = torch.inverse(ego_transforms[bid, reference_idx] @ lidar_transforms[bid, reference_idx])
                    diff_pose = tfm_inverse @ ego_transforms[bid, fid] @ lidar_transforms[bid, fid]
                    action[bid, fid] = diff_pose

        elif act_type == "dwm":
            # action = torch.zeros_like(batch_size, num_frames, 2).to(device)

            # for bid in range(batch_size):
            #     for fid in range(num_frames-1):
            #         diff_xy = torch.matmul(batch["ego_transforms"][bid, fid+1], torch.inverse(batch["ego_transforms"][bid, fid]))[0:2, 3]
            #         action[bid, fid] = diff_xy
            action = batch["ego_transforms"].new_zeros((batch_size, num_frames, 3)).to(device)
            ego_transforms = batch['ego_transforms']            # b/t/k
            lidar_transforms = batch['lidar_transforms']        # b/t

            for bid in range(batch_size):
                for fid in range(num_frames-1):
                    tfm_inverse = torch.inverse(ego_transforms[bid, fid+1] @ lidar_transforms[bid, fid+1])
                    diff_pose = tfm_inverse @ ego_transforms[bid, fid] @ lidar_transforms[bid, fid]

                    action[bid, fid] = diff_pose[0:3, -1]
            action = action[..., None]

        action = rearrange(action, "b l m n -> (b l) m n")

        return action

    def save_checkpoint(self, output_path: str, steps: int):
        if self.should_save:
            os.makedirs(
                os.path.join(output_path, "checkpoints"), exist_ok=True)
            torch.save({"bi_directional_Transformer": self.bi_directional_Transformer.state_dict(),
                        "bev_layout_encoder": self.bev_layout_encoder.state_dict()}, os.path.join(output_path, "checkpoints", "{}.pth".format(steps)))
            os.makedirs(os.path.join(output_path, "optimizers"), exist_ok=True)
            torch.save(self.optimizer.state_dict(), os.path.join(output_path, "optimizers", "{}.pth".format(steps)))
            if self.lr_scheduler is not None:
                os.makedirs(os.path.join(output_path, "schedulers"), exist_ok=True)
                torch.save(
                self.lr_scheduler.state_dict(),
                os.path.join(output_path, "schedulers", "{}.pth".format(steps)))
        if self.ddp:
            torch.distributed.barrier()

    def log(self, global_step: int, log_steps: int, log_type: str='wandb'):
        if self.should_save:
            if len(self.loss_list) > 0:
                log_dict = {
                    k: sum([
                        self.loss_list[i][k]
                        for i in range(len(self.loss_list))
                    ]) / len(self.loss_list)
                    for k in self.loss_list[0].keys()
                }
                log_string = ", ".join(
                    ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()])
                print(
                    "Step {} ({:.1f} s/step), LR={} Norm={} {}".format(
                        global_step, self.step_duration / log_steps, self.lr, self.grad_norm,
                        log_string))
                if self.summary is not None:
                    for k, v in log_dict.items():
                        self.summary.add_scalar(
                            "train/{}".format(k), v, global_step)
                if log_type == 'wandb' and wandb.run is not None:
                    wandb.log(log_dict)

        self.loss_list.clear()
        self.step_duration = 0

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
        # torch.argsort return the original index of the sorted elements. The position 0 is the index of the smallest element in the original array.
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)
        return x_masked, mask, ids_restore, ids_keep

    def mask_code(self, code, code_indices, mask_token, mask_ratio=None, eta=20, with_noise=False):
        # if with_noise:
        #     return self._mask_noise_code(code, mask_token, cond, mask_ratio, eta)
        # else:
            return self._mask_code(code, code_indices, mask_token, mask_ratio)

    def _mask_code(self, code, code_indices, mask_token, mask_ratio=None):
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
        mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)  # no cls token
        # after gather -> 16,6400,1024
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x_indices = code_indices.clone()
        x_indices[mask] = -1
        return x, x_indices, mask, ids_restore

    # NOTE discrete diffusion
    def _mask_noise_code(self, code, mask_token, cond=None, mask_ratio=None, eta=20,):
        if mask_ratio == None:
            mask_ratio = self.gamma(np.random.uniform())

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(code, mask_ratio)
        # append mask tokens to sequence

        mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        # NOTE add noise token to unmaksed tokens
        N, Lx, D = x.shape
        noise_ratio = torch.rand(1) * eta * 0.01
        noise_len = int(torch.round(Lx * noise_ratio))
        print(noise_len, noise_ratio, Lx, "In train")

        # random choose noise_len tokens from codebook
        codebook_size = self.vq_point_cloud.vector_quantizer.n_e
        noise_indices = np.random.choice(codebook_size, (N, noise_len), replace=True)

        noise_indices = torch.tensor(noise_indices).to(x.device)

        noisy_token = self.vq_point_cloud.vector_quantizer.get_codebook_entry(noise_indices) # N, noise_len, d

        replaced_x = x.clone()
        new_mask = mask.clone()
        for i in range(N):
            selected_indices = torch.randperm(Lx)[:noise_len]
            replaced_x[i, selected_indices, :] = noisy_token[i]
            replace_ids_keep = ids_keep[i, selected_indices]
            new_mask[i, replace_ids_keep] = 1

        x = torch.cat([replaced_x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # (x-code)*new_mask.sum()

        if cond is not None:
            x = torch.cat([x, cond], dim=-1)

        return x, new_mask, ids_restore

    def process_inference_input(self, x, num_frames, code, ctype='pre', skip_past=0):
        if ctype == 'pre' or ctype == 'fin':
            code = rearrange(code, "(b f) ... -> b f ...", f=num_frames)
            x = rearrange(x, "(b f) ... -> b f ...", f=(num_frames+1)//2)
            batch_size = code.shape[0]
            x_past = code[:,]
            if skip_past > 0:
                x_past = x_past[:, skip_past:]
            x = torch.cat([x_past, x], dim=1).flatten(0, 1)
        elif ctype == 'after':
            x = rearrange(x, "(b f) ... -> b f ...", f=num_frames)
            x = x[:, num_frames_past:].flatten(0, 1)
        else:
            raise NotImplementedError
        return x

    def mutlitask_mask_code(self, code, code_indices, infer=False):
        """
        1. sample a task
        """
        use_discrete_diffusion = self.common_config.get('use_discrete_diffusion', False)
        if infer:
            batch_size = code.shape[0]
            x_future = self.bi_directional_Transformer.mask_token.repeat(batch_size,
                self.bi_directional_Transformer.img_size[0] * self.bi_directional_Transformer.img_size[1], 1)
            x_indices = torch.ones(*x_future.shape[:2]).to(x_future.device) * -1
            return x_future, x_indices, None, None
        else:
            x, x_indices, mask, ids_restore  = self.mask_code(code, code_indices,
                    self.bi_directional_Transformer.mask_token, mask_ratio=None, with_noise=use_discrete_diffusion)
            return x, x_indices, mask, ids_restore

    def debug_print_points(self, points, suffix):
        points = sum([[j[:, :3] for j in i] for i in points], [])
        with torch.no_grad():
            voxels = self.vq_point_cloud.voxelizer([[_] for _ in points])
            voxels = voxels.to(self.device)
        draw_bev_lidar_from_list(voxels, f"/mnt/storage/user/nijingcheng/workspace/codes/DWM/work_dirs/debug/vis/{suffix}_voxel_gt.mp4")

    def _data_process(self, batch, **kwargs):
        # gpu dataprocessor
        lidar_points = None
        if self.common_config.get('use_reference_coordinate', False):
            # dict_keys(['fps', 'pts', 'lidar_transforms', 'ego_transforms', 'lidar_points'])
            lidar_transforms = batch['lidar_transforms']
            ego_transforms = batch['ego_transforms']            # b,t...
            lidar_points = copy.deepcopy(batch["lidar_points_raw"])
            reference_idx = kwargs["num_frames_past"] - 1
            offsets = [[] for _ in range(len(lidar_points))]
            # As lidar_points is list(list(Tensor)), here we iterate on this data

            for bid in range(len(lidar_points)):
                tfm_inverse = torch.inverse(ego_transforms[bid, reference_idx] @ lidar_transforms[bid, reference_idx])
                for fid in range(len(lidar_points[0])):
                    lidar_points[bid][fid][..., -1] = 1         # to homo
                    tfm = tfm_inverse @ ego_transforms[bid, fid] @ lidar_transforms[bid, fid]
                    lidar_points[bid][fid] = torch.matmul(tfm, lidar_points[bid][fid][..., None]).squeeze(-1)
                    offsets[bid].append(tfm[:3, -1])

            batch["offsets"] = offsets
        if self.common_config.get('gpu_roi_filter', False):
            assert lidar_points is not None         # avoid warning
            range_x, range_y, range_z = self.common_config['roi_range']
            for bid in range(len(lidar_points)):
                for fid in range(len(lidar_points[0])):
                    mask = (lidar_points[bid][fid][:, 0] >= range_x[0]) & (lidar_points[bid][fid][:, 1] >= range_y[0]) & \
                        (lidar_points[bid][fid][:, 2] >= range_z[0]) & (lidar_points[bid][fid][:, 0] <= range_x[1]) & \
                            (lidar_points[bid][fid][:, 1] <= range_y[1]) & (lidar_points[bid][fid][:, 2] <= range_z[1])
                    lidar_points[bid][fid] = lidar_points[bid][fid][mask]
        assert lidar_points is not None
        if lidar_points is not None:
            batch["lidar_points"] = lidar_points
        return batch

    def _draw_points(self, voxels, suffix):         # debug code, e.g. self._draw_points(voxels[i], suffix=i)
        with torch.no_grad():
            image = (voxels.max(dim=0)[0][:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(f'/mnt/storage/user/nijingcheng/workspace/codes/DWM/work_dirs/tmp/debug_{suffix}.jpg', image)

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
                        bev_layout_encoder(_3dbox_images, return_features=True)
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
                        bev_layout_encoder(hdmap_images, return_features=True)
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


    def get_autocast_context(self):
        if "autocast" in self.common_config:
            return torch.autocast(**self.common_config["autocast"])
        else:
            return contextlib.nullcontext()


    def train_step(self, batch: dict, global_step: int):
        t0 = time.time()
        batch_size, num_frames = len(batch["lidar_points"]), len(batch["lidar_points"][0])
        # ====== Data process (GPU)
        points = DWM.preprocess_points(batch, self.device)
        self.bi_directional_Transformer_wrapper.train()
        with torch.no_grad():
            voxels = self.vq_point_cloud.voxelizer(points).squeeze(1)
            lidar_feats = self.vq_point_cloud.lidar_encoder(voxels)
            code, _, code_indices = self.vq_point_cloud.vector_quantizer(lidar_feats, self.vq_point_cloud.code_age,
                                                            self.vq_point_cloud.code_usage)

        # ========= Process
        with self.get_autocast_context():
            # 2D conditions. During training, we randomly drop the one of the condition
            _3dbox_condition_mask = (
                torch.rand((batch_size,), generator=self.generator) <
                self.training_config.get("3dbox_condition_ratio", 1.0))\
                .to(self.device)
            _hdmap_condition_mask = (
                torch.rand((batch_size,), generator=self.generator) <
                self.training_config.get("hdmap_condition_ratio", 1.0))\
                .to(self.device)
            both_false = ~torch.logical_or(_3dbox_condition_mask, _hdmap_condition_mask)
            _3dbox_condition_mask[both_false] = True
            _hdmap_condition_mask[both_false] = True
            maskgit_conditions = MaskGITPipeline.get_maskgit_conditions(
                    self.bev_layout_encoder_wrapper, self.common_config, batch, self.device,
                    torch.float16 if "autocast" in self.common_config else torch.float32,
                    _3dbox_condition_mask,
                    _hdmap_condition_mask
                )
            x, x_indices, mask, ids_restore = self.mutlitask_mask_code(code, code_indices, infer=False)
            # === Model forward/loss
            pred = self.bi_directional_Transformer_wrapper(x, x_indices, context = maskgit_conditions)
            loss = (
                F.cross_entropy(pred.flatten(0, 1), code_indices.flatten(0, 1), reduction="none", label_smoothing=0.1) * mask.flatten(0, 1)
            ).sum() / (mask.flatten(0, 1).sum() + 1e-5)

        acc = (pred.max(dim=-1)[1] == code_indices)[mask > 0].float().mean().item()

        losses = {
            "ce_loss": loss,
            "acc_0": acc
        }
        loss = sum([losses[i] for i in losses if "loss" in i])

        # optimize parameters
        should_optimize = \
            ("gradient_accumulation_steps" not in self.config) or \
            ("gradient_accumulation_steps" in self.config and
                (global_step + 1) %
             self.config["gradient_accumulation_steps"] == 0)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        if should_optimize:
            # print grad norm           # # self.bi_directional_Transformer_wrapper.module.pred.weight
            total_norm = 0
            norm_dict, shape_dict = dict(), dict()
            for n, p in self.bi_directional_Transformer_wrapper.named_parameters():
                if p.grad is not None and p.requires_grad:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                    # norm_dict[n] = param_norm
                    # shape_dict[n] = p.shape
            # total_norm = total_norm ** 0.5
            # yy=sorted(norm_dict.items(), key=lambda x: x[1])
            self.grad_norm = total_norm         # now, only one iter

            # parameters = [p for p in self.bi_directional_Transformer_wrapper.parameters() if p.grad is not None and p.requires_grad]
            # for p in parameters:
            #     param_norm = p.grad.detach().data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            # self.grad_norm = total_norm         # now, only one iter

            if "max_norm_for_grad_clip" in self.config:
                torch.nn.utils.clip_grad_norm_(
                    self.bi_directional_Transformer_wrapper.parameters(),
                    self.config["max_norm_for_grad_clip"])

            if self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        if self.training_config.get('warmup_iters', None) is not None:
            if self.warmup_scheduler.last_epoch > self.warmup_scheduler.total_iters:
                self.lr_scheduler.step()
                cur_lr = self.lr_scheduler.get_last_lr()
            else:
                self.warmup_scheduler.step()
                cur_lr = self.warmup_scheduler.get_last_lr()
            self.lr = cur_lr
        else:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.lr = self.lr_scheduler.get_last_lr()
        self.loss_list.append(losses)
        self.step_duration += time.time() - t0

    def voxels2points(self, voxels):
        non_zero_indices = torch.nonzero(voxels)
        xy = (non_zero_indices[:, 2:] * self.vq_point_cloud.voxelizer.step) + self.vq_point_cloud.voxelizer.y_min
        z = (non_zero_indices[:, 1] * self.vq_point_cloud.voxelizer.z_step) + self.vq_point_cloud.voxelizer.z_min
        xyz = torch.cat([xy, z.unsqueeze(1)], dim=1)

        return xyz

    def preview_pipeline(
        self, batch: dict, output_path: str, global_step: int
    ):
        if "lidar_points" in batch:
            batch_size, num_frames = len(batch["lidar_points"]), len(batch["lidar_points"][0])
        else:
            batch_size, num_frames = len(batch["lidar_points_raw"]), len(batch["lidar_points_raw"][0])
        # batch_size, num_frames = len(batch["lidar_points"]), len(batch["lidar_points"][0])
        results = self.inference_pipeline(batch)
        voxels = results['gt_voxels']
        generated_sample_v = results['pred_voxels']

        if self.should_save:
            # folder_name = os.path.join(output_path, "preview", str(global_step))
            # os.makedirs(folder_name, exist_ok=True)
            preview_lidar_size = (640, 640)
            preview_lidar = Image.new(
            "L", (
                2 * preview_lidar_size[0],
                len(batch["lidar_points"]) * preview_lidar_size[1]
            ))

            for i in range(len(batch["lidar_points"])):
                images = [
                    torchvision.transforms.functional
                    .to_pil_image(torch.amax(voxels[i], 0))
                    .resize(preview_lidar_size),
                    torchvision.transforms.functional
                    .to_pil_image(torch.amax(generated_sample_v[i], 0))
                    .resize(preview_lidar_size)
                ]
                for j, image in enumerate(images):
                    preview_lidar.paste(
                        image, (j * preview_lidar_size[0], i * preview_lidar_size[1]))
            if self.should_save:
                os.makedirs(os.path.join(output_path, "preview"), exist_ok=True)
                preview_lidar.save(
                    os.path.join(
                        output_path, "preview", "{}_lidar.png".format(global_step)))


    def inference_pipeline(self, batch, output_for_eval=False, output_from_ray=False):
        # TODO: now: gt_points + uncond points -> uncond infer blank code + past (from gt) to future (pred)
        batch_size, num_frames = len(batch["lidar_points"]), len(batch["lidar_points"][0])
        points = DWM.preprocess_points(batch, self.device)
        use_blank_code = self.inference_config.get("use_blank_code", True)
        use_maskgit = self.inference_config.get("use_maskgit", False)           # maskgit style sample

        self.bi_directional_Transformer_wrapper.eval()
        with torch.no_grad():
            voxels = self.vq_point_cloud.voxelizer(points).squeeze(1)
            voxels = voxels.to(self.device)
            lidar_feats = self.vq_point_cloud.lidar_encoder(voxels)
            code, _, code_indices = self.vq_point_cloud.vector_quantizer(lidar_feats, self.vq_point_cloud.code_age,
                                                            self.vq_point_cloud.code_usage)

            # NOTE generation
            choice_temperature = 2.0

            # x = self.bi_directional_Transformer.mask_token.repeat(1, self.bi_directional_Transformer.img_size**2, 1)
            # code_idx = torch.ones((1, self.bi_directional_Transformer.img_size**2), dtype=torch.int64, device=x.device) * -1
            # num_unknown_code = (code_idx == -1).sum(dim=-1)

            # ===Sample task code
            # now, only support task_code == 0
            x, x_indices, mask, _ = self.mutlitask_mask_code(code, code_indices, infer=True)
            code_idx = torch.ones((x.shape[0], x.shape[1]), dtype=torch.int64, device=x.device) * -1
            num_unknown_code = (code_idx == -1).sum(dim=-1)
            with self.get_autocast_context():
                maskgit_conditions = MaskGITPipeline.get_maskgit_conditions(
                    self.bev_layout_encoder, self.common_config, batch, self.device,
                    torch.float16 if "autocast" in self.common_config else torch.float32,
                    None,
                    None
                )
                for t in range(self.T):
                    pred = self.bi_directional_Transformer_wrapper(x, x_indices, context = maskgit_conditions)
                    if t < 10 and use_blank_code:
                        pred[..., self.BLANK_CODE] = -10000

                    if self.inference_config.get("sample_from_topk", None) is not None:
                        k = self.inference_config["sample_from_topk"]

                        val, ind = pred.topk(k, dim = -1)
                        probs = torch.full_like(pred, float('-inf'))
                        probs.scatter_(-1, ind, val)
                        if use_maskgit:
                            sample_ids = probs.argmax(-1)
                            # pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
                        else:
                            sample_ids = torch.distributions.Categorical(logits=probs).sample()

                        cnt = 0
                        for v in sample_ids.flatten():
                            if v.item() in self.BLANK_CODE:
                                cnt += 1
                        print("Step: ", t, sample_ids.numel(), cnt, cnt / (sample_ids.numel()+1))

                        # topk_v, topk_ids = pred.topk(k=sample_topk, dim=-1)
                        # sample_ids = torch.distributions.Categorical(logits=topk_v).sample()
                        # sample_ids = torch.gather(topk_ids, -1, sample_ids.unsqueeze(-1)).squeeze(-1)
                        print("Debug prob to topk")
                        prob = torch.softmax(probs, dim=-1)         # bug probs: -> pred    # 影响不大
                        prob = torch.gather(prob, -1, sample_ids.unsqueeze(-1)).squeeze(-1)
                    else:
                        sample_ids = torch.distributions.Categorical(logits=pred).sample()
                        prob = torch.softmax(pred, dim=-1)
                        prob = torch.gather(prob, -1, sample_ids.unsqueeze(-1)).squeeze(-1)

                    sample_ids[code_idx != -1] = code_idx[code_idx != -1]
                    prob[code_idx != -1] = 1e10

                    ratio = 1.0 * (t + 1) / self.T
                    mask_ratio = self.gamma(ratio)

                    mask_len = num_unknown_code * mask_ratio # all code len
                    mask_len = torch.minimum(mask_len, num_unknown_code - 1)
                    mask_len = mask_len.clamp(min=1).long()

                    if use_maskgit:
                        confidence = prob.log()
                    else:
                        temperature = choice_temperature * (1.0 - ratio)
                        # gumbels = -torch.empty_like(prob, memory_format=torch.legacy_contiguous_format).exponential_().log()
                        gumbels = torch.zeros_like(prob).uniform_(0, 1)
                        gumbels = -log(-log(gumbels))
                        confidence = prob.log() + temperature * gumbels

                    cutoff = torch.sort(confidence, dim=-1)[0][
                        torch.arange(mask_len.shape[0], device=mask_len.device), mask_len
                    ].unsqueeze(1)
                    mask = confidence < cutoff
                    x = self.vq_point_cloud.vector_quantizer.get_codebook_entry(sample_ids)
                    code_idx = sample_ids.clone()

                    if t != self.T - 1:
                        code_idx[mask] = -1
                        x[mask] = self.bi_directional_Transformer.mask_token
                        x_indices = code_idx.clone()

            # NOTE original decoder

            lidar_density, lidar_voxel = self.vq_point_cloud.lidar_decoder(x)
            generated_sample_v = gumbel_sigmoid(lidar_voxel, hard=True, generator=self.generator)
            generated_points_v = BEVWorldVAE.voxels2points(self.vq_point_cloud.grid_size,
                                                            generated_sample_v.unsqueeze(1))

            if "offsets" in batch:
                offsets = batch["offsets"]
                offsets = sum(offsets, [])
            else:
                offsets = None


        results = {}
        results['raw_points'] = batch["lidar_points"]
        results['gt_points'] = BEVWorldVAE.voxels2points(self.vq_point_cloud.grid_size,
                                                         voxels.unsqueeze(1))
        results['gt_voxels'] = voxels
        results['pred_voxels'] = generated_sample_v
        results['pred_points'] = generated_points_v

        return results

    def evaluate_pipeline(
        self, global_step: int, dataset_length: int,
        validation_dataloader: torch.utils.data.DataLoader,
        validation_datasampler=None,
        log_type="wandb"
    ):
        # self._evaluate_COPILOT4D(should_save, global_step, dataset_length, validation_dataloader, validation_datasampler)
        with torch.no_grad():
            for batch in tqdm(validation_dataloader):
                if self.ddp:
                    torch.distributed.barrier()
                if "lidar_points" in batch:
                    batch_size, num_frames = len(batch["lidar_points"]), len(batch["lidar_points"][0])
                else:
                    batch_size, num_frames = len(batch["lidar_points_raw"]), len(batch["lidar_points_raw"][0])
                # batch_size, num_frames = len(batch["lidar_points"]), len(batch["lidar_points"][0])
                results = self.inference_pipeline(batch)
                voxels = results['gt_voxels']
                pred_voxels = results['pred_voxels']
                gt_points = results['gt_points']
                pred_points = results['pred_points']
                voxels, pred_voxels = voxels.to(int), pred_voxels.to(int)
                if "voxel_diff" in self.metrics:
                    self.metrics["voxel_diff"].update(voxels, pred_voxels)
                if "voxel_iou" in self.metrics:
                    self.metrics["voxel_iou"].update(voxels, pred_voxels)
                for k in self.metrics:
                    if "chamfer" in k:
                        self.metrics[k].update(pred_points, gt_points, self.device)
                if self.config.get("save_results", False):
                    # save pred voxels
                    paths = [
                        os.path.join(self.output_path, 'pred_voxel_' + k)
                        for i in batch["sample_data"]
                        for j in i
                        for k in j["filename"] if k.endswith(".bin")
                    ]
                    pred_voxel_pc = pred_points
                    pred_voxel_pc = DWM.postprocess_points(batch, pred_voxel_pc)
                    pred_voxel_pc = [
                                    j
                                    for i in pred_voxel_pc
                                    for j in i
                                ]
                    for path, points in zip(paths, pred_voxel_pc):
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        points = points.numpy()
                        padded_points = np.concatenate([
                            points, np.zeros((points.shape[0], 2), dtype=np.float32)
                        ], axis=-1)
                        with open(path, "wb") as f:
                            f.write(padded_points.tobytes())
                    # save raw points
                    paths = [
                            os.path.join(self.output_path, 'raw_' + k)
                            for i in batch["sample_data"]
                            for j in i
                            for k in j["filename"] if k.endswith(".bin")
                        ]
                    raw_points = [
                                        j
                                        for i in results['raw_points']
                                        for j in i
                                    ]
                    for path, points in zip(paths, raw_points):
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            points = points.numpy()
                            padded_points = np.concatenate([
                                points, np.zeros((points.shape[0], 2), dtype=np.float32)
                            ], axis=-1)
                            with open(path, "wb") as f:
                                f.write(padded_points.tobytes())
                    # save gt points
                    paths = [
                            os.path.join(self.output_path, 'gt_' + k)
                            for i in batch["sample_data"]
                            for j in i
                            for k in j["filename"] if k.endswith(".bin")
                        ]
                    gt_points = DWM.postprocess_points(batch, gt_points)
                    gt_points = [
                                    j
                                    for i in gt_points
                                    for j in i
                                ]
                    for path, points in zip(paths, gt_points):
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        points = points.numpy()
                        padded_points = np.concatenate([
                            points, np.zeros((points.shape[0], 2), dtype=np.float32)
                        ], axis=-1)
                        with open(path, "wb") as f:
                            f.write(padded_points.tobytes())

            for k, metric in self.metrics.items():
                value = metric.compute()
                metric.reset()
                if self.should_save:
                    print("{}: {:.3f}, count: {}".format(k, value, metric.num_samples))
                    if log_type == "tensorboard":
                        self.summary.add_scalar(
                            "evaluation/{}".format(k), value, global_step)
                    elif log_type == "wandb" and wandb.run is not None:
                        wandb.log({f"evaluation_{k}": value})