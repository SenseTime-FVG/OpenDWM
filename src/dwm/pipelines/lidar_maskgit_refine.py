import dwm.common
import dwm.models.dvgo_utils
import os
import cv2
import numpy as np
import safetensors.torch
import time
import torch
import torch.cuda.amp
import torch.nn.functional as F
import torch.utils.tensorboard

import open3d as o3d
import concurrent.futures
from tqdm import tqdm
from functools import partial
from einops import rearrange
from collections import defaultdict
import imageio
import pickle
import math
from easydict import EasyDict as edict

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
    # import pdb
    # pdb.set_trace()

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


class MaskGITRefine(torch.nn.Module):
    @staticmethod
    def load_state(path: str):
        if path.endswith(".safetensors"):
            state = safetensors.torch.load_file(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu")
        return state
    
    def __init__(
        self, output_path: str, config: dict, device, vq_point_cloud,
        bi_directional_Transformer, 
        vq_point_cloud_checkpoint_path: str = None,
        bi_directional_Transformer_checkpoint_path: str = None, 
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
            2. add temporal flip transform
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
        if vq_point_cloud_checkpoint_path is not None:
            state_dict = MaskGITRefine.load_state(vq_point_cloud_checkpoint_path)
            if 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']
            missing_keys, unexpected_keys = self.vq_point_cloud.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print("Missing keys in state dict:", missing_keys)
            if unexpected_keys:
                print("Unexpected keys in state dict:", unexpected_keys)
        self.vq_point_cloud.eval()
        
        self.bi_directional_Transformer_wrapper = self.bi_directional_Transformer = bi_directional_Transformer
        self.bi_directional_Transformer.to(self.device)
        if resume_from is not None:
            resume_from_path = os.path.join(output_path, resume_from)
            if self.should_save:
                print(f"===Resume from {resume_from_path}...")
            self.bi_directional_Transformer.load_state_dict(
                MaskGITRefine.load_state(resume_from_path))
        elif bi_directional_Transformer_checkpoint_path is not None:
            if self.should_save:
                print(f"===Load from {bi_directional_Transformer_checkpoint_path}...")
            self.bi_directional_Transformer.load_state_dict(
                MaskGITRefine.load_state(bi_directional_Transformer_checkpoint_path))
        # if bi_directional_Transformer_checkpoint_path is not None:
        #     self.bi_directional_Transformer.load_state_dict(
        #         MaskGITRefine.load_state(bi_directional_Transformer_checkpoint_path))
        
        self.gamma = gamma_func("cosine")
        self.iter = 0
        self.T = 30
        self.BLANK_CODE = None

        # self.vis_flag = vis_flag
        self.code_dict = {}
        for i in range(self.vq_point_cloud.vector_quantizer.n_e):
            self.code_dict[i] = 0
        
        # setup training parts
        self.loss_list = []
        self.step_duration = 0

        if config["device"] == "cuda":
            self.grad_scaler = torch.cuda.amp.GradScaler()

        if self.ddp:
            self.bi_directional_Transformer_wrapper = torch.nn.parallel.DistributedDataParallel(
                self.bi_directional_Transformer,
                device_ids=[int(os.environ["LOCAL_RANK"])])

        if self.should_save:
            self.summary = torch.utils.tensorboard.SummaryWriter(
                os.path.join(output_path, "log"))

        self.optimizer = dwm.common.create_instance_from_config(
            config["optimizer"],
            params=self.bi_directional_Transformer_wrapper.parameters())
        self.lr_scheduler = dwm.common.create_instance_from_config(
            config["lr_scheduler"], optimizer=self.optimizer)

        # === Load BLANK CODE
        with open(os.path.join('/mnt/storage/user/wangxiaodong/DWM_work_dir/blank_code_eval/', "blank_code.pkl"), 'rb') as f:
            blank_code = pickle.load(f)
            self.BLANK_CODE = blank_code
            print("=== Load BLANK CODE: ", blank_code)
    
    def save_checkpoint(self, output_path: str, steps: int):
        if self.should_save:
            os.makedirs(
                os.path.join(output_path, "checkpoints"), exist_ok=True)
            torch.save(self.bi_directional_Transformer.state_dict(), os.path.join(output_path, "checkpoints", "{}.pth".format(steps)))

        if self.ddp:
            torch.distributed.barrier()
    
    def log(self, global_step: int, log_steps: int):
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
                    "Step {} ({:.1f} s/step), {}".format(
                        global_step, self.step_duration / log_steps,
                        log_string))
                for k, v in log_dict.items():
                    self.summary.add_scalar(
                        "train/{}".format(k), v, global_step)

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
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def mask_code(self, code, mask_token, cond=None, mask_ratio=None, eta=20, with_noise=False):
        if with_noise:
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
        mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)  # no cls token
        # after gather -> 16,6400,1024
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        if cond is not None:
            x = torch.cat([x, cond], dim=-1)

        return x, mask, ids_restore
    
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

        if cond is not None:
            x = torch.cat([x, cond], dim=-1)

        return x, new_mask, ids_restore

    def process_inference_input(self, task_code, x, num_frames, code, ctype='pre'):
        assert task_code == 0
        num_frames_past = num_frames // 2
        if ctype == 'pre' or ctype == 'fin':
            code = rearrange(code, "(b f) ... -> b f ...", f=num_frames)
            x = rearrange(x, "(b f) ... -> b f ...", f=(num_frames+1)//2)
            batch_size = code.shape[0]
            x_past = code[:, :num_frames_past]
            x = torch.cat([x_past, x], dim=1).flatten(0, 1)
        elif ctype == 'after':
            x = rearrange(x, "(b f) ... -> b f ...", f=num_frames)
            x = x[:, num_frames_past:].flatten(0, 1)
        else:
            raise NotImplementedError
        return x

    def mutlitask_mask_code(self, task_code, num_frames, code, infer=False):
        """
        1. sample a task
        """
        use_discrete_diffusion = self.common_config.get('use_discrete_diffusion', False)
        if infer:
            assert task_code == 0
            batch_size = code.shape[0] // num_frames
            x_future = self.bi_directional_Transformer.mask_token[:, None].repeat(batch_size, (num_frames+1)//2, 
                self.bi_directional_Transformer.img_size**2, 1).flatten(0, 1)
            return x_future, None, None
        else:
            if task_code == 0:
                code = rearrange(code, "(b f) ... -> b f ...", f=num_frames)
                batch_size = code.shape[0]
                num_frames_past = num_frames // 2
                past = code[:, :num_frames_past].flatten(0, 1)
                future = code[:, num_frames_past:].flatten(0, 1)
                x_past, mask_past, ids_restore_past = self.mask_code(past, 
                    self.bi_directional_Transformer.mask_token, mask_ratio=0, with_noise=False)
                x_future, mask_future, ids_restore_future = self.mask_code(future, 
                    self.bi_directional_Transformer.mask_token, mask_ratio=None, with_noise=use_discrete_diffusion)

                # merge
                x = torch.cat([rearrange(x_past, "(b f) ... -> b f ...", b=batch_size), 
                    rearrange(x_future, "(b f) ... -> b f ...", b=batch_size)], dim=1).flatten(0, 1)
                mask = torch.cat([rearrange(mask_past, "(b f) ... -> b f ...", b=batch_size), 
                    rearrange(mask_future, "(b f) ... -> b f ...", b=batch_size)], dim=1).flatten(0, 1)
                ids_restore = torch.cat([rearrange(ids_restore_past, "(b f) ... -> b f ...", b=batch_size), 
                    rearrange(ids_restore_future, "(b f) ... -> b f ...", b=batch_size)], dim=1).flatten(0, 1)
            else:
                x, mask, ids_restore = self.mask_code(code, 
                    self.bi_directional_Transformer.mask_token, mask_ratio=None, with_noise=use_discrete_diffusion)
            return x, mask, ids_restore

    def _data_process(self, batch, **kwargs):
        # gpu dataprocessor
        if self.common_config.get('use_reference_coordinate', False):
            # dict_keys(['fps', 'pts', 'lidar_transforms', 'ego_transforms', 'lidar_points'])
            lidar_transforms = batch['lidar_transforms']
            ego_transforms = batch['ego_transforms']            # b,t...
            lidar_points = batch["lidar_points_raw"]
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
            batch["lidar_points"] = lidar_points
            batch["offsets"] = offsets
        return batch

    def train_step(self, batch: dict, global_step: int):
        t0 = time.time()
        
        batch_size, num_frames = len(batch["lidar_points"]), len(batch["lidar_points"][0])
        num_frames_past = num_frames // 2
        # ====== Data process (GPU)
        batch = self._data_process(batch, num_frames_past=num_frames_past)

        # === Process input/target (voxelizer/vae)
        # ====== 3D
        points = sum([[j[:, :3] for j in i] for i in batch["lidar_points"]], [])
        # TODO: add temporal flip transform
        
        self.bi_directional_Transformer_wrapper.train()
        with torch.no_grad():
            voxels = self.vq_point_cloud.voxelizer([[_] for _ in points])
            voxels = voxels.to(self.device)
            lidar_feats = self.vq_point_cloud.lidar_encoder(voxels)
            code, _, code_indices = self.vq_point_cloud.vector_quantizer(lidar_feats, self.vq_point_cloud.code_age,
                                                            self.vq_point_cloud.code_usage)
        # ====== 2D

        # ====== Condition (layout/cxt/mask)


        # === Prepare training (scheduler/multi-task/mask code)
        # ====== Multi-task temporal
        # ========= Sample multi task
        task_prob = self.common_config.get('task_prob', [0.5, 0.4, 0.1])
        task_code = np.random.choice(3, size=1, p=task_prob)[0]
        # ========= Process
        x, mask, ids_restore = self.mutlitask_mask_code(task_code, num_frames, code, infer=False)
        if task_code in [0, 1]:
            # causal mask   note: this is not the actual batch_size in self-att
            attention_mask_temporal = _make_causal_mask((batch_size, num_frames), dtype=x.dtype, device=x.device)[0, 0]
        else:
            attention_mask_temporal = _make_eye_mask((batch_size, num_frames), dtype=x.dtype, device=x.device)[0, 0]
        # === Model forward/loss
        pred = self.bi_directional_Transformer_wrapper(x, attention_mask_temporal=attention_mask_temporal)

        if task_code == 0:
            pred = rearrange(pred, "(b f) ... -> b f ...", b=batch_size)[:, num_frames_past:].flatten(0, 1)
            target = rearrange(code_indices, "(b f s) -> b f s", b=batch_size, f=num_frames)[:, num_frames_past:].flatten()
            mask = rearrange(mask, "(b f) ... -> b f ...", b=batch_size)[:, num_frames_past:].flatten(0, 1)
        else:
            pred, target = pred, code_indices

        mask = mask.flatten(0, 1)

        loss = (
            F.cross_entropy(pred.flatten(0, 1), target, reduction="none", label_smoothing=0.1) * mask
        ).sum() / (mask.sum() + 1e-5)

        acc = (pred.flatten(0, 1).max(dim=-1)[1] == target)[mask > 0].float().mean()

        losses = {
            "loss_code_pred": loss,
            "acc": acc,
        }
        loss = sum([losses[i] for i in losses if "loss" in i])

        # optimize parameters
        should_optimize = \
            ("gradient_accumulation_steps" not in self.config) or \
            ("gradient_accumulation_steps" in self.config and
                (global_step + 1) %
             self.config["gradient_accumulation_steps"] == 0)
        if self.config["device"] == "cuda":
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        if should_optimize:
            if "max_norm_for_grad_clip" in self.config:
                torch.nn.utils.clip_grad_norm_(
                    self.bi_directional_Transformer_wrapper.parameters(),
                    self.config["max_norm_for_grad_clip"])

            if self.config["device"] == "cuda":
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        self.lr_scheduler.step()
        self.loss_list.append(losses)
        self.step_duration += time.time() - t0
    
    def voxels2points(self, voxels):
        non_zero_indices = torch.nonzero(voxels)
        xy = (non_zero_indices[:, 2:] * self.vq_point_cloud.voxelizer.step) + self.vq_point_cloud.voxelizer.y_min
        z = (non_zero_indices[:, 1] * self.vq_point_cloud.voxelizer.z_step) + self.vq_point_cloud.voxelizer.z_min
        xyz = torch.cat([xy, z.unsqueeze(1)], dim=1)

        return xyz

    def batch_voxels2points(self, voxels):
        rt = []
        for i in range(len(voxels)):
            cur = self.voxels2points(voxels[i:i+1])
            rt.append(cur)

        return rt

    def preview_pipeline(
        self, batch: dict, output_path: str, global_step: int
    ):        
        batch_size, num_frames = len(batch["lidar_points"]), len(batch["lidar_points"][0])
        voxels, generated_sample_v, generated_sample_r, _ = self.inference_pipeline(batch)
        if self.should_save:
            folder_name = os.path.join(output_path, "preview", str(global_step))
            os.makedirs(folder_name, exist_ok=True)

            for bid in range(batch_size):
                st, fin = bid*num_frames, (bid+1)*num_frames
                draw_bev_lidar_from_list(voxels[st:fin], f"{folder_name}/{bid}_voxel_gt.mp4")
                draw_bev_lidar_from_list(generated_sample_v[st:fin], f"{folder_name}/{bid}_generated_v.mp4")
                draw_bev_lidar_from_list(generated_sample_r[st:fin], f"{folder_name}/{bid}_generated_r.mp4")

    def inference_pipeline(self, batch, output_for_eval=False):
        def count(idx):
            unique_elements, counts = torch.unique(idx, return_counts=True)
            sorted_indices = torch.argsort(counts, descending=True)
            sorted_elements = unique_elements[sorted_indices]
            sorted_counts = counts[sorted_indices]
            return sorted_elements, sorted_counts
        
        # TODO: use should_save
        # TODO: now: gt_points + uncond points -> uncond infer blank code + past (from gt) to future (pred)
        batch_size, num_frames = len(batch["lidar_points"]), len(batch["lidar_points"][0])
        num_frames_past = num_frames // 2
        batch = self._data_process(batch, num_frames_past=num_frames_past)
        points = sum([[j[:, :3] for j in i] for i in batch["lidar_points"]], [])

        self.bi_directional_Transformer_wrapper.eval()

        with torch.no_grad():
            voxels = self.vq_point_cloud.voxelizer([[_] for _ in points])
            voxels = voxels.to(self.device)
            lidar_feats = self.vq_point_cloud.lidar_encoder(voxels)
            code, _, code_indices = self.vq_point_cloud.vector_quantizer(lidar_feats, self.vq_point_cloud.code_age,
                                                            self.vq_point_cloud.code_usage)

            # original code

            # if self.BLANK_CODE is None:
            #     # NOTE blank code
            #     for _ in range(100):
            #         # TODO: how to infer in temporal, task_code 2 + 1/8 frames?
            #         import pdb
            #         pdb.set_trace()
            #         x = self.bi_directional_Transformer.mask_token.repeat(num_frames, self.bi_directional_Transformer.img_size**2, 1)
            #         code_idx = torch.ones((num_frames, self.bi_directional_Transformer.img_size**2), dtype=torch.int64, device=x.device) * -1
            #         num_unknown_code = (code_idx == -1).sum(dim=-1)
            #         pred = self.bi_directional_Transformer(x)
            #         sample_ids = torch.distributions.Categorical(logits=pred).sample()
            #         codes, counts = count(sample_ids)
            #         for code_idx, conut_nbs in zip(codes, counts):
            #             self.code_dict[int(code_idx.data)] += conut_nbs
            #         blank_code = [k for k, v in sorted(self.code_dict.items(), key=lambda item: item[1], reverse=True)[:20]]
            #     self.BLANK_CODE = blank_code
            #     print('----------------Obtain blank code-----------------------------')
            
            # NOTE generation
            choice_temperature = 2.0

            # x = self.bi_directional_Transformer.mask_token.repeat(1, self.bi_directional_Transformer.img_size**2, 1)
            # code_idx = torch.ones((1, self.bi_directional_Transformer.img_size**2), dtype=torch.int64, device=x.device) * -1
            # num_unknown_code = (code_idx == -1).sum(dim=-1)
            
            # ===Sample task code
            # now, only support task_code == 0
            task_code = 0
            x, _, _ = self.mutlitask_mask_code(task_code, num_frames, code, infer=True)
            code_idx = torch.ones((x.shape[0], x.shape[1]), dtype=torch.int64, device=x.device) * -1
            num_unknown_code = (code_idx == -1).sum(dim=-1)

            for t in range(self.T):
                # TODO: cat
                x = self.process_inference_input(task_code, x, num_frames, code, ctype='pre')
                if task_code in [0, 1]:
                    # causal mask   note: this is not the actual batch_size in self-att
                    attention_mask_temporal = _make_causal_mask((batch_size, num_frames), dtype=x.dtype, device=x.device)[0, 0]
                else:
                    attention_mask_temporal = _make_eye_mask((batch_size, num_frames), dtype=x.dtype, device=x.device)[0, 0]
                pred = self.bi_directional_Transformer(x, attention_mask_temporal=attention_mask_temporal)
                pred = self.process_inference_input(task_code, pred, num_frames, code, ctype='after')
                # TODO：detach

                if t < 10:
                    pred[..., self.BLANK_CODE] = -10000

                sample_ids = torch.distributions.Categorical(logits=pred).sample()

                prob = torch.softmax(pred, dim=-1)
                prob = torch.gather(prob, -1, sample_ids.unsqueeze(-1)).squeeze(-1)
                if self.common_config.get('use_discrete_diffusion', False) and self.common_config['update_unmask_pos']:
                    pass            # now, do nothing
                else:
                    sample_ids[code_idx != -1] = code_idx[code_idx != -1]
                # sample_ids[code_idx != -1] = code_idx[code_idx != -1]

                if self.common_config.get('use_discrete_diffusion', False):
                    if self.common_config['discrete_diffusion_max_exist']:
                        prob[code_idx != -1] = 1e10
                else:
                    prob[code_idx != -1] = 1e10

                ratio = 1.0 * (t + 1) / self.T
                mask_ratio = self.gamma(ratio)

                mask_len = num_unknown_code * mask_ratio # all code len
                mask_len = torch.minimum(mask_len, num_unknown_code - 1)
                mask_len = mask_len.clamp(min=1).long()

                temperature = choice_temperature * (1.0 - ratio)
                gumbels = -torch.empty_like(prob, memory_format=torch.legacy_contiguous_format).exponential_().log()
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
            
            x = self.process_inference_input(task_code, x, num_frames, code, ctype='fin')
            # NOTE original decoder
            lidar_density, lidar_voxel = self.vq_point_cloud.lidar_decoder(x)
            generated_sample_v = gumbel_sigmoid(lidar_voxel, hard=True, generator=self.generator)
            geneted_points_v = self.batch_voxels2points(generated_sample_v)

            # NOTE ray casting
            lidar_voxel = gumbel_sigmoid(lidar_voxel, hard=True, generator=self.generator)
            pooled_voxels = F.max_pool3d(lidar_voxel, (4, 8, 8))
            _, _, lidar_rec = self.vq_point_cloud.ray_render_dvgo(lidar_density, points, pooled_voxels)

            lidar_rec = self.vq_point_cloud.voxelizer([[_] for _ in lidar_rec])
            generated_sample_r = lidar_rec
            geneted_points_r = self.batch_voxels2points(lidar_rec)

            # GT points
        gt_points = self.batch_voxels2points(voxels)

        # type
        gt_points = [pt.detach().cpu().numpy() for pt in gt_points]
        geneted_points_v = [pt.detach().cpu().numpy() for pt in geneted_points_v]
        geneted_points_r = [pt.detach().cpu().numpy() for pt in geneted_points_r]

        if output_for_eval:
            if task_code == 0:
                gt_points = list(map(lambda x: x[1], filter(lambda x: (x[0]%num_frames) >= num_frames_past, enumerate(gt_points))))
                geneted_points_v = list(map(lambda x: x[1], filter(lambda x: (x[0]%num_frames) >= num_frames_past, enumerate(geneted_points_v))))
                geneted_points_r = list(map(lambda x: x[1], filter(lambda x: (x[0]%num_frames) >= num_frames_past, enumerate(geneted_points_r))))
                if "offsets" in batch:
                    offsets = sum(batch["offsets"], [])
                    offsets = list(map(lambda x: x[1], filter(lambda x: (x[0]%num_frames) >= num_frames_past, enumerate(offsets))))
                else:
                    offsets = None
            else:
                raise NotImplementedError
            return gt_points, geneted_points_v, geneted_points_r, offsets
        else:
            return voxels, generated_sample_v, generated_sample_r, None

    def evaluate_pipeline(
        self, global_step: int, dataset_length: int,
        validation_dataloader: torch.utils.data.DataLoader,
        validation_datasampler=None
    ): 
        self._evaluate_COPILOT4D(self.should_save, global_step, dataset_length, validation_dataloader, validation_datasampler)

    def _evaluate_ultralidar(
        self, global_step: int, dataset_length: int,
        validation_dataloader: torch.utils.data.DataLoader,
        validation_datasampler=None
    ): 
        arrays_gt = []
        arrays_voxel = []

        # TODO: refine it?
        voxelizer = Voxelizer(-50, 50, -50, 50, 0.15625, -3.73, 2.27, 0.15)

        def array_to_histograms(samples, src):
            hist = []
            for sample in samples:
                if src == "gt":
                    voxels = voxelizer([[torch.from_numpy(sample)]])
                    non_zero_indices = torch.nonzero(voxels)

                    xy = (non_zero_indices[:, 2:] * voxelizer.step) + voxelizer.y_min
                    z = (non_zero_indices[:, 1] * voxelizer.z_step) + voxelizer.z_min
                    point_cloud = torch.cat([xy, z.unsqueeze(1)], dim=1).detach().cpu().numpy()
                else:
                    point_cloud = sample

                histogram = point_cloud_to_histogram(160, 100, point_cloud)[0]
                hist.append(histogram)
            return hist

        with torch.no_grad():

            for batch in tqdm(validation_dataloader):
                if self.ddp:
                    torch.distributed.barrier()

                gt_points_all, geneted_points_v_all, _, _ = self.inference_pipeline(batch, output_for_eval=True)
                
                for fid in range(len(gt_points_all)):
                    gt_points = gt_points_all[fid]
                    geneted_points_v = geneted_points_v_all[fid]

                    # gt
                    point_cloud_gt = o3d.geometry.PointCloud()
                    point_cloud_gt.points = o3d.utility.Vector3dVector(gt_points)

                    pcd = np.array(point_cloud_gt.points)
                    # TODO: from here to eval
                    xyz = torch.from_numpy(pcd)
                    bev = voxelizer([[torch.cat([xyz, torch.zeros_like((xyz[:, [0]]))], dim=1)]]).float()
                    if bev[:, :, 350:370, 310:330].sum() < 200:
                        xyz[:, 1] = -xyz[:, 1]
                    
                    if self.ddp:
                        arrays_gt.append(xyz.numpy())
                    else:
                        arrays_gt.append(xyz.numpy())

                    # voxel
                    point_cloud_v = o3d.geometry.PointCloud()
                    point_cloud_v.points = o3d.utility.Vector3dVector(geneted_points_v)

                    pcd = np.array(point_cloud_v.points)
                    xyz = torch.from_numpy(pcd)
                    bev = voxelizer([[torch.cat([xyz, torch.zeros_like((xyz[:, [0]]))], dim=1)]]).float()
                    if bev[:, :, 350:370, 310:330].sum() < 200:
                        xyz[:, 1] = -xyz[:, 1]
                    
                    if self.ddp:
                        arrays_voxel.append(xyz.numpy())
                    else:
                        arrays_voxel.append(xyz.numpy())

        if self.ddp:
            all_arrays_gt = arrays_gt
            all_arrays_voxel = arrays_voxel
        # jsd
        else:
            all_arrays_gt = arrays_gt
            all_arrays_voxel = arrays_voxel
        
        gt_histograms = array_to_histograms(all_arrays_gt, src="gt")            # list of array: (100, 100)
        model_histograms = array_to_histograms(all_arrays_voxel, src="model")
        if self.ddp:
            all_gt_histograms = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(all_gt_histograms, gt_histograms)
            all_gt_histograms = sum(all_gt_histograms, [])

            all_model_histograms = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(all_model_histograms, model_histograms)
            all_model_histograms = sum(all_model_histograms, [])

            gt_histograms = all_gt_histograms
            model_histograms = all_model_histograms

        model_p = np.stack(model_histograms, axis=0)
        model_p = np.sum(model_p, axis=0)

        kitti_p = np.stack(gt_histograms, axis=0)
        kitti_p = np.sum(kitti_p, axis=0)

        jsd_score = jsd_2d(kitti_p, model_p)


        # TODO: match is not needed in task 0?
        # kitti_model_distance = compute_mmd(gt_histograms, model_histograms, gaussian, is_hist=True)

        if self.should_save:
            print(f'jsd score: {jsd_score}\n')
            # print(f'mmd score: {kitti_model_distance}\n')
        torch.distributed.barrier()

    def _evaluate_COPILOT4D(
        self, global_step: int, dataset_length: int,
        validation_dataloader: torch.utils.data.DataLoader,
        validation_datasampler=None
    ):
        metrics = defaultdict(lambda: 0)
        with torch.no_grad():

            for batch in tqdm(validation_dataloader):
                if self.ddp:
                    torch.distributed.barrier()

                # output_for_eval: output future frames only
                gt_points_all, geneted_points_v_all, _, offsets = self.inference_pipeline(batch, output_for_eval=True)
                
                for fid in range(len(gt_points_all)):
                    gt_points = gt_points_all[fid]
                    geneted_points_v = geneted_points_v_all[fid]

                    # gt
                    point_cloud_gt = o3d.geometry.PointCloud()
                    point_cloud_gt.points = o3d.utility.Vector3dVector(gt_points)

                    # output_origin = None # torch.zeros_like(output_origin)
                    gt_pcd = torch.from_numpy(np.array(point_cloud_gt.points))             # n,3
                    # TODO: from here to eval
                    # xyz = torch.from_numpy(pcd)
                    # bev = voxelizer([[torch.cat([xyz, torch.zeros_like((xyz[:, [0]]))], dim=1)]]).float()
                    # if bev[:, :, 350:370, 310:330].sum() < 200:
                    #     xyz[:, 1] = -xyz[:, 1]
                    
                    # if self.ddp:
                    #     arrays_gt.append(xyz.numpy())
                    # else:
                    #     arrays_gt.append(xyz.numpy())

                    # voxel
                    point_cloud_v = o3d.geometry.PointCloud()
                    point_cloud_v.points = o3d.utility.Vector3dVector(geneted_points_v)

                    pred_pcd = torch.from_numpy(np.array(point_cloud_v.points))           # n,3
                    # xyz = torch.from_numpy(pcd)
                    # bev = voxelizer([[torch.cat([xyz, torch.zeros_like((xyz[:, [0]]))], dim=1)]]).float()
                    # if bev[:, :, 350:370, 310:330].sum() < 200:
                    #     xyz[:, 1] = -xyz[:, 1]
                    
                    # if self.ddp:
                    
                    #     arrays_voxel.append(xyz.numpy())
                    # else:
                    #     arrays_voxel.append(xyz.numpy())
                    if offsets is not None:
                        origin = offsets[fid].to(pred_pcd.device)
                    else:
                        origin = pred_pcd.new_zeros(pred_pcd.shape[-1])           # in batch, fid -> 3
                    if self.common_config.get('use_reverse_origin', False):          # debug code
                        origin = -origin
                    # TODO: pc_range is set as default in metrics_copilot4d, check the correctness of it
                    metrics["count"] += 1
                    metrics["chamfer_distance"] += compute_chamfer_distance(pred_pcd, gt_pcd, self.device).item()
                    metrics["chamfer_distance_inner"] += compute_chamfer_distance_inner(pred_pcd, gt_pcd, self.device, pc_range=[-50.0, -50.0, -5, 50.0, 50.0, 3]).item()
                    # The dtype of origin code is not very clear
                    l1_error, absrel_error, l1_error_med, absrel_error_med = compute_ray_errors(pred_pcd, gt_pcd, origin, self.device, pipeline=self, pc_range=[-50.0, -50.0, -5, 50.0, 50.0, 3])
                    metrics["l1_error_mean"] += l1_error
                    metrics["absrel_error_mean"] += absrel_error
                    metrics["l1_error_median"] += l1_error_med
                    metrics["absrel_error_median"] += absrel_error_med

                # print("Debug eval")
                # break
            
            metrics = dict(metrics)
            if self.ddp:
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
                if self.should_save:
                    self.summary.add_scalar("metrics/{}".format(k), v_avg, global_step)
                    print(f'{k}: {v_avg} with count {count}\n')

