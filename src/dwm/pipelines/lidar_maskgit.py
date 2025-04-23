import dwm.common
import os
import numpy as np
import safetensors.torch
import time
import torch
import torch.amp
import torch.nn.functional as F
import contextlib
import torch.utils.tensorboard
import wandb
from torch import nn
from tqdm import tqdm
import pickle
import math
import re
from easydict import EasyDict as edict
from typing import Optional
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed.fsdp.sharded_grad_scaler
import torch.distributed.checkpoint.state_dict
from dwm.utils.preview import make_lidar_preview_tensor, save_tensor_to_video
from dwm.utils.lidar import preprocess_points, postprocess_points, voxels2points
from torchvision import transforms

# schedule the mask level

task_type_dict = {
    0: "generation",
    1: "prediction",
}


def gamma_func(mode="cosine"):
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
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(
        shape, generator=generator)
    return torch.log(U) - torch.log(1 - U)


def _sigmoid_sample(logits, tau=1, generator=None):
    """
    Implementation of Bernouilli reparametrization based on Maddison et al. 2017
    """
    dims = logits.dim()
    logistic_noise = _sample_logistic(
        logits.size(), out=logits.data.new(), generator=generator)
    y = logits + logistic_noise
    return torch.sigmoid(y / tau)


# Transform the logits to the mask level
def gumbel_sigmoid(logits, tau=1, hard=False, generator=None):

    gumbel_sigmoid_coeff = 1.0
    y_soft = _sigmoid_sample(
        logits * gumbel_sigmoid_coeff, tau=tau, generator=generator)
    if hard:
        y_hard = torch.where(y_soft > 0.5, torch.ones_like(
            y_soft), torch.zeros_like(y_soft))
        y = y_hard.data - y_soft.data + y_soft
    else:
        y = y_soft
    return y


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


class MaskGITPipeline(torch.nn.Module):
    @staticmethod
    def load_state(path: str):
        if path.endswith(".safetensors"):
            state = safetensors.torch.load_file(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu", weights_only=False)
        return state

    def __init__(
        self, output_path: str, config: dict,
        device, vq_point_cloud, bi_directional_Transformer,
        bev_layout_encoder: nn.Module = None,
        vq_point_cloud_ckpt_path: str = None,
        vq_blank_code_path: str = None,
        metrics: dict = dict(),
        training_config: dict = dict(),
        inference_config: dict = dict(),
        common_config: dict = dict(),
        resume_from=None,
        model_ckpt_path: str = None
    ):
        r"""
        Args:
            training_config (`dict`):
                training related parameters, e.g. dropout
            inference_config (`dict`):
                inference related parameters, e.g. cfg
            common_config (`dict`):
                config for model, used for both train/val
            bev_layout_encoder: This feature has been moved into the model. But you can still use it in the pipeline.
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
            missing_keys, unexpected_keys = self.vq_point_cloud.load_state_dict(
                state_dict, strict=False)
            if missing_keys:
                print("Missing keys in state dict:", missing_keys)
            if unexpected_keys:
                print("Unexpected keys in state dict:", unexpected_keys)
        self.vq_point_cloud.eval()

        self.bi_directional_Transformer_wrapper = self.bi_directional_Transformer = bi_directional_Transformer
        self.bi_directional_Transformer.to(self.device)

        if self.training_config.get('set_vq_no_grad', False):
            # no influence, only for test security
            print("Set vq no grad")
            self.vq_point_cloud.requires_grad_(False)

        # self.vq_point_cloud.vector_quantizer.embedding
        if self.training_config.get('weight_tying', False):
            print("Set weight tying!!!")
            self.bi_directional_Transformer.pred.requires_grad_(False)
            self.bi_directional_Transformer.pred.weight.copy_(
                self.vq_point_cloud.vector_quantizer.embedding.weight)

        if resume_from is not None:
            self.resume_from = resume_from
            model_state_dict = MaskGITPipeline.load_state(
                os.path.join(
                    output_path, "checkpoints", "{}.pth".format(resume_from)))
            self.bi_directional_Transformer.load_state_dict(
                model_state_dict["bi_directional_Transformer"])

        if model_ckpt_path is not None:
            self.bi_directional_Transformer.load_state_dict(
                MaskGITPipeline.load_state(model_ckpt_path)["bi_directional_Transformer"])

        self.gamma = gamma_func("cosine")
        self.iter = 0
        self.T = self.inference_config.get("sample_steps", 30)
        self.BLANK_CODE = None
        self.grad_scaler = None
        if self.training_config.get("enable_grad_scaler", False):
            self.grad_scaler = torch.GradScaler()
            if self.ddp:
                self.distribution_framework = self.common_config.get(
                    "distribution_framework", "ddp")
                if self.distribution_framework == "fsdp":
                    self.grad_scaler = torch.distributed.fsdp.sharded_grad_scaler\
                        .ShardedGradScaler()

        # setup training parts
        self.loss_list = []
        self.step_duration = 0
        self.metrics = metrics
        # for key, _ in self.metrics.items():
            # self.metrics[key].device = self.device

        if self.ddp:
            find_unused_parameters = self.training_config.get(
                "find_unused_parameters", False)
            if self.distribution_framework == "fsdp":
                for name, param in self.bi_directional_Transformer.named_parameters():
                    if not param.requires_grad:
                        print(name, param.requires_grad)
                self.bi_directional_Transformer_wrapper = FSDP(
                    self.bi_directional_Transformer,
                    device_id=torch.cuda.current_device(),
                    **self.common_config["ddp_wrapper_settings"])
            else:
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
            self.warmup_scheduler = LinearLR(
                self.optimizer, start_factor=0.001, total_iters=total_iters)
            self.total_iters = total_iters
        if "lr_scheduler" in config:
            self.lr_scheduler = dwm.common.create_instance_from_config(
                config["lr_scheduler"], optimizer=self.optimizer)
        else:
            self.lr_scheduler = None

        if resume_from is not None and self.optimizer is not None:
            if os.path.exists(os.path.join(output_path, "optimizers")):
                optimizer_state_dict = torch.load(
                    os.path.join(
                        output_path, "optimizers",
                        "{}.pth".format(resume_from)),
                    map_location="cpu")
                if torch.distributed.is_initialized() \
                        and self.distribution_framework == "fsdp":
                    options = torch.distributed.checkpoint.state_dict\
                        .StateDictOptions(full_state_dict=True, cpu_offload=True)
                    torch.distributed.checkpoint.state_dict\
                        .set_optimizer_state_dict(
                            self.bi_directional_Transformer_wrapper, self.optimizer, optimizer_state_dict,
                            options=options)
                else:
                    self.optimizer.load_state_dict(optimizer_state_dict)

            if os.path.exists(os.path.join(output_path, "schedulers")):
                scheduler_state_dict = torch.load(
                    os.path.join(
                        output_path, "schedulers",
                        "{}.pth".format(resume_from)),
                    map_location="cpu")
                if torch.distributed.is_initialized():
                    options = torch.distributed.checkpoint.state_dict\
                        .StateDictOptions(full_state_dict=True, cpu_offload=True)
                    torch.distributed.checkpoint.state_dict\
                        .set_optimizer_state_dict(
                            self.lr_scheduler, self.optimizer, scheduler_state_dict,
                            options=options)
                else:
                    self.lr_scheduler.load_state_dict(scheduler_state_dict)
        self.output_path = output_path
        # === Load BLANK CODE
        with open(vq_blank_code_path, 'rb') as f:
            blank_code = pickle.load(f)
            self.BLANK_CODE = blank_code
            print("=== Load BLANK CODE: ", blank_code)

    def save_checkpoint(self, output_path: str, steps: int):
        if torch.distributed.is_initialized():
            options = torch.distributed.checkpoint.state_dict.StateDictOptions(
                full_state_dict=True, cpu_offload=True)
            model_state_dict, optimizer_state_dict = torch.distributed\
                .checkpoint.state_dict.get_state_dict(
                    self.bi_directional_Transformer_wrapper, self.optimizer, options=options)
            model_save_dict = {
                "bi_directional_Transformer": model_state_dict,
            }
            if self.lr_scheduler is not None:
                scheduler_state_dict = torch.distributed\
                    .checkpoint.state_dict.get_state_dict(
                        self.lr_scheduler, options=options)
        elif self.should_save:
            model_state_dict = self.bi_directional_Transformer.state_dict()
            optimizer_state_dict = self.optimizer.state_dict()
            model_save_dict = {
                "bi_directional_Transformer": model_state_dict,
            }
            if self.lr_scheduler is not None:
                scheduler_state_dict = self.lr_scheduler.state_dict()

        if self.should_save:
            print(
                f"Save checkpoint to {os.path.join(output_path, 'checkpoints', '{}.pth'.format(steps))}")
            os.makedirs(
                os.path.join(output_path, "checkpoints"), exist_ok=True)
            torch.save(model_save_dict, os.path.join(
                output_path, "checkpoints", "{}.pth".format(steps)))
            os.makedirs(os.path.join(output_path, "optimizers"), exist_ok=True)
            torch.save(optimizer_state_dict, os.path.join(
                output_path, "optimizers", "{}.pth".format(steps)))
            if self.lr_scheduler is not None:
                os.makedirs(os.path.join(
                    output_path, "schedulers"), exist_ok=True)
                torch.save(
                    scheduler_state_dict,
                    os.path.join(output_path, "schedulers", "{}.pth".format(steps)))

        if self.ddp:
            torch.distributed.barrier()

    def log(self, global_step: int, log_steps: int, log_type: str = 'wandb'):
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
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x.clone(), dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)
        return x_masked, mask, ids_restore, ids_keep

    def mask_code_for_generation(self, code, code_indices, mask_ratio=None):
        # code -> 16, 6400, 1024; mask_token -> 1, 1, 1024
        if mask_ratio == None:
            mask_ratio = self.gamma(torch.rand((1,), generator=self.generator))

        # masking: length -> length * mask_ratio
        # x -> 16, 6400*mask_ratio, 1024; mask -> 16, 6400
        # ids_restore -> 16, 6400; ids_keep -> 16, 6400*mask_ratio
        _, mask, ids_restore, ids_keep = self.random_masking(code, mask_ratio)
        # We will modify the code in the model.
        x = code

        # after cat -> 16,6400,1024
        # mask_token -> 1,1,1024; requires_grad
        x_indices = code_indices.clone()
        x_indices = torch.where(
            mask,
            -1,
            x_indices
        )

        return x, x_indices, mask, ids_restore

    def mask_code_for_prediction(self, code, code_indices,  mask_ratio=None, num_reference_frame=None, batch_size=None, num_frames=None):
        # code -> 16, 6400, 1024; mask_token -> 1, 1, 1024
        if mask_ratio == None:
            mask_ratio = self.gamma(torch.rand((1,), generator=self.generator))
        mask_for_prediction = torch.zeros(
            (batch_size, num_frames), device=code.device, dtype=torch.bool)
        mask_for_prediction[:, num_reference_frame:] = 1
        mask_for_prediction = mask_for_prediction.flatten(0, 1)
        # masking: length -> length * mask_ratio
        # x -> 16, 6400*mask_ratio, 1024; mask -> 16, 6400
        # ids_restore -> 16, 6400; ids_keep -> 16, 6400*mask_ratio
        all_frame_mask = torch.zeros(
            (code.shape[0], code.shape[1]), device=code.device, dtype=torch.bool)
        _, mask, ids_restore, ids_keep = self.random_masking(
            code[mask_for_prediction], mask_ratio)
        # We will modify the code in the model.
        x = code
        # after cat -> 16,6400,1024
        # mask_token -> 1,1,1024; requires_grad

        x_indices = code_indices.clone()
        x_indices[mask_for_prediction] = torch.where(
            mask,
            -1,
            x_indices[mask_for_prediction]
        )
        all_frame_mask[mask_for_prediction] = mask
        return x, x_indices, all_frame_mask, ids_restore

    def mutlitask_mask_code(self, code, code_indices, infer=False, task_type=None, num_reference_frame=None, batch_size=None, num_frames=None):
        """
        Description:
            This function is used to mask the code for different tasks.
        Args:
            code: torch.Tensor. the code to be masked
            code_indices: torch.Tensor. the indices of the code
            infer: bool. whether to infer the code
            task_type: str. the type of the task, which can be "generation" or "prediction"
            num_reference_frame: int. the number of the reference frames
            batch_size: int. the batch size
            num_frames: int. the number of the frames
        Returns:
            x: torch.Tensor. the masked code
            x_indices: torch.Tensor. the indices of the masked code
            mask: torch.Tensor. the mask of the code
            ids_restore: torch.Tensor. the restore indices of the code
        """
        if infer:
            if task_type == "generation":
                # generate the empty code for the generation task
                x_future = torch.zeros_like(code).to(code.device).to(
                    self.bi_directional_Transformer.mask_token.dtype)
                x_indices = torch.ones(
                    *x_future.shape[:2]).to(x_future.device) * -1
            elif task_type == "prediction":
                x_future = torch.zeros_like(code).to(code.device).to(
                    self.bi_directional_Transformer.mask_token.dtype)
                x_indices = torch.ones(
                    *x_future.shape[:2]).to(x_future.device) * -1
                # For the future frames, we mask them
                reference_frame_mask = torch.ones(
                    (batch_size, num_frames), device=code.device, dtype=torch.bool)
                reference_frame_mask[:, num_reference_frame:] = 0
                reference_frame_mask = reference_frame_mask.flatten(0, 1)
                # Set the value of the reference frames
                x_future[reference_frame_mask] = code[reference_frame_mask].to(
                    x_future.dtype).clone()
                x_indices[reference_frame_mask] = code_indices[reference_frame_mask].to(
                    x_indices.dtype).clone()
            else:
                raise ValueError(f"Invalid task type: {task_type}")
            return x_future, x_indices, None, None
        else:
            # if task_type is generation, we train the lidar generation task
            if task_type == "generation":
                x, x_indices, mask, ids_restore = self.mask_code_for_generation(
                    code, code_indices)
            # if task_type == 1, we train the lidar prediction task
            elif task_type == "prediction":
                x, x_indices, mask, ids_restore = self.mask_code_for_prediction(
                    code, code_indices, num_reference_frame=num_reference_frame, batch_size=batch_size, num_frames=num_frames)
            else:
                raise ValueError(f"Invalid task type: {task_type}")
            return x, x_indices, mask, ids_restore

    @staticmethod
    def get_maskgit_conditions(
        common_config: dict, batch: dict, device, dtype,
        _3dbox_condition_mask: Optional[torch.Tensor] = None,
        hdmap_condition_mask: Optional[torch.Tensor] = None,
        do_classifier_free_guidance: bool = False
    ):
        condition_embedding_list = []
        # layout condition
        if "3dbox_bev_images" in batch:
            _3dbox_images = batch["3dbox_bev_images"]
            _3dbox_images = _3dbox_images.to(device).flatten(0, 1)
            if _3dbox_condition_mask is not None:
                for i in range(_3dbox_condition_mask.shape[0]):
                    if not _3dbox_condition_mask[i]:
                        _3dbox_images[i] = 0

            if do_classifier_free_guidance:
                _3dbox_images = torch.cat(
                    [torch.zeros_like(_3dbox_images), _3dbox_images])
            condition_embedding_list.append(
                _3dbox_images)

        if "hdmap_bev_images" in batch:
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
                hdmap_images)

        if len(condition_embedding_list) > 0:
            # [batch_size, token_count, embedding_feature_size]
            encoder_hidden_states = torch.cat(condition_embedding_list, 1)\
                .to(dtype=dtype)
            encoder_hidden_states = encoder_hidden_states.flatten(
                2, 3).permute(0, 2, 1)
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
        batch_size, num_frames = len(
            batch["lidar_points"]), len(batch["lidar_points"][0])
        # ====== Data process (GPU)
        points = preprocess_points(batch, self.device)
        self.bi_directional_Transformer_wrapper.train()
        with torch.no_grad():
            voxels = self.vq_point_cloud.voxelizer(points)
            voxels = voxels.flatten(0, 1)
            lidar_feats = self.vq_point_cloud.lidar_encoder(voxels)
            code, _, code_indices = self.vq_point_cloud.vector_quantizer(
                lidar_feats, self.vq_point_cloud.code_age, self.vq_point_cloud.code_usage)

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
            maskgit_conditions = MaskGITPipeline.get_maskgit_conditions(
                self.common_config, batch, self.device,
                torch.float16 if "autocast" in self.common_config else torch.float32,
                _3dbox_condition_mask,
                _hdmap_condition_mask
            )
            # There are two types of tasks: lidar prediction and generation.
            # We randomly choose one of the tasks to train.
            # By default, we train the lidar generation task.
            task_type = (
                torch.rand((1,)) <
                self.training_config.get("task_type_ratio", 0))\
                .to(self.device).to(torch.int)
            task_type = task_type.item()
            task_type = task_type_dict[task_type]
            # refernce frame
            max_reference_frame = min(
                num_frames, self.training_config.get("max_reference_frame", 3))
            num_reference_frame = (
                torch.randint(1, max_reference_frame + 1,
                              (batch_size,), generator=self.generator)
            )
            x, x_indices, mask, ids_restore = self.mutlitask_mask_code(
                code, code_indices, infer=False, task_type=task_type, num_reference_frame=num_reference_frame,
                batch_size=batch_size, num_frames=num_frames)
            # === Model forward/loss
            pred = self.bi_directional_Transformer_wrapper(x, x_indices, context=maskgit_conditions,
                                                           batch_size=batch_size, num_frames=num_frames)
            loss = (
                F.cross_entropy(pred.flatten(0, 1), code_indices.flatten(
                    0, 1), reduction="none", label_smoothing=0.1) * mask.flatten(0, 1)
            ).sum() / (mask.flatten(0, 1).sum() + 1e-5)

        acc = (pred.max(dim=-1)[1] ==
               code_indices)[mask > 0].float().mean().item()

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
            if "max_norm_for_grad_clip" in self.training_config:
                if self.grad_scaler is not None:
                    self.grad_scaler.unscale_(self.optimizer)
                if (
                    torch.distributed.is_initialized() and
                    self.distribution_framework == "fsdp"
                ):
                    self.bi_directional_Transformer_wrapper.clip_grad_norm_(
                        self.training_config["max_norm_for_grad_clip"])
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.bi_directional_Transformer_wrapper.parameters(),
                        self.training_config["max_norm_for_grad_clip"])

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
        xy = (non_zero_indices[:, 2:] * self.vq_point_cloud.voxelizer.step) + \
            self.vq_point_cloud.voxelizer.y_min
        z = (non_zero_indices[:, 1] * self.vq_point_cloud.voxelizer.z_step) + \
            self.vq_point_cloud.voxelizer.z_min
        xyz = torch.cat([xy, z.unsqueeze(1)], dim=1)

        return xyz

    def preview_pipeline(
        self, batch: dict, output_path: str, global_step: int
    ):
        batch_size, num_frames = len(
            batch["lidar_points"]), len(batch["lidar_points"][0])
        results = self.inference_pipeline(batch)
        voxels = results['gt_voxels']
        generated_sample_v = results['pred_voxels']
        vq_voxels = results['vq_voxels']
        if "task_type" in results and num_frames > 1:
            task_type = results["task_type"]
            suffix = f"_{task_type}"
        else:
            suffix = ""

        if self.should_save:
            preview_lidar = make_lidar_preview_tensor(
                voxels.unflatten(0, (batch_size, num_frames)),
                [vq_voxels.unflatten(0, (batch_size, num_frames)),
                 generated_sample_v.unflatten(0, (batch_size, num_frames)),],
                batch, self.inference_config)

            if len(preview_lidar.shape) == 4:
                # change the shape from [num_frames, 3, batch_size, H, W] to [batch_size, 3,  num_frames * H, W]
                preview_lidar = preview_lidar.unflatten(2, (batch_size, -1))
                preview_lidar = preview_lidar.permute(
                    2, 1, 0, 3, 4).flatten(2, 3)
                preview_lidar = [preview_lidar[i]
                    for i in range(preview_lidar.shape[0])]
            else:
                preview_lidar = [preview_lidar]

            if self.should_save:
                os.makedirs(os.path.join(
                    output_path, "preview"), exist_ok=True)
                if len(preview_lidar) == 1:
                    transforms.ToPILImage()(preview_lidar[0]).save(
                        os.path.join(
                            output_path, "preview", "{}{}.png".format(global_step, suffix)))
                    video_frame_list = preview_lidar[0].unflatten(
                        1, (num_frames, -1)).permute(1, 0, 2, 3).detach().cpu()
                    if num_frames > 1:
                        save_tensor_to_video(
                            os.path.join(output_path, "preview", "{}{}.mp4".
                                         format(global_step, suffix)),
                            "libx264", 2, video_frame_list)
                else:
                    for i in range(len(preview_lidar)):
                        transforms.ToPILImage()(preview_lidar[i]).save(
                            os.path.join(
                                output_path, "preview", "{}{}{}.png".format(global_step, suffix, i)))
                        video_frame_list = preview_lidar[0].unflatten(
                            1, (num_frames, -1)).permute(1, 0, 2, 3).detach().cpu()
                        if num_frames > 1:
                            save_tensor_to_video(
                                os.path.join(output_path, "preview", "{}{}{}.mp4".
                                             format(global_step, suffix, i)),
                                "libx264", 2, video_frame_list)

        torch.cuda.empty_cache() # might help to avoid OOM

    def save_results(self, results, batch, batch_size, num_frames):
        suffix = str(self.resume_from) + \
            "_" if hasattr(self, 'resume_from') else ""

        gt_voxels = results['gt_voxels'] if "gt_voxels" in results else None
        pred_voxels = results['pred_voxels'] if "pred_voxels" in results else None

        # Save visualization results
        if self.inference_config.get("save_preview", True):
            preview_lidar = make_lidar_preview_tensor(
                gt_voxels.unflatten(0, (batch_size, -1)),
                pred_voxels.unflatten(0, (batch_size, -1)),
                batch, self.inference_config)
            if preview_lidar.shape[0] != 3:
                preview_lidar = preview_lidar.permute(
                    1, 0, 2, 3).flatten(1, 2)
                paths = [os.path.join(
                    self.output_path, f'pred_voxel_{suffix}preview', batch["sample_data"][0][0]["filename"][0])]
            else:
                paths = [
                    os.path.join(self.output_path,
                                 f'pred_voxel_{suffix}preview', k)
                    for i in batch["sample_data"]
                    for j in i
                    for k in j["filename"] if k.endswith(".bin")
                ]
            preview_lidar_height = preview_lidar.shape[1]
            preview_img_height = preview_lidar_height // len(paths)
            for i in range(len(paths)):
                os.makedirs(os.path.join(self.output_path,
                            f'pred_voxel_{suffix}preview'), exist_ok=True)
                cur_image = preview_lidar[:, i *
                    preview_img_height:(i + 1) * preview_img_height]
                cur_image = transforms.ToPILImage()(cur_image)
                cur_image.save(paths[i].replace('samples/LIDAR_TOP/', '').
                                        replace('/velodyne_points/data/', '_').
                                        replace('.bin', '.png'))
                
                if num_frames > 1:
                    video_frame_list = preview_lidar[:, i *
                        preview_img_height:(i + 1) * preview_img_height]
                    video_frame_list = video_frame_list.unflatten(
                        1, (num_frames, -1)).permute(1, 0, 2, 3).detach().cpu()
                    save_tensor_to_video(
                        paths[i].replace('samples/LIDAR_TOP/', '').
                                 replace('/velodyne_points/data/', '_').
                                 replace('.bin', '.mp4'),
                        "libx264", 2, video_frame_list)

        # save pred voxels
        if self.inference_config.get("save_pred_results", False):
            paths = [
                os.path.join(self.output_path, 'pred_voxel_' + suffix + k)
                for i in batch["sample_data"]
                for j in i
                for k in j["filename"] if k.endswith(".bin")
            ]
            pred_voxel_pc = results['pred_points']
            pred_voxel_pc = postprocess_points(batch, pred_voxel_pc)
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

        # save raw points (the points before voxelization)
        if self.inference_config.get("save_raw_results", False):
            paths = [
                os.path.join(self.output_path, 'raw_' + suffix + k)
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

        # save gt points (the points that are voxelized and then devoxelized)
        if self.inference_config.get("save_gt_results", False):
            paths = [
                os.path.join(self.output_path, 'gt_' + suffix + k)
                for i in batch["sample_data"]
                for j in i
                for k in j["filename"] if k.endswith(".bin")
            ]
            gt_points = results['gt_points']
            gt_points = postprocess_points(batch, gt_points)
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

        # save vq points
        if self.inference_config.get("save_vq_results", False):
            paths = [
                os.path.join(self.output_path, 'vq_' + suffix + k)
                for i in batch["sample_data"]
                for j in i
                for k in j["filename"] if k.endswith(".bin")
            ]
            vq_points = results['vq_points']
            vq_points = postprocess_points(batch, vq_points)
            vq_points = [
                j
                for i in vq_points
                for j in i
            ]
            for path, points in zip(paths, vq_points):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                points = points.numpy()
                padded_points = np.concatenate([
                    points, np.zeros((points.shape[0], 2), dtype=np.float32)
                ], axis=-1)
                with open(path, "wb") as f:
                    f.write(padded_points.tobytes())

    @torch.no_grad()
    def autoregressive_inference_pipeline(self, batch, output_for_eval=False, output_from_ray=False):
        batch_size, num_frames = len(
            batch["lidar_points"]), len(batch["lidar_points"][0])
        points = preprocess_points(batch, self.device)
        use_blank_code = self.inference_config.get("use_blank_code", True)
        use_maskgit = self.inference_config.get(
            "use_maskgit", False) # maskgit style sample
        do_classifier_free_guidance = self.inference_config.get(
            "do_classifier_free_guidance", False)
        if do_classifier_free_guidance:
            guidance_scale = self.inference_config.get("guidance_scale", 3.0)
        num_reference_frame = self.common_config.get("max_reference_frame", 3)
        num_training_frames = self.training_config.get(
            "num_training_frames", 8)
        prediction_steps = math.ceil(
            (num_frames - num_reference_frame) / float(num_training_frames - num_reference_frame))
        self.bi_directional_Transformer_wrapper.eval()
        reference_points = [points[i][:num_reference_frame]
                                for i in range(batch_size)]
        voxels = self.vq_point_cloud.voxelizer(reference_points)
        voxels = voxels.flatten(0, 1)
        lidar_feats = self.vq_point_cloud.lidar_encoder(voxels)
        # Select the first few frames as the reference frames
        if self.inference_config.get("use_ground_truth_as_reference", False):
            reference_code, _, reference_code_indices = self.vq_point_cloud.vector_quantizer(
                lidar_feats, self.vq_point_cloud.code_age, self.vq_point_cloud.code_usage)
            # prepare the code and code_indices for the init step
            code = torch.zeros((batch_size * num_training_frames, *reference_code.shape[1:]),
                    device=reference_code.device, dtype=reference_code.dtype)
            code_indices = torch.ones((batch_size * num_training_frames, *reference_code_indices.shape[1:]),
                    device=reference_code_indices.device, dtype=reference_code_indices.dtype) * -1
            reference_frame_mask = torch.ones((batch_size, num_training_frames),
                                            device=code.device, dtype=torch.bool)
            reference_frame_mask[:, num_reference_frame:] = False
            reference_frame_mask = reference_frame_mask.flatten(0, 1)
            code[reference_frame_mask] = reference_code
            code_indices[reference_frame_mask] = reference_code_indices
        else:
            code = torch.zeros_like(lidar_feats,
                    device=lidar_feats.device, dtype=lidar_feats.dtype)
            code_indices = torch.ones_like(lidar_feats,
                    device=lidar_feats.device, dtype=torch.int64) * -1

        results = {}
        results['pred_points'] = [[] for _ in range(batch_size)]
        results['pred_voxels'] = []
        all_maskgit_conditions = MaskGITPipeline.get_maskgit_conditions(
            self.common_config, batch, self.device,
            torch.float16 if "autocast" in self.common_config else torch.float32,
            None, None, do_classifier_free_guidance=do_classifier_free_guidance
        )
        all_maskgit_conditions["context"] = all_maskgit_conditions["context"].unflatten(
            0, (batch_size, -1))
        for i in range(prediction_steps):
            choice_temperature = 2.0
            start_idx = i * (num_training_frames - num_reference_frame)
            maskgit_conditions = {
                "context": all_maskgit_conditions["context"][:, start_idx: start_idx + num_training_frames].flatten(0, 1),
            }
            # ===Sample task code
            x, x_indices = code.clone(), code_indices.clone()

            code_idx = x_indices.to(dtype=torch.int64).clone()
            num_unknown_code = (code_idx == -1).sum(dim=-1)

            with self.get_autocast_context():
                for t in range(self.T):
                    if do_classifier_free_guidance:
                        x = torch.cat([x] * 2)
                        x_indices = torch.cat([x_indices] * 2)
                    pred = self.bi_directional_Transformer_wrapper(
                        x, x_indices, context=maskgit_conditions, batch_size=batch_size, num_frames=num_training_frames)
                    # Replace predicted value of the blank code with -10000. Therefore, BLANK_CODE will not be selected during the first t steps.
                    if t < 10 and use_blank_code:  # TODO: variable
                        pred[..., self.BLANK_CODE] = -10000
                    if do_classifier_free_guidance:
                        uncond_pred, cond_pred = pred.chunk(2)
                        pred = uncond_pred + guidance_scale * \
                            (cond_pred - uncond_pred)
                    sample_ids = torch.distributions.Categorical(
                        logits=pred).sample()
                    prob = torch.softmax(pred, dim=-1)
                    prob = torch.gather(
                        prob, -1, sample_ids.unsqueeze(-1)).squeeze(-1)

                    # Update the sample_ids and prob
                    # For the reference frames, the sample_ids will not be updated
                    sample_ids[code_idx != -1] = code_idx[code_idx != -1]
                    prob[code_idx != -1] = 1e10

                    ratio = 1.0 * (t + 1) / self.T
                    mask_ratio = self.gamma(ratio)

                    mask_len = num_unknown_code * mask_ratio  # all code len
                    mask_len = torch.minimum(mask_len, num_unknown_code - 1)
                    mask_len = mask_len.clamp(min=1).long()

                    if use_maskgit:
                        confidence = prob.log()
                    else:
                        temperature = choice_temperature * (1.0 - ratio)
                        gumbels = torch.zeros_like(prob).uniform_(0, 1)
                        gumbels = -log(-log(gumbels))
                        confidence = prob.log() + temperature * gumbels

                    cutoff = torch.sort(confidence, dim=-1)[0][
                        torch.arange(mask_len.shape[0], device=mask_len.device), mask_len
                    ].unsqueeze(1)
                    mask = confidence < cutoff
                    x = self.vq_point_cloud.vector_quantizer.get_codebook_entry(
                        sample_ids)
                    code_idx = sample_ids.clone()

                    if t != self.T - 1:
                        mask[reference_frame_mask] = False
                        code_idx[mask] = -1
                        x[mask] = self.bi_directional_Transformer.mask_token.to(x.dtype)
                        x_indices = code_idx.clone()

            _, lidar_voxel = self.vq_point_cloud.lidar_decoder(x)
            generated_sample_v = gumbel_sigmoid(
                lidar_voxel, hard=True, generator=self.generator)
            generated_points_v = voxels2points(self.vq_point_cloud.grid_size,
                                               generated_sample_v.unsqueeze(1))
            crop_idx = 0 if i == 0 else num_reference_frame
            for j in range(batch_size):
                next_points = [pts[0] for pts in generated_points_v[
                    j * num_training_frames:(j+1)*num_training_frames][crop_idx:]]
                results['pred_points'][j] += next_points
            results['pred_voxels'].append(generated_sample_v.unflatten(
                0, (batch_size, -1))[:, crop_idx:])
            # prepare code and code_indices for the next step
            code = torch.zeros_like(code).unflatten(0, (batch_size, -1))
            code_indices = torch.ones_like(code_indices).unflatten(0, (batch_size, -1)) * -1
            x = x.unflatten(0, (batch_size, -1))
            sample_ids = sample_ids.unflatten(0, (batch_size, -1))
            code[:, :num_reference_frame] = x[:, -num_reference_frame:].clone()
            code_indices[:, :num_reference_frame] = sample_ids[:, -num_reference_frame:].clone()
            code = code.flatten(0, 1)
            code_indices = code_indices.flatten(0, 1)

        results['pred_voxels'] = torch.cat(results['pred_voxels'], dim=1).flatten(0, 1)
        results['gt_voxels'] = self.vq_point_cloud.voxelizer(points).flatten(0, 1)

        return results

    def inference_pipeline(self, batch, output_for_eval=False, output_from_ray=False):
        batch_size, num_frames = len(
            batch["lidar_points"]), len(batch["lidar_points"][0])
        points = preprocess_points(batch, self.device)
        use_blank_code = self.inference_config.get("use_blank_code", True)
        use_maskgit = self.inference_config.get(
            "use_maskgit", False) # maskgit style sample
        do_classifier_free_guidance = self.inference_config.get(
            "do_classifier_free_guidance", False)
        if do_classifier_free_guidance:
            guidance_scale = self.inference_config.get("guidance_scale", 3.0)

        if num_frames > 1:
            task_type = (
                torch.rand((1,)) <
                self.training_config.get("task_type_ratio", 0))\
                .to(self.device).to(torch.int)
        else:
            task_type = torch.tensor([0], device=self.device)

        task_type = task_type.item()
        task_type = task_type_dict[task_type]

        num_reference_frame = self.common_config.get("max_reference_frame", 3)
        self.bi_directional_Transformer_wrapper.eval()
        with torch.no_grad():
            voxels = self.vq_point_cloud.voxelizer(points)
            voxels = voxels.flatten(0, 1)
            lidar_feats = self.vq_point_cloud.lidar_encoder(voxels)
            code, _, code_indices = self.vq_point_cloud.vector_quantizer(lidar_feats, self.vq_point_cloud.code_age,
                                                                         self.vq_point_cloud.code_usage)

            # NOTE generation
            choice_temperature = 2.0

            # ===Sample task code
            x, x_indices, mask, _ = self.mutlitask_mask_code(
                code, code_indices, infer=True, task_type=task_type, num_reference_frame=num_reference_frame,
                batch_size=batch_size, num_frames=num_frames)
            if task_type == "generation":
                code_idx = torch.ones(
                    (x.shape[0], x.shape[1]), dtype=torch.int64, device=x.device) * -1
                num_unknown_code = (code_idx == -1).sum(dim=-1)
            elif task_type == "prediction":
                code_idx = x_indices.to(dtype=torch.int64)
                num_unknown_code = (code_idx == -1).sum(dim=-1)
                frame_mask = torch.ones(
                    (batch_size, num_frames), device=code.device, dtype=torch.bool)
                frame_mask[:, :num_reference_frame] = False
                frame_mask = frame_mask.flatten(0, 1)
                # change the value of the frames to be generated to -1
                code_idx[frame_mask] = -1
            else:
                raise ValueError(f"Invalid task type: {task_type}")

            with self.get_autocast_context():
                maskgit_conditions = MaskGITPipeline.get_maskgit_conditions(
                    self.common_config, batch, self.device,
                    torch.float16 if "autocast" in self.common_config else torch.float32,
                    None, None, do_classifier_free_guidance=do_classifier_free_guidance
                )
                for t in range(self.T):
                    if do_classifier_free_guidance:
                        x = torch.cat([x] * 2)
                        x_indices = torch.cat([x_indices] * 2)
                    pred = self.bi_directional_Transformer_wrapper(
                        x, x_indices, context=maskgit_conditions, 
                        batch_size=batch_size, num_frames=num_frames)
                    # Replace predicted value of the blank code with -10000. Therefore, BLANK_CODE will not be selected during the first t steps.
                    if t < 10 and use_blank_code:  # TODO: variable
                        pred[..., self.BLANK_CODE] = -10000
                    if do_classifier_free_guidance:
                        uncond_pred, cond_pred = pred.chunk(2)
                        pred = uncond_pred + guidance_scale * \
                            (cond_pred - uncond_pred)
                    sample_ids = torch.distributions.Categorical(
                        logits=pred).sample()
                    prob = torch.softmax(pred, dim=-1)
                    prob = torch.gather(
                        prob, -1, sample_ids.unsqueeze(-1)).squeeze(-1)

                    sample_ids[code_idx != -1] = code_idx[code_idx != -1]
                    prob[code_idx != -1] = 1e10

                    ratio = 1.0 * (t + 1) / self.T
                    mask_ratio = self.gamma(ratio)

                    mask_len = num_unknown_code * mask_ratio  # all code len
                    mask_len = torch.minimum(mask_len, num_unknown_code - 1)
                    mask_len = mask_len.clamp(min=1).long()

                    if use_maskgit:
                        confidence = prob.log()
                    else:
                        temperature = choice_temperature * (1.0 - ratio)
                        gumbels = torch.zeros_like(prob).uniform_(0, 1)
                        gumbels = -log(-log(gumbels))
                        confidence = prob.log() + temperature * gumbels

                    cutoff = torch.sort(confidence, dim=-1)[0][
                        torch.arange(mask_len.shape[0], device=mask_len.device), mask_len
                    ].unsqueeze(1)
                    mask = confidence < cutoff
                    x = self.vq_point_cloud.vector_quantizer.get_codebook_entry(
                        sample_ids)
                    code_idx = sample_ids.clone()

                    if t != self.T - 1:
                        code_idx[mask] = -1
                        x[mask] = self.bi_directional_Transformer.mask_token.to(
                            x.dtype)
                        x_indices = code_idx.clone()

            _, lidar_voxel = self.vq_point_cloud.lidar_decoder(x)
            _, vq_lidar_voxel = self.vq_point_cloud.lidar_decoder(code)
            generated_sample_v = gumbel_sigmoid(
                lidar_voxel, hard=True, generator=self.generator)
            generated_points_v = voxels2points(self.vq_point_cloud.grid_size,
                                               generated_sample_v.unsqueeze(1))
            vq_sample_v = gumbel_sigmoid(
                vq_lidar_voxel, hard=True, generator=self.generator)
            vq_points_v = voxels2points(self.vq_point_cloud.grid_size,
                                        vq_sample_v.unsqueeze(1))

            if "offsets" in batch:
                offsets = batch["offsets"]
                offsets = sum(offsets, [])
            else:
                offsets = None

        results = {}
        results['raw_points'] = batch["lidar_points"]
        results['gt_points'] = voxels2points(self.vq_point_cloud.grid_size,voxels.unsqueeze(1))
        results['gt_voxels'] = voxels

        results['pred_points'] = generated_points_v
        results['pred_voxels'] = generated_sample_v

        results['vq_points'] = vq_points_v
        results['vq_voxels'] = vq_sample_v

        results['task_type'] = task_type

        return results

    @torch.no_grad()
    def evaluate_pipeline(
        self, global_step: int, dataset_length: int,
        validation_dataloader: torch.utils.data.DataLoader,
        validation_datasampler=None,
        log_type="wandb"
    ):
        for idx, batch in tqdm(enumerate(validation_dataloader)):
            torch.cuda.empty_cache()
            batch_size, num_frames = len(
                batch["lidar_points"]), len(batch["lidar_points"][0])
            if self.ddp:
                torch.distributed.barrier()
            if self.inference_config.get("enable_autoregressive_inference", False):
                results = self.autoregressive_inference_pipeline(batch)
            else:
                results = self.inference_pipeline(batch)
            if self.common_config.get("should_evaluate", True):
                voxels = results['gt_voxels']
                pred_voxels = results['pred_voxels']
                gt_points = results['gt_points']
                pred_points = results['pred_points']
                voxels, pred_voxels = voxels.to(int), pred_voxels.to(int)
                for k in self.metrics:
                    if "chamfer" in k or "mmd" in k or "jsd" in k:
                        self.metrics[k].update(pred_points, gt_points, self.device)
                    elif "iou" in k:
                        self.metrics[k].update(voxels, pred_voxels)
            if self.config.get("save_results", False):
                self.save_results(results, batch, batch_size, num_frames)

        if self.common_config.get("should_evaluate", True):
            for k, metric in self.metrics.items():
                value = metric.compute()
                metric.reset()
                if self.should_save:
                    print("{}: {:.8f}".format(
                        k, value))
                    if log_type == "tensorboard":
                        self.summary.add_scalar(
                            "evaluation/{}".format(k), value, global_step)
                    elif log_type == "wandb" and wandb.run is not None:
                        wandb.log({f"evaluation_{k}": value})
