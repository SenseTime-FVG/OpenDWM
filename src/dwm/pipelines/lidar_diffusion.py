import contextlib
from easydict import EasyDict as edict
from diffusers import FlowMatchEulerDiscreteScheduler
import dwm.common
from dwm.functional import gumbel_sigmoid
from dwm.models.vq_point_cloud import VQPointCloud
from dwm.models.vae_point_cloud import VAEPointCloud
from dwm.models.crossview_temporal_dit import DiTCrossviewTemporalConditionModel
from dwm.utils.lidar import preprocess_points, postprocess_points, voxels2points
from dwm.utils.preview import make_lidar_preview_tensor, save_tensor_to_video
import math
import numpy as np
import os
import re
import safetensors.torch
from tqdm import tqdm
import time
import torch
import torch.cuda.amp
import torch.distributed.checkpoint.state_dict
import torch.distributed.fsdp.sharded_grad_scaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn.functional as F
import torch.utils.tensorboard
from torchvision import transforms
from typing import Optional
import wandb

task_type_dict = {
    0: "generation",
    1: "prediction",
}

class LidarDiffusionPipeline(torch.nn.Module):
    @staticmethod
    def load_state(path: str):
        if path.endswith(".safetensors"):
            state = safetensors.torch.load_file(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        return state

    @staticmethod
    def flow_match_compute_density_for_timestep_sampling(
        weighting_scheme: str, batch_size: int, logit_mean: float = None,
        logit_std: float = None, mode_scale: float = None
    ):
        """Compute the density for sampling the timesteps when doing SD3
        training.

        Courtesy: This was contributed by Rafie Walker in
        https://github.com/huggingface/diffusers/pull/8528.

        SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
        """
        if weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            u = torch.normal(
                mean=logit_mean, std=logit_std, size=(batch_size,),
                device="cpu")
            u = torch.nn.functional.sigmoid(u)
        elif weighting_scheme == "mode":
            u = torch.rand(size=(batch_size,), device="cpu")
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=(batch_size,), device="cpu")

        return u

    @staticmethod
    def flow_match_get_sigmas(noise_scheduler, timestep_indices, n_dim, device, dtype):
        sigmas = noise_scheduler.sigmas[timestep_indices].to(
            device=device, dtype=dtype)
        while len(sigmas.shape) < n_dim:
            sigmas = sigmas.unsqueeze(-1)

        return sigmas

    def __init__(
        self, output_path: str, config: dict,
        device,
        diffusion_model,
        autoencoder,
        train_diffusion_scheduler,
        test_diffusion_scheduler,
        autoencoder_ckpt_path: str = None,
        diffusion_model_ckpt_path: str = None,
        metrics: dict = dict(),
        training_config: dict = dict(),
        inference_config: dict = dict(),
        common_config: dict = dict(),
        resume_from: str = None,
    ):
        r"""
        Args:
            training_config (`dict`):
                training related parameters, e.g. dropout
            inference_config (`dict`):
                inference related parameters, e.g. cfg
            common_config (`dict`):
                config for model, used for both train/val
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

        self.autoencoder_wrapper = self.autoencoder = autoencoder
        self.autoencoder.to(self.device)
        if autoencoder_ckpt_path is not None:
            state_dict = LidarDiffusionPipeline.load_state(
                autoencoder_ckpt_path)
            if isinstance(self.autoencoder, VAEPointCloud):
                if "vae_bev_mm" in state_dict.keys():
                    state_dict = state_dict["vae_bev_mm"]
                elif "lidar_vae" in state_dict.keys():
                    state_dict = state_dict["lidar_vae"]
            elif isinstance(self.autoencoder, VQPointCloud):
                if 'state_dict' in state_dict.keys():
                    state_dict = state_dict['state_dict']
            else:
                raise ValueError(f"Unknown autoencoder: {type(self.autoencoder)}")
            missing_keys, unexpected_keys = self.autoencoder.load_state_dict(
                state_dict, strict=False)
            if missing_keys:
                print("Missing keys in state dict:", missing_keys)
            if unexpected_keys:
                print("Unexpected keys in state dict:", unexpected_keys)
        self.autoencoder.eval()

        if self.common_config.get("disable_condition", False):
            diffusion_model.disable_condition_model()
        self.diffusion_model_wrapper = self.diffusion_model = diffusion_model
        if self.common_config.get("enable_gradient_checkpointing", True):
            self.diffusion_model.enable_gradient_checkpointing()
        self.diffusion_model.to(self.device)

        self.train_diffusion_scheduler = train_diffusion_scheduler
        self.test_diffusion_scheduler = test_diffusion_scheduler

        # no influence, only for test security
        print("Set autoencoder no grad")
        self.autoencoder.requires_grad_(False)

        if resume_from is not None:
            self.resume_from = resume_from
            if self.should_save:
                print(f"===Resume from {resume_from}...")
            model_state_dict = LidarDiffusionPipeline.load_state(
                os.path.join(
                    output_path, "checkpoints",
                    "{}.pth".format(resume_from)))
            self.diffusion_model.load_state_dict(
                model_state_dict["diffusion_model"], strict=False)

        if diffusion_model_ckpt_path is not None:
            if self.should_save:
                print(f"===Load from {diffusion_model_ckpt_path}...")
            self.diffusion_model.load_state_dict(
                LidarDiffusionPipeline.load_state(diffusion_model_ckpt_path)["diffusion_model"])

        self.iter = 0

        if self.ddp:
            self.distribution_framework = self.common_config.get(
                "distribution_framework", "nccl")
        self.grad_scaler = None
        if self.training_config.get("enable_grad_scaler", False):
            self.grad_scaler = torch.GradScaler()
            if self.ddp:
                if self.distribution_framework == "fsdp":
                    self.grad_scaler = torch.distributed.fsdp.sharded_grad_scaler\
                        .ShardedGradScaler()

        # setup training parts
        self.loss_list = []
        self.step_duration = 0
        self.metrics = metrics

        if self.ddp:
            find_unused_parameters = self.training_config.get(
                "find_unused_parameters", False)
            if self.distribution_framework == "fsdp":
                self.diffusion_model_wrapper = FSDP(
                    self.diffusion_model,
                    device_id=torch.cuda.current_device(),
                    **self.common_config["ddp_wrapper_settings"])
            else:
                self.diffusion_model_wrapper = torch.nn.parallel.DistributedDataParallel(
                    self.diffusion_model,
                    device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=find_unused_parameters)

        if self.should_save:
            self.summary = torch.utils.tensorboard.SummaryWriter(
                os.path.join(output_path, "log"))

        # change decay by name
        if len(self.training_config.get('to_skip_decay', [])) > 0:
            to_skip_decay = self.training_config.get('to_skip_decay', [])
            params1, params2 = [], []
            for name, params in self.diffusion_model_wrapper.named_parameters():
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
                params=self.diffusion_model_wrapper.parameters())

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

        if resume_from is not None:
            optimizer_state_dict = torch.load(
                os.path.join(
                    output_path, "optimizers",
                    "{}.pth".format(resume_from)),
                map_location="cpu", weights_only=True)
            if torch.distributed.is_initialized() \
                    and self.distribution_framework == "fsdp":
                options = torch.distributed.checkpoint.state_dict\
                    .StateDictOptions(full_state_dict=True, cpu_offload=True)
                torch.distributed.checkpoint.state_dict\
                    .set_optimizer_state_dict(
                        self.diffusion_model_wrapper, self.optimizer, optimizer_state_dict,
                        options=options)
            else:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None:
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

    def save_checkpoint(self, output_path: str, steps: int):
        if torch.distributed.is_initialized():
            options = torch.distributed.checkpoint.state_dict.StateDictOptions(
                full_state_dict=True, cpu_offload=True)
            model_state_dict, optimizer_state_dict = torch.distributed\
                .checkpoint.state_dict.get_state_dict(
                    self.diffusion_model_wrapper, self.optimizer, options=options)
            model_save_dict = {
                "diffusion_model": model_state_dict,
            }
            if self.lr_scheduler is not None:
                scheduler_state_dict = torch.distributed\
                    .checkpoint.state_dict.get_state_dict(
                        self.lr_scheduler, options=options)
        elif self.should_save:
            model_state_dict = self.diffusion_model.state_dict()
            optimizer_state_dict = self.optimizer.state_dict()
            model_save_dict = {
                "diffusion_model": model_state_dict,
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

    @staticmethod
    def get_diffusion_conditions(
        common_config: dict, batch: dict, device, dtype,
        _3dbox_condition_mask: Optional[torch.Tensor] = None,
        hdmap_condition_mask: Optional[torch.Tensor] = None,
        text_condition_mask: Optional[torch.Tensor] = None,
        do_classifier_free_guidance: bool = False,
        model_type: str = None
    ):
        """
        Description:
            Get maskgit conditions
        Args:
            common_config: common config
            batch: batch data
            device: device
            dtype: dtype
            _3dbox_condition_mask: 3dbox condition mask
            hdmap_condition_mask: hdmap condition mask
            do_classifier_free_guidance: whether to do classifier free guidance. When True, the condition will be concatenated with zero-initialized condition.
        """
        condition_embedding_list = []
        batch_size, num_frames = len(batch["lidar_points"]), len(batch["lidar_points"][0])
        if do_classifier_free_guidance:
            batch_size = batch_size * 2

        # layout condition
        if "3dbox_bev_images" in batch:
            _3dbox_images = batch["3dbox_bev_images"]
            if _3dbox_condition_mask is not None:
                for i in range(_3dbox_condition_mask.shape[0]):
                    if not _3dbox_condition_mask[i]:
                        _3dbox_images[i] = 0
            _3dbox_images = _3dbox_images.to(device).flatten(0, 1)
            if do_classifier_free_guidance:
                _3dbox_images = torch.cat(
                    [torch.zeros_like(_3dbox_images), _3dbox_images])
            condition_embedding_list.append(
                _3dbox_images)
        if "hdmap_bev_images" in batch:
            hdmap_images = batch["hdmap_bev_images"]
            if hdmap_condition_mask is not None:
                for i in range(hdmap_condition_mask.shape[0]):
                    if not hdmap_condition_mask[i]:
                        hdmap_images[i] = 0
            hdmap_images = hdmap_images.to(device).flatten(0, 1)
            if do_classifier_free_guidance:
                hdmap_images = torch.cat(
                    [torch.zeros_like(hdmap_images), hdmap_images])
            condition_embedding_list.append(
                hdmap_images)
        if len(condition_embedding_list) > 0:
            # [batch_size * num_frames, feature_dim, *embedding_feature_size]
            layout_hidden_states = torch.cat(condition_embedding_list, 1)\
                .to(dtype=dtype).unflatten(0, (batch_size, num_frames))
        else:
            layout_hidden_states = None

        result = {
            "condition_image_tensor": layout_hidden_states,
        }
        
        if "text_description_embeddings" in batch:
            result["encoder_hidden_states"] = batch["text_description_embeddings"]
            if text_condition_mask is not None:
                for i in range(text_condition_mask.shape[0]):
                    if not text_condition_mask[i]:
                        result["text_description"][i] = 0
            if do_classifier_free_guidance:
                result["encoder_hidden_states"] = torch.cat(
                    [torch.zeros_like(result["encoder_hidden_states"]), result["encoder_hidden_states"]])
        else:
            text_embeddings_dim = common_config.get("text_embeddings_dim", 1024)
            result["encoder_hidden_states"] = torch.zeros(batch_size, num_frames, 1, text_embeddings_dim)
        result["encoder_hidden_states"] = result["encoder_hidden_states"].to(device)

        if "feature_collect_range" in common_config:
            result["feature_collect_range"] = \
                common_config["feature_collect_range"]
        if model_type == DiTCrossviewTemporalConditionModel:
            pooled_text_embeddings_dim = common_config.get("pooled_text_embeddings_dim", 2048)
            result["pooled_projections"] = torch.zeros(batch_size, num_frames, pooled_text_embeddings_dim)
        return result

    def get_autocast_context(self):
        if "autocast" in self.common_config:
            return torch.autocast(**self.common_config["autocast"])
        else:
            return contextlib.nullcontext()

    def encode_points(self, points):
        with torch.no_grad():
            voxels = self.autoencoder.voxelizer(points)
            voxels = voxels.flatten(0, 1)
            lidar_feats = self.autoencoder.lidar_encoder(voxels)
            if isinstance(self.autoencoder, VAEPointCloud):
                latents, _ = self.autoencoder.variational_model(lidar_feats)
            elif isinstance(self.autoencoder, VQPointCloud):
                latents, _, _ = self.autoencoder.vector_quantizer(
                    lidar_feats, self.autoencoder.code_age, self.autoencoder.code_usage)
                H = W = int(latents.shape[1] ** 0.5)
                latents = latents.view(latents.shape[0], H, W, -1)
            else:
                raise ValueError(
                    f"Unknown autoencoder: {type(self.autoencoder)}")
        return latents, voxels

    def decode_points(self, latents, voxels):
        if isinstance(self.autoencoder, VAEPointCloud):
            return self.autoencoder.decode_lidar(latents, None, voxels, None)
        elif isinstance(self.autoencoder, VQPointCloud):
            latents = latents.flatten(1, 2)
            code, _, code_indices = self.autoencoder.vector_quantizer(
                latents, self.autoencoder.code_age, self.autoencoder.code_usage)
            _, pred_voxel = self.autoencoder.lidar_decoder(code)
            return {
                "pred_voxel": pred_voxel
            }
        else:
            raise ValueError(f"Unknown autoencoder: {type(self.autoencoder)}")

    @staticmethod
    def try_make_input_for_prediction(
        noisy_input: torch.Tensor, latents: torch.Tensor, timesteps: torch.Tensor, 
        training_config: dict, common_config: dict,
        generator: torch.Generator = None, task_type: str = "train"
    ):
        # for the reference frame augmentation
        rf_scale = (
            (
                torch.randn(latents.shape[:2], generator=generator) *
                common_config["reference_frame_scale_std"] + 1
            ).view(*latents.shape[:2], 1, 1, 1, 1).to(latents.device)
            if "reference_frame_scale_std" in common_config else 1
        )
        rf_offset = (
            (
                torch.randn(latents.shape[:2], generator=generator) *
                common_config["reference_frame_offset_std"]
            ).view(*latents.shape[:2], 1, 1, 1, 1).to(latents.device)
            if "reference_frame_offset_std" in common_config else 0
        )        
        batch_size, num_frames = noisy_input.shape[:2]

        if task_type == "train":
            # tasks divided by generation_task_ratio:
            # * Y: video generation tasks divided by image_generation_ratio
            #   * Y: image generation with temporal module disabled
            #   * N: video generation
            # * N: video prediction tasks, temporal module enabled
            generation_task_indicator = \
                    torch.rand((batch_size, 1), generator=generator) < \
                    training_config.get("generation_task_ratio", 0.0)
            disable_temporal = torch.logical_and(
                torch.rand((batch_size, 1), generator=generator) <
                training_config.get("image_generation_ratio", 0.0),
                generation_task_indicator)

            num_reference_frame = common_config.get("reference_frame_count", 3)
            num_reference_frame = torch.where(
                disable_temporal,
                torch.zeros((batch_size,)),
                torch.randint(1, num_reference_frame + 1, (batch_size,), generator=generator)
            ).to(torch.int32)
        elif task_type == "prediction":
            num_reference_frame = common_config.get("reference_frame_count", 3)
            num_reference_frame = torch.ones((batch_size,), dtype=torch.int32) * num_reference_frame
            disable_temporal = torch.zeros((batch_size, 1), dtype=torch.bool)
        elif task_type == "generation":
            num_reference_frame = common_config.get("reference_frame_count", 3)
            num_reference_frame = torch.zeros((batch_size,), dtype=torch.int32)
            disable_temporal = torch.zeros((batch_size, 1), dtype=torch.bool)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        enable_temporal = common_config.get("enable_temporal", False)
        reference_frame_indicator = torch.zeros([batch_size, num_frames], dtype=torch.bool, device=latents.device)
        if enable_temporal:
            for i in range(num_reference_frame.shape[0]):
                reference_frame_indicator[i, :num_reference_frame[i]] = True
            made_noisy_input = torch.where(
                reference_frame_indicator.view(*latents.shape[:2], 1, 1, 1)
                .to(latents.device),
                latents * rf_scale + rf_offset, noisy_input)
            made_timesteps = torch.where(
                reference_frame_indicator.to(timesteps.device),
                torch.zeros_like(timesteps), timesteps)
        else:
            made_noisy_input = noisy_input
            made_timesteps = timesteps
            disable_temporal = None 

        return made_noisy_input, made_timesteps, disable_temporal, \
            reference_frame_indicator


    def train_step(self, batch: dict, global_step: int):
        t0 = time.time()
        self.diffusion_model_wrapper.train()
        batch_size, num_frames = len(
            batch["lidar_points"]), len(batch["lidar_points"][0])
        # Process points data
        points = preprocess_points(batch, self.device)

        # Encode points to latent space
        with torch.no_grad():
            latents, _ = self.encode_points(points)  # (bs * num_frames, h, w, c)
            latents = latents.unflatten(0, (batch_size, num_frames))
            scale = self.common_config.get("latent_scale", 1.0)
            bias = self.common_config.get("latent_bias", 0.0)
            latents = (latents - bias) / scale

        # Sample noise and timesteps
        noise = torch.randn(
            latents.shape, generator=self.generator).to(self.device)
        if isinstance(self.train_diffusion_scheduler, FlowMatchEulerDiscreteScheduler):
            # Default config for train diffusion scheduler
            weighting_scheme = self.training_config.get(
                "diffusion_sche_weighting_scheme", "logit_normal")
            logit_mean = self.training_config.get(
                "diffusion_sche_logit_mean", 0.0)
            logit_std = self.training_config.get(
                "diffusion_sche_logit_std", 1.0)
            mode_scale = self.training_config.get(
                "diffusion_sche_mode_scale", 1.29)
            u = LidarDiffusionPipeline.flow_match_compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme, batch_size=latents.shape[0],
                logit_mean=logit_mean, logit_std=logit_std, mode_scale=mode_scale)
            timestep_indices = (
                u * self.train_diffusion_scheduler.config.num_train_timesteps
            ).long()
            timesteps = self.train_diffusion_scheduler.timesteps[timestep_indices].to(
                self.device)
            # Here we assume the timesteps are the same for all frames in a batch
            timesteps = timesteps.unsqueeze(1).expand(-1, num_frames)
            timestep_indices = timestep_indices.unsqueeze(
                1).expand(-1, num_frames)

            latents, noise, timesteps = latents.flatten(
                0, 1), noise.flatten(0, 1), timesteps.flatten(0, 1)
            timestep_indices = timestep_indices.flatten(0, 1)
            # Add noise according to flow matching.
            sigmas = LidarDiffusionPipeline.flow_match_get_sigmas(
                self.train_diffusion_scheduler, timestep_indices, n_dim=latents.ndim,
                dtype=latents.dtype, device=latents.device)
            noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
            target = latents
        else:
            timesteps = torch.randint(
                0, self.train_diffusion_scheduler.config.num_train_timesteps,
                (batch_size,), generator=self.generator).to(self.device)
            timesteps = timesteps.unsqueeze(1).expand(-1, num_frames)

            latents, noise, timesteps = latents.flatten(
                0, 1), noise.flatten(0, 1), timesteps.flatten(0, 1)
            # Add noise to latents
            noisy_latents = self.train_diffusion_scheduler.add_noise(
                latents, noise, timesteps)
            if self.train_diffusion_scheduler.config.prediction_type == "v_prediction":
                target = self.train_diffusion_scheduler.get_velocity(
                    latents, noise, timesteps)
            elif self.train_diffusion_scheduler.config.prediction_type == "epsilon":
                target = noise
            else:
                raise ValueError(
                    f"Unknown target: {self.training_config['target']}")

        # Get conditions from layout encoder
        # Random condition dropout during training
        _3dbox_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("3dbox_condition_ratio", 1.0))\
            .to(self.device)
        _hdmap_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("hdmap_condition_ratio", 1.0))\
            .to(self.device)
        _text_condition_mask = (
            torch.rand((batch_size,), generator=self.generator) <
            self.training_config.get("text_condition_ratio", 1.0))\
            .to(self.device)

        # Model forward pass
        with self.get_autocast_context():
            if self.common_config["disable_condition"]:
                conditions = {
                    "encoder_hidden_states": None,
                    "condition_image_tensor": None,
                }
            else:
                conditions = LidarDiffusionPipeline.get_diffusion_conditions(
                    self.common_config,
                    batch,
                    self.device,
                    torch.float16 if "autocast" in self.common_config else torch.float32,
                    _3dbox_condition_mask,
                    _hdmap_condition_mask,
                    _text_condition_mask,
                    model_type=type(self.diffusion_model)
                )
            latents = latents.unflatten(0, (batch_size, num_frames)).permute(0, 1, 4, 2, 3)
            noisy_latents = noisy_latents.unflatten(
                0, (batch_size, num_frames)).permute(0, 1, 4, 2, 3)
            target = target.unflatten(0, (batch_size, num_frames)).permute(0, 1, 4, 2, 3)
            timesteps = timesteps.unflatten(0, (batch_size, num_frames))
            made_noisy_input, made_timesteps, disable_temporal, reference_frame_indicator = self.try_make_input_for_prediction(
                noisy_latents, latents, timesteps, self.training_config, self.common_config, self.generator)
            
            if isinstance(self.train_diffusion_scheduler, FlowMatchEulerDiscreteScheduler):
                sigmas = sigmas.unflatten(0, (batch_size, num_frames))
            results = self.diffusion_model_wrapper(
                made_noisy_input.to(torch.float32),
                made_timesteps,
                disable_temporal=disable_temporal,
                return_dict=True,
                **conditions
            )
            noise_pred = results["noise_pred"] * (-sigmas) + made_noisy_input if isinstance(
                self.train_diffusion_scheduler, FlowMatchEulerDiscreteScheduler) else results["noise_pred"]
            intermediate_x_state = {}
            if "intermediate_x" in results:
                intermediate_x_state = {
                    "x_mean": results["intermediate_x"].mean(),
                    "x_var": results["intermediate_x"].var(),
                    "x_max": results["intermediate_x"].max(),
                    "x_min": results["intermediate_x"].min(),
                }
            # Calculate loss
            mse_loss = F.mse_loss(noise_pred, target, reduce=False)
            mse_loss = mse_loss.flatten(0, 1)[~reference_frame_indicator.flatten(0, 1)].mean()

        losses = {
            "mse_loss": mse_loss
        }
        if intermediate_x_state:
            losses.update(intermediate_x_state)
        if self.common_config.get("get_latent_variance", False):
            losses["latent_mean"] = noisy_latents.mean()
            losses["latent_variance"] = noisy_latents.var()
            losses["original_latent_mean"] = latents.mean()
            losses["original_latent_variance"] = latents.var()
        loss = torch.tensor(0.0).to(self.device)
        loss += sum([v for k, v in losses.items() if "loss" in k])


        # Optimize parameters
        should_optimize = \
            ("gradient_accumulation_steps" not in self.config) or \
            ("gradient_accumulation_steps" in self.config and
                (global_step + 1) % self.config["gradient_accumulation_steps"] == 0)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        if should_optimize:
            # Gradient clipping if configured
            if "max_norm_for_grad_clip" in self.training_config:
                if self.grad_scaler is not None:
                    self.grad_scaler.unscale_(self.optimizer)
                if (
                    torch.distributed.is_initialized() and
                    self.distribution_framework == "fsdp"
                ):
                    self.diffusion_model_wrapper.clip_grad_norm_(
                        self.training_config["max_norm_for_grad_clip"])
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.diffusion_model_wrapper.parameters(),
                        self.training_config["max_norm_for_grad_clip"])

            # Optimizer step
            if self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        # Learning rate scheduling
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

    @torch.no_grad()
    def preview_pipeline(
        self, batch: dict, output_path: str,
        global_step: int
    ):
        batch_size, num_frames = len(
            batch["lidar_points"]), len(batch["lidar_points"][0])
        results = self.inference_pipeline(batch)
        gt_voxels = results['gt_voxels']
        pred_voxels = results['pred_voxels']
        vae_voxels = results['vae_voxels']
        if "task_type" in results and num_frames > 1:
            task_type = results["task_type"]
            suffix = f"_{task_type}"
        else:
            suffix = ""

        if self.should_save:
            preview_lidar = make_lidar_preview_tensor(
                gt_voxels.unflatten(0, (batch_size, num_frames)),
                [vae_voxels.unflatten(0, (batch_size, num_frames)),
                 pred_voxels.unflatten(0, (batch_size, num_frames))],
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

    @torch.no_grad()
    def autoregressive_inference_pipeline(self, batch, output_for_eval=False, output_from_ray=False):
        torch.cuda.empty_cache()
        batch_size, num_frames = len(
            batch["lidar_points"]), len(batch["lidar_points"][0])
        points = preprocess_points(batch, self.device)
        do_classifier_free_guidance = "guidance_scale" in self.inference_config and self.inference_config[
            "guidance_scale"] > 0.0
        guidance_scale = self.inference_config.get("guidance_scale", 1)
        num_reference_frame = self.common_config.get("max_reference_frame", 3)
        num_training_frames = self.common_config.get("num_training_frames", 8)
        prediction_steps = math.ceil(
            (num_frames - num_reference_frame) / float(num_training_frames - num_reference_frame))
        self.diffusion_model_wrapper.eval()

        reference_points = [points[i][:num_reference_frame]
                                for i in range(batch_size)]
        with self.get_autocast_context():
            lidar_latents, _ = self.encode_points(reference_points)  # (bs * num_frames, h, w, c)
        # (bs, num_frames, c, h, w)
        lidar_latents = lidar_latents.unflatten(
            0, (batch_size, num_reference_frame)).permute(0, 1, 4, 2, 3)
        conditions = LidarDiffusionPipeline.get_diffusion_conditions(
            self.common_config,
            batch,
            self.device,
            torch.float16 if "autocast" in self.common_config else torch.float32,
            None, None, None,
            do_classifier_free_guidance,
            model_type=type(self.diffusion_model)
        )
        
        reference_latents = torch.zeros(batch_size, num_training_frames, *lidar_latents.shape[-3:]).to(self.device)
        # Select the first few frames as the reference frames
        if self.inference_config.get("use_ground_truth_as_reference", True):
            reference_latents[:, :num_reference_frame] = lidar_latents[:, :num_reference_frame]

        output_dict = {}
        output_dict['pred_points'] = [[] for _ in range(batch_size)]
        output_dict['pred_voxels'] = []
        
        for i in range(prediction_steps):
            self.test_diffusion_scheduler.set_timesteps(
                self.inference_config["inference_steps"], self.device)
            start_idx = i * (num_training_frames - num_reference_frame)
            end_idx = min(start_idx + num_training_frames, num_frames)
            # noisy latents to be used for generation
            latents = torch.randn([batch_size, num_training_frames, 
                *lidar_latents.shape[-3:]], generator=self.generator).to(self.device)
            # context
            cur_conditions = {}
            for key, value in conditions.items():
                if value is not None:
                    cur_conditions[key] = value[:, start_idx: end_idx]
                    residual_frames = num_training_frames - (end_idx - start_idx)
                    if residual_frames > 0:
                        cur_conditions[key] = torch.cat([cur_conditions[key]] + \
                            [value[:, -1:]] * residual_frames, dim=1)
            if do_classifier_free_guidance:
                reference_latents = torch.cat([reference_latents] * 2)
            for t in self.test_diffusion_scheduler.timesteps:
                latents = latents.flatten(0, 1)
                scaled_latents = latents.to(dtype=torch.float32).clone()
                if hasattr(self.test_diffusion_scheduler, "scale_model_input"):
                    scaled_latents = self.test_diffusion_scheduler.scale_model_input(
                        latents, t).to(dtype=torch.float32)

                timestep = t.expand(latents.size(0))
                scaled_latents = scaled_latents.unflatten(
                    0, (batch_size, num_training_frames))
                timestep = timestep.unflatten(0, (batch_size, num_training_frames))

                if do_classifier_free_guidance:
                    scaled_latents = torch.cat([scaled_latents] * 2)
                    timestep = torch.cat([timestep] * 2)
                    
                scaled_latents, timestep, disable_temporal, reference_frame_indicator = self.try_make_input_for_prediction(
                    scaled_latents, reference_latents, timestep, self.training_config, self.common_config, self.generator, 
                    task_type = "generation" if i == 0 and not self.inference_config.get("use_ground_truth_as_reference", True) else "prediction"
                )

                with self.get_autocast_context():
                    # To save memory, we separate the forward pass for conditional and unconditional latents.
                    if self.inference_config.get("separate_forward", False) and do_classifier_free_guidance:
                        uncond_conditions = {}
                        cond_conditions = {}
                        for key, value in cur_conditions.items():
                            if value is not None:
                                uncond_conditions[key] = value[:batch_size]
                                cond_conditions[key] = value[batch_size:]
                        results_uncond = self.diffusion_model_wrapper(
                            scaled_latents[:batch_size],
                            timestep[:batch_size].to(self.device),
                            disable_temporal=disable_temporal[:batch_size],
                            return_dict=True,
                            **uncond_conditions
                        )
                        results_cond = self.diffusion_model_wrapper(
                            scaled_latents[batch_size:],
                            timestep[batch_size:].to(self.device),
                            disable_temporal=disable_temporal[batch_size:],
                            return_dict=True,
                            **cond_conditions
                        )
                        noise_pred = results_uncond["noise_pred"] + guidance_scale * \
                            (results_cond["noise_pred"] -
                                results_uncond["noise_pred"])
                    else:
                        results = self.diffusion_model_wrapper(
                            scaled_latents,
                            timestep.to(self.device),
                            disable_temporal=disable_temporal,
                            return_dict=True,
                            **cur_conditions
                        )
                        noise_pred = results["noise_pred"]
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(
                                2)
                            noise_pred = noise_pred_uncond + guidance_scale * \
                                (noise_pred_cond - noise_pred_uncond)
                noise_pred = noise_pred.flatten(0, 1)
                latents = self.test_diffusion_scheduler.step(
                    noise_pred,
                    t,
                    latents
                ).prev_sample
                latents = latents.to(self.device)
                latents = latents.unflatten(0, (batch_size, num_training_frames))
            # replace the reference frames with the "ground truth"
            latents[:, :num_reference_frame] = reference_latents[:batch_size, :num_reference_frame].clone()
            # reshape it to (bs * num_frames, h, w, c)
            latents = latents.flatten(0, 1).permute(0, 2, 3, 1)
            # shift and scale the latents to the original range
            scale = self.common_config.get("latent_scale", 1.0)
            bias = self.common_config.get("latent_bias", 0.0)
            latents = ((latents * scale) + bias).to(torch.float32)
            with self.get_autocast_context():
                pred_output = self.decode_points(latents, None)
            generated_sample_v = gumbel_sigmoid(
                pred_output["pred_voxel"], hard=True, generator=self.generator)
            generated_points_v = voxels2points(self.autoencoder.grid_size,
                                               generated_sample_v.unflatten(0, (batch_size, -1)))
            crop_idx = 0 if i == 0 else num_reference_frame
            for j in range(batch_size):
                next_points = generated_points_v[j][crop_idx:]
                output_dict['pred_points'][j] += next_points
            output_dict['pred_voxels'].append(generated_sample_v.unflatten(
                0, (batch_size, -1))[:, crop_idx:])
            # prepare code and code_indices for the next step
            latents = latents.unflatten(0, (batch_size, num_training_frames)).permute(0, 1, 4, 2, 3)
            reference_latents = torch.zeros_like(latents)
            reference_latents[:, :num_reference_frame] = latents[:, -num_reference_frame:].clone()

        output_dict['pred_voxels'] = torch.cat(output_dict['pred_voxels'], dim=1)[:, :num_frames].flatten(0, 1)
        output_dict['pred_points'] = [pts[:num_frames] for pts in output_dict['pred_points']]
        output_dict['gt_voxels'] = self.autoencoder.voxelizer(points).flatten(0, 1)
        output_dict['gt_points'] = points

        return output_dict


    @torch.no_grad()
    def inference_pipeline(self, batch, output_for_eval=False, output_from_ray=False):
        batch_size, num_frames = len(
            batch["lidar_points"]), len(batch["lidar_points"][0])
        points = preprocess_points(batch, self.device)
        do_classifier_free_guidance = "guidance_scale" in self.inference_config and self.inference_config[
            "guidance_scale"] > 0.0
        guidance_scale = self.inference_config.get("guidance_scale", 1)
        self.test_diffusion_scheduler.set_timesteps(
            self.inference_config["inference_steps"], self.device)
        self.diffusion_model_wrapper.eval()

        original_latents, voxels = self.encode_points(points)  # (bs * num_frames, h, w, c)
        original_latents = original_latents.unflatten(
            0, (batch_size, num_frames))

        _3dbox_condition_mask, _hdmap_condition_mask, _text_condition_mask = None, None, None
        if self.common_config["disable_condition"] or not do_classifier_free_guidance:
            _3dbox_condition_mask = torch.zeros(
                (batch_size,)).to(bool)
            _hdmap_condition_mask = torch.zeros(
                (batch_size,)).to(bool)
            _text_condition_mask = torch.zeros(
                (batch_size,)).to(bool)
            conditions = {
                "layout_context": None,
                "text_context": None,
            }
        else:
            conditions = LidarDiffusionPipeline.get_diffusion_conditions(
                self.common_config,
                batch,
                self.device,
                torch.float16 if "autocast" in self.common_config else torch.float32,
                _3dbox_condition_mask,  # No dropout during inference
                _hdmap_condition_mask,
                _text_condition_mask,
                do_classifier_free_guidance,
                model_type=type(self.diffusion_model)
            )
        latents = torch.randn(
            original_latents.shape,
            generator=self.generator,
        ).to(self.device).permute(0, 1, 4, 2, 3)
        gt_latents = original_latents.clone().permute(0, 1, 4, 2, 3)
        if do_classifier_free_guidance:
            gt_latents = torch.cat([gt_latents] * 2)
        
        task_type = (
            torch.rand((1,)) < 0.5)\
            .to(self.device).to(torch.int)
        task_type = task_type.item()
        task_type = task_type_dict[task_type]
        
        for t in self.test_diffusion_scheduler.timesteps:
            latents = latents.flatten(0, 1)
            scaled_latents = latents.to(dtype=torch.float32).clone()
            if hasattr(self.test_diffusion_scheduler, "scale_model_input"):
                scaled_latents = self.test_diffusion_scheduler.scale_model_input(
                    latents, t).to(dtype=torch.float32)

            timestep = t.expand(latents.size(0))
            scaled_latents = scaled_latents.unflatten(
                0, (batch_size, num_frames))
            timestep = timestep.unflatten(0, (batch_size, num_frames))

            if do_classifier_free_guidance:
                scaled_latents = torch.cat([scaled_latents] * 2)
                timestep = torch.cat([timestep] * 2)

            scaled_latents, timestep, disable_temporal, reference_frame_indicator = self.try_make_input_for_prediction(
                scaled_latents, gt_latents, timestep, self.training_config, self.common_config, self.generator, task_type = task_type)

            with self.get_autocast_context():
                # To save memory, we separate the forward pass for conditional and unconditional latents.
                if self.inference_config.get("separate_forward", False) and do_classifier_free_guidance:
                    uncond_conditions = {}
                    cond_conditions = {}
                    for key, value in conditions.items():
                        if value is not None:
                            uncond_conditions[key] = value[:batch_size]
                            cond_conditions[key] = value[batch_size:]
                    results_uncond = self.diffusion_model_wrapper(
                        scaled_latents[:batch_size],
                        timestep[:batch_size].to(self.device),
                        disable_temporal=disable_temporal[:batch_size] if disable_temporal is not None else None,
                        return_dict=True,
                        **uncond_conditions
                    )
                    results_cond = self.diffusion_model_wrapper(
                        scaled_latents[batch_size:],
                        timestep[batch_size:].to(self.device),
                        disable_temporal=disable_temporal[batch_size:] if disable_temporal is not None else None,
                        return_dict=True,
                        **cond_conditions
                    )
                    noise_pred = results_uncond["noise_pred"] + guidance_scale * \
                        (results_cond["noise_pred"] - results_uncond["noise_pred"])
                else:
                    results = self.diffusion_model_wrapper(
                        scaled_latents,
                        timestep.to(self.device),
                        disable_temporal=disable_temporal,
                        return_dict=True,
                        **conditions
                    )
                    noise_pred = results["noise_pred"]
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(
                            2)
                        noise_pred = noise_pred_uncond + guidance_scale * \
                            (noise_pred_cond - noise_pred_uncond)
            noise_pred = noise_pred.flatten(0, 1)
            latents = self.test_diffusion_scheduler.step(
                noise_pred,
                t,
                latents
            ).prev_sample
            latents = latents.to(self.device)
            latents = latents.unflatten(0, (batch_size, num_frames))
        
        
        # reshape it to (bs * num_frames, h, w, c)
        latents = latents.flatten(0, 1).permute(0, 2, 3, 1)
        original_latents = original_latents.flatten(0, 1).to(torch.float32)
        # shift and scale the latents to the original range
        scale = self.common_config.get("latent_scale", 1.0)
        bias = self.common_config.get("latent_bias", 0.0)
        latents = ((latents * scale) + bias).to(torch.float32)
        
        if task_type == "prediction":
            reference_frame_indicator = reference_frame_indicator[:batch_size].flatten(0, 1) # (bs * num_frames)
            latents[reference_frame_indicator] = original_latents[reference_frame_indicator].clone()
        pred_output = self.decode_points(latents, voxels)
        vae_output = self.decode_points(original_latents, voxels)
        # Convert to binary voxels using gumbel sigmoid
        generated_sample_v = gumbel_sigmoid(
            pred_output["pred_voxel"], hard=True, generator=self.generator)
        vae_generated_sample_v = gumbel_sigmoid(
            vae_output["pred_voxel"], hard=True, generator=self.generator)
        generated_points_v = voxels2points(
            self.autoencoder.grid_size, generated_sample_v.unflatten(0, (batch_size, -1)))
        vae_generated_points_v = voxels2points(
            self.autoencoder.grid_size, vae_generated_sample_v.unflatten(0, (batch_size, -1)))
        gt_points_v = voxels2points(
            self.autoencoder.grid_size, voxels.unflatten(0, (batch_size, -1)))
    
        results = {}
        results['gt_voxels'] = voxels
        results['pred_voxels'] = generated_sample_v
        results['vae_voxels'] = vae_generated_sample_v

        results['gt_points'] = gt_points_v
        results['pred_points'] = generated_points_v
        results['vae_points'] = vae_generated_points_v

        results['pred_latents'] = latents
        results['task_type'] = task_type

        return results

    def save_results(self, results, batch, batch_size, num_frames):
        suffix = str(self.resume_from) + \
            "_" if hasattr(self, 'resume_from') else ""

        gt_voxels = results['gt_voxels'] if "gt_voxels" in results else None
        pred_voxels = results['pred_voxels'] if "pred_voxels" in results else None

        # Save visualization results
        if self.inference_config.get("save_preview", True):
            num_frames = gt_voxels.shape[0] // batch_size
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
                cur_image.save(paths[i].replace(
                    'samples/LIDAR_TOP/', '').replace('.bin', '.png'))
                if num_frames > 0:
                    video_frame_list = preview_lidar[:, i *
                        preview_img_height:(i + 1) * preview_img_height]
                    video_frame_list = video_frame_list.unflatten(
                        1, (num_frames, -1)).permute(1, 0, 2, 3).detach().cpu()
                    save_tensor_to_video(
                        paths[i].replace('samples/LIDAR_TOP/', '').replace('.bin', '.mp4'),
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
        if self.inference_config.get("save_vae_results", False):
            paths = [
                os.path.join(self.output_path, 'vae_' + suffix + k)
                for i in batch["sample_data"]
                for j in i
                for k in j["filename"] if k.endswith(".bin")
            ]
            vae_points = results['vae_points']
            vae_points = postprocess_points(batch, vae_points)
            vae_points = [
                j
                for i in vae_points
                for j in i
            ]
            for path, points in zip(paths, vae_points):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                points = points.numpy()
                padded_points = np.concatenate([
                    points, np.zeros((points.shape[0], 2), dtype=np.float32)
                ], axis=-1)
                with open(path, "wb") as f:
                    f.write(padded_points.tobytes())

    @torch.no_grad()
    def evaluate_pipeline(
        self, global_step: int, dataset_length: int,
        validation_dataloader: torch.utils.data.DataLoader,
        validation_datasampler=None,
        log_type="wandb"
    ):
        self.diffusion_model_wrapper.eval()
        for idx, batch in tqdm(enumerate(validation_dataloader)):
            if self.ddp:
                torch.distributed.barrier()
            batch_size, num_frames = len(
                batch["lidar_points"]), len(batch["lidar_points"][0])
            if self.inference_config.get("enable_autoregressive_inference", False):
                results = self.autoregressive_inference_pipeline(batch)
            else:
                results = self.inference_pipeline(batch)

            if self.common_config.get("should_evaluate", True):
                gt_voxels = results['gt_voxels']
                pred_voxels = results['pred_voxels']
                gt_pts = results['gt_points']
                pred_pts = results['pred_points']
                gt_voxels, pred_voxels = gt_voxels.to(int), pred_voxels.to(int)
                for k in self.metrics:
                    if "chamfer" in k or "mmd" in k or "jsd" in k:
                        self.metrics[k].update(gt_pts, pred_pts, self.device)
                    elif "iou" in k:
                        self.metrics[k].update(gt_voxels, pred_voxels)
                if idx % 10 == 0:
                    for k, metric in self.metrics.items():
                        value = metric.compute()
                        if self.should_save:
                            print("Index {}: {}: {:.3f}".format(
                                idx, k, value))
            if self.config.get("save_results", False):
                self.save_results(results, batch, batch_size, num_frames)

        for k, metric in self.metrics.items():
            value = metric.compute()
            metric.reset()
            if self.should_save:
                print("{}: {:.3f}".format(
                    k, value))
                if log_type == "tensorboard":
                    self.summary.add_scalar(
                        "evaluation/{}".format(k), value, global_step)
                elif log_type == "wandb" and wandb.run is not None:
                    wandb.log({f"evaluation_{k}": value})
