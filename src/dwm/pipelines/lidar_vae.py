import os
import time
from PIL import Image
import contextlib
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional
import torch.utils.tensorboard
import safetensors.torch
import dwm.common
import dwm.functional
from typing import Optional
import wandb
import itertools
from dwm.utils.lidar import preprocess_points
import pdb

class Clamp(nn.Module):
    def __init__(self, min_val, max_val):
        super(Clamp, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)

class LidarVAE():
    @staticmethod
    def load_state(path: str):
        if path.endswith(".safetensors"):
            state = safetensors.torch.load_file(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu")
        return state

    @staticmethod
    def stable_BCE_loss_with_logits(pred, target):
        neg_abs = - pred.abs()
        loss = pred.clamp(min=0) - pred * target + \
            (1 + neg_abs.exp() + 1e-7).log()

        return loss.mean()

    def __init__(self,
                 output_path: str,
                 config: dict,
                 common_config: dict,
                 training_config: dict,
                 inference_config: dict,
                 device: str,
                 lidar_vae: nn.Module,
                 lidar_vae_checkpoint_path: Optional[str] = None,
                 metrics: dict = {},
                 shift_factor: int = 0,
                 resume_from = None):
        self.should_save = not torch.distributed.is_initialized() or \
            torch.distributed.get_rank() == 0
        self.output_path = output_path
        self.config = config
        self.common_config = common_config
        self.training_config = training_config
        self.inference_config = inference_config
        self.device = device
        self.ddp = torch.distributed.is_initialized()
        self.metrics = metrics

        self.lidar_vae = lidar_vae
        self.lidar_vae_checkpoint_path = lidar_vae_checkpoint_path
        self.lidar_vae_wrapper = self.lidar_vae = lidar_vae
        self.lidar_vae.to(self.device)

        self.grad_scaler = torch.GradScaler() \
            if self.training_config.get("enable_grad_scaler", False) else None
        if torch.distributed.is_initialized():
            self.lidar_vae_wrapper = nn.parallel.DistributedDataParallel(
                self.lidar_vae,
                device_ids=[int(os.environ["LOCAL_RANK"])],
                **self.common_config.get("ddp_wrapper_settings", {}),
                broadcast_buffers=False)
        self.summary = torch.utils.tensorboard.SummaryWriter(
            os.path.join(output_path, "log"))
        # load model state
        if resume_from is not None:
            model_state_dict = LidarVAE.load_state(
                os.path.join(
                    output_path, "checkpoints", "{}.pth".format(resume_from)))
            self.lidar_vae.load_state_dict(model_state_dict["lidar_vae"])
        elif lidar_vae_checkpoint_path is not None:
            # Load the state dict
            model_state_dict = LidarVAE.load_state(
                lidar_vae_checkpoint_path)
            
            # Create a new dictionary with renamed keys instead of modifying during iteration
            if "vae_bev_mm" in model_state_dict:
                new_state_dict = {}
                for name, param in model_state_dict["vae_bev_mm"].items():
                    if "bev_decoder" in name:
                        new_name = name.replace("bev_decoder", "lidar_decoder")
                        new_state_dict[new_name] = param
                    else:
                        new_state_dict[name] = param
                
                # Replace the original dictionary with the new one
                model_state_dict["vae_bev_mm"] = new_state_dict
                
            self.lidar_vae.load_state_dict(model_state_dict["vae_bev_mm"])

        # optimizer
        training_params = [{'params': self.lidar_vae_wrapper.parameters()}]
        self.optimizer = dwm.common.create_instance_from_config(
            config["optimizer"],
            params=training_params)

        # scheduler
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
            optimizer_state_path = os.path.join(
                output_path, "optimizer", "{}.pth".format(resume_from))
            optimizer_state_dict = torch.load(
                optimizer_state_path, map_location="cpu")
            self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None:
                scheduler_state_path = os.path.join(
                    output_path, "scheduler", "{}.pth".format(resume_from))
                scheduler_state_dict = torch.load(
                    scheduler_state_path, map_location="cpu")
                self.lr_scheduler.load_state_dict(scheduler_state_dict)

        self.shift_factor = shift_factor
        # setup training parts
        self.loss_list = []
        self.step_duration = 0
        self.iter = 0

    def get_loss_coef(self, name):
        loss_coef = 1
        if "loss_coef_dict" in self.training_config:
            loss_coef = self.training_config["loss_coef_dict"].get(name, 0)
        return loss_coef

    def get_autocast_context(self):
        if "autocast" in self.common_config:
            return torch.autocast(**self.common_config["autocast"])
        else:
            return contextlib.nullcontext()

    def save_checkpoint(self, output_path: str, steps: int):
        lidar_vae_model_state_dict = self.lidar_vae.state_dict()
        model_state_dict = {
            "lidar_vae": lidar_vae_model_state_dict,
        }
        optimizer_state_dict = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            scheduler_state_dict = self.lr_scheduler.state_dict()

        if self.should_save:
            model_root = os.path.join(output_path, "checkpoints")
            os.makedirs(model_root, exist_ok=True)
            torch.save(
                model_state_dict,
                os.path.join(model_root, "{}.pth".format(steps)))

            optimizer_root = os.path.join(output_path, "optimizers")
            os.makedirs(optimizer_root, exist_ok=True)
            torch.save(
                optimizer_state_dict,
                os.path.join(optimizer_root, "{}.pth".format(steps)))
            if self.lr_scheduler is not None:
                scheduler_root = os.path.join(output_path, "scheduler")
                os.makedirs(scheduler_root, exist_ok=True)
                torch.save(
                    scheduler_state_dict,
                    os.path.join(scheduler_root, "{}.pth".format(steps)))

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def log(self, global_step: int, log_steps: int, log_type: str = "wandb"):
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

    def train_step(self, batch: dict, global_step: int):
        t0 = time.time()
        self.lidar_vae_wrapper.train()
        torch.cuda.empty_cache()
        points = preprocess_points(batch, self.device)
        points = list(itertools.chain.from_iterable(points))
        points = [[p] for p in points]
        with self.get_autocast_context():
            ray_cast_center = self.common_config.get(
                "ray_cast_center", None)
            if ray_cast_center is not None:
                batch_size = len(batch["lidar_points"])
                ray_cast_center = torch.tensor([ray_cast_center])\
                    .repeat(batch_size, 1)
            results = self.lidar_vae_wrapper(
                points, depth_ray_cast_center=ray_cast_center)
            losses = {}
            # The following losses include depth_loss (and maybe kl_loss or emb_loss)
            for k, v in results.items():
                if k.endswith("_loss") and v is not None:
                    losses[k] = (v.sum() if torch.is_tensor(
                        v) else sum(v)) * self.get_loss_coef(k)

            losses["pred_voxel_loss"] = self.get_loss_coef("pred_voxel_loss") * \
                LidarVAE.stable_BCE_loss_with_logits(results['pred_voxel'].float(), results['voxels'])
        # backpropagation
        loss = sum(losses.values()) / \
            self.training_config.get("gradient_accumulation_steps", 1)
        if torch.isnan(loss):
            print("loss is nan")
            loss = loss * 0.
            for k, v in losses.items():
                losses[k] = losses[k] * 0. if torch.isnan(v).any() else v
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        should_optimize = \
            ("gradient_accumulation_steps" not in self.training_config) or \
            ("gradient_accumulation_steps" in self.training_config and
                (global_step + 1) %
             self.training_config["gradient_accumulation_steps"] == 0)
        if should_optimize:
            if "max_norm_for_grad_clip" in self.training_config:
                if self.grad_scaler is not None:
                    self.grad_scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.lidar_vae_wrapper.parameters(),
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

    @torch.no_grad()
    def preview_pipeline(
        self, batch: dict, output_path: str, global_step: int
    ):
        self.lidar_vae_wrapper.eval()
        points = preprocess_points(batch, self.device)
        points = list(itertools.chain.from_iterable(points))
        points = [[p] for p in points]
        with self.get_autocast_context():
            ray_cast_center = self.common_config.get("ray_cast_center", None)
            if ray_cast_center is not None:
                batch_size = len(batch["lidar_points"])
                ray_cast_center = torch.tensor([ray_cast_center])\
                    .repeat(batch_size, 1)
            results = self.lidar_vae_wrapper(
                points, depth_ray_cast_center=ray_cast_center)
            if "lidar_rec" in results and results["lidar_rec"] is not None:
                results["lidar_rec"] = self.lidar_vae.voxelizer(
                    [[_] for _ in results["lidar_rec"]])[:, 0]
            for k, v in results.items():
                if isinstance(v, torch.Tensor):
                    results[k] = v.detach().cpu()
        # LiDAR preview
        # columns: GT, ray reconstruction
        if ("lidar_rec" in results and results["lidar_rec"] is not None) or \
            ("pred_voxel" in results and results["pred_voxel"] is not None):
            preview_lidar_size = self.inference_config["preview_lidar_img_size"]
            preview_lidar = Image.new(
                "L", (
                    3 * preview_lidar_size[0],
                    len(batch["lidar_points"]) * preview_lidar_size[1]
                ))
            if not "lidar_rec" in results or results["lidar_rec"] is None:
                results["lidar_rec"] = torch.zeros_like(results["voxels"])
            if not "pred_voxel" in results or results["pred_voxel"] is None:
                results["pred_voxel"] = torch.zeros_like(results["voxels"])
            for i in range(len(points)):
                images = [
                    torchvision.transforms.functional
                    .to_pil_image(torch.amax(results["voxels"][i], 0))
                    .resize(preview_lidar_size),
                    torchvision.transforms.functional
                    .to_pil_image(torch.amax(results["lidar_rec"][i], 0))
                    .resize(preview_lidar_size),
                    torchvision.transforms.functional
                    .to_pil_image(torch.amax(dwm.functional.gumbel_sigmoid(results["pred_voxel"][i]), 0))
                ]
                for j, image in enumerate(images):
                    preview_lidar.paste(
                        image, (j * preview_lidar_size[0], i * preview_lidar_size[1]))
            if self.should_save:
                os.makedirs(os.path.join(
                    output_path, "preview"), exist_ok=True)
                preview_lidar.save(
                    os.path.join(
                        output_path, "preview", "{}_lidar.png".format(global_step)))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate_pipeline(
        self, global_step: int, dataset_length: int,
        validation_dataloader: torch.utils.data.DataLoader,
        validation_datasampler=None, log_type: str = "wandb"
    ):
        # NOTE
        # the batch size of evaluation should be same with training
        if self.should_save:
            print(f"eval {len(validation_dataloader)} samples")
        if self.ddp:
            validation_datasampler.set_epoch(0)
        for (idx, batch) in enumerate(validation_dataloader):
            self.lidar_vae_wrapper.eval()
            torch.cuda.empty_cache()
            if self.ddp:
                torch.distributed.barrier()
            points = preprocess_points(batch, self.device)
            points = list(itertools.chain.from_iterable(points))
            points = [[p] for p in points]
            with self.get_autocast_context():
                ray_cast_center = self.common_config.get(
                    "ray_cast_center", None)
                if ray_cast_center is not None:
                    batch_size = len(batch["lidar_points"])
                    ray_cast_center = torch.tensor([ray_cast_center])\
                        .repeat(batch_size, 1)
                results = self.lidar_vae_wrapper(
                    points, depth_ray_cast_center=ray_cast_center)

            pred_voxel = dwm.functional.gumbel_sigmoid(
                results['pred_voxel'], hard=True) >= 0.5
            gt_voxel = results['voxels'] >= 0.5

        if self.should_save:
            print("Step {}:".format(global_step))
        if self.common_config.get("should_evaluate", True):
            for k in self.metrics:
                self.metrics[k].update(gt_voxel, pred_voxel)
        for k, metric in self.metrics.items():
            value = metric.compute()
            metric.reset()
            if self.should_save:
                print("{}: {:.3f}, count: {}".format(
                    k, value, metric.num_samples))
                if log_type == "tensorboard":
                    self.summary.add_scalar(
                        "evaluation/{}".format(k), value, global_step)
                elif log_type == "wandb" and wandb.run is not None:
                    wandb.log({f"evaluation_{k}": value})
