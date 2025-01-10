import os
import time
from PIL import Image
import contextlib
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms.functional
import torch.utils.tensorboard
import safetensors.torch
import dwm.common
import dwm.functional
from taming.modules.discriminator.model import weights_init
import functools
from taming.modules.util import ActNorm
from taming.modules.losses.lpips import LPIPS
from taming.modules.losses.vqperceptual import hinge_d_loss
from typing import Optional
import wandb
import numpy as np
import torchmetrics
from collections import defaultdict
import itertools
from metrics.general_metrics import CustomMeanMetrics


def save_dict_items(dictionary, save_dir="/mnt/storage/user/liuyichen/repo/DWM/scripts/lyc_noupload/test_data"):
    import numpy as np
    """
    Save all items in a dictionary to files, converting PyTorch tensors to NumPy arrays.
    Handles lists of tensors with varying lengths by saving each tensor separately.

    Args:
    dictionary (dict): The dictionary containing items to save.
    save_dir (str): The directory to save the items in.
    """
    os.makedirs(save_dir, exist_ok=True)

    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            file_path = os.path.join(save_dir, f"{key}.npy")
            np.save(file_path, value.detach().cpu().numpy())

        elif isinstance(value, list):
            if all(isinstance(item, torch.Tensor) for item in value):
                item_dir = os.path.join(save_dir, key)
                os.makedirs(item_dir, exist_ok=True)
                for i, tensor in enumerate(value):
                    file_path = os.path.join(item_dir, f"{i}.npy")
                    np.save(file_path, tensor.detach().cpu().numpy())

            elif all(isinstance(item, list) and all(isinstance(subitem, torch.Tensor) for subitem in item) for item in value):
                item_dir = os.path.join(save_dir, key)
                os.makedirs(item_dir, exist_ok=True)
                for i, sublist in enumerate(value):
                    sublist_dir = os.path.join(item_dir, str(i))
                    os.makedirs(sublist_dir, exist_ok=True)
                    for j, tensor in enumerate(sublist):
                        file_path = os.path.join(sublist_dir, f"{j}.npy")
                        np.save(file_path, tensor.detach().cpu().numpy())

            else:
                print(f"Warning: Skipping {key}, not a supported type.")

        else:
            print(f"Warning: Skipping {key}, not a supported type.")

    print(f"All supported items saved to {save_dir}")


def load_dict_items(load_dir="/mnt/storage/user/liuyichen/repo/DWM/scripts/lyc_noupload/test_data"):
    import numpy as np
    """
    Load items from files, converting NumPy arrays to PyTorch tensors.
    Handles nested directory structures created by save_dict_items.

    Args:
    load_dir (str): The directory to load the items from.

    Returns:
    dict: A dictionary containing the loaded items.
    """
    loaded_dict = {}

    for item in os.listdir(load_dir):
        item_path = os.path.join(load_dir, item)

        if item.endswith('.npy'):
            # Single tensor
            key = item[:-4]  # Remove .npy extension
            loaded_dict[key] = torch.from_numpy(np.load(item_path))

        elif os.path.isdir(item_path):
            # List of tensors or list of lists of tensors
            sub_items = os.listdir(item_path)
            if any(subitem.endswith('.npy') for subitem in sub_items):
                # List of tensors
                tensor_list = []
                for i in range(len(sub_items)):
                    file_path = os.path.join(item_path, f"{i}.npy")
                    if os.path.exists(file_path):
                        tensor_list.append(
                            torch.from_numpy(np.load(file_path)))
                loaded_dict[item] = tensor_list
            else:
                # List of lists of tensors
                nested_tensor_list = []
                for subdir in sorted(sub_items):
                    subdir_path = os.path.join(item_path, subdir)
                    if os.path.isdir(subdir_path):
                        sub_tensor_list = []
                        for j in range(len(os.listdir(subdir_path))):
                            file_path = os.path.join(subdir_path, f"{j}.npy")
                            if os.path.exists(file_path):
                                sub_tensor_list.append(
                                    torch.from_numpy(np.load(file_path)))
                        nested_tensor_list.append(sub_tensor_list)
                loaded_dict[item] = nested_tensor_list

    print(f"All items loaded from {load_dir}")
    return loaded_dict

# Example usage:
# loaded_dict = load_dict_items("/path/to/load/directory")


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


class Clamp(nn.Module):
    def __init__(self, min_val, max_val):
        super(Clamp, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)


class BevwImageDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix. Modified from the taming.modules.discriminator.model.NLayerDiscriminator
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False, eps=1e-4):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(BevwImageDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult, eps=eps),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult, eps=eps),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class BEVWorldVAE():
    @staticmethod
    def load_state(path: str):
        if path.endswith(".safetensors"):
            state = safetensors.torch.load_file(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu")
        return state

    @staticmethod
    def preprocess_points(batch, common_config, device):
        if common_config.get("point_space", "ego") == "ego":
            return [
                [
                    (dwm.functional.make_homogeneous_vector(
                        p_j.to(device)) @ t_j.permute(1, 0))[:, :3]
                    for p_j, t_j in zip(p_i, t_i.flatten(0, 1))
                ]
                for p_i, t_i in zip(
                    batch["lidar_points"],
                    batch["lidar_transforms"].to(device))
            ]
        else:
            return [[j.to(device) for j in i] for i in batch["lidar_points"]]

    @staticmethod
    def postprocess_points(batch, ego_space_points, common_config):
        if common_config.get("point_space", "ego") == "ego":
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
        else:
            ego_space_points

    @staticmethod
    def stable_BCE_loss_with_logits(pred, target, keep_balance=False):
        # if keep_balance:
        #     pos_pred = pred[target == 1.0]
        #     neg_pred = pred[~(target == 1.0)]
        #     positive_weight = neg_pred.shape[0] / pos_pred.shape[0]
        #     pos_neg_abs = -pos_pred.abs()
        #     pos_loss = positive_weight * (pos_pred.clamp(min=0) - pos_pred * torch.ones_like(pos_pred) + (1 + pos_neg_abs.exp() + 1e-7).log())
        #     neg_neg_abs = -neg_pred.abs()
        #     neg_loss = (neg_pred.clamp(min=0) - neg_pred * torch.ones_like(neg_pred) + (1 + neg_neg_abs.exp() + 1e-7).log())
        #     loss = torch.concat([pos_loss, neg_loss])
        # else:
        neg_abs = - pred.abs()
        loss = pred.clamp(min=0) - pred * target + \
            (1 + neg_abs.exp() + 1e-7).log()
        if keep_balance:
            loss[target == 1.0] = loss[target == 1.0] * 100
        return loss.mean()

    @staticmethod
    def voxels2points(grid_size, voxels):
        interval = torch.tensor([grid_size["interval"]])
        min = torch.tensor([grid_size["min"]])
        return [
            [
                torch.nonzero(v_j).flip(-1).cpu() * interval + min
                for v_j in v_i
            ]
            for v_i in voxels
        ]

    def __init__(self,
                 output_path: str,
                 config: dict,
                 common_config: dict,
                 training_config: dict,
                 inference_config: dict,
                 device: str,
                 vae_bev_mm: nn.Module,
                 vae_be_mm_checkpoint_path: Optional[str] = None,
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
        # image precessor
        # self.image_processor = diffusers.image_processor.VaeImageProcessor(do_resize = False)

        self.vae_bev_mm = vae_bev_mm
        self.vae_be_mm_checkpoint_path = vae_be_mm_checkpoint_path
        self.vae_bev_mm_wrapper = self.vae_bev_mm = vae_bev_mm
        self.vae_bev_mm.to(self.device)

        # lpips for perceptual loss
        self.lpips_preprocess = lambda x: x * 2. - 1.
        self.lpips_wrapper = self.lpips = LPIPS().to(device)
        self.lpips.eval()

        # discriminator for gan loss
        if training_config.get("enable_discriminator", False):
            self.discriminator_wrapper = self.discriminator = BevwImageDiscriminator(
                **training_config["loss"]["discriminator_config"]).apply(weights_init)
            self.discriminator.to(device)
            self.discriminator.train()
            self.disc_factor = training_config["loss"]["disc_factor"]
            self.discriminator_iter_start = training_config["loss"]["discriminator_iter_start"]
        if "rgb_loss_type" in training_config["loss"]:
            self.rgb_loss_type = training_config["loss"]["rgb_loss_type"]
            if self.rgb_loss_type == "l1":
                self.rgb_loss = F.l1_loss
            elif self.rgb_loss_type == "l2":
                self.rgb_loss = F.mse_loss
            else:
                raise ValueError(
                    f"Unknown rgb loss type: {self.rgb_loss_type}")
        else:
            self.rgb_loss = F.mse_loss
        # self.discriminator_iter_start = 1

        self.grad_scaler = torch.cuda.amp.GradScaler() \
            if ("autocast" in self.common_config) else None

        if torch.distributed.is_initialized():
            self.vae_bev_mm_wrapper = nn.parallel.DistributedDataParallel(
                self.vae_bev_mm,
                device_ids=[int(os.environ["LOCAL_RANK"])],
                **self.common_config.get("ddp_wrapper_settings", {}),
                broadcast_buffers=False)
            if training_config.get("enable_discriminator", False):
                self.discriminator_wrapper = nn.parallel.DistributedDataParallel(
                    self.discriminator,
                    device_ids=[int(os.environ["LOCAL_RANK"])],
                    **self.common_config.get("ddp_wrapper_settings", {}),
                    broadcast_buffers=False)

            # self.lpips_wrapper = nn.parallel.DistributedDataParallel(
            #     self.lpips,
            #     device_ids=[int(os.environ["LOCAL_RANK"])],
            #     **self.common_config.get("ddp_wrapper_settings", {}))
        self.summary = torch.utils.tensorboard.SummaryWriter(
            os.path.join(output_path, "log"))
        # load model state
        if resume_from is not None:
            model_state_dict = BEVWorldVAE.load_state(
                os.path.join(
                    output_path, "checkpoints", "{}.pth".format(resume_from)))
            self.vae_bev_mm.load_state_dict(model_state_dict["vae_bev_mm"])
            if training_config.get("enable_discriminator", False):

                self.discriminator.load_state_dict(
                    model_state_dict["discriminator"])
        elif vae_be_mm_checkpoint_path is not None:
            model_state_dict = BEVWorldVAE.load_state(
                vae_be_mm_checkpoint_path)
            self.vae_bev_mm.load_state_dict(model_state_dict)

        # optimizer
        training_params = [{'params': self.vae_bev_mm_wrapper.parameters()}]
        if training_config.get("enable_discriminator", False):
            training_params.append(
                {'params': self.discriminator_wrapper.parameters()})
        self.optimizer = dwm.common.create_instance_from_config(
            config["optimizer"],
            params=training_params)
        # scheduler
        self.lr_scheduler = dwm.common.create_instance_from_config(
            config["lr_scheduler"], optimizer=self.optimizer) \
            if "lr_scheduler" in config else None
        if resume_from is not None:
            optimizer_state_path = os.path.join(
                output_path, "optimizer", "{}.pth".format(resume_from))
            optimizer_state_dict = torch.load(
                optimizer_state_path, map_location="cpu")
            self.optimizer.load_state_dict(optimizer_state_dict)
        if self.lr_scheduler is not None and resume_from is not None:
            scheduler_state_path = os.path.join(
                output_path, "scheduler", "{}.pth".format(resume_from))
            scheduler_state_dict = torch.load(
                scheduler_state_path, map_location="cpu")
            self.lr_scheduler.load_state_dict(scheduler_state_dict)
        self.shift_factor = shift_factor
        # setup training parts
        self.loss_report_list = []
        self.metric_report_list = []
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
        vae_bev_mm_model_state_dict = self.vae_bev_mm.state_dict()
        model_state_dict = {
            "vae_bev_mm": vae_bev_mm_model_state_dict,
        }
        if self.training_config.get("enable_discriminator", False):
            discriminator_model_state_dict = self.discriminator.state_dict()
            model_state_dict["discriminator"] = discriminator_model_state_dict
        optimizer_state_dict = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            scheduler_state_dict = self.lr_scheduler.state_dict()

        if self.should_save:
            model_root = os.path.join(output_path, "checkpoints")
            os.makedirs(model_root, exist_ok=True)
            torch.save(
                model_state_dict,
                os.path.join(model_root, "{}.pth".format(steps)))

            optimizer_root = os.path.join(output_path, "optimizer")
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
            joint_list = [
                {**i[0], **i[1]}
                for i in zip(self.loss_report_list, self.metric_report_list)
            ]
            if len(joint_list) > 0:
                all_keys = set()
                for i in range(len(joint_list)):
                    all_keys.update(joint_list[i].keys())
                log_dict = {
                    k: sum([
                        joint_list[i][k] for i in range(len(joint_list)) if k in joint_list[i]
                    ]) / len([1 for i in range(len(joint_list)) if k in joint_list[i]])
                    for k in all_keys
                }
                if self.lr_scheduler is not None:
                    log_dict["lr"] = self.lr_scheduler.get_last_lr()[0]
                log_string = ", ".join(
                    ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()])
                print(
                    "Step {} ({:.1f} s/step), {}".format(
                        global_step, self.step_duration / log_steps,
                        log_string))
                if log_type == "wandb" and wandb.run is not None:
                    wandb.log(log_dict)
                else:
                    for k, v in log_dict.items():
                        self.summary.add_scalar(
                            "train/{}".format(k), v, global_step)

        self.loss_report_list.clear()
        self.metric_report_list.clear()
        self.step_duration = 0

    def train_discriminator(self, global_step: int):
        """
        Args:
            global_step: int - Current gloable step
        Returns:
            bool - Whether to train the discriminator
        """
        return self.training_config.get("enable_discriminator", False) and global_step % 2 != 0 and global_step >= self.discriminator_iter_start

    def train_step(self, batch: dict, global_step: int):
        # batch = self._data_process(batch)
        """
        Args:
            batch: dict. All keys: 'fps', 'pts', 'camera_transforms', 'camera_intrinsics', 'image_size', 'lidar_transforms', 'ego_transforms', 'bev_translation', 'bev_rotation', '3dbox_images', '3dbox_bev_images', 'vae_images', 'bev_images', 'clip_text', 'lidar_points'
                Currently the dataset inherit from HoloDrive so not all the keys are useful in this step. The data used in training are:
                    vae_images: torch.Tensor, [batch_size, sequence_length, view_count, C, H, W].
                    lidar_points: List[List[torch.Tensor]], len(lidar_points) == batch_size, len(lidar_points[ind]) == view_count, the shape of each point cloud is [num_points, 3].
            global_step: int
        """
        # save_dict_items(batch, "/mnt/storage/user/liuyichen/repo/DWM/scripts/lyc_noupload/test_data")
        # batch = load_dict_items("/mnt/storage/user/liuyichen/repo/DWM/scripts/lyc_noupload/test_data")
        t0 = time.time()
        self.vae_bev_mm_wrapper.train()
        torch.cuda.empty_cache()
        if "vae_images" in batch:
            multi_view_images = batch["vae_images"]

            image_tensor = multi_view_images.clone().flatten(
                0, 1).to(self.device).contiguous()
        else:
            image_tensor = None
        if self.training_config.get("random_sample", 0) > 1:
            image_tensor = image_tensor[:, torch.randperm(image_tensor.size(1))[
                :self.training_config.get("random_sample", 1)]]
        points = BEVWorldVAE.preprocess_points(
            batch, self.common_config, self.device)
        points = list(itertools.chain.from_iterable(points))
        points = [[p] for p in points]
        camera_transforms = batch["camera_transforms"].flatten(0, 1).to(
            self.device) if "camera_transforms" in batch else None
        camera_intrinsics = batch["camera_intrinsics"].flatten(0, 1).to(
            self.device) if "camera_intrinsics" in batch else None
        ego_transforms = batch["ego_transforms"].flatten(0, 1).to(self.device)
        with self.get_autocast_context():
            # Disable gradients of params in the main model when training the discriminator
            # In odd steps, the discriminator is trained
            with torch.set_grad_enabled(not self.train_discriminator(global_step)):
                ray_cast_center = self.common_config.get(
                    "ray_cast_center", None)
                if ray_cast_center is not None:
                    batch_size = len(batch["lidar_points"])
                    ray_cast_center = torch.tensor([ray_cast_center])\
                        .repeat(batch_size, 1)
                results = self.vae_bev_mm_wrapper(
                    points, image_tensor,
                    camera_transforms, camera_intrinsics, ego_transforms,
                    depth_ray_cast_center=ray_cast_center)
            losses = {}
            # The following losses include depth_loss (and maybe kl_loss or emb_loss)
            for k, v in results.items():
                if k.endswith("_loss"):
                    losses[k] = (v.sum() if torch.is_tensor(
                        v) else sum(v)) * self.get_loss_coef(k)
            # Enable discriminator loss in odd steps
            if "pred_imgs" in results:
                losses["rgb_loss"] = self.rgb_loss(
                    results['pred_imgs'], image_tensor) * self.get_loss_coef("rgb_loss")
                if self.training_config.get("enable_discriminator", False):
                    with torch.set_grad_enabled(self.train_discriminator(global_step)):
                        # Second pass for discriminator update
                        # Make sure the output loss is float32 type
                        logits_fake = self.discriminator_wrapper(
                            results['pred_imgs'].flatten(0, 1)).float().contiguous()
                        logits_real = self.discriminator_wrapper(
                            image_tensor.flatten(0, 1)).float().contiguous()
                        disc_factor = adopt_weight(
                            self.disc_factor, global_step, threshold=self.discriminator_iter_start)
                    if self.train_discriminator(global_step):
                        losses["discriminator_loss"] = self.get_loss_coef(
                            "discriminator_loss") * disc_factor * hinge_d_loss(logits_real, logits_fake)
                    else:
                        losses["generator_loss"] = self.get_loss_coef(
                            "generator_loss") * disc_factor * (-torch.mean(logits_fake))

            if self.training_config.get("use_voxel_decoder", False):
                losses["pred_voxel_loss"] = self.get_loss_coef("pred_voxel_loss") * BEVWorldVAE.stable_BCE_loss_with_logits(results['pred_voxel'].float(),
                                                                                                                            results['voxels'],
                                                                                                                            keep_balance=self.training_config["loss"].get("keep_logistc_balance", False))

        if "pred_imgs" in results:
            # Normalization for LPIPS
            lpips_pred_imgs = self.lpips_preprocess(
                results['pred_imgs'].flatten(0, 1).float()).contiguous()
            lpips_target_imgs = self.lpips_preprocess(
                image_tensor.flatten(0, 1).float()).contiguous()
            # LPIPS loss
            losses["perceptual_loss"] = self.lpips_wrapper(lpips_pred_imgs, lpips_target_imgs).mean() * \
                self.get_loss_coef("perceptual_loss")
        metrics = {}
        # backpropagation
        loss = sum(losses.values()) / \
            self.training_config.get("gradient_accumulation_steps", 1)
        loss += sum([p.data.sum() *
                    0 for p in self.vae_bev_mm.parameters() if p.grad is None])
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
                    self.vae_bev_mm_wrapper.parameters(),
                    self.training_config["max_norm_for_grad_clip"])
                if self.training_config.get("enable_discriminator", False):
                    nn.utils.clip_grad_norm_(
                        self.discriminator_wrapper.parameters(),
                        self.training_config["max_norm_for_grad_clip"])
            if self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        self.loss_report_list.append(losses)
        self.metric_report_list.append(metrics)
        self.step_duration += time.time() - t0

    @torch.no_grad()
    def preview_pipeline(
        self, batch: dict, output_path: str, global_step: int
    ):
        self.vae_bev_mm_wrapper.eval()
        # In bev_vae, we do not consider the sequence length, so we need to flatten the batch
        if "vae_images" in batch:
            multi_view_images = batch["vae_images"]
            image_tensor = multi_view_images.clone().flatten(
                0, 1).to(self.device).contiguous()

        else:
            image_tensor = None
        points = BEVWorldVAE.preprocess_points(
            batch, self.common_config, self.device)
        points = list(itertools.chain.from_iterable(points))
        points = [[p] for p in points]
        camera_transforms = batch["camera_transforms"].flatten(0, 1).to(
            self.device) if "camera_transforms" in batch else None
        camera_intrinsics = batch["camera_intrinsics"].flatten(0, 1).to(
            self.device) if "camera_intrinsics" in batch else None

        ego_transforms = batch["ego_transforms"].flatten(0, 1).to(self.device)
        with self.get_autocast_context():
            ray_cast_center = self.common_config.get("ray_cast_center", None)
            if ray_cast_center is not None:
                batch_size = len(batch["lidar_points"])
                ray_cast_center = torch.tensor([ray_cast_center])\
                    .repeat(batch_size, 1)
            results = self.vae_bev_mm_wrapper(
                points, image_tensor,
                camera_transforms, camera_intrinsics, ego_transforms,
                depth_ray_cast_center=ray_cast_center)
            if "lidar_rec" in results:
                results["lidar_rec"] = self.vae_bev_mm.voxelizer(
                    [[_] for _ in results["lidar_rec"]])[:, 0]
            for k, v in results.items():
                if isinstance(v, torch.Tensor):
                    results[k] = v.detach().cpu()
        # LiDAR preview
        # columns: GT, ray reconstruction

        if "lidar_rec" in results or "pred_voxel" in results:
            preview_lidar_size = self.inference_config["preview_lidar_img_size"]
            preview_lidar = Image.new(
                "L", (
                    3 * preview_lidar_size[0],
                    len(batch["lidar_points"]) * preview_lidar_size[1]
                ))
            if not "lidar_rec" in results:
                results["lidar_rec"] = torch.zeros_like(results["voxels"])
            if not "pred_voxel" in results:
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
        # image preview
        if "pred_imgs" in results:
            # columes: view count ｜ rows: GT, prediction, GT, prediction, ...
            preview_img_size = self.inference_config["preview_image_size"]
            pred_imgs = results["pred_imgs"].contiguous()
            batch_size, view_count, C, H, W = pred_imgs.shape
            preview_imgs = Image.new(
                "RGB", (
                    view_count * preview_img_size[0],
                    2 * batch_size * preview_img_size[1]
                ))
            for i in range(batch_size):
                # paste GT
                for j, image in enumerate(image_tensor[i]):
                    preview_imgs.paste(
                        torchvision.transforms.functional.to_pil_image(image), (j * preview_img_size[0], 2 * i * preview_img_size[1]))
                # paste pred imgs
                for j, image in enumerate(pred_imgs[i]):
                    preview_imgs.paste(
                        torchvision.transforms.functional.to_pil_image(image), (j * preview_img_size[0], (2 * i + 1) * preview_img_size[1]))
            if self.should_save:
                os.makedirs(os.path.join(
                    output_path, "preview"), exist_ok=True)

                preview_imgs.save(
                    os.path.join(
                        output_path, "preview", "{}_imgs.png".format(global_step)))

        torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate_pipeline(
        self, global_step: int, dataset_length: int,
        validation_dataloader: torch.utils.data.DataLoader,
        validation_datasampler=None, log_type: str = "wandb"
    ):
        # NOTE
        # the batch size of evaluation should be same with training
        """
        Args:
            batch: dict. All keys: 'fps', 'pts', 'camera_transforms', 'camera_intrinsics', 'image_size', 'lidar_transforms', 'ego_transforms', 'bev_translation', 'bev_rotation', '3dbox_images', '3dbox_bev_images', 'vae_images', 'bev_images', 'clip_text', 'lidar_points'
                Currently the dataset inherit from HoloDrive so not all the keys are useful in this step. The data used in training are:
                    vae_images: torch.Tensor, [batch_size, sequence_length, view_count, C, H, W].
                    lidar_points: List[List[torch.Tensor]], len(lidar_points) == batch_size, len(lidar_points[ind]) == view_count, the shape of each point cloud is [num_points, 3].
            global_step: int
        """
        t0 = time.time()
        if self.should_save:
            print(f"eval {len(validation_dataloader)} samples")
        if self.ddp:
            validation_datasampler.set_epoch(0)
        all_metrics = []
        step_metrics = defaultdict(lambda: 0)
        for (idx, batch) in enumerate(validation_dataloader):
            os.makedirs("temp_saving", exist_ok=True)
            self.vae_bev_mm_wrapper.eval()
            torch.cuda.empty_cache()
            if self.ddp:
                torch.distributed.barrier()
            if "vae_images" in batch:
                multi_view_images = batch["vae_images"]
                image_tensor = multi_view_images.clone().flatten(
                    0, 1).to(self.device).contiguous()
            else:
                image_tensor = None
            points = BEVWorldVAE.preprocess_points(
                batch, self.common_config, self.device)
            points = list(itertools.chain.from_iterable(points))
            points = [[p] for p in points]
            camera_transforms = batch["camera_transforms"].flatten(0, 1).to(
                self.device) if "camera_transforms" in batch else None
            camera_intrinsics = batch["camera_intrinsics"].flatten(0, 1).to(
                self.device) if "camera_intrinsics" in batch else None
            ego_transforms = batch["ego_transforms"].flatten(
                0, 1).to(self.device)
            with self.get_autocast_context():
                ray_cast_center = self.common_config.get(
                    "ray_cast_center", None)

                if ray_cast_center is not None:
                    batch_size = len(batch["lidar_points"])
                    ray_cast_center = torch.tensor([ray_cast_center])\
                        .repeat(batch_size, 1)
                results = self.vae_bev_mm_wrapper(
                    points, image_tensor,
                    camera_transforms, camera_intrinsics, ego_transforms,
                    depth_ray_cast_center=ray_cast_center)

            if "pred_imgs" in results:
                if not "rgb_loss" in self.metrics:
                    self.metrics["rgb_loss"] = CustomMeanMetrics()
                self.metrics["rgb_loss"].update(
                    self.rgb_loss(results['pred_imgs'], image_tensor))
            for k, v in results.items():
                if k.endswith("_loss"):
                    if not k in self.metrics:
                        self.metrics[k] = CustomMeanMetrics().to(self.device)
                    self.metrics[k].update(v.sum())
            if self.training_config.get("use_voxel_decoder", False):
                if not "voxel_loss" in self.metrics:
                    self.metrics["voxel_loss"] = CustomMeanMetrics().to(
                        self.device)
                self.metrics["voxel_loss"].update(BEVWorldVAE.stable_BCE_loss_with_logits(results['pred_voxel'].float(),
                                                                                          results['voxels'],
                                                                                          keep_balance=self.training_config.get("keep_logistc_balance", False)))
                pred_voxel = dwm.functional.gumbel_sigmoid(
                    results['pred_voxel'], hard=True) >= 0.5
                gt_voxel = results['voxels'] >= 0.5
                if "voxel_diff" in self.metrics:
                    self.metrics["voxel_diff"].update(pred_voxel, gt_voxel)
                if "voxel_iou" in self.metrics:
                    self.metrics["voxel_iou"].update(pred_voxel, gt_voxel)

            # save the results for downstream evaluation
            if self.config.get("save_results", False):
                if "pred_imgs" in results:
                    preview_img_size = self.inference_config["preview_image_size"]
                    pred_imgs = results["pred_imgs"].contiguous()
                    batch_size, view_count, C, H, W = pred_imgs.shape
                    preview_imgs = Image.new(
                        "RGB", (
                            view_count * preview_img_size[0],
                            2 * batch_size * preview_img_size[1]
                        ))
                    for i in range(batch_size):
                        # paste GT
                        for j, image in enumerate(image_tensor[i]):
                            preview_imgs.paste(
                                torchvision.transforms.functional.to_pil_image(image), (j * preview_img_size[0], 2 * i * preview_img_size[1]))
                        # paste pred imgs
                        for j, image in enumerate(pred_imgs[i]):
                            preview_imgs.paste(
                                torchvision.transforms.functional.to_pil_image(image), (j * preview_img_size[0], (2 * i + 1) * preview_img_size[1]))
                    os.makedirs(os.path.join(self.output_path,
                                "pred_imgs"), exist_ok=True)
                    preview_imgs.save(
                        os.path.join(
                            self.output_path, "pred_imgs", "{}_imgs.png".format(idx)))
                # save the voxel
                if "pred_voxel" in results:
                    paths = [
                        os.path.join(self.output_path, 'pred_voxel_' + k)
                        for i in batch["sample_data"]
                        for j in i
                        for k in j["filename"] if k.endswith(".bin")
                    ]
                    # convert the voxel to pc
                    pred_voxel_pc = BEVWorldVAE.voxels2points(self.vae_bev_mm.grid_size,
                                                              pred_voxel[:, None, ...])
                    pred_voxel_pc = BEVWorldVAE.postprocess_points(
                        batch, pred_voxel_pc, self.common_config)
                    pred_voxel_pc = [
                        j
                        for i in pred_voxel_pc
                        for j in i
                    ]

                    for path, points in zip(paths, pred_voxel_pc):
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        points = points.numpy()
                        padded_points = np.concatenate([
                            points, np.zeros(
                                (points.shape[0], 2), dtype=np.float32)

                        ], axis=-1)
                        with open(path, "wb") as f:
                            f.write(padded_points.tobytes())
                paths = [
                    os.path.join(self.output_path, 'raw_' + k)
                    for i in batch["sample_data"]
                    for j in i
                    for k in j["filename"] if k.endswith(".bin")
                ]
                raw_points = [
                    j
                    for i in batch["lidar_points"]
                    for j in i
                ]
                for path, points in zip(paths, raw_points):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    points = points.numpy()
                    padded_points = np.concatenate([
                        points, np.zeros(
                            (points.shape[0], 2), dtype=np.float32)
                    ], axis=-1)
                    with open(path, "wb") as f:
                        f.write(padded_points.tobytes())

        if self.should_save:
            print("Step {}:".format(global_step))
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
