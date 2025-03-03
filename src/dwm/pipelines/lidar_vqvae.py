import contextlib
import dwm.common
import dwm.functional
import os
import safetensors.torch
import time
import torch
import torch.cuda.amp
import torch.distributed
import torch.nn.functional as F
import torch.utils.tensorboard
import torchvision


class LidarCodebook():

    @staticmethod
    def load_state(path: str, pop_reservoir: bool = True):
        if path.endswith(".safetensors"):
            state = safetensors.torch.load_file(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu")

        if pop_reservoir and "vector_quantizer.reservoir" in state:
            state.pop("vector_quantizer.reservoir")

        return state

    @staticmethod
    def preprocess_points(batch, common_config, device):
        if common_config.get("point_space", "ego") == "ego":
            mhv = dwm.functional.make_homogeneous_vector
            return [
                [
                    (mhv(p_j.to(device)) @ t_j.permute(1, 0))[:, :3]
                    for p_j, t_j in zip(p_i, t_i.flatten(0, 1))
                ]
                for p_i, t_i in zip(
                    batch["lidar_points"],
                    batch["lidar_transforms"].to(device))
            ]
        else:
            return [[j.to(device) for j in i] for i in batch["lidar_points"]]

    def __init__(
        self, output_path: str, config: dict, device, common_config: dict,
        training_config: dict, inference_config: dict,
        vq_point_cloud: torch.nn.Module,
        vq_point_cloud_checkpoint_path: str = None, resume_from = None
    ):
        self.should_save = not torch.distributed.is_initialized() or \
            torch.distributed.get_rank() == 0

        self.output_path = output_path
        self.config = config
        self.common_config = common_config
        self.training_config = training_config
        self.inference_config = inference_config
        self.device = device

        self.vq_point_cloud_wrapper = self.vq_point_cloud = vq_point_cloud
        self.vq_point_cloud.to(self.device)

        self.generator = torch.Generator()
        if "generator_seed" in self.config:
            self.generator.manual_seed(self.config["generator_seed"])
        else:
            self.generator.seed()

        if resume_from is not None:
            model_state_dict = LidarCodebook.load_state(
                os.path.join(
                    output_path, "checkpoints", "{}.pth".format(resume_from)))
            self.vq_point_cloud.load_state_dict(model_state_dict, strict=False)

        elif vq_point_cloud_checkpoint_path is not None:
            model_state_dict = LidarCodebook.load_state(
                vq_point_cloud_checkpoint_path)
            self.vq_point_cloud.load_state_dict(model_state_dict, strict=False)

        # setup training parts
        self.loss_report_list = []
        self.metric_report_list = []
        self.step_duration = 0
        self.iter = 0
        self.code_dict = {}
        for i in range(self.vq_point_cloud.vector_quantizer.n_e):
            self.code_dict[i] = 0

        self.grad_scaler = torch.cuda.amp.GradScaler() \
            if self.training_config.get("enable_grad_scaler", False) else None

        if torch.distributed.is_initialized():
            self.vq_point_cloud_wrapper = torch.nn.parallel.DistributedDataParallel(
                self.vq_point_cloud,
                device_ids=[int(os.environ["LOCAL_RANK"])],
                **self.common_config.get("ddp_wrapper_settings", {}))

        if self.should_save:
            self.summary = torch.utils.tensorboard.SummaryWriter(
                os.path.join(output_path, "log"))

        self.optimizer = dwm.common.create_instance_from_config(
            config["optimizer"],
            params=self.vq_point_cloud_wrapper.parameters())

        if resume_from is not None:
            optimizer_state_path = os.path.join(
                output_path, "optimizer", "{}.pth".format(resume_from))
            optimizer_state_dict = torch.load(
                optimizer_state_path, map_location="cpu")
            self.optimizer.load_state_dict(optimizer_state_dict)

        self.lr_scheduler = dwm.common.create_instance_from_config(
            config["lr_scheduler"], optimizer=self.optimizer) \
            if "lr_scheduler" in config else None

    def get_loss_coef(self, name):
        loss_coef = 1
        if "loss_coef_dict" in self.training_config:
            loss_coef = self.training_config["loss_coef_dict"].get(name, 1.0)

        return loss_coef

    def get_autocast_context(self):
        if "autocast" in self.common_config:
            return torch.autocast(**self.common_config["autocast"])
        else:
            return contextlib.nullcontext()

    def save_checkpoint(self, output_path: str, steps: int):
        model_state_dict = self.vq_point_cloud.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()

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

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def log(self, global_step: int, log_steps: int):
        if self.should_save:
            joint_list = [
                {**i[0], **i[1]}
                for i in zip(self.loss_report_list, self.metric_report_list)
            ]
            if len(joint_list) > 0:
                log_dict = {
                    k: sum([
                        joint_list[i][k] for i in range(len(joint_list))
                    ]) / len(joint_list)
                    for k in joint_list[0].keys()
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

        self.loss_report_list.clear()
        self.metric_report_list.clear()
        self.step_duration = 0

    def train_step(self, batch: dict, global_step: int):
        t0 = time.time()

        self.vq_point_cloud_wrapper.train()

        points = LidarCodebook.preprocess_points(
            batch, self.common_config, self.device)
        with self.get_autocast_context():
            ray_cast_center = self.common_config.get("ray_cast_center", None)
            if ray_cast_center is not None:
                batch_size = len(batch["lidar_points"])
                ray_cast_center = torch\
                    .tensor([ray_cast_center], device=self.device)\
                    .repeat(batch_size, 1)

            result = self.vq_point_cloud_wrapper(
                points, offsets=ray_cast_center)

        losses = {
            "voxel_loss": F.binary_cross_entropy_with_logits(
                result["lidar_voxel"].float(), result["voxels"],
                reduction="mean") *
            self.get_loss_coef("voxel_loss"),
            "emb_loss": sum(result["emb_loss"]) *
            self.get_loss_coef("emb_loss"),
        }

        if "depth_loss" in result:
            losses["depth_loss"] = result["depth_loss"] * \
                self.get_loss_coef("depth_loss")

        if "sdf_loss" in result:
            losses["sdf_loss"] = result["sdf_loss"] * \
                self.get_loss_coef("sdf_loss")

        voxel_rec = dwm.functional.gumbel_sigmoid(
            result["lidar_voxel"], hard=True, generator=self.generator)
        voxel_rec_iou = \
            ((voxel_rec >= 0.5) & (result["voxels"] >= 0.5)).sum() / \
            ((voxel_rec >= 0.5) | (result["voxels"] >= 0.5)).sum()
        voxel_rec_diff = \
            (voxel_rec - result["voxels"]).abs().sum() / \
            result["voxels"].shape[0]
        lidar_rec = self.vq_point_cloud.voxelizer(result["sampled_points"])
        lidar_rec_iou = \
            ((lidar_rec >= 0.5) & (result["voxels"] >= 0.5)).sum() / \
            ((lidar_rec >= 0.5) | (result["voxels"] >= 0.5)).sum()
        lidar_rec_diff = (
            lidar_rec - result["voxels"]).abs().sum() / \
            result["voxels"].shape[0]
        code_util = (
            self.vq_point_cloud.code_age <
            self.vq_point_cloud.vector_quantizer.dead_limit).sum() / \
            self.vq_point_cloud.code_age.numel()
        code_uniformity = \
            self.vq_point_cloud.code_usage.topk(10)[0].sum() / \
            self.vq_point_cloud.code_usage.sum()
        metrics = {
            "voxel_rec_iou": voxel_rec_iou.item(),
            "voxel_rec_diff": voxel_rec_diff.item(),
            "lidar_rec_iou": lidar_rec_iou.item(),
            "lidar_rec_diff": lidar_rec_diff.item(),
            "code_util": code_util.item(),
            "code_uniformity": code_uniformity.item()
        }

        loss = sum(losses.values())

        if self.training_config.get("enable_grad_scaler", False):
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        # optimize parameters
        should_optimize = \
            ("gradient_accumulation_steps" not in self.training_config) or \
            ("gradient_accumulation_steps" in self.training_config and
                (global_step + 1) %
             self.training_config["gradient_accumulation_steps"] == 0)
        if should_optimize:
            if "max_norm_for_grad_clip" in self.training_config:
                if self.training_config.get("enable_grad_scaler", False):
                    self.grad_scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.vq_point_cloud_wrapper.parameters(),
                    self.training_config["max_norm_for_grad_clip"])

            if self.training_config.get("enable_grad_scaler", False):
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
        self.vq_point_cloud_wrapper.eval()

        points = LidarCodebook.preprocess_points(
            batch, self.common_config, self.device)

        with self.get_autocast_context():
            ray_cast_center = self.common_config.get("ray_cast_center", None)
            if ray_cast_center is not None:
                batch_size = len(batch["lidar_points"])
                ray_cast_center = \
                    torch.tensor([ray_cast_center], device=self.device)\
                    .repeat(batch_size, 1)

            voxels = self.vq_point_cloud.voxelizer(points)
            lidar_features = self.vq_point_cloud.lidar_encoder(
                voxels.flatten(0, 1))
            lidar_quantized_features, _, _ = \
                self.vq_point_cloud.vector_quantizer(lidar_features)
            density, lidar_voxel = self.vq_point_cloud.lidar_decoder(
                lidar_quantized_features)

            lidar_voxel = lidar_voxel.unflatten(0, voxels.shape[:2])
            voxel_rec = dwm.functional.gumbel_sigmoid(
                lidar_voxel, hard=True, generator=self.generator)
            coarse_mask = F.max_pool3d(voxel_rec, (4, 8, 8))
            _, _, sampled_points = self.vq_point_cloud.ray_render_dvgo(
                density.unflatten(0, voxels.shape[:2]),
                points, coarse_mask, offsets=ray_cast_center)
            lidar_rec = self.vq_point_cloud.voxelizer(sampled_points)

        # columns: GT, reconstruction, ray reconstruction
        stacked_images = torch.stack([voxels, voxel_rec, lidar_rec])\
            .amax(-3, keepdim=True)
        resized_images = torch.nn.functional.interpolate(
            stacked_images.flatten(0, 2),
            tuple(self.inference_config["preview_image_size"][::-1])
        ).unflatten(0, stacked_images.shape[:3])

        # [C, B * T * H, S * W]
        preview_image = resized_images.permute(3, 1, 2, 4, 0, 5).flatten(-2)\
            .flatten(1, 3)
        if self.should_save:
            os.makedirs(os.path.join(output_path, "preview"), exist_ok=True)
            preview_image_path = os.path.join(
                output_path, "preview", "{}.png".format(global_step))
            torchvision.transforms.functional.to_pil_image(preview_image)\
                .save(preview_image_path)

    def evaluate_pipeline(
        self, global_step: int, dataset_length: int,
        validation_dataloader: torch.utils.data.DataLoader,
        validation_datasampler=None
    ):
        # TODO: evaluate the metrics on the validation set
        pass
