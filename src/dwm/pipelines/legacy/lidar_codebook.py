import dwm.common
import dwm.functional
import dwm.models.vq_point_cloud
import os
from PIL import Image
import pickle
import safetensors.torch
import time
import torch
import torch.cuda.amp
import torch.nn.functional as F
import torch.utils.tensorboard
import torchvision
from tqdm import tqdm


class LidarCodebook():

    def load_state(path: str):
        if path.endswith(".safetensors"):
            state = safetensors.torch.load_file(path, device="cpu")
        else:
            state = torch.load(path, map_location="cpu")

        return state

    def __init__(
        self, ddp: bool, should_save: bool, output_path: str, config: dict,
        device, vq_point_cloud, vq_point_cloud_checkpoint_path: str = None
    ):
        self.should_save = not torch.distributed.is_initialized() or \
            (torch.distributed.is_initialized() and
             torch.distributed.get_rank() == 0)

        self.config = config
        self.device = device
        self.generator = torch.Generator()
        if "generator_seed" in config:
            self.generator.manual_seed(config["generator_seed"])
        else:
            self.generator.seed()

        self.vq_point_cloud_wrapper = self.vq_point_cloud = vq_point_cloud
        self.vq_point_cloud.to(self.device)
        if vq_point_cloud_checkpoint_path is not None:
            self.vq_point_cloud.load_state_dict(
                LidarCodebook.load_state(vq_point_cloud_checkpoint_path), strict=False)
        
        self.output_path = output_path
        self.ddp = ddp

        # setup training parts
        self.loss_list = []
        self.step_duration = 0
        self.iter = 0
        self.code_dict = {}
        for i in range(self.vq_point_cloud.vector_quantizer.n_e):
            self.code_dict[i] = 0

        if config["device"] == "cuda":
            self.grad_scaler = torch.cuda.amp.GradScaler()

        if torch.distributed.is_initialized():
            self.vq_point_cloud_wrapper = torch.nn.parallel.DistributedDataParallel(
                self.vq_point_cloud,
                device_ids=[int(os.environ["LOCAL_RANK"])])

        if self.should_save:
            self.summary = torch.utils.tensorboard.SummaryWriter(
                os.path.join(output_path, "log"))

        self.optimizer = dwm.common.create_instance_from_config(
            config["optimizer"],
            params=self.vq_point_cloud_wrapper.parameters())
        self.lr_scheduler = dwm.common.create_instance_from_config(
            config["lr_scheduler"], optimizer=self.optimizer)

    def save_checkpoint(self, _1: bool, output_path: str, steps: int):
        state_dict = self.vq_point_cloud.state_dict()

        if self.should_save:
            os.makedirs(
                os.path.join(output_path, "checkpoints"), exist_ok=True)
            torch.save(
                state_dict,
                os.path.join(
                    output_path, "checkpoints", "{}.pth".format(steps)))

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def log(self, _1: bool, global_step: int, log_steps: int):
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

    def train_step(self, batch: dict, global_step: int):
        t0 = time.time()
        
        points = sum([[j[:, :3] for j in i] for i in batch["lidar_points"]], [])
        # b t [4 4]
        # b*t [N 3]
        
        # lidar_tranforms = batch["lidar_transforms"].flatten(0, 2)
        # points_new = []
        # for p, trans in zip(points, lidar_tranforms):
        #     p = dwm.functional.make_homogeneous_vector(p)
        #     p = p @ (trans.permute(1, 0))
        #     points_new.append(p[:, :3])
        # points = points_new

        # points = [i[0][:, :3] for i in batch["lidar_points"]]
        for i in range(len(points)):
            if torch.rand((1,), generator=self.generator).item() < 0.5:
                points[i][:, 0] *= -1

        self.vq_point_cloud_wrapper.train()
        with torch.autocast(device_type=self.config["device"]):
            losses = self.vq_point_cloud_wrapper(
                [i.to(self.device) for i in points],
                self.config["depth_sdf_loss_coef"])

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
                    self.vq_point_cloud_wrapper.parameters(),
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

    def preview_pipeline(
        self, _1: bool, batch: dict, output_path: str,
        global_step: int
    ):
        # points = [i[0][:, :3] for i in batch["lidar_points"]]
        points = sum([[j[:, :3] for j in i] for i in batch["lidar_points"]], [])
        # b t [4 4]
        # b*t [N 3]
        
        # lidar_tranforms = batch["lidar_transforms"].flatten(0, 2)
        # points_new = []
        # for p, trans in zip(points, lidar_tranforms):
        #     p = dwm.functional.make_homogeneous_vector(p)
        #     p = p @ (trans.permute(1, 0))
        #     points_new.append(p[:, :3])
        # points = points_new

        self.vq_point_cloud_wrapper.eval()
        with torch.no_grad():
            voxels = self.vq_point_cloud.voxelizer(
                [[_.to(self.device)] for _ in points])
            lidar_feats = self.vq_point_cloud.lidar_encoder(voxels)
            lidar_quant, emb_loss, _ = self.vq_point_cloud.vector_quantizer(
                lidar_feats, self.vq_point_cloud.code_age,
                self.vq_point_cloud.code_usage)  # [bsz, 6400, 1024]

            lidar_density, lidar_voxel = self.vq_point_cloud.lidar_decoder(
                lidar_quant)
            lidar_voxel = dwm.models.vq_point_cloud.gumbel_sigmoid(
                lidar_voxel, hard=True)

            pooled_voxels = F.max_pool3d(lidar_voxel, (4, 8, 8))
            depth_loss, sdf_loss, lidar_rec = self.vq_point_cloud.ray_render_dvgo(
                lidar_density, points, pooled_voxels)
            lidar_rec = self.vq_point_cloud.voxelizer(
                [[_] for _ in lidar_rec])  # [bsz, 64, 640, 640]

        # columns: GT, reconstruction, ray reconstruction
        preview_size = self.config["preview_image_size"]
        preview_image = Image.new(
            "L", (
                3 * preview_size[0],
                len(batch["lidar_points"]) * preview_size[1]
            ))
        for i in range(len(points)):
            images = [
                torchvision.transforms.functional\
                    .to_pil_image(torch.amax(voxels[i], 0))\
                    .resize(preview_size),
                torchvision.transforms.functional\
                    .to_pil_image(torch.amax(lidar_voxel[i], 0))\
                    .resize(preview_size),
                torchvision.transforms.functional\
                    .to_pil_image(torch.amax(lidar_rec[i], 0))\
                    .resize(preview_size)
            ]
            for j, image in enumerate(images):
                preview_image.paste(
                    image, (j * preview_size[0], i * preview_size[1]))

        if self.should_save:
            os.makedirs(os.path.join(
                output_path, "preview"), exist_ok=True)
            preview_image.save(
                os.path.join(
                    output_path, "preview", "{}.png".format(global_step)))

    def evaluate_pipeline(
        self, _: bool, global_step: int, dataset_length: int,
        validation_dataloader: torch.utils.data.DataLoader,
        validation_datasampler=None
    ):
        # NOTE
        # not real evaluate, only save blank code in this function
        # the batch size of evaluation should be same with training
        def count(idx):
            unique_elements, counts = torch.unique(idx, return_counts=True)
            sorted_indices = torch.argsort(counts, descending=True)
            sorted_elements = unique_elements[sorted_indices]
            sorted_counts = counts[sorted_indices]
            return sorted_elements, sorted_counts
        
        # add the blank code saving
        blank_code = None
        with torch.no_grad():

            for batch in tqdm(validation_dataloader):
                self.iter += 1
                if self.iter > 100:
                    break
                if self.ddp:
                    torch.distributed.barrier()
                points = [
                    i[0][i[0][:, :3].norm(dim=-1) > 2.4, :3]
                    for i in batch["lidar_points"]
                ]
                
                lidar_tranforms = batch["lidar_transforms"].flatten(0, 2)
                points_new = []
                for p, trans in zip(points, lidar_tranforms):
                    p = dwm.functional.make_homogeneous_vector(p)
                    p = p @ (trans.T.float())
                    points_new.append(p[:, :3])
                points = points_new
            
                voxels = self.vq_point_cloud.voxelizer([[_] for _ in points])
                voxels = voxels.to(self.device)
                lidar_feats = self.vq_point_cloud.lidar_encoder(voxels)
                _, _, code_indices = self.vq_point_cloud.vector_quantizer(lidar_feats, self.vq_point_cloud.code_age,
                                                                self.vq_point_cloud.code_usage)
                
                codes, counts = count(code_indices)
                for code_idx, conut_nbs in zip(codes, counts):
                    self.code_dict[int(code_idx.data)] += conut_nbs
                blank_code = [k for k, v in sorted(self.code_dict.items(), key=lambda item: item[1], reverse=True)[:20]]
        
        # save pickle
        if self.should_save:
            with open(os.path.join(self.output_path, f"blank_code_{global_step}.pkl"), 'wb') as f:
                pickle.dump(blank_code, f)
                print(f'Saved BLACK CODE file')
