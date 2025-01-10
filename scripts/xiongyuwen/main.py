from mmcv import Config, DictAction
import torch
import argparse
import os
import warnings
import time
import subprocess
import mmcv
import os.path as osp
from mmcv.runner import get_dist_info, build_optimizer

from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader

import diffusers
from transformers import CLIPTextModel, AutoTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="ids of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both specified, "
            "--options is deprecated in favor of --cfg-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --cfg-options")
        args.cfg_options = args.options

    return args


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs, flush=True)

    __builtin__.print = print


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = rank % torch.cuda.device_count()

        world_size = int(os.environ["SLURM_NTASKS"])

        args.rank = rank
        args.gpu = local_rank
        args.local_rank = local_rank
        args.world_size = world_size

        try:
            local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
        except:
            local_size = int(os.environ.get("LOCAL_SIZE", 1))

        if "MASTER_PORT" not in os.environ:
            port = 22110

            print(f"MASTER_PORT = {port}")
            os.environ["MASTER_PORT"] = str(port)

            time.sleep(3)

        node_list = os.environ["SLURM_STEP_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr

        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(local_size)
        os.environ["WORLD_SIZE"] = str(world_size)

    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(local_rank)
    args.dist_backend = "nccl"
    # print('| distributed init (rank {}): {}'.format(
    #     rank, args.dist_url), flush=True)
    # torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                                      world_size=args.world_size, rank=args.rank)
    dist_backend = "nccl"
    torch.distributed.init_process_group(
        backend=dist_backend,  # init_method=args.dist_url,
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )
    torch.distributed.barrier()
    # print(dist.get_world_size())
    setup_for_distributed(rank == 0)


def save_checkpoint(
    unet: torch.nn.Module, should_save: bool, output_path: str, filename: str
):
    # state_dict = unet.state_dict()

    if should_save:
        os.makedirs(output_path, exist_ok=True)
        unet.save_pretrained(os.path.join(output_path, filename))
        # torch.save(state_dict, os.path.join(output_path, filename))

    torch.distributed.barrier()


def main():

    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # import modules from plguin/xx, registry will be updated
    import sys

    sys.path.append(os.path.abspath("."))

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_distributed_mode(args)
        # init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")

    dataset = [build_dataset(cfg.data.train)]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
        )
        for ds in dataset
    ][0]

    pretrained_model_name_or_path = "assets/CompVis/stable-diffusion-v1-4"

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False
    )

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder.requires_grad_(False)
    text_encoder.cuda()

    # VAE
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.cuda()
    vae_image_processor = diffusers.image_processor.VaeImageProcessor(
        vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1)
    )

    # noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
    # noise_scheduler = PNDMScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )

    # unet
    unet_wrapper = unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet"
    )
    unet.cuda()
    unet.enable_gradient_checkpointing()

    unet_wrapper = torch.nn.parallel.DistributedDataParallel(
        unet, device_ids=[torch.cuda.current_device()], find_unused_parameters=True
    )
    unet_wrapper.train()

    parameters_to_optimize = [] # unet_wrapper.parameters()

    # for n, p in unet_wrapper.named_parameters():
    #     if 'to_q' in n or 'to_k' in n or 'to_v' in n:
    #         parameters_to_optimize.append(p)
    #     else:
    #         p.requires_grad_(False)

    # optimizer = torch.optim.AdamW(parameters_to_optimize, lr=1e-4, weight_decay=1e-4)
 
    optimizer = build_optimizer(unet_wrapper, cfg.optimizer)


    grad_scaler = torch.cuda.amp.GradScaler()
    loss_list = []
    step_duration = 0
    global_step = 0

    for epoch in range(10000):
        data_loaders.sampler.set_epoch(epoch)

        for step, batch in enumerate(data_loaders):

            t0 = time.time()

            with torch.no_grad():

                image_tensor = batch["img"].data[0].cuda()

                image_tensor = vae_image_processor.preprocess(image_tensor)

                latents = vae.encode(image_tensor).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (image_tensor.shape[0],),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # encoder_hidden_states = torch.zeros(image_tensor.shape[0], 77, 768).cuda()
                # encoder_hidden_states = text_encoder(zero_text_prompt.cuda(), return_dict=False)[0]
                uncond_input = tokenizer(
                    [""] * image_tensor.shape[0],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                encoder_hidden_states = text_encoder(uncond_input.input_ids.cuda())[0]

                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast(device_type="cuda"):

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)

                # for sd2.1 torch.Size([bs, 4, 24, 48]) torch.Size([bs]) torch.Size([bs, 77, 1024])
                # for sd1.4torch.Size([bs, 4, 24, 48]) torch.Size([bs]) torch.Size([bs, 77, 768]
                model_pred = unet_wrapper(
                    noisy_latents, timesteps, encoder_hidden_states, return_dict=False
                )[0]
                loss = torch.nn.functional.mse_loss(
                    model_pred.float(), target.float(), reduction="mean"
                )

            loss_list.append(loss.item())
            step_duration += time.time() - t0

            # update the model parameters
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            # loss.backward()
            # optimizer.step()
            optimizer.zero_grad()

            # lr_scheduler.step()

            global_step += 1

            if global_step % cfg.log_steps == 0:
                loss_value = sum(loss_list) / len(loss_list)
                print(
                    "Step {} ({:.1f} s/step), Epoch {}, loss: {:.4f}".format(
                        global_step, step_duration / cfg.log_steps, epoch, loss_value
                    )
                )
                loss_list.clear()
                step_duration = 0

            if (
                cfg.checkpointing_steps > 0
                and global_step % cfg.checkpointing_steps == 0
            ):
                save_checkpoint(
                    unet,
                    torch.distributed.get_rank() == 0,
                    cfg.work_dir,
                    "ckpt-{}".format(global_step),
                )


if __name__ == "__main__":

    main()
