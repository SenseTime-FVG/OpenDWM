import torch
import torchvision
import transformers
from torchmetrics.image.fid import FrechetInceptionDistance

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
from PIL import Image
from tqdm.auto import tqdm
import argparse
from mmcv import Config, DictAction
import os
import warnings


from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader

import time
import subprocess

from mmcv.runner import get_dist_info


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


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def concat_images(images, direction="horizontal"):
    """
    Concatenate a list of PIL images either horizontally or vertically.

    Parameters:
        images (list): List of PIL images to concatenate.
        direction (str): Direction of concatenation, either 'horizontal' or 'vertical'.

    Returns:
        PIL Image: Concatenated image.
    """
    if direction == "horizontal":
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_img = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.size[0]
    elif direction == "vertical":
        widths, heights = zip(*(i.size for i in images))
        max_width = max(widths)
        total_height = sum(heights)
        new_img = Image.new("RGB", (max_width, total_height))
        y_offset = 0
        for img in images:
            new_img.paste(img, (0, y_offset))
            y_offset += img.size[1]
    else:
        raise ValueError("Direction should be 'horizontal' or 'vertical'")

    return new_img


def generate(
    prompt,
    tokenizer,
    text_encoder,
    vae,
    unet,
    scheduler,
    height,
    width,
    latents_device="cuda",
):

    batch_size = len(prompt)

    latents_shape = (batch_size, unet.in_channels, height // 8, width // 8)

    latents = torch.randn(
        latents_shape,
        device=latents_device,
    )

    scheduler.set_timesteps(50)

    do_classifier_free_guidance = True
    guidance_scale = 5.0

    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_input.input_ids.cuda())[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.cuda())[0]

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    text_embeddings = torch.cat([uncond_embeddings, uncond_embeddings])

    # text_embeddings = torch.zeros(batch_size * 2, 77, 768).cuda()

    with torch.autocast(device_type="cuda", enabled=False):

        for i, t in enumerate(tqdm(scheduler.timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            # predict the noise residual
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents.float()).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()

    return numpy_to_pil(image)


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

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_distributed_mode(args)
        # init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    pretrained_model_name_or_path = "assets/CompVis/stable-diffusion-v1-4"

    # unet_pretrain_path = "work_dirs/nusc_mono/ckpt-35000"
    # unet_pretrain_path = "work_dirs/nusc_mono_pndm/ckpt-21000"
    unet_pretrain_path = "work_dirs/nusc_mono_rgb/ckpt-22000"
    # unet_pretrain_path = "work_dirs/nusc_mono_fp32/ckpt-7000"
    # unet_pretrain_path = "work_dirs/nusc_mono_512/ckpt-3000"
    # unet_pretrain_path = None

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False
    )

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder"
    ).cuda()
    text_encoder.requires_grad_(False)
    # text_encoder.to(device)

    noise_scheduler = DDPMScheduler.from_pretrained(
        # noise_scheduler = PNDMScheduler.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae"
    ).cuda()
    vae.requires_grad_(False)
    vae_image_processor = diffusers.image_processor.VaeImageProcessor(
        vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1)
    )
    # vae.to(device)
    print(f"Loading CLIP/VAE/... from {pretrained_model_name_or_path}")
    print(f"Loading unet from {unet_pretrain_path}")

    if unet_pretrain_path == None:
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet"
        ).cuda()
    else:
        unet = UNet2DConditionModel.from_pretrained(unet_pretrain_path).cuda()

    viz = False

    if not viz:

        fid = FrechetInceptionDistance(normalize=True, sync_on_compute=True).to("cuda")
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        dataset = [build_dataset(cfg.data.val)]

        data_loaders = [
            build_dataloader(
                ds,
                25,
                cfg.data.workers_per_gpu,
                # cfg.gpus will be ignored if distributed
                len(cfg.gpu_ids),
                dist=distributed,
                seed=cfg.seed,
                shuffle=False,
            )
            for ds in dataset
        ][0]

        for n, data in enumerate(data_loaders):
            if n >= 25:
                break
            fid.update(data["img"].data[0].to("cuda"), real=True)

        for i in range(5):
            print(f"{i}/5")
            with torch.no_grad():
                fake_image = generate(
                    ["car" for _ in range(125)],
                    tokenizer,
                    text_encoder,
                    vae,
                    unet,
                    noise_scheduler,
                    192,
                    384,
                )
                fake_image = torch.stack([transform(_) for _ in fake_image])
                fid.update(fake_image.to("cuda"), real=False)

        fid._should_unsync = False
        print(
            f"FID: {float(fid.compute())}, {fid.real_features_num_samples}, {fid.fake_features_num_samples}"
        )

    else:

        # names = []
        # for n, p in unet.named_parameters():
        #     if 'to_q' in n or 'attn_temp' in n or 'conv_temporal' in n or 'skeleton' in n:
        #         names.append(n)

        n = 1

        with torch.no_grad():
            fake_image = generate(
                ["car" for _ in range(16)],
                tokenizer,
                text_encoder,
                vae,
                unet,
                noise_scheduler,
                192,
                384,
            )

        concatenated_img_horizontal_1 = concat_images(
            fake_image[0:8], direction="horizontal"
        )
        concatenated_img_horizontal_2 = concat_images(
            fake_image[8:16], direction="horizontal"
        )
        all_img = concat_images(
            [concatenated_img_horizontal_1, concatenated_img_horizontal_2],
            direction="vertical",
        )
        all_img.save(f"preview_sd_2.1_{n}.png")
        print(f"save preview_sd_2.1_{n}.png")


if __name__ == "__main__":
    main()

