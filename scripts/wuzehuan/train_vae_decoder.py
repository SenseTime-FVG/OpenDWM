import argparse
import diffusers
import diffusers.image_processor
import dwm.common
import itertools
import json
import numpy as np
import os
from PIL import Image
import safetensors
import taming.modules.discriminator.model
import taming.modules.losses.lpips
import taming.modules.losses.vqperceptual
import time
import torch
import torch.utils.tensorboard
import torchvision
from typing import Tuple


class LPIPS(torch.nn.Module):

    # Learned perceptual metric
    def __init__(
        self, features: torch.nn.Module, feature_layer_segments: list,
        use_dropout: bool = True
    ):
        super().__init__()
        self.scaling_layer = taming.modules.losses.lpips.ScalingLayer()
        self.features = features
        self.feature_layer_segments = feature_layer_segments
        for i_id, i in enumerate(feature_layer_segments):
            setattr(
                self, "lin{}".format(i_id),
                taming.modules.losses.lpips.NetLinLayer(
                    i[2], use_dropout=use_dropout))

    def forward(self, input, target):
        result = 0.0
        f = self.scaling_layer(torch.cat([input, target]))
        for i_id, i in enumerate(self.feature_layer_segments):
            f = self.features[i[0]:i[1]](f)
            nf0, nf1 = taming.modules.losses.lpips.normalize_tensor(f).chunk(2)
            diff = (nf0 - nf1) ** 2
            result += taming.modules.losses.lpips.spatial_average(
                getattr(self, "lin{}".format(i_id)).model(diff), keepdim=True)

        return result


class DatasetAdapter(torch.utils.data.Dataset):
    def __init__(
        self, base_dataset: torch.utils.data.Dataset,
        expected_vae_size: tuple
    ):
        self.base_dataset = base_dataset
        self.vae_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(expected_vae_size),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        item = self.base_dataset[index]
        images = item.pop("images")
        item["vae_images"] = torch.stack(
            [self.vae_transform(i) for i in images])
        return item


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to train a VAE decoder of the world model.")
    parser.add_argument(
        "-c", "--config-path", default="train_vae_decoder.json",
        type=str, help="The config to load the train model and dataset.")
    parser.add_argument(
        "-i", "--pretrained-model-name-or-path", type=str, required=True,
        help=(
            "Path to pretrained model or model identifier from huggingface.co/"
            "models."))
    parser.add_argument(
        "-o", "--output-path", default="output", type=str, required=True,
        help="The path to save checkpoint files.")
    parser.add_argument(
        "-l", "--log-path", default="log", type=str, required=True,
        help="The path to store the summary data of the tensor board.")
    parser.add_argument(
        "--pretrained-vgg16-model-path", type=str, required=True,
        help=(
            "Path to pretrained VGG-16 model downloaded from "
            "https://download.pytorch.org/models/vgg16-397923af.pth."))
    parser.add_argument(
        "--pretrained-lpips-model-path", type=str, required=True,
        help=(
            "Path to pretrained VGG lpips model downloaded from "
            "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1."))
    parser.add_argument(
        "--checkpoint-path-to-load", default=None, type=str,
        help="The checkpoint path to load and recover the training state.")
    parser.add_argument(
        "--revision", default=None, type=str, help=(
            "Revision of pretrained model identifier from huggingface.co/"
            "models. Trainable model components should be float32 precision."))
    parser.add_argument("--train-epochs", default=10, type=int)
    parser.add_argument(
        "--log-steps", default=100, type=int,
        help="Print log and update the tensorboard per log_steps.")
    parser.add_argument(
        "--preview-steps", default=1000, type=int,
        help="Print log and update the tensorboard per log_steps.")
    parser.add_argument(
        "--checkpointing-steps", default=0, type=int, help=(
            "Save a checkpoint of the training state every X updates. "
            "Checkpoints can be used for resuming training or inference."))
    return parser


def flatten_b_s(tensor: torch.Tensor):
    shape = tensor.shape
    new_shape = (shape[0] * shape[1],) + shape[2:]
    return torch.reshape(tensor, new_shape)


@torch.no_grad()
def test_pipeline(vae, image_processor, batch, config, device):
    sequence_length = batch["vae_images"].shape[1]
    image_tensor = image_processor.preprocess(
        batch["vae_images"][0].to(device))
    with torch.autocast(config["autocast_device_type"]):
        reconstructions = vae(
            image_tensor, num_frames=sequence_length).sample

    predicted_images = image_processor.postprocess(
        torch.cat([image_tensor, reconstructions]))

    # make a preview image of the condition image and the generated image
    preview_image = Image.new(
        "RGB",
        (2 * config["preview_image_size"][0],
         sequence_length * config["preview_image_size"][1]), "black")
    for i_id, i in enumerate(predicted_images):
        preview_image.paste(
            i.resize(config["preview_image_size"]),
            ((i_id // sequence_length * config["preview_image_size"][0]),
             (i_id % sequence_length) * config["preview_image_size"][1]))
    return preview_image


def save_checkpoint(
    vae: torch.nn.Module, output_path: str, filename: str
):
    os.makedirs(output_path, exist_ok=True)
    torch.save(vae.state_dict(), os.path.join(output_path, filename))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert args.log_steps % 2 == 0
    assert args.preview_steps % config["gradient_accumulation_steps"] == 0

    # set distributed training (if enabled), log, random number generator, and
    # load the checkpoint (if required).
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(config["device"], local_rank)
        if config["device"] == "cuda":
            torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(backend=config["ddp_backend"])
    else:
        device = torch.device(config["device"])

    should_log = (ddp and local_rank == 0) or not ddp
    should_save = "RANK" not in os.environ or \
        ("RANK" in os.environ and os.environ["RANK"] == "0")
    generator = torch.Generator(device=device)
    if "generator_seed" in config:
        generator.manual_seed(config["generator_seed"])

    # load the models
    lpips = LPIPS(
        torchvision.models.vgg.make_layers(
            **config["loss"]["vgg_config"]),
        config["loss"]["vgg_feature_layer_segments"])
    lpips.load_state_dict(
        torch.load(args.pretrained_vgg16_model_path), strict=False)
    lpips.load_state_dict(
        torch.load(args.pretrained_lpips_model_path), strict=False)
    lpips.requires_grad_(False)
    lpips.to(device, torch.float16)

    parameters_to_optimize = []
    discriminator_wrapper = discriminator = \
        taming.modules.discriminator.model.NLayerDiscriminator(
            **config["loss"]["discriminator_config"])\
        .apply(taming.modules.discriminator.model.weights_init)
    discriminator.to(device)
    discriminator.train()
    if ddp:
        discriminator_wrapper = torch.nn.parallel.DistributedDataParallel(
            discriminator, device_ids=[local_rank], output_device=local_rank)

    parameters_to_optimize.append(discriminator_wrapper.parameters())

    vae_wrapper = vae = dwm.common.create_instance_from_config(config["model"])
    state = safetensors.torch.load_file(
        os.path.join(
            args.pretrained_model_name_or_path, "vae",
            "diffusion_pytorch_model.safetensors"))
    vae.load_state_dict(state, strict=False)
    if args.checkpoint_path_to_load:
        state = torch.load(args.checkpoint_path_to_load)
        vae.load_state_dict(state)

    vae.encoder.requires_grad_(False)
    vae.to(device)
    vae.enable_gradient_checkpointing()
    vae.train()
    vae_image_processor = diffusers.image_processor.VaeImageProcessor(
        vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1))
    if ddp:
        vae_wrapper = torch.nn.parallel.DistributedDataParallel(
            vae, device_ids=[local_rank], output_device=local_rank)

    parameters_to_optimize.append(
        [i for i in vae_wrapper.parameters() if i.requires_grad])

    # create optimizer
    optimizer = dwm.common.create_instance_from_config(
        config["optimizer"], params=itertools.chain(*parameters_to_optimize))

    lr_scheduler = diffusers.optimization.get_scheduler(
        **config["optimization_scheduler"], optimizer=optimizer)

    # load the dataset
    dataset = DatasetAdapter(
        dwm.common.create_instance_from_config(config["dataset"]),
        **config["dataset_adapter"])
    if ddp:
        datasampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=config["data_shuffle"])
        dataloader = torch.utils.data.DataLoader(
            dataset, **config["dataloader"], sampler=datasampler)
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, **config["dataloader"], shuffle=config["data_shuffle"])

    if should_save:
        summary = torch.utils.tensorboard.SummaryWriter(args.log_path)

    # train loop
    if config["autocast_device_type"] == "cuda":
        grad_scaler = torch.cuda.amp.GradScaler()

    loss_list = []
    global_step = 0
    step_duration = 0
    for epoch in range(args.train_epochs):
        if ddp:
            datasampler.set_epoch(epoch)

        for batch in dataloader:
            t0 = time.time()

            # convert the GT images to latent space
            batch_size, sequence_length = batch["vae_images"].shape[0:2]

            target = vae_image_processor.preprocess(
                flatten_b_s(batch["vae_images"].to(device=device)))
            with torch.autocast(config["autocast_device_type"]):
                if global_step % 2 == 0:
                    # train discriminator
                    with torch.no_grad():
                        # fold encoding to save GPU memory
                        reconstructions = vae(
                            target, num_frames=sequence_length).sample

                    logits = discriminator_wrapper(
                        torch.cat([target, reconstructions.float()]))
                    loss = config["loss"]["discriminator_factor"] * \
                        taming.modules.losses.vqperceptual.hinge_d_loss(
                            *logits.chunk(2))
                else:
                    # use discriminator
                    reconstructions = vae(
                        target, num_frames=sequence_length).sample

                    reconstructions_float = reconstructions.float()
                    reconstruct_loss = torch.nn.functional.mse_loss(
                        target, reconstructions_float)
                    perceptual_loss = config["loss"]["perceptual_weight"] * \
                        lpips(target.to(dtype=torch.float16),
                              reconstructions).mean()

                    with torch.no_grad():
                        logits_fake = discriminator(
                            torch.cat([target, reconstructions_float]))

                    gan_loss = config["loss"]["discriminator_weight"] * \
                        config["loss"]["discriminator_factor"] * \
                        -torch.mean(logits_fake)
                    loss_1 = reconstruct_loss + perceptual_loss
                    loss = loss_1 + gan_loss
                    loss_list.append(loss_1.item())

            # update the model parameters
            should_optimize = ("gradient_accumulation_steps" not in config) or \
                ("gradient_accumulation_steps" in config and
                 (global_step + 1) % config["gradient_accumulation_steps"] == 0)
            if config["autocast_device_type"] == "cuda":
                grad_scaler.scale(loss).backward()
                if should_optimize:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if should_optimize:
                    optimizer.step()
                    optimizer.zero_grad()

            lr_scheduler.step()

            step_duration += time.time() - t0
            global_step += 1
            if should_log:

                # log
                if global_step % args.log_steps == 0:
                    loss_value = sum(loss_list) / len(loss_list)
                    print(
                        "Step {} ({:.1f} s/step), loss: {:.3f}".format(
                            global_step, step_duration / args.log_steps,
                            loss_value))
                    if should_save:
                        summary.add_scalar(
                            "train/Loss", loss_value, global_step)

                    loss_list.clear()
                    step_duration = 0

                # save step checkpoint
                if should_save and args.checkpointing_steps > 0 and \
                        global_step % args.checkpointing_steps == 0:
                    save_checkpoint(
                        vae, args.output_path, "{}.pth".format(global_step))

                # visualization
                if should_save and global_step % args.preview_steps == 0:
                    preview_image = test_pipeline(
                        vae, vae_image_processor, batch, config, device)
                    os.makedirs(
                        os.path.join(args.output_path, "preview"),
                        exist_ok=True)
                    preview_image.save(
                        os.path.join(
                            args.output_path,
                            "preview", "{}.png".format(global_step)))

        # save epoch checkpoint
        if should_log:
            print("Epoch {} done.".format(epoch))
            if should_save and args.checkpointing_steps == 0:
                save_checkpoint(
                    vae, args.output_path, "epoch-{}.pth".format(epoch))
