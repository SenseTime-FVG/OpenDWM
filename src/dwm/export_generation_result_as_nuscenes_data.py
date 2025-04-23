import argparse
import dwm.common
import json
import numpy as np
import os
import torch


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to run the diffusion model to generate data for"
        "detection evaluation.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config to load the train model and dataset.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path to save checkpoint files.")
    return parser


"""
This script requires:
  1. The validation dataset should be only nuScenes.
  2. Add `"enable_sample_data": true` to the dataset arguments, so the Dataset
     load the filename (path) to save the generated images.
  3. Add `"sample_data"` to "validation_dataloader.collate_fn.keys" to pass the
     object data directly to the script here, no need to collate to tensors.

Note:
  *. Set the "model_checkpoint_path" with the trained checkpoint, rather than
     the pretrained checkpoint.
  *. Set the fps_stride to [0, 1] for the image dataset, or
     [2, 0.5 * sequence_length] for the origin video dataset, or
     [12, 0.1 * sequence_length] for the 12Hz video dataset. Make sure no
     sample is missed.
"""


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

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

    # setup the global state
    if "global_state" in config:
        for key, value in config["global_state"].items():
            dwm.common.global_state[key] = \
                dwm.common.create_instance_from_config(value)

    should_log = (ddp and local_rank == 0) or not ddp

    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=args.output_path, config=config,
        device=device)
    if should_log:
        print("The pipeline is loaded.")

    # load the dataset
    validation_dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    if ddp:
        validation_datasampler = \
            torch.utils.data.distributed.DistributedSampler(
                validation_dataset)
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"]),
            sampler=validation_datasampler)
    else:
        validation_datasampler = None
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"]))

    if should_log:
        print("The validation dataset is loaded with {} items.".format(
            len(validation_dataset)))

    if ddp:
        validation_datasampler.set_epoch(0)

    for batch in validation_dataloader:
        batch_size, sequence_length, view_count = batch["vae_images"].shape[:3]
        latent_height = batch["vae_images"].shape[-2] // \
            (2 ** (len(pipeline.vae.config.down_block_types) - 1))
        latent_width = batch["vae_images"].shape[-1] // \
            (2 ** (len(pipeline.vae.config.down_block_types) - 1))
        latent_shape = (
            batch_size, sequence_length, view_count,
            pipeline.vae.config.latent_channels, latent_height,
            latent_width
        )

        with torch.no_grad():
            pipeline_output = pipeline.inference_pipeline(
                latent_shape, batch, "pil")

        if "images" in pipeline_output:
            paths = [
                os.path.join(args.output_path, k["filename"])
                for i in batch["sample_data"]
                for j in i
                for k in j if not k["filename"].endswith(".bin")
            ]
            image_results = pipeline_output["images"]
            image_sizes = batch["image_size"].flatten(0, 2)
            for path, image, image_size in zip(paths, image_results, image_sizes):
                dir = os.path.dirname(path)
                os.makedirs(dir, exist_ok=True)
                image.resize(tuple(image_size.int().tolist()))\
                    .save(path, quality=95)

        if "raw_points" in pipeline_output:
            paths = [
                os.path.join(args.output_path, k["filename"])
                for i in batch["sample_data"]
                for j in i
                for k in j if k["filename"].endswith(".bin")
            ]
            raw_points = [
                j
                for i in pipeline_output["raw_points"]
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
