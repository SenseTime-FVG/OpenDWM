import argparse

import torch.distributed
import dwm.common
import einops
import json
import numpy as np
import os
import torch
import torchvision


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

    should_log = (ddp and local_rank == 0) or not ddp
    should_save = "RANK" not in os.environ or \
        ("RANK" in os.environ and os.environ["RANK"] == "0")

    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], ddp=ddp, should_save=should_save,
        output_path=args.output_path, config=config, device=device)
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

    batch_ind = 0
    select_index = []
    for a in range(3, 6):
        for b in range(6):
            select_index.append(a*6+b)

    for batch in validation_dataloader:
        paths = [
                os.path.join(args.output_path, k["filename"])
                for i in batch["sample_data"]
                for j in i
                for k in j if not k["filename"].endswith(".bin")
            ]
        flag = any([True if "samples" in j else False for j in paths[len(paths)//2:]])

        if flag:
            with torch.no_grad():
                pipeline_output = pipeline.inference_pipeline(batch, "pt")
                
            image_results = pipeline_output["images"]
            image_sizes = batch["image_size"].flatten(0, 2)
            for ind, (path, image, image_size) in enumerate(zip(paths, image_results, image_sizes)):
                if "samples" not in path:
                    continue
                if ind in select_index:
                    image = torchvision.transforms.functional.to_pil_image(image)
                    dir = os.path.dirname(path)
                    os.makedirs(dir, exist_ok=True)
                    if not os.path.exists(path):
                        image.resize(tuple(image_size.int().tolist()))\
                            .save(path, quality=95)
                    
        if torch.distributed.get_rank() == 0:
            print("batch_ind:", batch_ind)
        batch_ind += 1