import argparse
import dwm.common
import json
import os
import torch


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to finetune a stable diffusion model to the "
        "driving dataset.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config to load the train model and dataset.")
    parser.add_argument(
        "-o", "--output-path", type=str, default=None,
        help="The path to save checkpoint files.")
    parser.add_argument(
        "--log-steps", default=100, type=int,
        help="The step count to print log and update the tensorboard.")
    parser.add_argument(
        "--preview-steps", default=400, type=int,
        help="The step count to preview the pipeline result.")
    parser.add_argument(
        "--evaluation-steps", default=1000, type=int,
        help="The step count to preview the pipeline result.")
    parser.add_argument(
        "--checkpointing-steps", default=4000, type=int,
        help="The step count to save the checkpoint.")
    parser.add_argument(
        "--resume-from", default=None, type=int,
        help="The step to resume from")
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

    # load the pipeline including the models
    # As resume is implemented by pipeline (for avoiding multiple loading), config["resume_from"] is defined
    assert "resume_from" not in config, f"Conflict for define resume_from multiple times."
    config["resume_from"] = args.resume_from
    if args.output_path is None:
        args.output_path = config["output_path"]

    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], ddp=ddp, should_save=should_save,
        output_path=args.output_path, config=config, device=device)
    if should_log:
        print("The pipeline is loaded.")
        from mmengine.runner import set_random_seed
        set_random_seed(seed=2022)
        print("Set random seed to 2022")

    validation_dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    # print("Debug filter ...")
    # validation_dataset.base_dataset.filter_indexs(
    #     "/mnt/afs/user/nijingcheng/workspace/codes/sup_codes2/data/uniad/ids_v1.pkl",
    #     dict(
    #         cross = ['intersection', 'crossing', 'crossroads'],
    #         bridge = ['bridge', 'overpass', 'viaduct']
    #     )
    # )
    # import pdb; pdb.set_trace()
    if ddp:
        # make equal sample count for each process to simplify the result
        # gathering
        total_batch_size = int(os.environ["WORLD_SIZE"]) * \
            config["validation_dataloader"]["batch_size"]
        dataset_length = len(validation_dataset) // \
            total_batch_size * total_batch_size
        validation_dataset = torch.utils.data.Subset(
            validation_dataset, range(0, dataset_length))
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

    pipeline.evaluate_pipeline(
        should_save, 0, len(validation_dataset), validation_dataloader,
        validation_datasampler)
