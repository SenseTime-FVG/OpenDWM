import argparse
import dwm.common
import json
import os
import torch
from dwm.utils.sampler import VariableVideoBatchSampler


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
        "--checkpointing-steps", default=10000, type=int,
        help="The step count to save the checkpoint.")
    parser.add_argument(
        "--evaluation-steps", default=10000, type=int,
        help="The step count to preview the pipeline result.")
    parser.add_argument(
        "--resume-from", default=None, type=int,
        help="The step to resume from")
    parser.add_argument(
        "--wandb", action="store_true",
        help="Use wandb to log the training process.")
    parser.add_argument(
        "--wandb-project", type=str, default="dwm",
        help="The wandb project name.")
    parser.add_argument(
        "--wandb-run-name", type=str, default="train",
        help="The wandb run name.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    torch.manual_seed(config["generator_seed"])

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
    should_save = not torch.distributed.is_initialized() or \
        torch.distributed.get_rank() == 0

    # load the pipeline including the models
    output_path = config["output_path"] if args.output_path is None else args.output_path
    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=output_path, config=config,
        device=device, resume_from=args.resume_from)

    if should_log:
        print("The pipeline is loaded.")

    if args.wandb and should_save:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)

    # load the dataset
    training_dataset = dwm.common.create_instance_from_config(
        config["training_dataset"])
    validation_dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    if ddp:

        if "mix_config" in config.keys():
            process_group = torch.distributed.group.WORLD

            training_datasampler = VariableVideoBatchSampler(
                training_dataset,
                config["mix_config"],
                num_replicas=process_group.size(),
                rank=process_group.rank(),
                shuffle=config["data_shuffle"],
                seed=config["generator_seed"]
            )

            training_dataloader = torch.utils.data.DataLoader(
                training_dataset,
                **dwm.common.instantiate_config(config["training_dataloader"]),
                batch_sampler=training_datasampler)

        else:
            training_datasampler = torch.utils.data.distributed.DistributedSampler(
                training_dataset, shuffle=config["data_shuffle"],
                seed=config["generator_seed"])
            training_dataloader = torch.utils.data.DataLoader(
                training_dataset,
                **dwm.common.instantiate_config(config["training_dataloader"]),
                sampler=training_datasampler)

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
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            **dwm.common.instantiate_config(config["training_dataloader"]),
            shuffle=config["data_shuffle"])
        validation_datasampler = None
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"]))

    preview_dataloader = torch.utils.data\
        .DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["preview_dataloader"])) if \
        "preview_dataloader" in config else None
    if preview_dataloader is not None:
        preview_data_iterator = iter(preview_dataloader)

    if should_log:
        print("The training dataset is loaded with {} items.".format(
            len(training_dataset)))
        print("The validation dataset is loaded with {} items.".format(
            len(validation_dataset)))

    # train loop
    global_step = 0 if args.resume_from is None else args.resume_from
    for epoch in range(config["train_epochs"]):

        if ddp:
            # Fixing training data order reduces the accessed objects per rank,
            # therefore reduces the upper-bound of memory usage comsumed by the
            # Python reference counting of objects.
            sampler_epoch = 0 if config.get("fix_training_data_order", False) \
                else epoch
            training_datasampler.set_epoch(sampler_epoch)

        for batch in training_dataloader:
            pipeline.train_step(batch, global_step)
            global_step += 1

            # log
            if global_step % args.log_steps == 0:
                pipeline.log(global_step, args.log_steps)

            # preview
            if global_step % args.preview_steps == 0:
                if preview_dataloader is None:
                    pipeline.preview_pipeline(batch, output_path, global_step)
                else:
                    try:
                        preview_batch = next(preview_data_iterator)
                    except StopIteration:
                        preview_data_iterator = iter(preview_dataloader)
                        preview_batch = next(preview_data_iterator)

                    pipeline.preview_pipeline(
                        preview_batch, output_path, global_step)

            # save step checkpoint
            if global_step % args.checkpointing_steps == 0:
                pipeline.save_checkpoint(output_path, global_step)

            # evaluation
            if (
                args.evaluation_steps > 0 and
                global_step % args.evaluation_steps == 0
            ):
                pipeline.evaluate_pipeline(
                    global_step, len(validation_dataset),
                    validation_dataloader, validation_datasampler)

        if should_log:
            print("Epoch {} done.".format(epoch))

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
