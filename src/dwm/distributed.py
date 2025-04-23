import os
import torch.distributed
import torch.distributed.checkpoint.state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def distributed_save_optimizer_state(model, optimizer, folder, filename):
    if torch.distributed.is_initialized():
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # DDP: rank 0 save
            if torch.distributed.get_rank() == 0:
                state_dict = optimizer.state_dict()
                torch.save(
                    state_dict,
                    os.path.join(folder, "{}.pth".format(filename)))

        elif isinstance(model, FSDP):
            # Zero2, Zero3: all ranks save
            # Hybird Zero2, Zero3: ranks in the 1st group save
            state_dict = torch.distributed.checkpoint.state_dict\
                .get_optimizer_state_dict(model, optimizer)
            if model._device_mesh is None or (
                torch.distributed.get_rank() in
                model._device_mesh.mesh[0].tolist()
            ):
                torch.distributed.checkpoint.state_dict_saver.save(
                    state_dict, checkpoint_id=os.path.join(folder, filename),
                    process_group=(
                        None if model._device_mesh is None
                        else model._device_mesh.get_group(mesh_dim=-1)
                    ))

        else:
            raise Exception(
                "Unsupported distribution framework to save the optimizer "
                "state.")

    else:
        state_dict = optimizer.state_dict()
        torch.save(state_dict, os.path.join(folder, "{}.pth".format(filename)))


def distributed_load_optimizer_state(model, optimizer, folder, filename):
    if torch.distributed.is_initialized():
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # DDP: all ranks load the same
            state_dict = torch.load(
                os.path.join(folder, "{}.pth".format(filename)),
                map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state_dict)
        elif isinstance(model, FSDP):
            # Zero2, Zero3: all ranks load
            # Hybird Zero2, Zero3: ranks in the 1st group load
            state_dict = torch.distributed.checkpoint.state_dict\
                .get_optimizer_state_dict(model, optimizer)
            torch.distributed.checkpoint.state_dict_loader.load(
                state_dict, checkpoint_id=os.path.join(folder, filename),
                planner=torch.distributed.checkpoint.DefaultLoadPlanner(
                    allow_partial_load=True))

    else:
        state_dict = torch.load(
            os.path.join(folder, "{}.pth".format(filename)),
            map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state_dict)
