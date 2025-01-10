export PYTHONPATH="$PWD/src"
DWM_NAME=your_process_name
INPUT_CONFIG=scripts/liuyichen/bevw_vae/bevw_lidar_vae_nusc_full.json
OUTPUT_DIR=/your/output/directory/$DWM_NAME

WORLD_SIZE=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29600

mkdir -p $OUTPUT_DIR
cp $INPUT_CONFIG $OUTPUT_DIR

python -m torch.distributed.run \
--nnodes $WORLD_SIZE \
--nproc-per-node 2 \
--node-rank $RANK \
--master-addr $MASTER_ADDR \
--master-port $MASTER_PORT \
/mnt/storage/user/liuyichen/repo/DWM/src/dwm/train.py \
-c $INPUT_CONFIG \
-o $OUTPUT_DIR \
--wandb \
--wandb-project BEVW_AE \
--wandb-run-name $DWM_NAME \
--checkpointing-steps 100 \
--evaluation-steps 1 \
--log-steps 2 \
--resume-from 51000


# python -m torch.distributed.run \
# --nnodes $WORLD_SIZE \
# --nproc-per-node 2 \
# --node-rank $RANK \
# --master-addr $MASTER_ADDR \
# --master-port $MASTER_PORT \
# /mnt/storage/user/liuyichen/repo/DWM/src/dwm/evaluate.py \
# -c $INPUT_CONFIG \
# -o $OUTPUT_DIR \
# --resume-from 51000



# export PYTHONPATH="$PWD/src"
# DWM_NAME=bevw_vae_nusc_full-lidar_vae-diffusers_backbone-no_render
# INPUT_CONFIG=/mnt/storage/user/liuyichen/repo/DWM/scripts/lyc_noupload/bevw_vae_nusc_full_lidar.json
# OUTPUT_DIR=/mnt/storage/user/liuyichen/tasks/$DWM_NAME

# WORLD_SIZE=1
# RANK=0
# MASTER_ADDR=127.0.0.1
# MASTER_PORT=29500

# mkdir -p $OUTPUT_DIR
# cp $INPUT_CONFIG $OUTPUT_DIR
# chown -R liuyichen $OUTPUT_DIR

# python -m torch.distributed.run \
# --nnodes $WORLD_SIZE \
# --nproc-per-node 8 \
# --node-rank $RANK \
# --master-addr $MASTER_ADDR \
# --master-port $MASTER_PORT \
# /mnt/storage/user/liuyichen/repo/DWM/src/dwm/train.py \
# -c $INPUT_CONFIG \
# -o $OUTPUT_DIR \
# --wandb \
# --wandb-project BEVW_AE \
# --wandb-run-name $DWM_NAME \
# --checkpointing-steps 4000 \
# --evaluation-steps 4000 \
# --log-step 20