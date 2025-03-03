export PYTHONPATH="$PWD/src"

DWM_NAME=lidar_maskgit_nusc_full-align-line
OUTPUT_DIR=/mnt/storage/user/liuyichen/tasks/$DWM_NAME
INPUT_CONFIG=/mnt/storage/user/liuyichen/tasks/lidar_maskgit_nusc_full-align-line/lidar_maskgit_nusc_mini-align-line.json

mkdir -p $OUTPUT_DIR


WORLD_SIZE=1
RANK=0
MASTER_ADDR=127.0.0.1
NNODES=2
MASTER_PORT=29000

# python3 -m torch.distributed.run \
# --nnodes $WORLD_SIZE \
# --nproc-per-node $NNODES \
# --node-rank $RANK \
# --master-addr $MASTER_ADDR \
# --master-port $MASTER_PORT \
python /mnt/storage/user/liuyichen/repo/DWM/src/dwm/train.py \
-c $INPUT_CONFIG \
-o $OUTPUT_DIR \
--wandb-project MASKGIT_LIDAR \
--wandb-run-name $DWM_NAME \
--log-steps 1 \
--preview-steps 1 \
--resume-from 66000
