#!/usr/bin/env bash
set -x
T=`date +%Y%m%d_%H%M%S`

work_dir=/mnt/afs/user/nijingcheng/workspace/codes/sup_codes3/DWM/src/dwm
env='/mnt/afs/user/nijingcheng/miniconda3/envs/mmagic/bin'
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export LD_LIBRARY_PATH=/mnt/cache/share/cuda-11.8/lib64:$PATH
export PATH=/mnt/cache/share/cuda-11.8/bin:$PATH
export PYTHONPATH=/mnt/afs/user/nijingcheng/workspace/codes/sup_codes3/DWM/src:$PYTHONPATH
export PYTHONPATH=/mnt/afs/user/wuzehuan/Documents/DWM/externals/pytorch-fid/src:$PYTHONPATH

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

GPUS=$1
config=$2
PY_ARGS=${@:3}
$env/torchrun --nproc_per_node=$GPUS --nnodes=$NNODES --node_rank=$NODE_RANK \
  --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR \
  /mnt/afs/user/nijingcheng/workspace/codes/sup_codes3/DWM/scripts/nijingcheng/evaluate_simple.py -c ${config} ${PY_ARGS}