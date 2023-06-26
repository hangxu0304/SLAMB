#!/bin/bash

OPTIMIZER="lamb"
#OPTIMIZER="slamb"
#OPTIMIZER="adamw"

DATADIR="./imagenet1k"

num_gpus = 8
num_nodes = 1
node_rank = 0
master_addr = "127.0.0.1"
master_port = 1234

CMD=" ./main.py"
CMD+=" --cfg ./configs/swin/swin_base_patch4_window7_224.yaml"
CMD+=" --data-path $DATADIR --batch-size 128 --output=./output"
CMD+=" --optim=$OPTIMIZER"

CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus --nnodes=$num_nodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port $CMD"

$CMD
