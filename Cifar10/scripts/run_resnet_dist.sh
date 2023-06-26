#!/bin/bash

DATADIR="./cifar10"
num_gpus = 4

CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus ./main.py --data-dir=$DATADIR -a=resnet110 --batch-size=256 --lr=0.03 --optimizer=SGD"

CMD2="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus ./main.py --data-dir=$DATADIR -a=resnet110 --batch-size=256 --lr=0.01 --optimizer=LAMB"

CMD3="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus ./main.py --data-dir=$DATADIR -a=resnet110 --batch-size=256 --lr=0.01 --optimizer=SLAMB --compress_ratio=0.1 --beta3=0.99"

$CMD


