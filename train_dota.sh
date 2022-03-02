#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --coco_path ./DOTA --num_classes 15 --pix2seq_lr --large_scale_jitter --rand_target $@
