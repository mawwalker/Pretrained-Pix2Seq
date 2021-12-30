#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --coco_path ./coco --num_classes 12 --pix2seq_lr --large_scale_jitter --rand_target $@
