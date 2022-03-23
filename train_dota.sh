#!/bin/bash
GPUS=$1

python -m torch.distributed.launch --nproc_per_node=$GPUS --use_env main.py \
    --pix2seq_lr --large_scale_jitter --rand_target \
    --model pix2seq \
    --resume ./output/dota_v6_2/checkpoint.pth \
    --transfer \
    --backbone swin_L \
    --swin_path weights/swin_large_patch4_window7_224_22k.pth \
    --batch_size 4 \
    --coco_path ./DOTA \
    --num_classes 2 \
    --output_dir ./output/dota_v6_2
