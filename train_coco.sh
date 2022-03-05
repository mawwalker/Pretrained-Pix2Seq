#!/bin/bash
GPUS=$1
python -m torch.distributed.launch --nproc_per_node=$GPUS --use_env main.py \
    --pix2seq_lr --large_scale_jitter --rand_target \
    --model pix2seq \
    --resume ./coco_ap370.pth \
    --backbone swin_L \
    --transfer \
    --swin_path weights/swin_large_patch4_window7_224_22k.pth \
    --batch_size 4 \
    --coco_path ./coco \
    --num_classes 90 \
    --output_dir ./output/coco_v5
