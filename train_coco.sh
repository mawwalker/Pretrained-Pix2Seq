#!/bin/bash
GPUS=$1
# --transfer \
python -m torch.distributed.launch --nproc_per_node=$GPUS --use_env main.py \
    --pix2seq_lr --large_scale_jitter --rand_target \
    --model pix2seq \
    --resume ./output/coco_v6/checkpoint.pth \
    --backbone swin_L \
    --swin_path weights/swin_large_patch4_window7_224_22k.pth \
    --batch_size 4 \
    --coco_path ./coco \
    --num_classes 90 \
    --output_dir ./output/coco_v6
