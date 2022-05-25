#!/bin/bash
GPUS=$1
# --transfer \
# --large_scale_jitter \
python -m torch.distributed.launch --nproc_per_node=$GPUS --use_env main.py \
    --pix2seq_lr \
    --rand_target \
    --model pix2seq \
    --epochs 300 \
    --input_size 600 \
    --backbone swin_T \
    --swin_path weights/swin_tiny_patch4_window7_224.pth \
    --batch_size 1 \
    --coco_path ./coco \
    --num_classes 90 \
    --maxdet 300 \
    --output_dir ./output/coco_v7
