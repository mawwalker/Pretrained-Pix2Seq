#!/bin/bash
GPUS=$1

# --large_scale_jitter \
python -m torch.distributed.launch --nproc_per_node=$GPUS --use_env main.py \
    --pix2seq_lr \
    --rand_target \
    --model pix2seq \
    --epochs 300 \
    --input_size 800 \
    --resume ./output/dota_v6_2_new/checkpoint_240_472.pth \
    --transfer \
    --backbone swin_L \
    --swin_path weights/swin_large_patch4_window7_224_22k.pth \
    --batch_size 1 \
    --coco_path ./DOTA \
    --num_classes 2 \
    --output_dir ./output/dota_v6_2_new
