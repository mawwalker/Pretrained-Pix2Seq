#!/bin/bash
GPUS=$1

# --large_scale_jitter \
python -m torch.distributed.launch --nproc_per_node=$GPUS --use_env main.py \
    --pix2seq_lr \
    --rand_target \
    --model pix2seq \
    --epochs 300 \
    --input_size 688 \
    --resume ./output/dota_all_v7_600/checkpoint.pth \
    --backbone swin_L \
    --swin_path weights/swin_large_patch4_window7_224_22k.pth \
    --batch_size 48 \
    --coco_path ./DOTA_all \
    --num_classes 17 \
    --maxdet 150 \
    --output_dir ./output/dota_all_v7_600 \
    --eval
