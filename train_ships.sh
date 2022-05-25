#!/bin/bash
GPUS=$1
# --large_scale_jitter \
# --resume ./pix2seq_swin_3.pth \
# --transfer \
python -m torch.distributed.launch --nproc_per_node=$GPUS --use_env main.py \
    --pix2seq_lr \
    --large_scale_jitter \
    --rand_target \
    --model pix2seq \
    --lr 5e-5 \
    --input_size 800 \
    --backbone swin_L \
    --swin_path weights/swin_large_patch4_window7_224_22k.pth \
    --batch_size 1 \
    --coco_path ./HRSC \
    --num_classes 2 \
    --output_dir ./output/HRSC_1cls_v7_800
