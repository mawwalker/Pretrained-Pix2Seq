#!/bin/bash
GPUS=$1
# --large_scale_jitter \
python -m torch.distributed.launch --nproc_per_node=$GPUS --use_env main.py \
    --pix2seq_lr \
    --rand_target \
    --model pix2seq \
    --input_size 800 \
    --resume ./output/HRSC_4cls_v7_1cls/checkpoint_best.pth \
    --transfer \
    --backbone swin_L \
    --swin_path weights/swin_large_patch4_window7_224_22k.pth \
    --batch_size 2 \
    --coco_path ./HRSC \
    --num_classes 2 \
    --output_dir ./output/HRSC_4cls_v7_1cls
