#!/bin/bash
GPUS=$1
# --transfer \
python -m torch.distributed.launch --nproc_per_node=$GPUS --use_env main.py \
    --pix2seq_lr --large_scale_jitter --rand_target \
    --model pix2seq \
    --resume ./output/HRSC_4cls_v5/checkpoint.pth \
    --backbone swin_L \
    --swin_path weights/swin_large_patch4_window7_224_22k.pth \
    --batch_size 4 \
    --coco_path ./HRSC \
    --num_classes 4 \
    --output_dir ./output/HRSC_4cls_v5
