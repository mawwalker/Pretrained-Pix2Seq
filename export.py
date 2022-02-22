# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
# from models import build_model
from playground import build_all_model
from playground.pix2seq.pix2seq import PostProcess
import datasets.transforms as T
from util import box_ops
from timm.utils import NativeScaler

from PIL import Image
from PIL import ImageDraw,ImageFont
import cv2
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import nested_tensor_from_tensor_list

from playground.pix2seq.backbone import build_backbone
from playground.pix2seq.transformer import build_transformer
from util.box_ops import box_cxcywh_to_xyxy


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--amp_train', action='store_true', help='amp fp16 training or not')
    parser.add_argument('--eval_epoch', default=5, type=int)

    # Pix2Seq
    parser.add_argument('--model', type=str, default="pix2seq",
                        help="specify the model from playground")
    parser.add_argument('--pix2seq_lr', action='store_true', help='use warmup linear drop lr')
    parser.add_argument('--large_scale_jitter', action='store_true', help='large scale jitter')
    parser.add_argument('--rand_target', action='store_true',
                        help="randomly permute the sequence of input targets")
    parser.add_argument('--pred_eos', action='store_true', help='use eos token instead of predicting 100 objects')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # * Loss coefficients
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=32, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--num_classes', default=90, type=int, help='max ID of the datasets')
    
    parser.add_argument('--img_path', default='', type=str, help='the path to predict')
    return parser


class Pix2Seq(nn.Module):
    """ This is the Pix2Seq module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_bins=2000):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_bins: number of bins for each side of the input image
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.input_proj = nn.Sequential(
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=(1, 1)),
            nn.GroupNorm(32, hidden_dim))
        self.backbone = backbone

    def forward(self, samples):
        """ 
            samples[0]:
            The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            samples[1]:
                targets
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all vocabulary.
                                Shape= [batch_size, num_sequence, num_vocal]
        """
        image_tensor, targets = samples[0], samples[1]
        if isinstance(image_tensor, (list, torch.Tensor)):
            image_tensor = nested_tensor_from_tensor_list(image_tensor)
        features, pos = self.backbone(image_tensor)

        src, mask = features[-1].decompose()
        assert mask is not None
        mask = torch.zeros_like(mask).bool()

        src = self.input_proj(src)
        out = self.forward_inference(src, mask, pos[-1])

        out = {'pred_seq_logits': out}
        return out

    def forward_inference(self, src, mask, pos):
        out_seq = self.transformer(src, -1, mask, pos)
        return out_seq


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    num_classes = args.num_classes + 1
    backbone = build_backbone(args)

    num_bins = 2000
    num_vocal = num_bins + 1 + num_classes + 2

    transformer = build_transformer(args, num_vocal)
    
    model = Pix2Seq(
        backbone,
        transformer,
        num_classes=num_classes,
        num_bins=num_bins)
    # model, criterion, postprocessors = build_all_model[args.model](args)
    model.to(device)
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location=device, check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])

    #read image
    image = Image.open(args.img_path)
    image = image.convert('RGB')
    
    w_ori,h_ori = image.size
    # image = np.array(image).astype(np.uint8)
    
    #transform
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform = T.Compose([
        T.RandomResize([800], max_size=512),
        normalize,
    ])

    image_new = transform(image,None)

    c,h,w = image_new[0].shape
    image_new = image_new[0].view(1,c,h,w).to(device)
    seq = torch.ones(1, 1).to(device,dtype=torch.long) * 2001
    model.eval()

    print(image_new.shape)
    # get predictions
    # print('input_seq: {}'.format(seq.shape))
    # output = model([image_new,seq])
    # out_seq_logits = output['pred_seq_logits']
    input_names = ["input"]
    output_names = ["output"]
    # torch.onnx.export(model, 
    #               [image_new, seq],
    #               "pix2seq.onnx",
    #               verbose=True,
    #               opset_version=12,
    #               input_names=input_names,
    #               output_names=output_names,
    #               export_params=True,
    #               )
    torch.onnx.export(model, [image_new, seq], "pix2seq.onnx", verbose=True, opset_version=12,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True, export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                      input_names=input_names,
                      output_names=output_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pix2Seq training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
