# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import random
import time

import numpy as np
import torch
import util.misc as utils
from playground import build_all_model
import datasets.transforms as T
from PIL import Image
import cv2


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
    parser.add_argument('--backbone', default='swin', type=str,
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
    parser.add_argument('--coco_path', default='./coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='output/ships_v2/checkpoint_best.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--num_classes', default=12, type=int, help='max ID of the datasets')
    
    parser.add_argument('--img_path', default='/home/dsm/Datasets/ships/val2017/100001000.jpg', type=str, help='the path to predict')
    parser.add_argument('--swin_path', default='/home/dsm/graduate/Pretrained-Pix2Seq/weights/swin_tiny_patch4_window7_224.pth', help='resume from swin transformer')
    return parser

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    names = ['QHJ', 'XYJ', 'DLJ', 'YSJ', 'LGJ', 'HKMJ', 'ZHJ', 'QT', 'HC', 'KC', 'BZJ', 'YLC', 'ship']
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_all_model[args.model](args)
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model.to(device)
    #read image
    image = Image.open(args.img_path)
    image = image.convert('RGB')
    
    w_ori, h_ori = image.size
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

    image_new = transform(image, None)

    c,h,w = image_new[0].shape
    print(c, h, w)
    image_new = image_new[0].view(1, c, h, w).to(device)
    seq = torch.ones(1, 1).to(device,dtype=torch.long) * 2001
    model.eval()

    # get predictions
    # print('input_seq: {}'.format(seq.shape))
    output = model([image_new,seq])
    # print('output: {}'.format(output))
    out_seq_logits = output['pred_seq_logits']
    
    orig_size = torch.as_tensor([int(h_ori), int(w_ori)])
    size = torch.as_tensor([int(h), int(w)])
    origin_img_sizes = torch.stack([orig_size], dim=0)
    input_img_sizes = torch.stack([size], dim=0)
    ori_img_h, ori_img_w = origin_img_sizes.unbind(1)
    inp_img_h, inp_img_w = input_img_sizes.unbind(1)
    num_bins = 2000
    num_classes = args.num_classes
    scale_fct = torch.stack(
            [ori_img_w / inp_img_w, ori_img_h / inp_img_h,
             ori_img_w / inp_img_w, ori_img_h / inp_img_h], dim=1).unsqueeze(1).to(device)
    results = []
    image = cv2.imread(args.img_path)
    for b_i, pred_seq_logits in enumerate(out_seq_logits):
        # print('pred_seq_logits'.format(pred_seq_logits))
        seq_len = pred_seq_logits.shape[0]
        if seq_len < 5:
            results.append(dict())
            continue
        pred_seq_logits = pred_seq_logits.softmax(dim=-1)
        num_objects = seq_len // 5
        pred_seq_logits = pred_seq_logits[:int(num_objects * 5)].reshape(num_objects, 5, -1)
        pred_boxes_logits = pred_seq_logits[:, :4, :num_bins + 1]
        pred_class_logits = pred_seq_logits[:, 4, num_bins + 1: num_bins + 1 + num_classes]
        # print(pred_class_logits)
        scores_per_image, labels_per_image = torch.max(pred_class_logits, dim=1)
        boxes_per_image = pred_boxes_logits.argmax(dim=2) * 1333 / num_bins
        boxes_per_image = boxes_per_image * scale_fct[b_i]
        result = dict()
        result['scores'] = []
        result['labels'] = []
        result['boxes'] = []
        for score, cls, box in zip(scores_per_image.detach().cpu().numpy(),
                                         labels_per_image.detach().cpu().numpy(),
                                         boxes_per_image.detach().cpu().numpy()):
            box = box.tolist()
            if score > 0.25:
                result['scores'].append(score)
                result['labels'].append(cls)
                result['boxes'].append(box)
                # print('box: ', box)
                colors = Colors()
                c = int(cls)
                plot_one_box(box, image, label=names[c], color=colors(c, True), line_thickness=3)
            else:
                break
        cv2.imwrite('./result.jpg', image)
        results.append(result)
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pix2Seq training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
