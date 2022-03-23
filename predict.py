# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse

import numpy as np
import torch
from playground import build_all_model
import datasets.transforms as T
from PIL import Image
import cv2
import os
torch.set_grad_enabled(False);


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Pix2Seq
    parser.add_argument('--model', type=str, default="pix2seq",
                        help="specify the model from playground")
    parser.add_argument('--pix2seq_lr', action='store_true', help='use warmup linear drop lr')
    parser.add_argument('--large_scale_jitter', action='store_true', help='large scale jitter')
    parser.add_argument('--rand_target', action='store_true',
                        help="randomly permute the sequence of input targets")
    parser.add_argument('--pred_eos', action='store_true', help='use eos token instead of predicting 100 objects')

    # * Backbone
    parser.add_argument('--backbone', default='swin_L', type=str,
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
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='output/dota_v6_2/checkpoint_best.pth', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=2, type=int)
    
    parser.add_argument('--num_classes', default=2, type=int, help='max ID of the datasets')
    
    parser.add_argument('--img_path', default='./DOTA/train2017/P0023__1.0__1000___3918.png', type=str, help='the path to predict')
    parser.add_argument('--swin_path', default='./weights/swin_large_patch4_window7_224_22k.pth', help='resume from swin transformer')
    parser.add_argument('--activation', default='relu', help='transformer activation function')
    parser.add_argument('--input_size', default=1333, type=int, help='max ID of the datasets')
    parser.add_argument('--need_attn', default=False, action='store_true',
                        help='if return the deformable attention weights, for visualization only')
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
        
def plot_one_box_polyl(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    points = np.array(x).reshape(4, 2).reshape(4, 1, 2).astype(int)
    # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    cv2.polylines(im, [points], True, color, line_thickness)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (int(x[0]), int(x[1]) - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def isfakebox(box):
    border_flag = 0
    for p in box:
        if abs(int(p) - 1333) < 20 or int(p) < 20:
            border_flag += 1
    if border_flag == 8:
        return True
    else:
        return False

def PostProcess(args, origin_path, output, names, h_ori, w_ori, h, w):
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
             ori_img_w / inp_img_w, ori_img_h / inp_img_h,
             ori_img_w / inp_img_w, ori_img_h / inp_img_h,
             ori_img_w / inp_img_w, ori_img_h / inp_img_h], dim=1).unsqueeze(1).to(args.device)
    results = []
    image = cv2.imread(origin_path)
    for b_i, pred_seq_logits in enumerate(out_seq_logits):
        # print('pred_seq_logits'.format(pred_seq_logits))
        seq_len = pred_seq_logits.shape[0]
        if seq_len < 9:
            results.append(dict())
            continue
        pred_seq_logits = pred_seq_logits.softmax(dim=-1)
        num_objects = seq_len // 9
        pred_seq_logits = pred_seq_logits[:int(num_objects * 9)].reshape(num_objects, 9, -1)
        pred_boxes_logits = pred_seq_logits[:, :8, :num_bins + 1]
        pred_class_logits = pred_seq_logits[:, 8, num_bins + 1: num_bins + 1 + num_classes]
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
            if score > 0.25 and not isfakebox(box):
                result['scores'].append(score)
                result['labels'].append(cls)
                result['boxes'].append(box)
                # print('box: ', box)
                colors = Colors()
                c = int(cls - 1)
                plot_one_box_polyl(box, image, label=names[c], color=colors(c, True), line_thickness=3)
            else:
                break
        if len(result['boxes']) > 0:
            cv2.imwrite('./predict_results/' + os.path.basename(origin_path), image)
        results.append(result)
    print(results)
    return results

def main(args):
    names = ['QHJ', 'XYJ', 'DLJ', 'YSJ', 'LGJ', 'HKMJ', 'ZHJ', 'QT', 'HC', 'KC', 'BZJ', 'YLC', 'ship']
    names = ['ship', 'aircraft carrier', 'warcraft', 'merchant ship']
    names = ['plane']
    
    device = torch.device(args.device)

    model, _, _ = build_all_model[args.model](args)
    
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    model.to(device)
    model.eval()
    predict_dir = './DOTA/train2017/'
    for file in os.listdir(path=predict_dir):
        file_path = os.path.join(predict_dir, file)
        #read image
        image = Image.open(file_path)
        image = image.convert('RGB')
        
        w_ori, h_ori = image.size
        print(image.size)
        # image = np.array(image).astype(np.uint8)
        
        #transform
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform = T.Compose([
            # T.RandomResize([800], max_size=1333),
            normalize,
        ])

        image_new = transform(image, None)

        c,h,w = image_new[0].shape
        print(c, h, w)
        image_new = image_new[0].view(1, c, h, w).to(device)
        seq = torch.ones(1, 1).to(device,dtype=torch.long) * 2001

        # get predictions
        # print('input_seq: {}'.format(seq.shape))
        output = model([image_new, seq])
        # print('output: {}'.format(output))
        results = PostProcess(args, file_path, output, names, h_ori, w_ori, h, w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pix2Seq training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
