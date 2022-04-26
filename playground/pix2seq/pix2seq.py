# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Pix2Seq model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import IoU

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .transformer import build_transformer
from util.box_ops import box_cxcywh_to_xyxy, box_iou, get_wh
# from playground.box_iou_rotated_diff.box_iou_rotated_diff import box_iou_rotated_poly
from playground.box_iou_rotated_nn.box_iou_rotated_diff import box_iou_rotated_poly


class Pix2Seq(nn.Module):
    """ This is the Pix2Seq module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_bins=2000, input_size=1333):
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
        # self.input_proj = nn.Sequential(
        #     nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=(1, 1)),
        #     nn.GroupNorm(32, hidden_dim))
        self.backbone = backbone
        self.input_size = input_size

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
        features, poses = self.backbone(image_tensor)
        # print('features.shape: {}, pos.shape: {}'.format(features[0].tensors.shape, pos[0].shape))
        src, mask = features[-1].decompose()
        tensors = []
        masks = []
        for feature in features:
            tensor, mask = feature.decompose()
            tensors.append(tensor)
            masks.append(mask)
        assert mask is not None
        # mask = torch.zeros_like(mask).bool()

        # src = self.input_proj(src)
        if self.training:
            out = self.forward_train(tensors, targets, masks, poses)
        else:
            out = self.forward_inference(tensors, masks, poses)
        out = {'pred_seq_logits': out}
        return out

    def forward_train(self, src, targets, masks, poses):
        input_seq = self.build_input_seq(targets)
        similarity = self.transformer(src, input_seq, masks, poses)
        return similarity

    def forward_inference(self, src, masks, poses):
        out_seq = self.transformer(src, -1, masks, poses)
        return out_seq

    def build_input_seq(self, targets, max_objects=100):
        device = targets[0]["labels"].device
        input_seq_list = []
        for b_i, target in enumerate(targets):
            box = target["boxes"]
            label = target["labels"]
            img_size = target["size"]
            h, w = img_size[0], img_size[1]
            scale_factor = torch.stack([w, h, w, h, w, h, w, h], dim=0)

            label_token = label.unsqueeze(1) + self.num_bins + 1
            scaled_box = box * scale_factor
            # scaled_box = box_cxcywh_to_xyxy(scaled_box)
            box_tokens = (scaled_box / self.input_size * self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
            input_tokens = torch.cat([box_tokens, label_token], dim=1)

            num_objects = input_tokens.shape[0]
            num_noise = max_objects - num_objects

            random_class = torch.randint(0, self.num_classes, (num_noise, 1), device=device) + self.num_bins + 1
            random_box_x0y0 = torch.rand(num_noise, 2, device=device).clamp(min=0, max=1)
            random_box_x1y1 = torch.rand(num_noise, 2, device=device).clamp(min=0, max=1)
            random_box_x2y2 = torch.rand(num_noise, 2, device=device).clamp(min=0, max=1)
            random_box_x3y3 = torch.rand(num_noise, 2, device=device).clamp(min=0, max=1)
            # random_box_x1y1 = (random_box_x0y0 + random_box_wh).clamp(min=0, max=1)
            random_scaled_box = torch.cat([random_box_x0y0, random_box_x1y1, random_box_x2y2, random_box_x3y3], dim=1) * scale_factor
            random_box_tokens = (random_scaled_box / self.input_size * self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
            random_tokens = torch.cat([random_box_tokens, random_class], dim=1)

            if num_objects > 0 and num_noise > 0:
                jitter_box_idx = torch.randint(0, num_objects, (num_noise,), device=device)
                jitter_class = label_token[jitter_box_idx]
                jitter_box = box[jitter_box_idx]
                # jitter_box_wh = jitter_box[:, 2:].repeat(1, 2)
                jitter_box_wh = get_wh(jitter_box)
                # jitter_box = box_cxcywh_to_xyxy(jitter_box)
                jitter_box = (torch.rand((num_noise, 8), device=device) - 0.5) * 2 * 0.2 * jitter_box_wh + jitter_box
                scaled_jitter_box = jitter_box.clamp(min=0, max=1.0) * scale_factor
                jitter_box_tokens = (scaled_jitter_box / self.input_size * self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
                jitter_tokens = torch.cat([jitter_box_tokens, jitter_class], dim=1)

                fake_tokens = torch.stack([random_tokens, jitter_tokens], dim=1)
                select_idx = torch.randint(0, 2, (num_noise,), device=device)
                fake_tokens = fake_tokens[range(num_noise), select_idx]
            else:
                fake_tokens = random_tokens

            input_seq = torch.cat([input_tokens, fake_tokens], dim=0).flatten()
            input_seq_list.append(input_seq)
        return torch.stack(input_seq_list, dim=0)


class SoftArgmax1D(torch.nn.Module):
    """
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """
    def __init__(self, base_index=0, step_size=1):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....
        Assumes that the input to this layer will be a batch of 1D tensors (so a 2D tensor).
        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        """
        super(SoftArgmax1D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=-1)


    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:
        SoftArgMax(x) = \sum_i (i * softmax(x)_i)
        :param x: The input to the soft arg-max layer
        :return: Output of the soft arg-max layer
        """
        smax = self.softmax(x)
        end_index = self.base_index + x.size()[-1] * self.step_size
        indices = torch.arange(start=self.base_index, end=end_index, step=self.step_size).float().to(x.device)
        return torch.matmul(smax, indices)


class PostProcess_Train(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_bins, num_classes, input_size=1333):
        super().__init__()
        self.num_bins = num_bins
        self.num_classes = num_classes
        self.input_size = input_size

    def forward(self, outputs, targets):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            targets:
            # target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
            #               For evaluation, this must be the original image size (before any data augmentation)
            #               For visualization, this should be the image size after data augment, but before padding
        """
        origin_img_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        input_img_sizes = torch.stack([t["size"] for t in targets], dim=0)
        out_seq_logits = outputs['pred_seq_logits']

        assert len(out_seq_logits) == len(origin_img_sizes)

        # and from relative [0, 1] to absolute [0, height] coordinates
        ori_img_h, ori_img_w = origin_img_sizes.unbind(1)
        inp_img_h, inp_img_w = input_img_sizes.unbind(1)
        scale_fct = torch.stack(
            [ori_img_w / inp_img_w, ori_img_h / inp_img_h,
             ori_img_w / inp_img_w, ori_img_h / inp_img_h,
             ori_img_w / inp_img_w, ori_img_h / inp_img_h,
             ori_img_w / inp_img_w, ori_img_h / inp_img_h], dim=1).unsqueeze(1)

        results = []
        for b_i, pred_seq_logits in enumerate(out_seq_logits):
            seq_len = pred_seq_logits.shape[0]
            if seq_len < 9:
                results.append(dict())
                continue
            pred_seq_logits = pred_seq_logits.softmax(dim=-1)
            num_objects = seq_len // 9
            pred_seq_logits = pred_seq_logits[:int(num_objects * 9)].reshape(num_objects, 9, -1)
            pred_boxes_logits = pred_seq_logits[:, :8, :self.num_bins + 1]
            pred_class_logits = pred_seq_logits[:, 8, self.num_bins + 1: self.num_bins + 1 + self.num_classes]
            scores_per_image, labels_per_image = torch.max(pred_class_logits, dim=1)
            # boxes_per_image = pred_boxes_logits.argmax(dim=2) * self.input_size / self.num_bins
            boxes_per_image = SoftArgmax1D()(pred_boxes_logits) * self.input_size / self.num_bins
            boxes_per_image = boxes_per_image * scale_fct[b_i]
            result = dict()
            result['scores'] = scores_per_image
            result['labels'] = labels_per_image
            result['boxes'] = boxes_per_image
            results.append(result)

        return results


class IoULoss(nn.Module):
    def __init__(self, postprocessors, reduction='mean', weight=None, size_average=True):
        super(IoULoss, self).__init__()
        self.postprocessors = postprocessors
        self.reduction = reduction

    def forward(self, outputs, targets):
        gt_boxes = [target['boxes'] * torch.stack([target["size"][1], target["size"][0], 
                                                   target["size"][1], target["size"][0], 
                                                   target["size"][1], target["size"][0], 
                                                   target["size"][1], target["size"][0]], dim=0)
                    for target in targets]
        det_boxes = [output['boxes'] for output in self.postprocessors['bbox'](outputs, targets)]
        bs = len(gt_boxes)
        losses_IoU = []
        for i in range(bs):
            gt_box = gt_boxes[i]
            gt_box = gt_box.reshape(gt_box.shape[0], 4, 2)
            det_box = det_boxes[i][:gt_box.shape[0], ...]
            det_box = det_box.reshape(gt_box.shape[0], 4, 2)
            loss_giou =  (1 - torch.diag(box_iou_rotated_poly(gt_box, det_box)[2])).mean()
            losses_IoU.append(loss_giou)
        return torch.mean((torch.stack(losses_IoU)))


class SetCriterion(nn.Module):
    """
    This class computes the loss for Pix2Seq.
    """
    def __init__(self, num_classes, weight_dict, eos_coef, num_bins, num_vocal, input_size=1333):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_bins = num_bins
        self.num_vocal = num_vocal
        self.num_classes = num_classes
        empty_weight = torch.ones(self.num_vocal)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.weight_dict = weight_dict
        self.input_size = input_size
        self.postprocessors = {'bbox': PostProcess_Train(num_bins, num_classes, input_size)}
        self.iou_loss_M = IoULoss(self.postprocessors)

    def build_target_seq(self, targets, max_objects=100):
        device = targets[0]["labels"].device
        target_seq_list = []
        for target in targets:
            label = target["labels"]
            box = target["boxes"]
            img_size = target["size"]
            h, w = img_size[0], img_size[1]

            label = label.unsqueeze(1) + self.num_bins + 1
            box = box * torch.stack([w, h, w, h, w, h, w, h], dim=0)
            # box = box_cxcywh_to_xyxy(box)
            box = (box / self.input_size * self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
            target_tokens = torch.cat([box, label], dim=1).flatten()

            end_token = torch.tensor([self.num_vocal - 2], dtype=torch.int64).to(device)

            num_noise = max_objects - len(label)
            fake_target_tokens = torch.zeros((num_noise, 9), dtype=torch.int64).to(device)
            fake_target_tokens[:, :7] = -100
            fake_target_tokens[:, 7] = self.num_vocal - 1  # noise class
            fake_target_tokens[:, 8] = self.num_vocal - 2  # eos
            fake_target_tokens = fake_target_tokens.flatten()

            target_seq = torch.cat([target_tokens, end_token, fake_target_tokens], dim=0)
            target_seq_list.append(target_seq)

        return torch.stack(target_seq_list, dim=0)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        target_seq = self.build_target_seq(targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_pos = (target_seq > -1).sum()
        num_pos = torch.as_tensor([num_pos], dtype=torch.float, device=target_seq.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pos)
        num_pos = torch.clamp(num_pos / get_world_size(), min=1).item()

        pred_seq_logits = outputs['pred_seq_logits']

        if isinstance(pred_seq_logits, list) and not self.training:
            num_pred_seq = [len(seq) for seq in pred_seq_logits]
            pred_seq_logits = torch.cat(pred_seq_logits, dim=0).reshape(-1, self.num_vocal)
            target_seq = torch.cat(
                [t_seq[:p_seq] for t_seq, p_seq in zip(target_seq, num_pred_seq)], dim=0)
        else:
            pred_seq_logits = pred_seq_logits.reshape(-1, self.num_vocal)
            target_seq = target_seq.flatten()

        loss_seq = F.cross_entropy(pred_seq_logits, target_seq, weight=self.empty_weight, reduction='sum') / num_pos

        iou_loss = self.iou_loss_M(outputs, targets)
        # Compute all the requested losses
        losses = dict()
        losses["loss_seq"] = loss_seq
        losses['iou_loss'] = iou_loss
        return losses

def isfakebox(box):
    border_flag = 0
    for p in box:
        if abs(int(p) - 1333) < 20 or int(p) < 20:
            border_flag += 1
    if border_flag == 8:
        return True
    else:
        return False

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_bins, num_classes, input_size=1333):
        super().__init__()
        self.num_bins = num_bins
        self.num_classes = num_classes
        self.input_size = input_size

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            targets:
            # target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
            #               For evaluation, this must be the original image size (before any data augmentation)
            #               For visualization, this should be the image size after data augment, but before padding
        """
        origin_img_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        input_img_sizes = torch.stack([t["size"] for t in targets], dim=0)
        out_seq_logits = outputs['pred_seq_logits']

        assert len(out_seq_logits) == len(origin_img_sizes)

        # and from relative [0, 1] to absolute [0, height] coordinates
        ori_img_h, ori_img_w = origin_img_sizes.unbind(1)
        inp_img_h, inp_img_w = input_img_sizes.unbind(1)
        scale_fct = torch.stack(
            [ori_img_w / inp_img_w, ori_img_h / inp_img_h,
             ori_img_w / inp_img_w, ori_img_h / inp_img_h,
             ori_img_w / inp_img_w, ori_img_h / inp_img_h,
             ori_img_w / inp_img_w, ori_img_h / inp_img_h], dim=1).unsqueeze(1)

        results = []
        for b_i, pred_seq_logits in enumerate(out_seq_logits):
            seq_len = pred_seq_logits.shape[0]
            if seq_len < 9:
                results.append(dict())
                continue
            pred_seq_logits = pred_seq_logits.softmax(dim=-1)
            num_objects = seq_len // 9
            pred_seq_logits = pred_seq_logits[:int(num_objects * 9)].reshape(num_objects, 9, -1)
            pred_boxes_logits = pred_seq_logits[:, :8, :self.num_bins + 1]
            pred_class_logits = pred_seq_logits[:, 8, self.num_bins + 1: self.num_bins + 1 + self.num_classes]
            scores_per_image, labels_per_image = torch.max(pred_class_logits, dim=1)
            boxes_per_image = pred_boxes_logits.argmax(dim=2) * self.input_size / self.num_bins
            boxes_per_image = boxes_per_image * scale_fct[b_i]
            result = dict()
            result['scores'] = scores_per_image
            result['labels'] = labels_per_image
            result['boxes'] = boxes_per_image
            results.append(result)
            # result['scores'] = []
            # result['labels'] = []
            # result['boxes'] = []
            # for score, cls, box in zip(scores_per_image.detach().cpu().numpy(),
                                       # labels_per_image.detach().cpu().numpy(),
                                       # boxes_per_image.detach().cpu().numpy()):
                # box = box.tolist()
                # if not isfakebox(box):
                    # result['scores'].append(score)
                    # result['labels'].append(cls)
                    # result['boxes'].append(box)
                    # # print('box: ', box)
                # else:
                    # break
                # results.append(result)
            # print("result: ", results)
        return results


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    num_classes = args.num_classes + 1
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    num_bins = 2000
    # 0 - num_bin coordinate, num_bin+1 - num_bin+num_class class,
    # num_bin+num_class+1 end, num_bin+num_class+2 noise
    num_vocal = num_bins + 1 + num_classes + 2

    transformer = build_transformer(args, num_vocal)

    model = Pix2Seq(
        backbone,
        transformer,
        num_classes=num_classes,
        num_bins=num_bins,
        input_size=args.input_size)

    # weight_dict = {'loss_seq': 1, 'mse': 1, 'iou_loss': 1}
    weight_dict = {'loss_seq': 1, 'iou_loss': 2}
    # weight_dict = {'loss_seq': 1}
    criterion = SetCriterion(
        num_classes,
        weight_dict,
        eos_coef=args.eos_coef,
        num_bins=num_bins,
        num_vocal=num_vocal,
        input_size=args.input_size)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_bins, num_classes, args.input_size)}

    return model, criterion, postprocessors
