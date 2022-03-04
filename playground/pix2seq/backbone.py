# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
from numpy import not_equal

import torch
from torch import nn
from typing import Dict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .swin_transformer import SwinPatch

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, backbone_name, train_backbone: bool,
                 num_channels: int, return_interm_layers: bool, hidden_dim: int = 256):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        # print('backbone name: {}'.format(backbone_name))
        if backbone_name.startswith('swin'):
            # for swin transformer
            self.body = backbone
        else:
            # for ResNet
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.backbone_name = backbone_name
        self.num_channels = num_channels

        # self.input_proj1 = nn.Sequential(
        #             nn.Conv2d(num_channels//4, hidden_dim, kernel_size=(1, 1)),
        #             nn.GroupNorm(32, hidden_dim))
        # self.input_proj2 = nn.Sequential(
        #             nn.Conv2d(num_channels//2, hidden_dim, kernel_size=(1, 1)),
        #             nn.GroupNorm(32, hidden_dim))
        self.input_proj3 = nn.Sequential(
                    nn.Conv2d(num_channels, hidden_dim, kernel_size=(1, 1)),
                    nn.GroupNorm(32, hidden_dim))
        # self.input_proj4 = nn.Sequential(
        #             nn.Conv2d(num_channels, hidden_dim, kernel_size=(3, 3), stride=2),
        #             nn.GroupNorm(32, hidden_dim))

    def forward(self, tensor_list: NestedTensor):
        xx = self.body(tensor_list.tensors)
        if self.backbone_name.startswith('swin'):
            # for swin transformer
            xs = OrderedDict()
            # swin-T transformer 4 layers tuple() type
            # layer3.shape: (1, 768, 12, 16)
            # xs['0'] = xx[1]
            # xs['1'] = xx[2]
            xs['2'] = xx[3]
            # print('xs[0].shape: {}, xs[1].shape: {}, xs[2].shape: {}'.format(xs['0'].shape, xs['1'].shape, xs['2'].shape))
        else:
            xs = xx
        m = tensor_list.mask
        assert m is not None
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            scale_map = self.input_proj3(x)
            mask = F.interpolate(m[None].float(), size=scale_map.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(scale_map, mask)
            
        #     if name == '0':
        #         scale_map = self.input_proj1(x)
        #     elif name == '1':
        #         scale_map = self.input_proj2(x)
        #     else:
        #         scale_map = self.input_proj3(x)
        #     mask = F.interpolate(m[None].float(), size=scale_map.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(scale_map, mask)

        # c4 = self.input_proj4(xs['2'])
        # mask = F.interpolate(m[None].float(), size=c4.shape[-2:]).to(torch.bool)[0]
        # out['3'] = NestedTensor(c4, mask)

        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str, swin_path: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        if name.startswith('swin'):
            backbone = getattr(SwinPatch, name)(patch_size=4,
                                                in_chans=3,
                                                window_size=7)
            if swin_path != '':
                backbone.init_weights(swin_path)
        else:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        if name in ('swin_T', 'swin_S'):
            num_channels = 768
        elif name == 'swin_B':
            num_channels = 1024
        elif name == 'swin_L':
            num_channels = 1536
        super().__init__(backbone, name, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = False
    backbone = Backbone(args.backbone, args.swin_path, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
