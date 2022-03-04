# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Pix2Seq Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention_layer import Attention

# Self-attention with 2D relative position encoding
from .rpe_attention import RPEMultiheadAttention, irpe
from .deformable_attn import DeformableHeadAttention, generate_ref_points

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False, num_vocal=2094,
                 pred_eos=False, scales=1, k=4, last_height=16, last_width=16):
        super().__init__()
        rpe_config = irpe.get_rpe_config(
                        ratio=1.9,
                        method="product",
                        mode='ctx',
                        shared_head=True,
                        skip=0,
                        rpe_on='k',
                    )
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, k=k, scales=scales,
            last_feat_height=last_height,
            last_feat_width=last_width, 
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, normalize_before=normalize_before,
            rpe_config=rpe_config
            )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, k=k, scales=scales,
            last_feat_height=last_height,
            last_feat_width=last_width, 
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, normalize_before=normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self._reset_parameters()

        self.num_vocal = num_vocal
        self.vocal_classifier = nn.Linear(d_model, num_vocal)
        self.det_embed = nn.Embedding(1, d_model)
        self.vocal_embed = nn.Embedding(self.num_vocal - 2, d_model)
        self.pred_eos = pred_eos

        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.query_ref_point_proj = nn.Linear(d_model, 2)
        self.ref_vocal_classifier = nn.Linear(2, num_vocal)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, input_seq, masks, pos_embeds):
        """
        Args:
            src: shape[B, C, H, W]
            input_seq: shape[B, 501, C] for training and shape[B, 1, C] for inference
            mask: shape[B, H, W]
            pos_embed: shape[B, C, H, W]
        """
        # flatten NxCxHxW to HWxNxC
        # bs = src.shape[0]
        bs, c, h, w = src[0].shape
        
        # B, C H, W -> B, H, W, C
        for index in range(len(src)):
            # src[index] = src[index].flatten(2).permute(2, 0, 1)
            # pos_embeds[index] = pos_embeds[index].flatten(2).permute(2, 0, 1)
            
            src[index] = src[index].permute(0, 2, 3, 1)
            pos_embeds[index] = pos_embeds[index].permute(0, 2, 3, 1)
            
        # B, H, W, C
        ref_points = []
        for tensor in src:
            _, height, width, _ = tensor.shape
            ref_point = generate_ref_points(width=width,
                                            height=height)
            ref_point = ref_point.type_as(src[0])
            # H, W, 2 -> B, H, W, 2
            ref_point = ref_point.unsqueeze(0).repeat(bs, 1, 1, 1)
            ref_points.append(ref_point)

        # src = src.flatten(2).permute(2, 0, 1)
        # mask = mask.flatten(1)
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        memory = self.encoder(src, ref_points, src_key_padding_masks=masks, poses=pos_embeds, hw=(h, w))
        # for index in range(len(memory)):
        #     memory[index] = memory[index].flatten(1, 2).permute(1, 0, 2)
        #     masks[index] = masks[index].flatten(1)
        #     pos_embeds[index] = pos_embeds[index].flatten(1, 2).permute(1, 0, 2)
        pre_kv = [torch.as_tensor([[], []], device=memory[0].device)
                  for _ in range(self.num_decoder_layers)]

        if self.training:
            input_embed = torch.cat(
                [self.det_embed.weight.unsqueeze(0).repeat(bs, 1, 1),
                 self.vocal_embed(input_seq)], dim=1)
            input_embed = input_embed.transpose(0, 1)
            num_seq = input_embed.shape[0]
            self_attn_mask = torch.triu(torch.ones((num_seq, num_seq)), diagonal=1).bool(). \
                to(input_embed.device)
            # L, B, C
            query_ref_point = self.query_ref_point_proj(input_embed)
            query_ref_point = torch.sigmoid(query_ref_point)
            hs, pre_kv = self.decoder(
                input_embed,
                memory,
                query_ref_point,
                memory_key_padding_masks=masks,
                poses=pos_embeds,
                pre_kv_list=pre_kv,
                self_attn_mask=self_attn_mask)
            pred_seq_logits = self.vocal_classifier(hs.transpose(0, 1))
            query_ref_point = - torch.log(1 / (query_ref_point + 1e-10) - 1 + 1e-10)
            query_ref_point = self.ref_vocal_classifier(query_ref_point.transpose(0, 1))
            pred_seq_logits = pred_seq_logits + query_ref_point
            return pred_seq_logits
        else:
            end = torch.zeros(bs).bool().to(memory[0].device)
            end_lens = torch.zeros(bs).long().to(memory[0].device)
            input_embed = self.det_embed.weight.unsqueeze(0).repeat(bs, 1, 1).transpose(0, 1)
            pred_seq_logits = []
            # L, B, C
            query_ref_point = self.query_ref_point_proj(input_embed)
            query_ref_point = torch.sigmoid(query_ref_point)
            for seq_i in range(500):
                hs, pre_kv = self.decoder(
                    input_embed,
                    memory,
                    query_ref_point,
                    memory_key_padding_masks=masks,
                    poses=pos_embeds,
                    pre_kv_list=pre_kv)
                similarity = self.vocal_classifier(hs)
                pred_seq_logits.append(similarity.transpose(0, 1))

                if self.pred_eos:
                    is_eos = similarity[:, :, :self.num_vocal - 1].argmax(dim=-1)
                    stop_state = is_eos.squeeze(0).eq(self.num_vocal - 2)
                    end_lens += seq_i * (~end * stop_state)
                    end = (stop_state + end).bool()
                    if end.all() and seq_i > 4:
                        break

                pred_token = similarity[:, :, :self.num_vocal - 2].argmax(dim=-1)
                input_embed = self.vocal_embed(pred_token)

            if not self.pred_eos:
                end_lens = end_lens.fill_(500)
            pred_seq_logits = torch.cat(pred_seq_logits, dim=1)
            query_ref_point = - torch.log(1 / (query_ref_point + 1e-10) - 1 + 1e-10)
            query_ref_point = self.ref_vocal_classifier(query_ref_point.transpose(0, 1))
            pred_seq_logits = pred_seq_logits + query_ref_point
            pred_seq_logits = [psl[:end_idx] for end_idx, psl in zip(end_lens, pred_seq_logits)]
            return pred_seq_logits


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, ref_points,
                src_key_padding_masks: Optional[Tensor] = None,
                poses: Optional[Tensor] = None,
                hw=None):
        outputs = src

        for layer in self.layers:
            outputs = layer(outputs, ref_points, src_key_padding_masks=src_key_padding_masks, poses=poses, hw=hw)

        if self.norm is not None:
            for index, output in enumerate(outputs):
                outputs[index] = self.norm(output)
            # output = self.norm(output)

        return outputs


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, ref_point, memory_key_padding_masks, poses, pre_kv_list=None, self_attn_mask=None):
        output = tgt
        cur_kv_list = []
        for layer, pre_kv in zip(self.layers, pre_kv_list):
            output, cur_kv = layer(
                output,
                memory,
                ref_point,
                memory_key_padding_masks=memory_key_padding_masks,
                poses=poses,
                self_attn_mask=self_attn_mask,
                pre_kv=pre_kv)
            cur_kv_list.append(cur_kv)

        if self.norm is not None:
            output = self.norm(output)

        return output, cur_kv_list


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead,
                 k: int,
                 scales: int,
                 last_feat_height: int,
                 last_feat_width: int,
                 need_attn: bool = False,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 rpe_config=None):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = RPEMultiheadAttention(
        #     d_model, nhead, dropout=dropout, rpe_config=rpe_config)
        self.ms_deformbale_attn = DeformableHeadAttention(h=nhead,
                                                          d_model=d_model,
                                                          k=k,
                                                          scales=scales,
                                                          last_feat_height=last_feat_height,
                                                          last_feat_width=last_feat_width,
                                                          dropout=dropout,
                                                          need_attn=need_attn)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
        self.need_attn = need_attn
        self.attns = []

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src_tensors, ref_points,
                     src_key_padding_masks: Optional[Tensor] = None,
                     poses: Optional[Tensor] = None,
                     hw=None):

        if src_key_padding_masks is None:
            src_key_padding_masks = [None] * len(src_tensors)

        if poses is None:
            poses = [None] * len(src_tensors)
        # q = k = self.with_pos_embed(src, pos)
        # src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask, hw=hw)[0]
        feats = []
        src_tensors = [self.with_pos_embed(tensor, pos) for tensor, pos in zip(src_tensors, poses)]
        for src, ref_point, src_key_padding_mask, pos in zip(src_tensors,
                                                            ref_points,
                                                            src_key_padding_masks,
                                                            poses):
            # src = self.with_pos_embed(src, pos)
            # print('src_tensors length: {}'.format(len(src_tensors)))
            src2, attns = self.ms_deformbale_attn(src,
                                                  src_tensors,
                                                  ref_point,
                                                  query_mask=src_key_padding_mask,
                                                  key_masks=src_key_padding_masks)

            if self.need_attn:
                self.attns.append(attns)

            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            feats.append(src)

        # src = src + self.dropout1(src2)
        # src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return feats

    def forward_pre(self, src_tensors, ref_points,
                    src_key_padding_masks: Optional[Tensor] = None,
                    poses: Optional[Tensor] = None,
                    hw=None):
        if src_key_padding_masks is None:
            src_key_padding_masks = [None] * len(src_tensors)

        if poses is None:
            poses = [None] * len(src_tensors)

        feats = []

        src_tensors = [self.with_pos_embed(tensor, pos) for tensor, pos in zip(src_tensors, poses)]
        for src, ref_point, src_key_padding_mask, pos in zip(src_tensors,
                                                             ref_points,
                                                             src_key_padding_masks,
                                                             poses):
            src2 = self.norm1(src, pos)
            # src2 = self.with_pos_embed(src2, pos)
            src2, attns = self.ms_deformbale_attn(src2, src_tensors, ref_point, query_mask=src_key_padding_mask)

            if self.need_attn:
                self.attns.append(attns)

            src = src + self.dropout1(src2)
            src2 = self.norm2(src)

            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
            feats.append(src)
        
        # src2 = self.norm1(src)
        # q = k = self.with_pos_embed(src2, pos)
        # src2 = self.self_attn(q, k, value=src2, key_padding_mask=src_key_padding_mask, hw=hw)[0]
        # src = src + self.dropout1(src2)
        # src2 = self.norm2(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # src = src + self.dropout2(src2)
        return feats

    def forward(self, src_tensors, ref_points,
                src_key_padding_masks: Optional[Tensor] = None,
                poses: Optional[Tensor] = None,
                hw=None):
        if self.normalize_before:
            return self.forward_pre(src_tensors, ref_points, src_key_padding_masks, poses, hw=hw)
        return self.forward_post(src_tensors, ref_points, src_key_padding_masks, poses, hw=hw)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead,
                 k: int,
                 scales: int,
                 last_feat_height: int,
                 last_feat_width: int,
                 need_attn=False,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ms_deformbale_attn = DeformableHeadAttention(h=nhead,
                                                          d_model=d_model,
                                                          k=k,
                                                          scales=scales,
                                                          last_feat_height=last_feat_height,
                                                          last_feat_width=last_feat_width,
                                                          dropout=dropout,
                                                          need_attn=need_attn)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.need_attn = need_attn
        self.attns = []

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            tgt,
            memory,
            ref_point,
            memory_key_padding_masks: Optional[Tensor] = None,
            poses: Optional[Tensor] = None,
            self_attn_mask: Optional[Tensor] = None,
            pre_kv=None,
    ):
        tgt2, pre_kv = self.self_attn(tgt, pre_kv=pre_kv, attn_mask=self_attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # tgt2 = self.multihead_attn(
        #     query=tgt,
        #     key=self.with_pos_embed(memory, pos),
        #     value=memory,
        #     key_padding_mask=memory_key_padding_mask,
        # )[0]
        memory = [self.with_pos_embed(tensor, pos) for tensor, pos in zip(memory, poses)]

        # L, B, C -> B, L, 1, C
        tgt = tgt.transpose(0, 1).unsqueeze(dim=2)
        ref_point = ref_point.transpose(0, 1).unsqueeze(dim=2)

        # B, L, 1, C
        tgt2, attns = self.ms_deformbale_attn(tgt,
                                              memory,
                                              ref_point,
                                              query_mask=None,
                                              key_masks=memory_key_padding_masks)

        if self.need_attn:
            self.attns.append(attns)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        # B, L, 1, C -> L, B, C
        tgt = tgt.squeeze(dim=2)
        tgt = tgt.transpose(0, 1).contiguous()
        return tgt, pre_kv

    def forward_pre(
            self,
            tgt,
            memory,
            ref_point,
            memory_key_padding_masks: Optional[Tensor] = None,
            poses: Optional[Tensor] = None,
            self_attn_mask: Optional[Tensor] = None,
            pre_kv=None,
    ):
        tgt2 = self.norm1(tgt)
        tgt2, pre_kv = self.self_attn(tgt2, pre_kv=pre_kv, attn_mask=self_attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        # tgt2 = self.multihead_attn(
        #     query=tgt2,
        #     key=self.with_pos_embed(memory, pos),
        #     value=memory,
        #     key_padding_mask=memory_key_padding_mask,
        # )[0]
        
        memory = [self.with_pos_embed(tensor, pos) for tensor, pos in zip(memory, poses)]

        # L, B, C -> B, L, 1, C
        tgt2 = tgt2.transpose(0, 1).unsqueeze(dim=2)
        ref_point = ref_point.transpose(0, 1).unsqueeze(dim=2)

        # B, L, 1, 2
        tgt2, attns = self.ms_deformbale_attn(tgt2, memory, ref_point,
                                              query_mask=None,
                                              key_masks=memory_key_padding_masks)
        if self.need_attn:
            self.attns.append(attns)

        # B, L, 1, C -> L, B, C
        tgt2 = tgt2.squeeze(dim=2)
        tgt2 = tgt2.transpose(0, 1).contiguous()

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, pre_kv

    def forward(
            self,
            tgt,
            memory,
            ref_point,
            memory_key_padding_masks: Optional[Tensor] = None,
            poses: Optional[Tensor] = None,
            self_attn_mask: Optional[Tensor] = None,
            pre_kv=None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, ref_point, memory_key_padding_masks, poses, self_attn_mask, pre_kv)
        return self.forward_post(tgt, memory, ref_point, memory_key_padding_masks, poses, self_attn_mask, pre_kv)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args, num_vocal):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        activation=args.activation,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        num_vocal=num_vocal,
        pred_eos=args.pred_eos,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
