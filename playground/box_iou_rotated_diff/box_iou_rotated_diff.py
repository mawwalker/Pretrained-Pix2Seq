"""
Differentiable IoU calculation for rotated boxes
Most of the code is adapted from https://github.com/lilanxiao/Rotated_IoU
"""
import torch
from .box_intersection_2d import oriented_box_intersection_2d
from .box_area_2d import oriented_box_area_2d


def rotated_box_to_poly(rotated_boxes: torch.Tensor):
    """ Transform rotated boxes to polygons
    Args:
        rotated_boxes (Tensor): (x, y, w, h, a) with shape (n, 5)
    Return:
        polys (Tensor): 4 corner points (x, y) of polygons with shape (n, 4, 2)
    """
    cs = torch.cos(rotated_boxes[:, 4])
    ss = torch.sin(rotated_boxes[:, 4])
    w = rotated_boxes[:, 2] - 1
    h = rotated_boxes[:, 3] - 1

    x_ctr = rotated_boxes[:, 0]
    y_ctr = rotated_boxes[:, 1]
    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)

    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)

    polys = torch.stack([x1, y1, x2, y2, x3, y3, x4, y4], dim=-1)
    polys = polys.reshape(-1, 4, 2)  # to (n, 4, 2)

    return polys


def box_iou_rotated_differentiable(boxes1: torch.Tensor, boxes2: torch.Tensor, iou_only: bool = True):
    """Calculate IoU between rotated boxes

    Args:
        box1 (torch.Tensor): (n, 5)
        box2 (torch.Tensor): (n, 5)
        iou_only: Whether to keep other vars, e.g., polys, unions. Default True to drop these vars.

    Returns:
        iou (torch.Tensor): (n, )
        polys1 (torch.Tensor): (n, 4, 2)
        polys2 (torch.Tensor): (n, 4, 2)
        U (torch.Tensor): (n) area1 + area2 - inter_area
    """
    # transform to polygons
    polys1 = rotated_box_to_poly(boxes1)
    polys2 = rotated_box_to_poly(boxes2)
    # calculate insection areas
    inter_area, _ = oriented_box_intersection_2d(polys1, polys2)
    area1 = boxes1[..., 2] * boxes1[..., 3]
    area2 = boxes2[..., 2] * boxes2[..., 3]
    union = area1 + area2 - inter_area
    iou = inter_area / union
    if iou_only:
        return iou
    else:
        return iou, union, polys1, polys2,


def box_iou_rotated_poly(polys1: torch.Tensor, polys2: torch.Tensor, iou_only: bool = False):
    """Calculate IoU between rotated boxes

    Args:
        polys1 (torch.Tensor): (n, 4, 2)
        polys2 (torch.Tensor): (n, 4, 2)
        iou_only: Whether to keep other vars, e.g., polys, unions. Default True to drop these vars.

    Returns:
        iou (torch.Tensor): (n, )
        polys1 (torch.Tensor): (n, 4, 2)
        polys2 (torch.Tensor): (n, 4, 2)
        U (torch.Tensor): (n) area1 + area2 - inter_area
    """
    # calculate insection areas
    inter_area, _ = oriented_box_intersection_2d(polys1.unsqueeze(1), polys2)
    area1, _ = oriented_box_area_2d(polys1, polys1)
    area2, _ = oriented_box_area_2d(polys2, polys2)
    n = polys1.size(0)
    m = polys2.size(0)
    xmin = torch.cat([polys1.unsqueeze(1).repeat(1, m, 1, 1)[..., 0], polys2.unsqueeze(0).repeat(n, 1, 1, 1)[..., 0]], dim=-1).min(-1)[0]
    xmax = torch.cat([polys1.unsqueeze(1).repeat(1, m, 1, 1)[..., 0], polys2.unsqueeze(0).repeat(n, 1, 1, 1)[..., 0]], dim=-1).max(-1)[0]
    ymin = torch.cat([polys1.unsqueeze(1).repeat(1, m, 1, 1)[..., 1], polys2.unsqueeze(0).repeat(n, 1, 1, 1)[..., 1]], dim=-1).min(-1)[0]
    ymax = torch.cat([polys1.unsqueeze(1).repeat(1, m, 1, 1)[..., 1], polys2.unsqueeze(0).repeat(n, 1, 1, 1)[..., 1]], dim=-1).max(-1)[0]
    areac = (xmax - xmin) * (ymax - ymin)
    inter_area = torch.stack([area1.unsqueeze(1).repeat(1, m), area2.unsqueeze(0).repeat(n, 1), inter_area]).min(0)[0]
    union = area1.unsqueeze(1).repeat(1, m) + area2.unsqueeze(0).repeat(n, 1) - inter_area
    iou = inter_area / (union + 1e-2)
    if iou_only:
        return iou
    else:
        # iou, iou_loss, giou_loss
        return iou, 1 - iou, 1. - iou + torch.abs((areac - union) / (areac + 1e-2))