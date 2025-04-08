# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Implementation of DUSt3R training losses
# --------------------------------------------------------
from copy import copy, deepcopy
import torch
import torch.nn as nn

from slam3r.utils.geometry import inv, geotrf, depthmap_to_pts3d, multiview_normalize_pointcloud

def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    elif 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d

def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class LLoss (nn.Module):
    """ L-norm loss
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim-1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss()


class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, LLoss), f'{criterion} is not a proper criterion!'
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = 'none'  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details

class Jointnorm_Regr3D (Criterion, MultiLoss):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
        gt and pred are transformed into localframe1
    """

    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False, dist_clip=None):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.dist_clip = dist_clip

    def get_all_pts3d(self, gts, preds, ref_id, in_camera=None, norm_scale=None, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        # print(ref_id)
        if in_camera is None:
            in_camera = inv(gts[ref_id]['camera_pose'])
        gt_pts = []
        valids = []
        for gt in gts:
            gt_pts.append(geotrf(in_camera, gt['pts3d']))
            valids.append(gt['valid_mask'].clone())
            
        dist_clip = self.dist_clip if dist_clip is None else dist_clip
        if dist_clip is not None:
            # points that are too far-away == invalid
            for i in range(len(gts)):
                dis = gt_pts[i].norm(dim=-1)
                valids[i] = valids[i] & (dis<dist_clip)
        pred_pts = []
        if isinstance(ref_id, int):
            for i in range(len(preds)):
                pred_pts.append(get_pred_pts3d(gts[i], preds[i], use_pose=(i!=ref_id)))
        else:
            for i in range(len(preds)):
                pred_pts.append(get_pred_pts3d(gts[i], preds[i], use_pose=not (i in ref_id)))
        
        if norm_scale is not None:
            if isinstance(norm_scale, tuple):
                gt_pts = [(gt_pts[i]-norm_scale[0]) / norm_scale[1] for i in range(len(gt_pts))]
                # print('loss', norm_scale)
            else:
                # print(gt_pts[0].shape, norm_scale.shape) 
                gt_pts = [gt_pts[i] / norm_scale for i in range(len(gt_pts))]
                # print("norm_scale", norm_scale)
        else:
            # normalize 3d points(见原论文3D regression loss)
            if self.norm_mode:
                pred_pts = multiview_normalize_pointcloud(pred_pts, self.norm_mode, valids)
            if self.norm_mode and not self.gt_scale:
                gt_pts = multiview_normalize_pointcloud(gt_pts, self.norm_mode, valids)
        return gt_pts, pred_pts, valids, {}

    def compute_loss(self, gts, preds, ref_id=None, head='', ref_camera=None, norm_scale=None, **kw):
        if not isinstance(ref_id, int):
            assert ref_camera is not None
        gt_pts, pred_pts, valids, monitoring = \
            self.get_all_pts3d(gts, preds, ref_id=ref_id, 
                               in_camera=ref_camera, norm_scale=norm_scale, **kw)
        all_l = []
        details = {}
        for i in range(len(gts)):
            l1 = self.criterion(pred_pts[i][valids[i]], gt_pts[i][valids[i]])
            # self_name = type(self).__name__
            self_name = "Regr3D"
            if head != '':
                self_name = self_name + '_' + head
            details[self_name+f'_pts3d_{i+1}'] = float(l1.mean())
            # print(l1.shape)  #(valid_num,)
            all_l.append((l1,valids[i]))

        return Sum(*all_l), (details | monitoring)


class Jointnorm_ConfLoss (MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gts, preds, head='', **kw):
            # compute per-pixel loss
        losses_and_masks, details = self.pixel_loss(gts, preds, head=head, **kw)
        for i in range(len(losses_and_masks)):
            if losses_and_masks[i][0].numel() == 0:
                print(f'NO VALID POINTS in img{i+1}', force=True)

        res_loss = 0
        res_info = details
        for i in range(len(losses_and_masks)):
            loss = losses_and_masks[i][0]
            mask = losses_and_masks[i][1]
            conf, log_conf = self.get_conf_log(preds[i]['conf'][mask])
            conf_loss = loss * conf - self.alpha * log_conf
            conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0
            res_loss += conf_loss
            info_name = f"conf_loss_{i+1}" if head == '' else f"conf_loss_{head}_{i+1}"
            res_info[info_name] = float(conf_loss)

        return res_loss, res_info
