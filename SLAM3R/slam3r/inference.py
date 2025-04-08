# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import torch
import numpy as np

from .utils.misc import invalid_to_zeros
from .utils.geometry import geotrf, inv


def loss_of_one_batch(loss_func, batch, model, criterion, device, 
                      use_amp=False, ret=None, 
                      assist_model=None, train=False, epoch=0,
                      args=None):
    if loss_func == "i2p":
        return loss_of_one_batch_multiview(batch, model, criterion, 
                                           device, use_amp, ret, 
                                           args.ref_id)
    elif loss_func == "i2p_corr_score":
        return loss_of_one_batch_multiview_corr_score(batch, model, criterion, 
                                                      device, use_amp, ret, 
                                                      args.ref_id)
    elif loss_func == "l2w":
        return loss_of_one_batch_l2w(
                        batch, model, criterion, 
                        device, use_amp, ret, 
                        ref_ids=args.ref_ids, coord_frame_id=0, 
                        exclude_ident=True, to_zero=True
                    )
    else:
        raise NotImplementedError            
    
    
def loss_of_one_batch_multiview(batch, model, criterion, device, 
                                use_amp=False, ret=None, ref_id=-1):
    views = batch
    for view in views:
        for name in 'img pts3d valid_mask camera_pose'.split():  # pseudo_focal
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if ref_id == -1:
        ref_id = (len(views)-1)//2

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        preds = model(views, ref_id=ref_id)
        assert len(preds) == len(views)
        
        with torch.cuda.amp.autocast(enabled=False):
            if criterion is None:
                loss = None
            else:
                loss = criterion(views, preds, ref_id=ref_id)
    
    result = dict(views=views, preds=preds, loss=loss)
    for i in range(len(preds)):
        result[f'pred{i+1}'] = preds[i]
        result[f'view{i+1}'] = views[i]
    return result[ret] if ret else result


def loss_of_one_batch_multiview_corr_score(batch, model, criterion, device, 
                                use_amp=False, ret=None, ref_id=-1):
    views = batch
    for view in views:
        for name in 'img pts3d valid_mask camera_pose'.split():  # pseudo_focal
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)
            
    if ref_id == -1:
        ref_id = (len(views)-1)//2

    all_loss = [0, {}]
    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        preds = model(views, ref_id=ref_id, return_corr_score=True)
        assert len(preds) == len(views)
        for i,pred in enumerate(preds):
            if i == ref_id:
                continue
            patch_pseudo_conf = pred['pseudo_conf']  # (B,S)
            true_conf = (pred['conf']-1.).mean(dim=(1,2))   # (B,)  mean(exp(x))
            pseudo_conf = torch.exp(patch_pseudo_conf).mean(dim=1) # (B,)  mean(exp(batch(x)))
            pseudo_conf = pseudo_conf / (1+pseudo_conf)
            true_conf = true_conf / (1+true_conf)
            dis = torch.abs(pseudo_conf-true_conf)
            loss = dis.mean()
            # if loss.isinf():
            #     print(((patch_pseudo_conf-patch_true_conf)**2).max())
            all_loss[0] += loss
            all_loss[1][f'pseudo_conf_loss_{i}'] = loss
    
    result = dict(views=views, preds=preds, loss=all_loss)
    for i in range(len(preds)):
        result[f'pred{i+1}'] = preds[i]
        result[f'view{i+1}'] = views[i]
    return result[ret] if ret else result


def get_multiview_scale(pts:list, valid:list, norm_mode='avg_dis'):
    # adpat from DUSt3R
    for i in range(len(pts)):
        assert pts[i].ndim >= 3 and pts[i].shape[-1] == 3
    assert len(pts) == len(valid)
    norm_mode, dis_mode = norm_mode.split('_')

    if norm_mode == 'avg':
        # gather all points together (joint normalization)
        all_pts = []
        all_nnz = 0
        for i in range(len(pts)):
            nan_pts, nnz = invalid_to_zeros(pts[i], valid[i], ndim=3) 
            # print(nnz,nan_pts.shape) #(B,) (B,H*W,3)
            all_pts.append(nan_pts)
            all_nnz += nnz
        all_pts = torch.cat(all_pts, dim=1)
        # compute distance to origin
        all_dis = all_pts.norm(dim=-1) 
        if dis_mode == 'dis':
            pass  # do nothing
        elif dis_mode == 'log1p':
            all_dis = torch.log1p(all_dis)
        else:
            raise ValueError(f'bad {dis_mode=}')

        norm_factor = all_dis.sum(dim=1) / (all_nnz + 1e-8)    
    else: 
        raise ValueError(f'bad {norm_mode=}')

    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts[0].ndim:
        norm_factor.unsqueeze_(-1)
    # print('norm factor:', norm_factor)
    return norm_factor


def loss_of_one_batch_l2w(batch, model, criterion, device, 
                                use_amp=False, ret=None, 
                                ref_ids=-1, coord_frame_id=0, 
                                exclude_ident=True, to_zero=True):
    views = batch
    for view in views:
        for name in 'img pts3d pts3d_cam valid_mask camera_pose'.split():  # pseudo_focal
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)
    
    if coord_frame_id == -1:
        # ramdomly select a camera as the target camera
        coord_frame_id = np.random.randint(0, len(views))
    # print(coord_frame_id)
    c2w = views[coord_frame_id]['camera_pose']  
    w2c = inv(c2w) 

    if exclude_ident:
        views.pop(coord_frame_id)
    
    if ref_ids == -1:
        ref_ids = [i for i in range(len(views)-1)] # all views except the last one
    elif ref_ids == -2:
        #select half of the views randomly
        ref_ids = np.random.choice(len(views), len(views)//2, replace=False).tolist()
    else:
        assert isinstance(ref_ids, list)

    for id in ref_ids:
        views[id]['pts3d_world'] = geotrf(w2c, views[id]['pts3d'])  #转移到目标坐标系
    norm_factor_world = get_multiview_scale([views[id]['pts3d_world'] for id in ref_ids],
                                            [views[id]['valid_mask'] for id in ref_ids], 
                                            norm_mode='avg_dis')
    for id,view in enumerate(views):            
        if id in ref_ids:
            view['pts3d_world'] = view['pts3d_world'].permute(0,3,1,2) / norm_factor_world
        else:
            norm_factor_src = get_multiview_scale([view['pts3d_cam']],
                                                [view['valid_mask']], 
                                                norm_mode='avg_dis')
            view['pts3d_cam'] = view['pts3d_cam'].permute(0,3,1,2) / norm_factor_src

    if to_zero:    
        for id,view in enumerate(views):
            valid_mask = view['valid_mask'].unsqueeze(1).float() # B,1,H,W
            if id in ref_ids:
                # print(view['pts3d_world'].shape, valid_mask.shape, (-valid_mask+1).sum())
                view['pts3d_world'] = view['pts3d_world'] * valid_mask
            else:
                view['pts3d_cam'] = view['pts3d_cam'] * valid_mask

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        preds = model(views, ref_ids=ref_ids)
        assert len(preds) == len(views)
        with torch.cuda.amp.autocast(enabled=False):
            if criterion is None:
                loss = None
            else:
                loss = criterion(views, preds, ref_id=ref_ids, ref_camera=w2c, norm_scale=norm_factor_world)   

    result = dict(views=views, preds=preds, loss=loss)
    for i in range(len(preds)):
        result[f'pred{i+1}'] = preds[i]
        result[f'view{i+1}'] = views[i]
    return result[ret] if ret else result
