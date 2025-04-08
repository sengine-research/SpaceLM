# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for 7Scenes dataset
# --------------------------------------------------------
import os.path as osp
import os
import cv2
import numpy as np
import torch
import itertools
from glob import glob
import json
import trimesh

SLAM3R_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
import sys # noqa: E402
sys.path.insert(0, SLAM3R_DIR) # noqa: E402
from slam3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from slam3r.utils.image import imread_cv2    

class SevenScenes_Seq(BaseStereoViewDataset):
    def __init__(self,  
                 ROOT='data/7Scenes', 
                 scene_id='office',
                 seq_id=1,
                 num_views=1, 
                 sample_freq=1,
                 start_freq=1,
                 cycle=False,
                 ref_id=-1,
                 *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.cycle = cycle
        self.scene_id = scene_id
        self.scene_names = [scene_id+'_seq-'+f"{seq_id:02d}"]
        self.seq_id = seq_id
        self.sample_freq = sample_freq
        self.start_freq = start_freq
        self.ref_id = ref_id if ref_id >= 0 else (num_views-1) // 2
        self.num_views = num_views
        self.num_fetch_views = self.num_views
        self.data_dir = os.path.join(self.ROOT, scene_id, f'seq-{seq_id:02d}')
        self._load_data()
        print(self)

    def _load_data(self):
        self.intrinsics = np.array([[585, 0, 320],
                                   [0, 585, 240],
                                   [0, 0, 1]], dtype=np.float32)
        self.trajectories = []  #c2w  (4,4)
        self.pairs = []
        self.images = sorted(glob(osp.join(self.data_dir, '*.color.png'))) #frame-000000.color.png
        image_num = len(self.images)
        #这两行能否提速
        if not self.cycle:
            for i in range(0, image_num, self.start_freq):
                last_id = i+(self.num_views-1)*self.sample_freq
                if last_id >= image_num: break
                self.pairs.append([i+j*self.sample_freq 
                                    for j in range(self.num_views)])
        else:
            for i in range(0, image_num, self.start_freq):
                pair = []
                for j in range(0, self.num_fetch_views):
                    pair.append((i+(j-self.ref_id)*self.sample_freq+image_num)%image_num)
                self.pairs.append(pair)
        print(self.pairs)
    def __len__(self):
        return len(self.pairs)
        # return len(self.img_group)
    
    def _get_views(self, idx, resolution, rng):

        image_idxes = self.pairs[idx]
        views = []
        scene_dir = self.data_dir
        
        for view_idx in image_idxes:
        
            intrinsics = self.intrinsics
            img_path = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(img_path)
            # Load depthmap(16-bit, PNG, invalid depth is set to 65535)
            depthmap = imread_cv2(img_path.replace('.color.png','.depth.png'), cv2.IMREAD_UNCHANGED)
            depthmap[depthmap == 65535] = 0
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            camera_pose = np.loadtxt(img_path.replace('.color.png','.pose.txt'))

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='ScanNet++',
                label=img_path,
                instance=f'{str(idx)}_{str(view_idx)}',
            ))
        return views
 

class SevenScenes_Seq_Cali(BaseStereoViewDataset):
    def __init__(self,  
                 ROOT='data/7s_dsac/dsac', 
                 scene_id='office',
                 seq_id=1,
                 num_views=1, 
                 sample_freq=1,
                 start_freq=1,
                 cycle=False,
                 ref_id=-1,
                 *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.cycle = cycle
        self.scene_id = scene_id
        self.scene_names = [scene_id+'_seq-'+f"{seq_id:02d}"]
        self.seq_id = seq_id
        self.sample_freq = sample_freq
        self.start_freq = start_freq
        self.ref_id = ref_id if ref_id >= 0 else (num_views-1) // 2
        self.num_views = num_views
        self.num_fetch_views = self.num_views
        self.data_dir = os.path.join(self.ROOT, scene_id, f'seq-{seq_id:02d}')
        self._load_data()
        print(self)

    def _load_data(self):
        self.intrinsics = np.array([[525, 0, 320],
                                   [0, 525, 240],
                                   [0, 0, 1]], dtype=np.float32)
        self.trajectories = []  #c2w  (4,4)
        self.pairs = []
        self.images = sorted(glob(osp.join(self.data_dir, '*.color.png'))) #frame-000000.color.png
        image_num = len(self.images)
        #这两行能否提速
        if not self.cycle:
            for i in range(0, image_num, self.start_freq):
                last_id = i+(self.num_views-1)*self.sample_freq
                if last_id >= image_num: break
                self.pairs.append([i+j*self.sample_freq 
                                    for j in range(self.num_views)])
        else:
            for i in range(0, image_num, self.start_freq):
                pair = []
                for j in range(0, self.num_fetch_views):
                    pair.append((i+(j-self.ref_id)*self.sample_freq+image_num)%image_num)
                self.pairs.append(pair)
        # print(self.pairs)
    def __len__(self):
        return len(self.pairs)
        # return len(self.img_group)
    
    def _get_views(self, idx, resolution, rng):

        image_idxes = self.pairs[idx]
        views = []
        scene_dir = self.data_dir
        
        for view_idx in image_idxes:
        
            intrinsics = self.intrinsics
            img_path = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(img_path)
            # Load depthmap(16-bit, PNG, invalid depth is set to 65535)
            depthmap = imread_cv2(img_path.replace('.color.png','.depth_cali.png'), cv2.IMREAD_UNCHANGED)
            depthmap[depthmap == 65535] = 0
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            
            camera_pose = np.loadtxt(img_path.replace('.color.png','.pose.txt'))

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)
            print(intrinsics)
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='SevenScenes',
                label=img_path,
                instance=f'{str(idx)}_{str(view_idx)}',
            ))
        return views


if __name__ == "__main__":
    from slam3r.datasets.base.base_stereo_view_dataset import view_name
    from slam3r.viz import SceneViz, auto_cam_size
    from slam3r.utils.image import rgb

    num_views = 3
    dataset = SevenScenes_Seq(scene_id='office',seq_id=9,resolution=(224,224), num_views=num_views,
                            start_freq=1, sample_freq=20)
    save_dir = "visualization/7scenes_seq_views"
    os.makedirs(save_dir, exist_ok=True)

    # for idx in np.random.permutation(len(dataset))[:10]:
    for idx in range(len(dataset))[:500:100]:
        os.makedirs(osp.join(save_dir, str(idx)), exist_ok=True)
        views = dataset[(idx,0)]
        assert len(views) == num_views
        all_pts = []
        all_color=[]
        for i, view in enumerate(views):
            img = np.array(view['img']).transpose(1, 2, 0)
            # save_path = osp.join(save_dir, str(idx), f"{'_'.join(view_name(view).split('/')[1:])}.jpg")
            print(view['label'])
            save_path = osp.join(save_dir, str(idx), f"{i}_{os.path.basename(view['label'])}")
            # img=cv2.COLOR_RGB2BGR(img)
            img=img[...,::-1]
            img = (img+1)/2
            cv2.imwrite(save_path, img*255)
            print(f"save to {save_path}")
            pts3d = np.array(view['pts3d']).reshape(-1,3)
            pct = trimesh.PointCloud(pts3d, colors=img.reshape(-1, 3))
            pct.export(save_path.replace('.png','.ply'))
            all_pts.append(pts3d)
            all_color.append(img.reshape(-1, 3))
        all_pts = np.concatenate(all_pts, axis=0)
        all_color = np.concatenate(all_color, axis=0)
        pct = trimesh.PointCloud(all_pts, all_color)
        pct.export(osp.join(save_dir, str(idx), f"all.ply"))
    # for idx in range(len(dataset)):
    #     views = dataset[(idx,0)]
    #     print([view['label'] for view in views])