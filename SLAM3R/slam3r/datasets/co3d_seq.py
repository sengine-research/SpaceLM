# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed Co3d_v2
# dataset at https://github.com/facebookresearch/co3d - Creative Commons Attribution-NonCommercial 4.0 International
# See datasets_preprocess/preprocess_co3d.py
# --------------------------------------------------------
import os.path as osp
import json
import itertools
from collections import deque

import cv2
import numpy as np

SLAM3R_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
import sys # noqa: E402
sys.path.insert(0, SLAM3R_DIR) # noqa: E402
from slam3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from slam3r.utils.image import imread_cv2

TRAINING_CATEGORIES = [
    "apple","backpack","banana","baseballbat","baseballglove","bench","bicycle",
    "bottle","bowl","broccoli","cake","car","carrot","cellphone","chair","cup","donut","hairdryer","handbag","hydrant","keyboard",
    "laptop","microwave","motorcycle","mouse","orange","parkingmeter","pizza","plant","stopsign","teddybear","toaster","toilet",
    "toybus","toyplane","toytrain","toytruck","tv","umbrella","vase","wineglass",
]
TEST_CATEGORIES = ["ball", "book", "couch", "frisbee", "hotdog", "kite", "remote", "sandwich", "skateboard", "suitcase"]


class Co3d_Seq(BaseStereoViewDataset):
    def __init__(self, 
                 mask_bg=True, 
                 ROOT="data/co3d_processed", 
                 num_views=2,
                 degree=90,
                 sel_num=1,
                 *args, 
                 **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        self.degree = degree
        self.winsize = int(degree / 360 * 100)
        self.sel_num = sel_num
        self.sel_num_perseq = (101 - self.winsize) * self.sel_num
        self.num_views = num_views

        # load all scenes
        if self.split == 'train':
            self.categories = TRAINING_CATEGORIES
        elif self.split == 'test':
            self.categories = TEST_CATEGORIES
        else:
            raise ValueError(f"Unknown split {self.split}")
        self.scenes = {}
        for cate in TRAINING_CATEGORIES:
            with open(osp.join(self.ROOT, cate, f'selected_seqs_{self.split}.json'), 'r') as f:
                self.scenes[cate] = json.load(f)
        self.scenes = {(k, k2): v2 for k, v in self.scenes.items()
                        for k2, v2 in v.items()}
        self.scene_list = list(self.scenes.keys())
        self.scene_lens = [len(v) for k,v in self.scenes.items()]
        print(np.unique(np.array(self.scene_lens), return_counts=True))
        self.invalidate = {scene: {} for scene in self.scene_list}
        
        print(self)

    def __len__(self):
        return len(self.scene_list) * self.sel_num_perseq

    def get_img_idxes(self, idx, rng):
        sid = max(0, idx // self.sel_num - 1) #from 0 to 99-winsize
        eid = sid + self.winsize
        if idx % self.sel_num == 0:
            #生成sid与eid之间的均匀采样
            return np.linspace(sid, eid, self.num_views, endpoint=True, dtype=int)
            
        #首尾必须选择，中间随机选择n-2个
        if self.num_views == 2:
            return [sid, eid]
        sel_ids = rng.choice(range(sid+1, eid), self.num_views-2, replace=False)
        sel_ids.sort()
        return [sid] + list(sel_ids) + [eid]
    

    def _get_views(self, idx, resolution, rng):
        # choose a scene
        obj, instance = self.scene_list[idx // self.sel_num_perseq]
        image_pool = self.scenes[obj, instance]
        last = len(image_pool)-1
        if last <= self.winsize:
            return self._get_views(rng.integers(0, len(self)-1), resolution, rng)

        imgs_idxs = self.get_img_idxes(idx % self.sel_num_perseq, rng)
        
        
        for i, idx in enumerate(imgs_idxs):
            if idx > last:
                while idx > last:
                    idx -= last
                imgs_idxs[i] = idx 
        # print(imgs_idxs)

        if resolution not in self.invalidate[obj, instance]:  # flag invalid images
            self.invalidate[obj, instance][resolution] = [False for _ in range(len(image_pool))]

        # decide now if we mask the bg
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))

        views = []
        imgs_idxs = deque(imgs_idxs)
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            im_idx = imgs_idxs.popleft()

            if self.invalidate[obj, instance][resolution][im_idx]:
                # search for a valid image
                random_direction = 2 * rng.choice(2) - 1
                for offset in range(1, len(image_pool)):
                    tentative_im_idx = (im_idx + (random_direction * offset)) % len(image_pool)
                    if not self.invalidate[obj, instance][resolution][tentative_im_idx]:
                        im_idx = tentative_im_idx
                        break

            view_idx = image_pool[im_idx]

            impath = osp.join(self.ROOT, obj, instance, 'images', f'frame{view_idx:06n}.jpg')

            # load camera params
            input_metadata = np.load(impath.replace('jpg', 'npz'))
            camera_pose = input_metadata['camera_pose'].astype(np.float32)
            intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)

            # load image and depth
            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(impath.replace('images', 'depths') + '.geometric.png', cv2.IMREAD_UNCHANGED)
            depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(input_metadata['maximum_depth'])

            if mask_bg:
                # load object mask
                maskpath = osp.join(self.ROOT, obj, instance, 'masks', f'frame{view_idx:06n}.png')
                maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1

                # update the depthmap with mask
                depthmap *= maskmap

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                # problem, invalidate image and retry
                self.invalidate[obj, instance][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='Co3d_v2',
                label=f"{obj}_{instance}_frame{view_idx:06n}.jpg",
                instance=osp.split(impath)[1],
            ))
        return views


if __name__ == "__main__":
    from slam3r.datasets.base.base_stereo_view_dataset import view_name
    import os
    import trimesh

    num_views = 5
    dataset = Co3d_Seq(split='train', 
                       mask_bg='rand', resolution=224, aug_crop=16,
                       num_views=num_views, degree=90, sel_num=3)

    save_dir = "visualization/co3d_seq_views"
    os.makedirs(save_dir, exist_ok=True)

    for idx in np.random.permutation(len(dataset))[:10]:
    # for idx in range(len(dataset))[5:10000:2000]:
        os.makedirs(osp.join(save_dir, str(idx)), exist_ok=True)
        views = dataset[(idx,0)]
        assert len(views) == num_views
        all_pts = []
        all_color=[]
        for i, view in enumerate(views):
            img = np.array(view['img']).transpose(1, 2, 0)
            # save_path = osp.join(save_dir, str(idx), f"{'_'.join(view_name(view).split('/')[1:])}.jpg")
            save_path = osp.join(save_dir, str(idx), f"{i}_{view['label']}")
            # img=cv2.COLOR_RGB2BGR(img)
            img=img[...,::-1]
            img = (img+1)/2
            cv2.imwrite(save_path, img*255)
            print(f"save to {save_path}")
            pts3d = np.array(view['pts3d']).reshape(-1,3)
            pct = trimesh.PointCloud(pts3d, colors=img.reshape(-1, 3))
            pct.export(save_path.replace('.jpg','.ply'))
            all_pts.append(pts3d)
            all_color.append(img.reshape(-1, 3))
        all_pts = np.concatenate(all_pts, axis=0)
        all_color = np.concatenate(all_color, axis=0)
        pct = trimesh.PointCloud(all_pts, all_color)
        pct.export(osp.join(save_dir, str(idx), f"all.ply"))
    # np.random.seed(43)
    # # views = dataset[(431519,0)]
    # for idx in np.random.permutation(len(dataset))[::1000]:
    # # for idx in range(len(dataset)):
    #     print(idx)
    #     views = dataset[(idx,0)]
    #     print([view['label'] for view in views])