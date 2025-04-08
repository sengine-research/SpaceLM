# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# --------------------------------------------------------
# Dataloader for preprocessed Replica dataset provided by NICER-SLAM
# --------------------------------------------------------
import os.path as osp
import os
import cv2
import numpy as np
from glob import glob
import json
import trimesh

SLAM3R_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
import sys # noqa: E402
sys.path.insert(0, SLAM3R_DIR) # noqa: E402
from slam3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from slam3r.utils.image import imread_cv2


class Replica(BaseStereoViewDataset):
    def __init__(self,  
                 ROOT='data/Replica', 
                 num_views=2, 
                 num_fetch_views=None,
                 sel_view=None, 
                 scene_name=None,
                 sample_freq=20,
                 start_freq=1,
                 sample_dis=1,
                 cycle=False,
                 ref_id=-1,
                 print_mess=False,
                 *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.print_mess = print_mess
        self.sample_freq = sample_freq
        self.start_freq = start_freq
        self.sample_dis = sample_dis
        self.cycle=cycle
        self.num_fetch_views = num_fetch_views if num_fetch_views is not None else num_views
        self.sel_view = np.arange(num_views) if sel_view is None else np.array(sel_view)
        self.num_views = num_views
        assert ref_id < num_views
        self.ref_id = ref_id if ref_id >= 0 else (num_views-1) // 2
        self.scene_names = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
        if self.split == 'train':
            self.scene_names = ["room0", "room1", "room2", "office0", "office1", "office2"]
        elif self.split=='val':
            self.scene_names = ["office3", "office4"]
        if scene_name is not None:
            assert self.split is None
            if isinstance(scene_name, list): 
                self.scene_names = scene_name
            else:
                assert isinstance(scene_name, str)
                self.scene_names = [scene_name]
        self._load_data()
        print(self)

    def _load_data(self):
        self.sceneids = []  
        self.image_paths = []
        self.trajectories = []  #c2w
        self.pairs = []
        with open(os.path.join(self.ROOT,"cam_params.json"),'r') as f:
            self.intrinsic = json.load(f)['camera'] 
        K = np.eye(3)
        K[0, 0] = self.intrinsic['fx']
        K[1, 1] = self.intrinsic['fy']
        K[0, 2] = self.intrinsic['cx']
        K[1, 2] = self.intrinsic['cy']
        self.intri_mat = K
        num_count = 0
        for id, scene_name in enumerate(self.scene_names):
            scene_dir = os.path.join(self.ROOT, scene_name)
            image_paths = sorted(glob(os.path.join(scene_dir,"results","frame*.jpg")))

            image_paths = image_paths[::self.sample_freq]
            image_num = len(image_paths)

            if not self.cycle:
                for i in range(0, image_num, self.start_freq):
                    last_id = i+self.sample_dis*(self.num_fetch_views-1)
                    if last_id >= image_num:
                        break
                    self.pairs.append([j+num_count for j in range(i,last_id+1,self.sample_dis)])
            else:
                for i in range(0, image_num, self.start_freq):
                    pair = []
                    for j in range(0, self.num_fetch_views):
                        pair.append((i+(j-self.ref_id)*self.sample_dis+image_num)%image_num + num_count)
                    self.pairs.append(pair)

            self.trajectories.append(np.loadtxt(os.path.join(scene_dir,"traj.txt")).reshape(-1,4,4)[::self.sample_freq])
            self.image_paths +=  image_paths
            self.sceneids += [id,] * image_num
            num_count += image_num
        # print(self.sceneids, self.scene_names)    
        self.trajectories = np.concatenate(self.trajectories,axis=0)
        assert len(self.trajectories) == len(self.sceneids) and len(self.sceneids)==len(self.image_paths), f"{len(self.trajectories)}, {len(self.sceneids)}, {len(self.image_paths)}"   
        
    def __len__(self):
        return len(self.pairs)
    
    def _get_views(self, idx, resolution, rng):
        
        image_idxes = self.pairs[idx]
        assert len(image_idxes) == self.num_fetch_views
        image_idxes = [image_idxes[i] for i in self.sel_view]
        views = []
        for view_idx in image_idxes:
            scene_id = self.sceneids[view_idx]
            camera_pose = self.trajectories[view_idx]
            image_path = self.image_paths[view_idx]
            image_name = os.path.basename(image_path)
            depth_path = image_path.replace(".jpg",".png").replace("frame","depth")
            # Load RGB image
            rgb_image = imread_cv2(image_path)
            
            # Load depthmap
            depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) 
            depthmap[~np.isfinite(depthmap)] = 0  # TODO:invalid
            depthmap /= self.intrinsic['scale']

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, self.intri_mat, resolution, rng=rng, info=view_idx)
            # print(intrinsics)
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='Replica',
                label=self.scene_names[scene_id] + '_' + image_name,
                instance=f'{str(idx)}_{str(view_idx)}',
            ))
        if self.print_mess:
            print(f"loading {[view['label'] for view in views]}")
        return views


if __name__ == "__main__":
    num_views = 5
    dataset= Replica(ref_id=1, print_mess=True, cycle=True, resolution=224, num_views=num_views, sample_freq=100, seed=777, start_freq=1, sample_dis=1)
    save_dir = "visualization/replica_views"
    
    # combine the pointmaps from different views with c2ws
    # to check the correctness of the dataloader
    for idx in np.random.permutation(len(dataset))[:10]:
    # for idx in range(10):
        views = dataset[(idx,0)]
        os.makedirs(osp.join(save_dir, str(idx)), exist_ok=True)
        assert len(views) == num_views
        all_pts = []
        all_color = []
        for i, view in enumerate(views):
            img = np.array(view['img']).transpose(1, 2, 0)
            save_path = osp.join(save_dir, str(idx), f"{i}_{view['label']}")
            print(save_path)
            img=img[...,::-1]
            img = (img+1)/2
            cv2.imwrite(save_path, img*255)
            print(f"save to {save_path}")
            img = img[...,::-1]
            pts3d = np.array(view['pts3d']).reshape(-1,3)
            pct = trimesh.PointCloud(pts3d, colors=img.reshape(-1, 3))
            pct.export(save_path.replace('.jpg','.ply'))
            all_pts.append(pts3d)
            all_color.append(img.reshape(-1, 3))
        all_pts = np.concatenate(all_pts, axis=0)
        all_color = np.concatenate(all_color, axis=0)
        pct = trimesh.PointCloud(all_pts, all_color)
        pct.export(osp.join(save_dir, str(idx), f"all.ply"))
                    