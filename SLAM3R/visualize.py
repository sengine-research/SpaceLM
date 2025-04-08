import open3d as o3d
import numpy as np
import argparse
import trimesh
import torch
from glob import glob
from os.path import join
from tqdm import tqdm
import json

from slam3r.utils.recon_utils import estimate_focal_knowing_depth, estimate_camera_pose
from slam3r.viz import find_render_cam, render_frames, vis_frame_preds

parser = argparse.ArgumentParser(description="Inference on a wild captured scene")
parser.add_argument("--vis_cam", action="store_true", help="visualize camera poses")
parser.add_argument("--vis_dir", type=str, required=True, help="directory to the predictions for visualization")
parser.add_argument("--save_stride", type=int, default=1, help="the stride for visualizing per-frame predictions")
parser.add_argument("--enhance_z", action="store_true", help="enhance the z axis for better visualization")
parser.add_argument("--conf_thres_l2w", type=float, default=12, help="confidence threshold for filter out low-confidence points in L2W")

def vis(args):
    
    root_dir = args.vis_dir
    
    preds_dir = join(args.vis_dir, "preds") 
    local_pcds = np.load(join(preds_dir, 'local_pcds.npy'))  # (V, 224, 224, 3)
    registered_pcds = np.load(join(preds_dir, 'registered_pcds.npy'))  # (V, 224, 224, 3)
    local_confs = np.load(join(preds_dir, 'local_confs.npy'))  # (V, 224, 224)
    registered_confs = np.load(join(preds_dir, 'registered_confs.npy'))  # (V, 224, 224)
    rgb_imgs = np.load(join(preds_dir, 'input_imgs.npy')) # (V, 224, 224, 3)
    
    rgb_imgs = rgb_imgs/255.
    
    recon_res_path = glob(join(args.vis_dir, "*.ply"))[0]
    recon_res = trimesh.load(recon_res_path)
    whole_pcd = recon_res.vertices
    whole_colors = recon_res.visual.vertex_colors[:, :3]/255.

    # change to open3d coordinate  x->x y->-y z->-z
    whole_pcd[..., 1:] *= -1
    registered_pcds[..., 1:] *= -1
    
    recon_pcd = o3d.geometry.PointCloud()
    recon_pcd.points = o3d.utility.Vector3dVector(whole_pcd)
    recon_pcd.colors = o3d.utility.Vector3dVector(whole_colors)
    
    # extract information about the initial window in the reconstruction
    num_views = local_pcds.shape[0]
    with open(join(preds_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    init_winsize = metadata['init_winsize']
    kf_stride = metadata['kf_stride']
    init_ids = list(range(0, init_winsize*kf_stride, kf_stride))
    init_ref_id = metadata['init_ref_id'] * kf_stride

    if args.vis_cam:
        # estimate camera intrinsics and poses
        principal_point = torch.tensor((local_pcds[0].shape[0]//2, local_pcds[0].shape[1]//2))
        init_window_focal = estimate_focal_knowing_depth(torch.tensor(local_pcds[init_ref_id][None]), 
                                                         principal_point, 
                                                         focal_mode='weiszfeld')
        
        focals = []
        for i in tqdm(range(num_views), desc="estimating intrinsics"):
            if i in init_ids:
                focals.append(init_window_focal)
            else:
                focal = estimate_focal_knowing_depth(torch.tensor(local_pcds[i:i+1]), 
                                                     principal_point, 
                                                     focal_mode='weiszfeld')
                focals.append(focal)

        intrinsics = []
        for i in range(num_views):
            intrinsic = np.eye(3)
            intrinsic[0, 0] = focals[i]
            intrinsic[1, 1] = focals[i]
            intrinsic[:2, 2] = principal_point
            intrinsics.append(intrinsic) 
        
        mean_intrinsics = np.mean(np.stack(intrinsics,axis=0), axis=0)  
        init_window_intrinsics = intrinsics[init_ref_id]            

        c2ws = []
        for i in tqdm(range(0, num_views, 1), desc="estimating camera poses"):
            registered_pcd = registered_pcds[i]
            # c2w, succ = estimate_camera_pose(registered_pcd, init_window_intrinsics)
            c2w, succ = estimate_camera_pose(registered_pcd, mean_intrinsics)
            # c2w, succ = estimate_camera_pose(registered_pcd, intrinsics[i])
            if not succ:
                print(f"fail to estimate camera pose for view {i}")
            c2ws.append(c2w)

    # find the camera parameters for rendering incremental reconstruction process
    # It will show a window of open3d, and you can rotate and translate the camera
    # press space to save the camera parameters selected
    camera_parameters = find_render_cam(recon_pcd, c2ws if args.vis_cam else None)
    # render the incremental reconstruction process
    render_frames(registered_pcds, rgb_imgs, camera_parameters, root_dir, 
                  mask=(registered_confs > args.conf_thres_l2w),
                  init_ids=init_ids,
                  c2ws=c2ws if args.vis_cam else None,
                  sample_ratio=1/args.save_stride,
                  save_stride=args.save_stride,
                  fps=10, 
                  vis_cam=args.vis_cam,
                  )
    
    # save visualizations of per-frame predictions, and combine them into a video
    vis_frame_preds(local_confs[::args.save_stride], type="I2P_conf", 
                    save_path=root_dir)
    vis_frame_preds(registered_confs[::args.save_stride], type="L2W_conf",
                    save_path=root_dir)
    vis_frame_preds(local_pcds[::args.save_stride], type="I2P_pcds", 
                    save_path=root_dir,
                    enhance_z=args.enhance_z
                    )
    vis_frame_preds(registered_pcds[::args.save_stride], type="L2W_pcds", 
                    save_path=root_dir,
                    )
    vis_frame_preds(rgb_imgs[::args.save_stride], type="imgs", 
                    save_path=root_dir,
                    norm_dims=None,
                    cmap=False
                    )
    
if __name__ == "__main__":
    args = parser.parse_args()
    vis(args)