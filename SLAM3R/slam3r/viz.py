# --------------------------------------------------------
# Visualization utilities. The code is adapted from Spann3r: 
# https://github.com/HengyiWang/spann3r/blob/main/spann3r/tools/vis.py
# --------------------------------------------------------

import numpy as np
from tqdm import tqdm
import open3d as o3d
import imageio
import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import exposure


def render_scene(vis, geometry, camera_parameters, bg_color=[1,1,1], point_size=1., 
                 uint8=True):
    vis.clear_geometries()
    for g in geometry:
        vis.add_geometry(g)

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera_parameters, allow_arbitrary=True)

    opt = vis.get_render_option()
    #调整点的大小
    opt.point_size = point_size
    opt.background_color = np.array(bg_color)

    vis.poll_events()
    vis.update_renderer()

    image = vis.capture_screen_float_buffer(do_render=True)
    
    if not uint8:
        return image
    else:
        image_uint8 = (np.asarray(image) * 255).astype(np.uint8)
        return image_uint8

def render_frames(pts_all, image_all, camera_parameters, output_dir, mask=None, save_video=True, save_camera=True,
                  init_ids=[],
                  c2ws=None, 
                  vis_cam=False,
                  save_stride=1,
                  sample_ratio=1.,
                  incremental=True,
                  save_name='render_frames',
                  bg_color=[1, 1, 1],
                  point_size=1.,
                  fps=10):
    
    t, h, w, _ = pts_all.shape

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960, height=544)

    render_frame_path = os.path.join(output_dir, save_name)
    os.makedirs(render_frame_path, exist_ok=True)

    if save_camera:
        o3d.io.write_pinhole_camera_parameters(os.path.join(render_frame_path, 'camera.json'), camera_parameters)

    video_path = os.path.join(output_dir, f'{save_name}.mp4')
    if save_video:
        writer = imageio.get_writer(video_path, fps=fps)

    # construct point cloud for initial window
    pcd = o3d.geometry.PointCloud()
    if init_ids is None: init_ids = []
    if len(init_ids) > 0:
        init_ids = np.array(init_ids)
        init_masks = mask[init_ids]
        init_pts = pts_all[init_ids][init_masks]
        init_colors = image_all[init_ids][init_masks]
        if sample_ratio < 1.:
            sampled_idx = np.random.choice(len(init_pts), int(len(init_pts)*sample_ratio), replace=False)
            init_pts = init_pts[sampled_idx]
            init_colors = init_colors[sampled_idx]

        pcd.points = o3d.utility.Vector3dVector(init_pts)
        pcd.colors = o3d.utility.Vector3dVector(init_colors)
                    
    vis.add_geometry(pcd)

    # visualize incremental reconstruction
    for i in tqdm(range(t), desc="Rendering incremental reconstruction"):
        if i not in init_ids:
            new_pts = pts_all[i].reshape(-1, 3)
            new_colors = image_all[i].reshape(-1, 3)

            if mask is not None:
                new_pts = new_pts[mask[i].reshape(-1)]
                new_colors = new_colors[mask[i].reshape(-1)]
            if sample_ratio < 1.:
                sampled_idx = np.random.choice(len(new_pts), int(len(new_pts)*sample_ratio), replace=False)
                new_pts = new_pts[sampled_idx]
                new_colors = new_colors[sampled_idx]
            if incremental:
                pcd.points.extend(o3d.utility.Vector3dVector(new_pts))
                pcd.colors.extend(o3d.utility.Vector3dVector(new_colors))
            else:
                pcd.points = o3d.utility.Vector3dVector(new_pts)
                pcd.colors = o3d.utility.Vector3dVector(new_colors)

        if (i+1) % save_stride != 0:
            continue
        
        geometry = [pcd]
        if vis_cam:
            geometry = geometry + draw_camera(c2ws[i], img=image_all[i])
        
        image_uint8 = render_scene(vis, geometry, camera_parameters, bg_color=bg_color, point_size=point_size)
        frame_filename = f'frame_{i:03d}.png'
        imageio.imwrite(osp.join(render_frame_path, frame_filename), image_uint8)
        if save_video:
            writer.append_data(image_uint8)

    if save_video:
        writer.close()

    vis.destroy_window()


def create_image_plane(img, c2w, scale=0.1):
    # simulate a image in 3D with point cloud
    H, W, _ = img.shape
    points = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H))
    points = np.stack(points, axis=-1).reshape(-1, 2)
    #translate the center of focal
    points -= np.array([W/2, H/2])

    points *= 2*scale/W
    points = np.concatenate([points, 0.1*np.ones((len(points), 1))], axis=-1)
    
    colors = img.reshape(-1, 3)
    
    # no need for such resolution
    sample_stride = max(1, int(0.2/scale))
    points = points[::sample_stride]
    colors = colors[::sample_stride]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.transform(c2w)
    
    return pcd

def draw_camera(c2w, cam_width=0.2/2, cam_height=0.2/2, f=0.10, color=[0, 1, 0], 
                show_axis=True, img=None):
    points = [[0, 0, 0], [-cam_width, -cam_height, f], [cam_width, -cam_height, f],
              [cam_width, cam_height, f], [-cam_width, cam_height, f]]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = [color for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.transform(c2w)

    res = [line_set]

    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
        axis.scale(min(cam_width, cam_height), np.array([0., 0., 0.]))
        axis.transform(c2w)
        res.append(axis)
        
    if img is not None:
        # draw image in the plane of the camera
        img_plane = create_image_plane(img, c2w)
        res.append(img_plane)
        
    return res

def find_render_cam(pcd, poses_all=None, cam_width=0.016, cam_height=0.012, cam_f=0.02):
    last_camera_params = None

    def print_camera_pose(vis):
        nonlocal last_camera_params
        ctr = vis.get_view_control()
        camera_params = ctr.convert_to_pinhole_camera_parameters()
        last_camera_params = camera_params 
        
        print("Intrinsic matrix:")
        print(camera_params.intrinsic.intrinsic_matrix)
        print("\nExtrinsic matrix:")
        print(camera_params.extrinsic)
        
        return False
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=960, height=544)
    vis.add_geometry(pcd)
    if poses_all is not None:
        for pose in poses_all:
            for geometry in draw_camera(pose, cam_width, cam_height, cam_f):
                vis.add_geometry(geometry)

    opt = vis.get_render_option()
    opt.point_size = 1
    opt.background_color = np.array([0, 0, 0])

    print_camera_pose(vis)
    print("Press the space key to record the current rendering view.")
    vis.register_key_callback(32, print_camera_pose)  

    while vis.poll_events():
        vis.update_renderer()

    vis.destroy_window()

    return last_camera_params

def vis_frame_preds(preds, type, save_path, norm_dims=(0, 1, 2), 
                    enhance_z=False, cmap=True,
                    save_imgs=True, save_video=True, fps=10):

    if norm_dims is not None:
        min_val = preds.min(axis=norm_dims, keepdims=True)
        max_val = preds.max(axis=norm_dims, keepdims=True)        
        preds = (preds - min_val) / (max_val - min_val)
            
    save_path = osp.join(save_path, type)
    if save_imgs:
        os.makedirs(save_path, exist_ok=True)

    if save_video:
        video_path = osp.join(osp.dirname(save_path), f'{type}.mp4')
        writer = imageio.get_writer(video_path, fps=fps)

    for frame_id in tqdm(range(preds.shape[0]), desc=f"Visualizing {type}"):
        pred_vis = preds[frame_id].astype(np.float32)
        if cmap:
            if preds.shape[-1] == 3:
                h = 1-pred_vis[...,0]
                s = 1-pred_vis[...,1]
                v = 1-pred_vis[...,2]
                if enhance_z:
                    new_v = exposure.equalize_adapthist(v, clip_limit=0.01, nbins=256)
                    v = new_v*0.2 + v*0.8
                pred_vis = mcolors.hsv_to_rgb(np.stack([h, s, v], axis=-1))
            elif len(pred_vis.shape)==2 or pred_vis.shape[-1] == 1:
                pred_vis = plt.cm.jet(pred_vis)
        pred_vis_rgb_uint8 = (pred_vis * 255).astype(np.uint8)

        if save_imgs:
            plt.imsave(osp.join(save_path, f'{type}_{frame_id:04d}.png'), pred_vis_rgb_uint8)

        if save_video:
            writer.append_data(pred_vis_rgb_uint8)

    if save_video:
        writer.close()



