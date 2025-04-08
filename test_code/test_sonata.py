import torch
import os
import shutil
import json
import numpy as np
import sys
sys.path.append("/root/autodl-tmp/SpaceLM")
from layout.layout import Layout
from layout.entity import entity_label_list
from pcd.pcd_loader import load_o3d_pcd, get_points_and_colors_and_normals, cleanup_pcd
from dataset.preprocess_ply import preprocess_point_cloud_and_normals
from dataset.preprocess_data import read_dir
import torch
import open3d as o3d
import sonata
from fast_pytorch_kmeans import KMeans

try:
    import flash_attn
except ImportError:
    flash_attn = None

scene_path = "/root/autodl-tmp/SpaceLM/3D_dataset/Scannet/scene0000_00"
output_dataset_path = "test/Scannet/scene0000_00"
object_csv_file, ply_file = read_dir(scene_path)

import pandas as pd

object_csv = pd.read_csv(object_csv_file)

prompt_list = []

# # 新增点云处理部分
point_cloud = load_o3d_pcd(ply_file)
# point_cloud = cleanup_pcd(point_cloud)
points, colors, normals = get_points_and_colors_and_normals(point_cloud)

min_extent = np.min(points, axis=0)
max_extent = np.max(points, axis=0)
mid_extent = np.array([(min_extent[0] + max_extent[0]) / 2, (min_extent[1] + max_extent[1]) / 2, min_extent[2]])

info_list = []

for i in range(len(object_csv)):
    object_name = object_csv['nyu_label'][i]
    object_bbox_cx = object_csv['object_bbox_cx'][i]
    object_bbox_cy = object_csv['object_bbox_cy'][i]
    object_bbox_cz = object_csv['object_bbox_cz'][i]
    object_bbox_heading = object_csv['object_bbox_heading'][i]
    object_bbox_xlength = object_csv['object_bbox_xlength'][i]
    object_bbox_ylength = object_csv['object_bbox_ylength'][i]
    object_bbox_zlength = object_csv['object_bbox_zlength'][i]

    if object_name in entity_label_list:
        info = [object_name, object_bbox_cx, object_bbox_cy, object_bbox_cz, object_bbox_heading, object_bbox_xlength, object_bbox_ylength, object_bbox_zlength]
        info_list.append(info)

info_list = sorted(info_list, key=lambda x: entity_label_list.index(x[0]))

for i in range(len(info_list)):
    info = info_list[i]
    prompt = f"bbox_{i}=({','.join([str(i) for i in info])})"
    
    prompt_list.append(prompt)

prompt_str = "\n".join(prompt_list)

layout = Layout(prompt_str)
# 移动所有坐标到中心
layout.translate(-mid_extent)
layout.normalize_and_discretize()
prompt_str = layout.to_language_string()

if not os.path.exists(os.path.join(output_dataset_path)):
    os.makedirs(os.path.join(output_dataset_path))

ply_file_path = os.path.join(output_dataset_path, os.path.basename(ply_file))
shutil.copy(ply_file, ply_file_path)

grid_size = Layout.get_grid_size()
original_coord = points.copy()
coord, grid_coord, color, offset, inverse, feat = preprocess_point_cloud_and_normals(points, colors, normals, grid_size)

point = {
    "coord": coord,
    "grid_coord": grid_coord,
    "color": color,
    "offset": offset,
    "inverse": inverse,
    "feat": feat
}

for key in point.keys():
    if isinstance(point[key], torch.Tensor):
        point[key] = point[key].cuda(non_blocking=True)

def get_pca_color(feat, center=True):
    u, s, v = torch.pca_lowrank(feat, center=center, q=3, niter=5)
    projection = feat @ v
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div
    return color

sonata.utils.set_seed(24525867)
# Load model
if flash_attn is not None:
    model = sonata.load("facebook/sonata").cuda()
else:
    custom_config = dict(
        enc_patch_size=[1024 for _ in range(5)],  # reduce patch size if necessary
        enable_flash=False,
    )
    model = sonata.load("facebook/sonata", custom_config=custom_config).cuda()

with torch.inference_mode():
    for key in point.keys():
        if isinstance(point[key], torch.Tensor):
            point[key] = point[key].cuda(non_blocking=True)
    # model forward:
    point = model(point)
    for key in point.keys():
        print(key)
    # upcast point feature
    # Point is a structure contains all the information during forward
    for _ in range(2):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while "pooling_parent" in point.keys():
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent

    # here point is down-sampled by GridSampling in default transform pipeline
    # feature of point cloud in original scale can be acquired by:
    _ = point.feat[point.inverse]

    # PCA
    pca_color = get_pca_color(point.feat, center=True)

    # Auto threshold with k-means
    # (DINOv2 manually set threshold for separating background and foreground)
    N_CLUSTERS = 3
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        mode="cosine",
        max_iter=1000,
        init_method="random",
        tol=0.0001,
    )

    kmeans.fit(point.feat)
    cluster = (
        kmeans.cos_sim(point.feat, kmeans.centroids)
        * torch.tensor([1, 1.12, 1]).cuda()
    ).argmax(dim=-1)

pca_color_ = pca_color.clone()
pca_color_[cluster == 1] = get_pca_color(point.feat[cluster == 1], center=True)

# inverse back to original scale before grid sampling
# point.inverse is acquired from the GirdSampling transform
original_pca_color_ = pca_color_[point.inverse]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(original_coord)
pcd.colors = o3d.utility.Vector3dVector(original_pca_color_.cpu().detach().numpy())
# o3d.visualization.draw_geometries([pcd])
# or
# o3d.visualization.draw_plotly([pcd])

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point.coord.cpu().detach().numpy())
# pcd.colors = o3d.utility.Vector3dVector(pca_color_.cpu().detach().numpy())
o3d.io.write_point_cloud(os.path.join(output_dataset_path, "pca.ply"), pcd)
