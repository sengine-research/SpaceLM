import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.preprocess_data import load_o3d_pcd, cleanup_pcd, get_points_and_colors_and_normals, preprocess_point_cloud_and_normals
from layout.layout import Layout
import numpy as np

point_cloud_file = "preprocessed_data_scene_script/Scannet/scene0000_00/scene0000_00_pc_result.ply"
str_file = "preprocessed_data_scene_script/Scannet/scene_info_list.json"
import json

with open(str_file, "r") as f:
    scene_info_list = json.load(f)

for i, scene_info in enumerate(scene_info_list):
    ply_file = scene_info["ply_file"]
    if ply_file == point_cloud_file:
        scene_info = scene_info_list[i]
        break

prompt = scene_info["prompt"]

point_cloud = load_o3d_pcd(point_cloud_file)
point_cloud = cleanup_pcd(point_cloud)
points, colors, normals = get_points_and_colors_and_normals(point_cloud)
min_extent = np.min(points, axis=0)
max_extent = np.max(points, axis=0)
mid_extent = np.array([(min_extent[0] + max_extent[0]) / 2, (min_extent[1] + max_extent[1]) / 2, min_extent[2]])

layout = Layout(prompt)

layout.undiscretize_and_unnormalize()
layout.translate(min_extent)
pred_language_string = layout.to_language_string()

with open("test.txt", "w") as f:
    f.write(pred_language_string)