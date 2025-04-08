
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import shutil
import json
import torch
from layout.layout import Layout
from pcd.pcd_loader import load_o3d_pcd, get_points_and_colors_and_normals, cleanup_pcd

from preprocess_ply import preprocess_point_cloud_and_normals, get_point_cloud_preprocessor
from tqdm import tqdm
def read_dir(dir_path):

    for file in os.listdir(dir_path):
        if file.endswith("_object_result.csv"):
            object_csv_file = os.path.join(dir_path, file)
        if file.endswith("_pc_result.ply"):
            ply_file = os.path.join(dir_path, file)

    return object_csv_file, ply_file

def process_one_scene(scene_path, output_dataset_path):
    print(scene_path)
    object_csv_file, ply_file = read_dir(scene_path)

    import pandas as pd

    object_csv = pd.read_csv(object_csv_file)

    prompt_list = []

    # # 新增点云处理部分
    point_cloud = load_o3d_pcd(ply_file)
    point_cloud = cleanup_pcd(point_cloud)
    points, colors, normals = get_points_and_colors_and_normals(point_cloud)

    min_extent = np.min(points, axis=0)
    max_extent = np.max(points, axis=0)
    mid_extent = np.array([(min_extent[0] + max_extent[0]) / 2, (min_extent[1] + max_extent[1]) / 2, min_extent[2]])

    info_list = []

    for i in range(len(object_csv)):
        object_name = object_csv['raw_label'][i]
        object_bbox_cx = object_csv['object_bbox_cx'][i]
        object_bbox_cy = object_csv['object_bbox_cy'][i]
        object_bbox_cz = object_csv['object_bbox_cz'][i]
        object_bbox_heading = object_csv['object_bbox_heading'][i]
        object_bbox_xlength = object_csv['object_bbox_xlength'][i]
        object_bbox_ylength = object_csv['object_bbox_ylength'][i]
        object_bbox_zlength = object_csv['object_bbox_zlength'][i]

        info = [object_name, object_bbox_cx, object_bbox_cy, object_bbox_cz, object_bbox_heading, object_bbox_xlength, object_bbox_ylength, object_bbox_zlength]
        info_list.append(info)
        prompt = f"bbox_{i}=Bbox({object_name},{object_bbox_cx},{object_bbox_cy},{object_bbox_cz},{object_bbox_heading},{object_bbox_xlength},{object_bbox_ylength},{object_bbox_zlength})"
        
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

    point_feat = model(point)
    print(point_feat.feat.shape)
    
    ply_json_path = ply_file_path.replace(".ply", ".json")
    os.makedirs(os.path.dirname(ply_json_path), exist_ok=True)
    with open(ply_json_path, "w") as f:
        json.dump({
            "coord": coord.tolist(),
            "grid_coord": grid_coord.tolist(),
            "color": color.tolist(),
            "offset": offset.tolist(),
            "inverse": inverse.tolist(),
            "feat": feat.tolist(),
            "point_feat": point_feat.feat.tolist()
        }, f)

    return {"prompt": prompt_str, "ply_file": os.path.join(output_dataset_path, os.path.basename(ply_file)), "ply_json_path": ply_json_path}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="3D_dataset")
    parser.add_argument("--dataset_name", type=str, default="3RScan")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_path = os.path.join(args.data_path, dataset_name)

    model = get_point_cloud_preprocessor()

    if os.path.exists(os.path.join("preprocessed_data_test", dataset_name)):
        shutil.rmtree(os.path.join("preprocessed_data_test", dataset_name))
    os.makedirs(os.path.join("preprocessed_data_test", dataset_name))

    scene_info_list = []

    for test_path in tqdm(os.listdir(dataset_path)[:1]):
        output_dataset_path = os.path.join("preprocessed_data_test", dataset_name, test_path)
        scene_info = process_one_scene(os.path.join(dataset_path, test_path), output_dataset_path)
        scene_info_list.append(scene_info)
    
    with open(os.path.join("preprocessed_data_test", dataset_name, "scene_info_list.json"), "w") as f:
        json.dump(scene_info_list, f, indent=4)

