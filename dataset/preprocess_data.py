import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import shutil
import json
import torch
from layout.layout import Layout
from layout.entity import entity_label_list
from pcd.pcd_loader import load_o3d_pcd, get_points_and_colors_and_normals, cleanup_pcd

from dataset.preprocess_ply import preprocess_point_cloud_and_normals, get_point_cloud_preprocessor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_dir(dir_path):

    for file in os.listdir(dir_path):
        if file.endswith("_object_result.csv"):
            object_csv_file = os.path.join(dir_path, file)
        if file.endswith("_pc_result.ply"):
            ply_file = os.path.join(dir_path, file)

    return object_csv_file, ply_file

@torch.no_grad()
def process_one_scene(scene_path, output_dataset_path, model):
    print(scene_path)
    object_csv_file, ply_file = read_dir(scene_path)

    import pandas as pd

    object_csv = pd.read_csv(object_csv_file)

    prompt_list = []

    # add the point cloud processing part
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
    # move all the coord to be larger than 0
    layout.translate(-min_extent)
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
    # print(point_feat.feat.shape)
    # all the coord should be larger than 0
    for i in range(len(point_feat.coord)):
        point_feat.coord[i] = point_feat.coord[i].cpu()
    
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
            "point_feat": point_feat.feat.tolist(),
            "point_feat_coord": point_feat.coord.tolist(),
            "min_extent": torch.tensor(min_extent).tolist(),
            }, f)

    torch.cuda.empty_cache()
    return {"prompt": prompt_str, "ply_file": os.path.join(output_dataset_path, os.path.basename(ply_file)), "ply_json_path": ply_json_path}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="3D_dataset")
    parser.add_argument("--dataset_name", type=str, default="3RScan")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    models = []
    for i in range(args.num_workers):
        # initialize the model (single initialization)
        model = get_point_cloud_preprocessor().cuda()
        model.eval()
        models.append(model)

    # prepare the output directory
    output_root = os.path.join("preprocessed_data", args.dataset_name)
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    # get all the scene paths to be processed
    dataset_path = os.path.join(args.data_path, args.dataset_name)
    scene_paths = [os.path.join(dataset_path, p) for p in os.listdir(dataset_path)]

    # multi-thread processing
    scene_info_list = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # submit all the tasks
        futures = {
            executor.submit(
                process_one_scene,
                scene_path,
                os.path.join(output_root, os.path.basename(scene_path)),
                models[i % args.num_workers]
            ): scene_path 
            for i, scene_path in enumerate(scene_paths)
        }

        # use tqdm to show the progress
        with tqdm(total=len(scene_paths), desc="Processing scenes") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    scene_info_list.append(result)
                except Exception as e:
                    scene_path = futures[future]
                    print(f"Error processing {scene_path}: {str(e)}")
                finally:
                    pbar.update(1)

    # save the final result
    with open(os.path.join(output_root, "scene_info_list.json"), "w") as f:
        json.dump(scene_info_list, f, indent=4)

