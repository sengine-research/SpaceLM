import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pcd.pcd_loader import load_o3d_pcd, get_points_and_colors_and_normals, cleanup_pcd
from layout.layout import Layout
import json
import tqdm


import numpy
import sonata
import torch

def preprocess_point_cloud_and_normals(points, colors, normals, grid_size, shift_type="center"):
    if shift_type == "center":
        config = [
                dict(type="CenterShift", apply_z=True),
                dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ]
    elif shift_type == "positive":
        config = [
                dict(type="PositiveShift"),
                dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ]

    transform = sonata.transform.Compose(config)

    # single point cloud
    point = {
        "coord": points,  # (N, 3)
        "color": colors,  # (N, 3)
        "normal": normals,  # (N, 3)
    }

    point = transform(point)
    return point["coord"], point["grid_coord"], point["color"], point["offset"], point["inverse"], point["feat"]

def get_point_cloud_preprocessor():
    # Load the pre-trained model from Huggingface
    # supported models: "sonata"
    # ckpt is cached in ~/.cache/sonata/ckpt, and the path can be customized by setting 'download_root'
    custom_config = dict(
        enc_patch_size=[1024 for _ in range(5)],
        enable_flash=False,  # reduce patch size if necessary
    )

    model = sonata.load("facebook/sonata", custom_config=custom_config).cuda().eval()

    return model

def get_transform():
    config = [
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=0.02,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=True,
        ),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "color", "inverse"),
            feat_keys=("coord", "color", "normal"),
        ),
    ]

    transform = sonata.transform.Compose(config)
    return transform