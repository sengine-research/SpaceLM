import logging
import numpy as np
import open3d as o3d

from pcd.registry import Registry

TRANSFORMS = Registry("transforms")
log = logging.getLogger(__name__)

"""
3D Point Cloud Preprocessing

Reference: PointCept

@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    year={2023}
}
"""


class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict


@TRANSFORMS.register_module()
class PositiveShift(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min
        return data_dict


@TRANSFORMS.register_module()
class NormalizeColor(object):
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 127.5 - 1
        return data_dict


@TRANSFORMS.register_module()
class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color", "normal", "segment"),
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
        max_grid_coord=None,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement
        self.max_grid_coord = max_grid_coord

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        if self.max_grid_coord is not None:
            grid_coord = np.clip(grid_coord, 0, self.max_grid_coord - 1)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == "test":  # test mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1]) + (count.max() // 2) % count
            )
            idx_part = idx_sort[idx_select]
            data_part = dict(index=idx_part)
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_part["grid_coord"] = grid_coord[idx_part]
            if self.return_min_coord:
                data_part["min_coord"] = min_coord.reshape([1, 3])
            for key in data_dict.keys():
                if key in self.keys:
                    data_part[key] = data_dict[key][idx_part]
                else:
                    data_part[key] = data_dict[key]
            return data_part
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


# Load a point cloud from a file
def load_o3d_pcd(file_path: str):
    return o3d.io.read_point_cloud(file_path)


# Get points and colors from a Open3D point cloud
def get_points_and_colors(pcd: o3d.geometry.PointCloud):
    points = np.asarray(pcd.points)
    colors = np.zeros_like(points, dtype=np.uint8)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        if colors.shape[1] == 4:
            colors = colors[:, :3]
        if colors.max() < 1.1:
            colors = (colors * 255).astype(np.uint8)
    return points, colors

# Get points and colors from a Open3D point cloud
def get_points_and_colors_and_normals(pcd: o3d.geometry.PointCloud):
    points = np.asarray(pcd.points)
    colors = np.zeros_like(points, dtype=np.uint8)
    normals = np.zeros_like(points, dtype=np.float32)
    colors = np.asarray(pcd.colors)
    if colors.shape[1] == 4:
        colors = colors[:, :3]
    if colors.max() < 1.1:
        colors = (colors * 255).astype(np.uint8)
    # 检查点云是否有法线
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=3)
        )
        normals = np.asarray(pcd.normals)
    return points, colors, normals


# Preprocess a point cloud
def cleanup_pcd(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.02,
    num_nb: int = 3,
    radius: float = 0.05,
):
    # voxelize the point cloud
    pcd = pcd.voxel_down_sample(voxel_size)
    # calculate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=num_nb)
    )
    # remove outliers
    pcd, _ = pcd.remove_radius_outlier(num_nb, radius)
    return pcd
