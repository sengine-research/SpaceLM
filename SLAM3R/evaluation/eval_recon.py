# --------------------------------------------------------
# Evaluation utilities. The code is adapted from nicer-slam: 
# https://github.com/cvg/nicer-slam/blob/main/code/evaluation/eval_rec.py
# --------------------------------------------------------
import os
from os.path import join 
import json
import trimesh
import argparse
import numpy as np
import random
import matplotlib.pyplot as pl
pl.ion()
import open3d as o3d
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Inference on a pair of images from ScanNet++")
parser.add_argument("--save_vis", action="store_true")
parser.add_argument("--seed", type=int, default=42, help="seed for python random")
parser.add_argument("--icp", type=str, default='plain', help='')
parser.add_argument("--root_dir", type=str, default='results', help='')
parser.add_argument('--dataset', type=str, default='replica')
parser.add_argument("--test_name", type=str, required=True, help='')
parser.add_argument("--gt_pcd", type=str, required=True, help='')


def save_vis(points, dis, vis_path):
    cmap = pl.get_cmap('Reds')
    color = cmap(dis/0.05)
    save_ply(points=points, save_path=vis_path, colors=color)

def save_ply(points:np.array, save_path, colors:np.array=None):
    #color:0-1
    pcd = trimesh.points.PointCloud(points, colors=colors)
    pcd.export(save_path)
    print("save_to ", save_path)

def eval_pointcloud(
    pointcloud, pointcloud_tgt, normals=None, normals_tgt=None, thresholds=np.linspace(1.0 / 1000, 1, 1000),
    vis_dir=None
):
    """Evaluates a point cloud.

    Args:
        pointcloud (numpy array): predicted point cloud
        pointcloud_tgt (numpy array): target point cloud
        normals (numpy array): predicted normals
        normals_tgt (numpy array): target normals
        thresholds (numpy array): threshold values for the F-score calculation
    """
    # Return maximum losses if pointcloud is empty
    assert len(pointcloud) > 0, "Empty pointcloud"

    pointcloud = np.asarray(pointcloud)
    pointcloud_tgt = np.asarray(pointcloud_tgt)

    # Completeness: how far are the points of the target point cloud
    # from thre predicted point cloud
    completeness, completeness_normals = distance_p2p(pointcloud_tgt, normals_tgt, pointcloud, normals)
    comp_ratio_5 = (completeness<0.05).astype(float).mean()
    if vis_dir is not None:
        save_vis(pointcloud_tgt, completeness, join(vis_dir, f"completeness.ply"))
    # print('completeness_normals', completeness_normals)
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness**2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(pointcloud, normals, pointcloud_tgt, normals_tgt)
    if vis_dir is not None:
        save_vis(pointcloud, accuracy, join(vis_dir, f"accuracy.ply"))
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals
    chamferL1 = 0.5 * (completeness + accuracy)

    # F-Score
    F = [2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-8) for i in range(len(precision))]
    
    out_dict = {
        "completeness": completeness*100,
        "accuracy": accuracy*100,
        "normals completeness": completeness_normals,
        "normals accuracy": accuracy_normals,
        "normals": normals_correctness,
        "completeness2": completeness2,
        "accuracy2": accuracy2,
        "chamfer-L2": chamferL2,
        "chamfer-L1": chamferL1,
        "f-score": F[9],  # threshold = 1.0%
        "f-score-15": F[14],  # threshold = 1.5%
        "f-score-20": F[19],  # threshold = 2.0%
        "comp_ratio-5": comp_ratio_5,
    }

    return out_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src, workers=8)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    """Evaluates a point cloud.

    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [(dist <= t).mean() for t in thresholds]
    return in_threshold


def umeyama_alignment(X, Y):
    """
    Perform Umeyama alignment to align two point sets with potential size differences.

    Parameters:
    X (numpy.ndarray): Source point set with shape (N, D).
    Y (numpy.ndarray): Target point set with shape (N, D).

    Returns:
    T (numpy.ndarray): Transformation matrix (D+1, D+1) that aligns X to Y.
    """

    # Calculate centroids
    centroid_X = np.median(X, axis=0)
    centroid_Y = np.median(Y, axis=0)

    # Center the point sets
    X_centered = X - centroid_X
    Y_centered = Y - centroid_Y

    '''
    # Covariance matrix
    sigma = np.dot(X_centered.T, Y_centered) / X.shape[0]
    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(sigma)
    # Ensure a right-handed coordinate system
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Vt[-1] = -Vt[-1]
        U[:, -1] = -U[:, -1]
    # Rotation matrix
    R = np.dot(Vt.T, U.T)
    #'''

    # solve rotation using svd with rectification.
    S = np.dot(X_centered.T, Y_centered)
    U, _, VT = np.linalg.svd(S)
    rectification = np.eye(3)
    rectification[-1,-1] = np.linalg.det(VT.T @ U.T)
    R = VT.T @ rectification @ U.T 

    # Scale factor
    sx = np.median(np.linalg.norm(X_centered, axis=1))
    sy = np.median(np.linalg.norm(Y_centered, axis=1))
    c = sy / sx

    # Translation
    t = centroid_Y - c * np.dot(R, centroid_X)

    # Transformation matrix
    T = np.zeros((X.shape[1] + 1, X.shape[1] + 1))
    T[:X.shape[1], :X.shape[1]] = c * R
    T[:X.shape[1], -1] = t
    T[-1, -1] = 1

    return T

def homogeneous(coordinates):
    homogeneous_coordinates = np.hstack((coordinates, np.ones((coordinates.shape[0], 1))))
    return homogeneous_coordinates


def SKU_RANSAC(src_pts, tar_pts):
    random.seed(args.seed)
    # generate and vote the best hypo.
    N_HYPO = 512
    ERR_MIN = 8888.
    Rt_init = np.identity(4)
    for hid in tqdm(range(N_HYPO), desc="Running umayama RANSAC"):
        ids = random.sample(range(len(src_pts)), 3)
        s_mini = src_pts[ids]
        t_mini = tar_pts[ids]
        hypo = umeyama_alignment(s_mini, t_mini)
        x = (hypo @ homogeneous(src_pts).transpose())[0:3] 
        y = homogeneous(tar_pts).transpose()[0:3]
        residuals = np.linalg.norm(x-y, axis=0)

        med_err = np.median(residuals)
        if ERR_MIN > med_err:
            ERR_MIN = med_err
            Rt_init = hypo
    # print("ERR_MIN", ERR_MIN)

    # todo: count inlier instead of median error.
    # todo: refine with inliers.

    return Rt_init



def voxelize_pcd(ori_pcd:trimesh.points.PointCloud, voxel_size=0.01):
    if voxel_size <= 0:
        return ori_pcd
    print(f"Downsample point cloud with voxel size {voxel_size}...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ori_pcd.vertices)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    return trimesh.points.PointCloud(downsampled_pcd.points)

    
def get_align_transformation(rec_pcd, gt_pcd, threshold=0.1):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    print("ICP alignment...")
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(rec_pcd))
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(gt_pcd))
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc, o3d_gt_pc, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    transformation = reg_p2p.transformation
    return transformation

    
def calcu_pcd_fscore(pcd_rec, pcd_gt, align=True, scale=1, vis_dir=None, voxel_size=0.01):
    """
    3D reconstruction metric.
    """
    pcd_rec.vertices /= scale
    pcd_gt.vertices /= scale

    pcd_rec = voxelize_pcd(pcd_rec, voxel_size=voxel_size)
    pcd_gt = voxelize_pcd(pcd_gt, voxel_size=voxel_size)

    if align:
        transformation = get_align_transformation(pcd_rec, pcd_gt, threshold=voxel_size*2)
        pcd_rec = pcd_rec.apply_transform(transformation)

    rec_pointcloud = pcd_rec.vertices.astype(np.float32)
    gt_pointcloud = pcd_gt.vertices.astype(np.float32)

    out_dict = eval_pointcloud(rec_pointcloud, gt_pointcloud, vis_dir=vis_dir)
    
    return out_dict


def align_pcd(source:np.array, target:np.array, icp=None, init_trans=None, mask=None, return_trans=True, voxel_size=0.1):
    """ Align the scale of source to target using umeyama,
    then refine the alignment using ICP.
    """
    if init_trans is not None:
        source = trimesh.transformations.transform_points(source, init_trans)
    #####################################
    # first step registration using umeyama.
    #####################################
    source_for_align = source if mask is None else source[mask]
    target = target if mask is None else target[mask]
    Rt_step1 = SKU_RANSAC(source_for_align, target)
    source_step1 = (Rt_step1 @ homogeneous(source_for_align).transpose())[0:3].transpose()
    #####################################
    # second step registration using icp.
    #####################################
    print("point-to-plane ICP...")
    icp_thr = voxel_size * 2

    pcd_source_step1 = o3d.geometry.PointCloud()
    pcd_source_step1.points = o3d.utility.Vector3dVector(source_step1)
    pcd_source_step1 = pcd_source_step1.voxel_down_sample(voxel_size=voxel_size)
    
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target)
    pcd_target = pcd_target.voxel_down_sample(voxel_size=voxel_size)
    pcd_target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    if icp == "point":
        icp_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    elif icp == 'plain':
        icp_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        raise ValueError
    reg_p2l = o3d.pipelines.registration.registration_icp(
        pcd_source_step1, pcd_target, icp_thr, np.identity(4), icp_method)

    Rt_step2 = reg_p2l.transformation
    
    # apply RT on initial source without downsample
    transformation_s2t = Rt_step2 @ Rt_step1
    transformed_source = trimesh.transformations.transform_points(source, transformation_s2t)
    if return_trans:
        return transformed_source, transformation_s2t
    else:
        return transformed_source
    
def calcu_pair_loss(align_gt_pcd, align_pred_pcd,
                    eval_gt_pcd=None, eval_pred_pcd=None, 
                    c2w=None, vis_dir=None,icp='plain',
                    voxel_size=0.01):
    """
    Keep the original scale of gt.
    First align the predicted pcd to the gt pcd with umeyama+icp,
    then calculating reconstruction metrics.
    """
    if eval_gt_pcd is None:
        eval_gt_pcd = align_gt_pcd
    if eval_pred_pcd is None:
        eval_pred_pcd = align_pred_pcd
    
    # align the predicted pcd to the gt pcd with umeyama+icp
    _, trans = align_pcd(align_pred_pcd, align_gt_pcd, 
                         init_trans=c2w, 
                         mask=None,
                         icp=icp, 
                         return_trans=True,
                         voxel_size=voxel_size*2)
    
    aligned_eval_pred_pcd = trimesh.transformations.transform_points(eval_pred_pcd, trans) 

    aligned_eval_pred_pcd = trimesh.points.PointCloud(aligned_eval_pred_pcd)
    gt_pcd = trimesh.points.PointCloud(eval_gt_pcd)

    # Calculate the reconstruction metrics
    res2 = calcu_pcd_fscore(aligned_eval_pred_pcd, gt_pcd, 
                            scale=1, align=True, vis_dir=vis_dir,
                            voxel_size=voxel_size)  
    align_flag = True
    if res2["completeness"] > 10 or res2["accuracy"] > 10:
        align_flag = False

    return res2, align_flag


if __name__ == "__main__":
    """
    The script consists of two parts:
    1. Align the predicted point cloud with the ground truth point cloud using the Umeyama and ICP algorithms. 
    2. calculate the reconstruction metrics.
    """
    args = parser.parse_args()
    print(args)
    res_dir = os.path.join(args.root_dir, args.test_name)
    save_dir = os.path.join(res_dir, 'eval')
    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(args.seed)
    
    eval_conf_thres = 3
    num_sample_points = 200000
    voxelize_size = 0.005
    
    gt_pcd = np.load(args.gt_pcd).astype(np.float32)
    valid_masks = np.load(args.gt_pcd.replace('_pcds', '_valid_masks')).astype(bool)
    pred_pcd = np.load(join(res_dir, 'preds', 'registered_pcds.npy')).astype(np.float32)
    pred_confs = np.load(join(res_dir, 'preds', 'registered_confs.npy')).astype(np.float32)
        
    # filter out points with conficence and valid masks
    pred_confs[~valid_masks] = 0
    valid_ids = pred_confs > eval_conf_thres
    gt_pcd = gt_pcd[valid_ids]
    pred_pcd = pred_pcd[valid_ids]

    
    # prepare the pcds for alignment and evaluation
    assert gt_pcd.shape[0] > num_sample_points
    sample_ids = np.random.choice(gt_pcd.shape[0], num_sample_points, replace=False)
    gt_pcd = gt_pcd[sample_ids]
    pred_pcd = pred_pcd[sample_ids]
        
    metric_dict, align_succ = calcu_pair_loss(gt_pcd, 
                                            pred_pcd,
                                            c2w=None,
                                            vis_dir=save_dir,
                                            voxel_size=voxelize_size,
                                            )

    metric_str = "Acc:{:.3f}, Comp:{:.3f}, f-score20:{:.3f}, f-score15:{:.3f}, f-score10:{:.3f}".format(
        metric_dict["accuracy"], metric_dict["completeness"], metric_dict["f-score-20"], 
        metric_dict["f-score-15"], metric_dict["f-score"])


    res_dict = {}
    res_dict['metric'] = metric_str    
    res_dict['align_succ'] = str(align_succ)
    print(res_dict)
    save_name = f"thres{eval_conf_thres}_recon_metric.json"
    with open(join(save_dir, save_name), 'w') as f:
        json.dump(res_dict, f, indent="")