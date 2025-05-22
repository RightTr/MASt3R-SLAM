import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement
import os
import csv
from evo.core.trajectory import PoseTrajectory3D
from evo.core import metrics
from evo.core.metrics import PoseRelation, StatisticsType
from evo.tools.plot import PlotMode, prepare_axis, traj, traj_colormap
from evo.core import sync
from matplotlib import pyplot as plt
import matplotlib
import copy

matplotlib.use('Agg')

def prepare_savedir(args, dataset):
    if "SmokeBasement" in str(dataset.dataset_path).split("/"):
        last_path = os.path.join(*os.path.normpath(dataset.dataset_path).split(os.sep)[-2:])
    else:
        last_path = os.path.basename(dataset.dataset_path)

    save_dir = pathlib.Path(f"logs/{last_path}")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name

def xyzw_to_wxyz(vec):
    t = vec[:3]
    q_xyzw = vec[3:]
    q_wxyz = np.roll(q_xyzw, 1)
    return np.concatenate([t, q_wxyz])

def evaluate_evo(poses_gt, poses_est, timestamps_kf,
                savedir, monocular=False):
    assert len(poses_gt) == len(poses_est)
    poses_gt = np.array(poses_gt)
    poses_est = np.array(poses_est)

    traj_est = PoseTrajectory3D(
    positions_xyz=poses_est[:, :3],
    orientations_quat_wxyz=poses_est[:, 3:], 
    timestamps=np.array(timestamps_kf, dtype=np.float64))

    traj_ref = PoseTrajectory3D(
        positions_xyz=poses_gt[:, :3],
        orientations_quat_wxyz=poses_gt[:, 3:],
        timestamps=np.array(timestamps_kf, dtype=np.float64))

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=monocular)

    ape_metric = metrics.APE(PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est_aligned))

    ape_stats = ape_metric.get_all_statistics()
    rmse = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    for key, value in ape_stats.items():
        ape_stats[key] = float(value)

    plot_mode = PlotMode.xy
    fig = plt.figure()
    ax = prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {rmse}")
    traj(ax, plot_mode, traj_ref, "--", "gray", "gt", plot_start_end_markers=True)
    traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
        plot_start_end_markers=True,
    )
    ax.legend()
    plt.savefig(os.path.join(savedir, "evo_2dplot.png"), dpi=90)

    with open(os.path.join(savedir, "metrics.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ATE"])
        writer.writerow([rmse])


def evaluate(savedir, poses_gt_input, 
             keyframes: SharedKeyframes, 
             intrinsics: Optional[Intrinsics] = None):
    poses_est, poses_gt, timestamps_kf = [], [], []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if intrinsics is None:
            T_WC = as_SE3(keyframe.T_WC)
        else:
            T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            
        poses_est.append(xyzw_to_wxyz(T_WC.data.numpy().reshape(-1))) 
        poses_gt.append(xyzw_to_wxyz(poses_gt_input[keyframe.frame_id]))
        timestamps_kf.append(keyframe.frame_id)
        
    evaluate_evo(poses_gt, poses_est, timestamps_kf, savedir, monocular=True)
            
def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_ply(savedir / filename, pointclouds, colors)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )


def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)
