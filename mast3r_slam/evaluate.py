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
import lpips
import torchvision.transforms as transforms
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import csv
from scipy.spatial.transform import Rotation as R
from evo.core.trajectory import PoseTrajectory3D
from evo.core import metrics
from evo.core.metrics import PoseRelation, StatisticsType
from evo.core import sync
import copy

def prepare_savedir(args, dataset):
    last_path = os.path.basename(str(dataset.dataset_path))
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

def compute_ate(poses_gt, poses_est, timestamps_kf, 
                monocular=False):
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

    return ape_metric.get_statistic(StatisticsType.rmse)

def evaluate(savedir, timestamps, imgsdir_gt, posesdir_gt, 
             keyframes: SharedKeyframes, 
             intrinsics: Optional[Intrinsics] = None,
             associated_frames = None):
    transform = transforms.Compose([transforms.ToTensor()])
    lpips_model = lpips.LPIPS(net='alex').to("cuda:0")
    psnrs, ssims, lpips_scores = [], [], []
    poses_est, poses_gt, timestamps_kf = [], [], []
    csv_path = os.path.join(savedir, "metrics.csv")
    pose_line_idx = -1
    with open(posesdir_gt, "r") as f:
        lines = [line for line in f.readlines() if not line.strip().startswith("#")]
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if intrinsics is None:
            T_WC = as_SE3(keyframe.T_WC)
        else:
            T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            
        poses_est.append(xyzw_to_wxyz(T_WC.data.numpy().reshape(-1))) 
        if associated_frames is not None:
            t = timestamps[keyframe.frame_id]
            t = os.path.splitext(t)[0]
            timestamps_kf.append(t)
            for frame in associated_frames:
                img_path = frame["rgb_files"]
                img_name_with_ext = os.path.basename(img_path) 
                img_time, _ = os.path.splitext(img_name_with_ext) 
                if abs(float(img_time) - float(t)) < 1e-6:
                    pose_tstamp = frame["pose_tstamp"]
                    break

            for idx, line in enumerate(lines):
                data = line.strip().split()
                tstamp = float(data[0])
                if abs(float(tstamp) - float(pose_tstamp)) < 1e-6:
                    pose_line_idx = idx
                    break
                
            poses_gt.append(xyzw_to_wxyz(np.array(list(map(float, lines[int(pose_line_idx)].split())))[1:8]))
        else: 
            t = timestamps[keyframe.frame_id]
            timestamps_kf.append(t)
            poses_gt.append(xyzw_to_wxyz(np.array(list(map(float, lines[int(t)].split())))[1:8]))

        image_gt = np.array(Image.open(os.path.join(imgsdir_gt, f"{t}.png")).convert("RGB"))
        image_est = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8)
        if image_gt.shape != image_est.shape:
            image_gt = Image.fromarray(image_gt).resize((image_est.shape[1], image_est.shape[0]), Image.BILINEAR)
            image_gt = np.array(image_gt)
        image_est_norm = image_est.astype(np.float32) / 255.0
        image_gt_norm = image_gt.astype(np.float32) / 255.0
        est_tensor = transform(image_est_norm).unsqueeze(0).to("cuda:0")
        gt_tensor = transform(image_gt_norm).unsqueeze(0).to("cuda:0")
        # ssim
        ssim_score = ssim(image_est_norm, image_gt_norm, channel_axis=-1, data_range=1.0)
        ssims.append(ssim_score)
        # psnr
        mse = torch.mean((est_tensor - gt_tensor) ** 2)
        psnr_score = 20 * torch.log10(1.0 / torch.sqrt(mse))
        psnrs.append(psnr_score.item())
        # lpips
        lpips_val = lpips_model(est_tensor, gt_tensor)
        lpips_scores.append(lpips_val.item())

    psnr_mean = np.mean(psnrs)
    ssim_mean = np.mean(ssims)
    lpips_mean = np.mean(lpips_scores)
    ate = compute_ate(poses_gt, poses_est, timestamps_kf, monocular=True)
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["PSNR", "SSIM", "LPIPS", "ATE"])
        writer.writerow([psnr_mean, ssim_mean, lpips_mean, ate])
            
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
