import pathlib
import re
import cv2
from natsort import natsorted
import numpy as np
import torch
import pyrealsense2 as rs
import yaml
import glob
import os

from mast3r_slam.vggt_utils import resize_img
from mast3r_slam.config import config
from scipy.spatial.transform import Rotation, Slerp

HAS_TORCHCODEC = True
try:
    from torchcodec.decoders import VideoDecoder
except Exception as e:
    HAS_TORCHCODEC = False


class MonocularDataset(torch.utils.data.Dataset):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
        self.rgb_files = []
        self.timestamps = []
        self.img_size = 512
        self.camera_intrinsics = None
        self.use_calibration = config["use_calib"]
        self.save_results = True

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Call get_image before timestamp for realsense camera
        img = self.get_image(idx)
        timestamp = self.get_timestamp(idx)
        return timestamp, img

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        img = cv2.imread(self.rgb_files[idx])
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_image(self, idx):
        img = self.read_img(idx)
        if self.use_calibration:
            img = self.camera_intrinsics.remap(img)
        return img.astype(self.dtype) / 255.0

    def get_img_shape(self):
        img = self.read_img(0)
        print(img.shape)
        w_raw, h_raw = img.shape[1], img.shape[0]
        img = resize_img(img)
        return img['img'][0].shape[0], img['img'][0].shape[1], w_raw, h_raw

    def subsample(self, subsample):
        self.rgb_files = self.rgb_files[::subsample]
        self.timestamps = self.timestamps[::subsample]

    def has_calib(self):
        return self.camera_intrinsics is not None
    
class StereoDataset(torch.utils.data.Dataset):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
        self.rgb_files_left = []
        self.rgb_files_right = []
        self.timestamps = []
        self.img_size = 512
        self.camera_intrinsics_left = None
        self.camera_intrinsics_right = None
        self.use_calibration = config["use_calib"]
        self.save_results = False # TODO:Stereo datasets do not save results by default

    def __len__(self):
        return len(self.rgb_files_left)

    def __getitem__(self, idx):
        # Call get_image before timestamp for realsense camera
        img_left = self.get_image_left(idx)
        img_right = self.get_image_right(idx)
        timestamp = self.get_timestamp(idx)
        return timestamp, img_left, img_right

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img_left(self, idx):
        img = cv2.imread(self.rgb_files_left[idx])
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def read_img_right(self, idx):
        img = cv2.imread(self.rgb_files_right[idx])
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_image_left(self, idx):
        img = self.read_img_left(idx)
        if self.use_calibration:
            img = self.camera_intrinsics_left.remap(img)
        return img.astype(self.dtype) / 255.0
    
    def get_image_right(self, idx):
        img = self.read_img_right(idx)
        if self.use_calibration:
            img = self.camera_intrinsics_right.remap(img)
        return img.astype(self.dtype) / 255.0

    def get_img_shape(self):
        img_left = self.read_img_left(0)
        img_right = self.read_img_right(0)
        assert len(img_left) == len(img_right), "Left and right image sizes do not match!"
        raw_img_shape = img_left.shape
        img = resize_img(img_left, self.img_size)
        # 3XHxW, HxWx3 -> HxW, HxW
        return img[0][0].shape, raw_img_shape[:2]

    def subsample(self, subsample):
        self.rgb_files_left = self.rgb_files_left[::subsample]
        self.rgb_files_right = self.rgb_files_right[::subsample]
        self.timestamps = self.timestamps[::subsample]

    def has_calib(self):
        return self.camera_intrinsics_left is not None and self.camera_intrinsics_right is not None

class TUMDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        rgb_list = self.dataset_path / "rgb.txt"
        tstamp_rgb = np.loadtxt(rgb_list, delimiter=" ", dtype=np.unicode_, skiprows=0)
        self.rgb_files = [self.dataset_path / f for f in tstamp_rgb[:, 1]]
        self.timestamps = tstamp_rgb[:, 0]

        match = re.search(r"freiburg(\d+)", dataset_path)
        idx = int(match.group(1))
        if idx == 1:
            calib = np.array(
                [517.3, 516.5, 318.6, 255.3, 0.2624, -0.9531, -0.0054, 0.0026, 1.1633]
            )
        if idx == 2:
            calib = np.array(
                [520.9, 521.0, 325.1, 249.7, 0.2312, -0.7849, -0.0033, -0.0001, 0.9172]
            )
        if idx == 3:
            calib = np.array([535.4, 539.2, 320.1, 247.6])
        W, H = 640, 480
        self.camera_intrinsics = Intrinsics.from_calib(self.img_size, W, H, calib)

class VIVIDDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        self.rgb_files = sorted(glob.glob(os.path.join(self.dataset_path, "RGB/data/*.png")))
        self.n_img = len(self.rgb_files)
        self.poses = self.load_poses(os.path.join(self.dataset_path, "gt_RGB.txt"))
        self.timestamps = [os.path.splitext(os.path.basename(f))[0] for f in self.rgb_files]
        calib = np.array([437.38861083256637, 437.29475745770907, 323.5284494924228, 256.36315482047905, 0, 0, 0, 0, 0])
        W, H = 640, 480
        self.camera_intrinsics = Intrinsics.from_calib(self.img_size, W, H, calib)

    def load_poses(self, path):
        poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(1, self.n_img):
            line = lines[i]
            vec = np.array(list(map(float, line.split()))[1:8])
            poses.append(vec)
        return poses

class RRXIODataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        self.max_dt = 0.08
        self.frame_rate = 30
        self.rgb_files = []
        self.poses = []
        imgs_data = np.loadtxt(os.path.join(self.dataset_path, "thermal_undistort.txt"), delimiter=" ", dtype=np.unicode_)
        poses_data = np.loadtxt(os.path.join(self.dataset_path, "gt_thermal.txt"), delimiter=" ", dtype=np.unicode_, skiprows=1)
        calib = np.array([334.19639643, 334.26241379, 318.48142004, 250.56663663, 0, 0, 0, 0, 0])
        W, H = 640, 512
        self.camera_intrinsics = Intrinsics.from_calib(self.img_size, W, H, calib)
        
        tstamp_image = imgs_data[:, 0].astype(np.float64)
        tstamp_pose = poses_data[:, 0].astype(np.float64)
        
        associations = self.associate_frames(tstamp_image, tstamp_pose)

        print('Found {} associations out of {} images and {} poses!'.format(
            len(associations), len(tstamp_image), len(tstamp_pose)))

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / self.frame_rate:
                indicies += [i]

        for ix in indicies:
            (i, j) = associations[ix]
            self.rgb_files += [os.path.join(self.dataset_path, imgs_data[i, 1])]
            vec= poses_data[j, 1:8].astype(np.float64)
            self.poses.append(vec)

        self.timestamps = [os.path.splitext(os.path.basename(f))[0] for f in self.rgb_files]
        
    def associate_frames(self, timestamp_image, timestamp_pose):
        associations = []
        for i, t in enumerate(timestamp_image):
            if timestamp_pose is not None:
                j = np.argmin(np.abs(timestamp_pose - t))
                if np.abs(timestamp_pose[j] - t) < self.max_dt:
                    associations.append((i, j))
        return associations
    
class SmokeBasementDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        self.poses = []
        run_dir = os.path.dirname(self.dataset_path)
        right_or_left = os.path.basename(self.dataset_path)
        poses_data = np.loadtxt(os.path.join(run_dir, "kissicp_poses.txt"), delimiter=" ", dtype=np.unicode_)

        with open(os.path.join(self.dataset_path, "times.txt")) as f:
            lines = f.readlines()
        timestamps = []
        for line in lines:
            line = line.strip()
            ts_str = line.split(",")[0].strip()
            timestamps.append(ts_str)
        imgs_tstamp = np.array(timestamps)

        calib = np.zeros(9)
        lidar2cam = np.eye(4)
        if right_or_left == "right":
            lidar2cam = np.array([[-1, 0, 0, 0.06],
                            [0, 0, -1, -0.072],
                            [0, -1, 0, -0.145],
                            [0, 0, 0, 1]])
            calib = np.array([358.484823, 357.311578, 317.492363, 267.103537, -0.212959, 0.039110, 0.000939, 0.001243, 0])
        elif right_or_left == "left":
            lidar2cam = np.array([[-1, 0, 0, -0.06],
                            [0, 0, -1, 0.072],
                            [0, -1, 0, 0.145],
                            [0, 0, 0, 1]])
            calib = np.array([358.009390, 356.631007, 320.649615, 268.477546, -0.210408, 0.037092, 0.000217, 0.000702, 0])

        for t in imgs_tstamp:
            self.rgb_files += [os.path.join(self.dataset_path, "image", t + '.png')]
            T = self.linear_interpol(poses_data, float(t))
            self.poses.append(self.matrix2vec(np.dot(T, np.linalg.inv(lidar2cam))))

        self.timestamps = [os.path.splitext(os.path.basename(f))[0] for f in self.rgb_files]

        W, H = 640, 512
        self.camera_intrinsics = Intrinsics.from_calib(self.img_size, W, H, calib)

    def linear_interpol(self, pose_data, time):
        times = pose_data[:, 0].astype(np.float64)
        poses = pose_data[:, 1:].astype(np.float64)

        interpolated_translation = np.array([np.interp(time, times, poses[:, i]) for i in range(3)])
        quaternions = Rotation.from_quat(poses[:, 3:7])
        if time <= times[0]:
            interpolated_rotation = quaternions[0]
        elif time >= times[-1]:
            interpolated_rotation = quaternions[-1]
        else:
            slerp = Slerp(times, quaternions)
            interpolated_rotation = slerp(time)

        rotation_matrix = interpolated_rotation.as_matrix()
        interpolated_pose = np.eye(4)
        interpolated_pose[0:3, 3] = interpolated_translation
        interpolated_pose[0:3, 0:3] = rotation_matrix
        return interpolated_pose

    def matrix2vec(self, mat):
        t = mat[:3, 3]
        rot_matrix = mat[:3, :3]
        quat = Rotation.from_matrix(rot_matrix).as_quat() 
        vec = np.concatenate([t, quat])
        return vec
    
class NTU4DRadLMDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        self.rgb_files = sorted(glob.glob(os.path.join(self.dataset_path, "thermal/*.png")))
        self.n_img = len(self.rgb_files)
        self.poses = []
        # self.poses = self.load_poses(os.path.join(self.dataset_path, "gt_thermal.txt"))
        self.timestamps = [os.path.splitext(os.path.basename(f))[0] for f in self.rgb_files]
        
        calib = np.array([471.96351324104091, 339.03066128694218, 472.48642748309049, 277.74073717116710, 
                          -1.8566954779749040e-01, 1.6745260846914475e-01, -1.8122010952647307e-04, 8.6534037842673963e-05, -1.0770856460153226e-01])
        W, H = 640, 512
        self.camera_intrinsics = Intrinsics.from_calib(self.img_size, W, H, calib)

    def load_poses(self, path):
        poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(1, self.n_img):
            line = lines[i]
            vec = np.array(list(map(float, line.split()))[1:8])
            poses.append(vec)
        return poses
    
class Nus822Dataset(StereoDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        self.rgb_files_left = sorted(glob.glob(os.path.join(self.dataset_path, "left/image/*.png")))
        self.rgb_files_right = sorted(glob.glob(os.path.join(self.dataset_path, "right/image/*.png")))
        assert len(self.rgb_files_left) == len(self.rgb_files_right), "Left and right image amounts do not match!"
        self.n_img = len(self.rgb_files_left)
        self.poses = []
        # self.poses = self.load_poses(os.path.join(self.dataset_path, "gt_thermal.txt"))
        self.timestamps = [os.path.splitext(os.path.basename(f))[0] for f in self.rgb_files_left]
        calib_left = np.array([358.009390, 356.631007, 320.649615, 268.477546, -0.210408, 0.037092, 0.000217, 0.000702, 0])
        calib_right = np.array([358.484823, 357.311578, 317.492363, 267.103537, -0.212959, 0.039110, 0.000939, 0.001243, 0])
        
        W, H = 640, 512
        self.camera_intrinsics_left = Intrinsics.from_calib(self.img_size, W, H, calib_left)
        self.camera_intrinsics_right = Intrinsics.from_calib(self.img_size, W, H, calib_right)

class EurocDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        # For Euroc dataset, the distortion is too much to handle for MASt3R.
        # So we always undistort the images, but the calibration will not be used for any later optimization unless specified.
        self.use_calibration = True
        self.dataset_path = pathlib.Path(dataset_path)
        rgb_list = self.dataset_path / "mav0/cam0/data.csv"
        tstamp_rgb = np.loadtxt(rgb_list, delimiter=",", dtype=np.unicode_, skiprows=0)
        self.rgb_files = [
            self.dataset_path / "mav0/cam0/data" / f for f in tstamp_rgb[:, 1]
        ]
        self.timestamps = tstamp_rgb[:, 0]
        with open(self.dataset_path / "mav0/cam0/sensor.yaml") as f:
            self.cam0 = yaml.load(f, Loader=yaml.FullLoader)
        W, H = self.cam0["resolution"]
        intrinsics = self.cam0["intrinsics"]
        distortion = np.array(self.cam0["distortion_coefficients"])
        self.camera_intrinsics = Intrinsics.from_calib(
            self.img_size, W, H, [*intrinsics, *distortion], always_undistort=True
        )

    def read_img(self, idx):
        img = cv2.imread(self.rgb_files[idx], cv2.IMREAD_GRAYSCALE)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class ETH3DDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        rgb_list = self.dataset_path / "rgb.txt"
        tstamp_rgb = np.loadtxt(rgb_list, delimiter=" ", dtype=np.unicode_, skiprows=0)
        self.rgb_files = [self.dataset_path / f for f in tstamp_rgb[:, 1]]
        self.timestamps = tstamp_rgb[:, 0]
        calibration = np.loadtxt(
            self.dataset_path / "calibration.txt",
            delimiter=" ",
            dtype=np.float32,
            skiprows=0,
        )
        _, (H, W) = self.get_img_shape()
        self.camera_intrinsics = Intrinsics.from_calib(self.img_size, W, H, calibration)


class SevenScenesDataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        self.rgb_files = natsorted(
            list((self.dataset_path / "seq-01").glob("*.color.png"))
        )
        self.timestamps = np.arange(0, len(self.rgb_files)).astype(self.dtype)
        fx, fy, cx, cy = 585.0, 585.0, 320.0, 240.0
        self.camera_intrinsics = Intrinsics.from_calib(
            self.img_size, 640, 480, [fx, fy, cx, cy]
        )


class RealsenseDataset(MonocularDataset):
    def __init__(self):
        super().__init__()
        self.dataset_path = None
        self.pipeline = rs.pipeline()
        # self.h, self.w = 720, 1280
        self.h, self.w = 480, 640
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.color, self.w, self.h, rs.format.bgr8, 30
        )
        self.profile = self.pipeline.start(self.rs_config)

        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        # self.rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        # self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        # self.rgb_sensor.set_option(rs.option.exposure, 200)
        self.rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )
        self.save_results = False

        if self.use_calibration:
            rgb_intrinsics = self.rgb_profile.get_intrinsics()
            self.camera_intrinsics = Intrinsics.from_calib(
                self.img_size,
                self.w,
                self.h,
                [
                    rgb_intrinsics.fx,
                    rgb_intrinsics.fy,
                    rgb_intrinsics.ppx,
                    rgb_intrinsics.ppy,
                ],
            )

    def __len__(self):
        return 999999

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        frameset = self.pipeline.wait_for_frames()
        timestamp = frameset.get_timestamp()
        timestamp /= 1000
        self.timestamps.append(timestamp)

        rgb_frame = frameset.get_color_frame()
        img = np.asanyarray(rgb_frame.get_data())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(self.dtype)
        return img


class Webcam(MonocularDataset):
    def __init__(self):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = None
        # load webcam using opencv
        self.cap = cv2.VideoCapture(-1)
        self.save_results = False

    def __len__(self):
        return 999999

    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def read_img(self, idx):
        ret, img = self.cap.read()
        if not ret:
            raise ValueError("Failed to read image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.timestamps.append(idx / 30)

        return img


class MP4Dataset(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = pathlib.Path(dataset_path)
        if HAS_TORCHCODEC:
            self.decoder = VideoDecoder(str(self.dataset_path))
            self.fps = self.decoder.metadata.average_fps
            self.total_frames = self.decoder.metadata.num_frames
        else:
            print("torchcodec is not installed. This may slow down the dataloader")
            self.cap = cv2.VideoCapture(str(self.dataset_path))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.stride = config["dataset"]["subsample"]

    def __len__(self):
        return self.total_frames // self.stride

    def read_img(self, idx):
        if HAS_TORCHCODEC:
            img = self.decoder[idx * self.stride]  # c,h,w
            img = img.permute(1, 2, 0)
            img = img.numpy()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx * self.stride)
            ret, img = self.cap.read()
            if not ret:
                raise ValueError("Failed to read image")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(self.dtype)
        timestamp = idx / self.fps
        self.timestamps.append(timestamp)
        return img

class RGBFiles(MonocularDataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.use_calibration = False
        self.dataset_path = pathlib.Path(dataset_path)
        self.rgb_files = natsorted(list((self.dataset_path).glob("*.png")))
        self.timestamps = np.arange(0, len(self.rgb_files)).astype(self.dtype) / 30.0


class Intrinsics:
    def __init__(self, img_size, W, H, K_orig, K, distortion, mapx, mapy):
        self.img_size = img_size
        self.W, self.H = W, H
        self.K_orig = K_orig
        self.K = K
        self.distortion = distortion
        self.mapx = mapx
        self.mapy = mapy
        _, (scale_w, scale_h, half_crop_w, half_crop_h) = resize_img(
            np.zeros((H, W, 3)), return_transformation=True
        )
        self.K_frame = self.K.copy()
        self.K_frame[0, 0] = self.K[0, 0] / scale_w
        self.K_frame[1, 1] = self.K[1, 1] / scale_h
        self.K_frame[0, 2] = self.K[0, 2] / scale_w - half_crop_w
        self.K_frame[1, 2] = self.K[1, 2] / scale_h - half_crop_h

    def remap(self, img):
        return cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)

    @staticmethod
    def from_calib(img_size, W, H, calib, always_undistort=False):
        if not config["use_calib"] and not always_undistort:
            return None
        fx, fy, cx, cy = calib[:4]
        distortion = np.zeros(4)
        if len(calib) > 4:
            distortion = np.array(calib[4:])
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        K_opt = K.copy()
        mapx, mapy = None, None
        center = config["dataset"]["center_principle_point"]
        K_opt, _ = cv2.getOptimalNewCameraMatrix(
            K, distortion, (W, H), 0, (W, H), centerPrincipalPoint=center
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            K, distortion, None, K_opt, (W, H), cv2.CV_32FC1
        )

        return Intrinsics(img_size, W, H, K, K_opt, distortion, mapx, mapy)


def load_dataset(dataset_path):
    split_dataset_type = dataset_path.split("/")
    if "tum" in split_dataset_type:
        return TUMDataset(dataset_path)
    if "euroc" in split_dataset_type:
        return EurocDataset(dataset_path)
    if "eth3d" in split_dataset_type:
        return ETH3DDataset(dataset_path)
    if "7-scenes" in split_dataset_type:
        return SevenScenesDataset(dataset_path)
    if "realsense" in split_dataset_type:
        return RealsenseDataset()
    if "webcam" in split_dataset_type:
        return Webcam()
    if "KAIST_VIVID" in split_dataset_type:
        return VIVIDDataset(dataset_path)
    if "rrxio" in split_dataset_type:
        return RRXIODataset(dataset_path)
    if "SmokeBasement" in split_dataset_type:
        return SmokeBasementDataset(dataset_path)
    if "NTU4DRadLM" in split_dataset_type:
        return NTU4DRadLMDataset(dataset_path)
    if "nus822" in split_dataset_type:
        return Nus822Dataset(dataset_path)

    ext = split_dataset_type[-1].split(".")[-1]
    if ext in ["mp4", "avi", "MOV", "mov"]:
        return MP4Dataset(dataset_path)
    return RGBFiles(dataset_path)
