import sys
import os.path as path
HERE_PATH = path.normpath(path.dirname(__file__))
VGGT_REPO_PATH = path.normpath(path.join(HERE_PATH, '../thirdparty/vggt'))
VGGT_LIB_PATH = path.join(VGGT_REPO_PATH, 'vggt')
if path.isdir(VGGT_LIB_PATH):
    sys.path.insert(0, VGGT_REPO_PATH)
else:
    raise ImportError(f"vggt is not initialized, could not find: {VGGT_LIB_PATH}.\n ")

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from mast3r_slam.frame import Frame
from vggt.models.vggt import VGGT
from torchvision import transforms as TF
import einops
import lietorch
from scipy.spatial.transform import Rotation as R_scipy
import mast3r_slam.matching as matching

from vggt.utils.pose_enc import pose_encoding_to_extri_intri

import numpy as np

@torch.inference_mode
def vggt_inference_mono(model, frame):
    img = frame.img.unsqueeze(0).unsqueeze(1)
    aggregated_tokens_list, patch_start_idx = model.aggregator(img)
    # if frame.feat is None:
    #     frame.feat = self.feature_extractor(aggregated_tokens_list, img, patch_start_idx)
    X, C = model.point_head(
                    aggregated_tokens_list, images=img, patch_start_idx=patch_start_idx
                )
    Xii = einops.rearrange(X[:, 0], "b h w c -> b (h w) c")
    Cii = einops.rearrange(C[:, 0], "b h w -> b (h w) 1")

    # b, a, c = Xii.shape
    # assert c == 3, "Each point must have 3 coordinates (x, y, z)"

    # points = Xii.cpu().numpy()

    # valid = np.isfinite(points).all(axis=2) & (points[:, :, 2] > 0)
    # points = points[valid]

    # with open("/home/pi/Documents/Right/MASt3R-SLAM/temp/Xii_points.ply", 'w') as f:
    #     f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
    #     f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
    #     for p in points:
    #         f.write(f"{p[0]} {p[1]} {p[2]}\n")

    # print(f"✅ Saved {len(points)} point")


    return Xii, Cii
    
@torch.inference_mode
def vggt_asymmetric_inference(model, frame_i, frame_j):
    img_i = frame_i.img.unsqueeze(0).unsqueeze(1)
    img_j = frame_j.img.unsqueeze(0).unsqueeze(1)
    img_size = frame_i.img.shape[-2:]
    imgs = torch.cat([img_i, img_j], dim=1)
    aggregated_tokens_list, patch_start_idx = model.aggregator(imgs)
    # if frame_i.feat is None:
    #     feats = model.feature_extractor(aggregated_tokens_list, imgs, patch_start_idx) 
    #     frame_i.feat = feats[:, 0]

    # if frame_j.feat is None:
    #     feats = model.feature_extractor(aggregated_tokens_list, imgs, patch_start_idx)
    #     frame_i.feat = feats[:, 0]

    P = model.camera_head(aggregated_tokens_list)[-1]

    extrinsics, _ = pose_encoding_to_extri_intri(P, img_size) 

    T_CiCj = closed_form_sim3(extrinsics[:, 1]) 
    
    X, C = model.point_head(
                    aggregated_tokens_list, imgs, patch_start_idx=patch_start_idx
                )
    Xii = X[:, 0]
    Xij = X[:, 1]
    Cii = C[:, 0]
    Cij = C[:, 1]

    return T_CiCj, Xii, Xij, Cii, Cij

def vggt_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):
    TCiCj, Xii, Xij, Cii, Cij = vggt_asymmetric_inference(model, frame_i, frame_j)

    idx_i2j, valid_match_j = matching.mymatch_iterative_proj(
        Xii, Xij, TCiCj, idx_i_to_j_init=idx_i2j_init
    )

    Xii = einops.rearrange(Xii[0, :], "h w c -> (h w) c")
    Cii = einops.rearrange(Cii[0, :], "h w -> (h w) 1")
    Xij = einops.rearrange(Xij[0, :], "h w c -> (h w) c")
    Cij = einops.rearrange(Cij[0, :], "h w -> (h w) 1")
    
    # b, a, c = Xij.shape
    # assert c == 3, "Each point must have 3 coordinates (x, y, z)"

    # points = Xij.cpu().numpy()

    # valid = np.isfinite(points).all(axis=2) & (points[:, :, 2] > 0)
    # points = points[valid]

    # with open("/home/pi/Documents/Right/MASt3R-SLAM/temp/Xij_points.ply", 'w') as f:
    #     f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
    #     f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
    #     for p in points:
    #         f.write(f"{p[0]} {p[1]} {p[2]}\n")

    # print(f"✅ Saved {len(points)} point")

    return idx_i2j, valid_match_j, TCiCj, Xii, Cii, Xij, Cij

def closed_form_sim3(se3, scale = 1.0, R=None, t=None):
    if se3.shape[-2:] == (3, 4):  # expand to 4x4 if needed
        bottom = torch.tensor([0, 0, 0, 1], device=se3.device, dtype=se3.dtype).view(1, 1, 4).repeat(se3.shape[0], 1, 1)
        se3 = torch.cat([se3, bottom], dim=1)

    if R is None:
        R = se3[:, :3, :3]
    if t is None:
        t = se3[:, :3, 3:]

    R_np = R.cpu().numpy()
    quats = []
    for i in range(R_np.shape[0]):
        r = R_scipy.from_matrix(R_np[i])
        q = r.as_quat()  # [x,y,z,w]
        quats.append(q)

    quats = np.array(quats)
    quats = torch.tensor(quats, device=R.device, dtype=R.dtype)

    t = t.squeeze(-1)
    scale = torch.full((se3.shape[0], 1), scale, dtype=se3.dtype, device=se3.device)

    return lietorch.Sim3(torch.cat([t, quats, scale], dim=-1))

def load_vggt(path=None, device="cuda"):
    model = VGGT()
    if path is not None:
        state_dict = torch.load(path, map_location=device)
    else:
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        state_dict = torch.hub.load_state_dict_from_url(_URL, map_location=device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    return model