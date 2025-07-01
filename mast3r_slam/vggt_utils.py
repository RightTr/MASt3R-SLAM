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
from vggt.models.vggt import VGGT
from torchvision import transforms as TF
import einops
import lietorch
from scipy.spatial.transform import Rotation as R_scipy
import mast3r_slam.matching as matching
from PIL import Image
from PIL import ImageOps
from mast3r_slam.config import config

from vggt.utils.pose_enc import pose_encoding_to_extri_intri

import numpy as np

@torch.inference_mode
def vggt_inference_mono(model, frame):
    img = frame.img.unsqueeze(0).unsqueeze(1)
    aggregated_tokens_list, patch_start_idx = model.aggregator(img)
    if frame.feat is None:
        frame.feat = model.track_head.feature_extractor(aggregated_tokens_list, img, patch_start_idx)
    
    P = model.camera_head(aggregated_tokens_list)[-1]
    if config['use_depth']:
        D, C = model.depth_head(
                        aggregated_tokens_list, images=img, patch_start_idx=patch_start_idx
                    )
        Dii = D[:, 0]
        Cii = C[:, 0]
        return Dii, Cii, P
    else:
        X, C = model.point_head(
                        aggregated_tokens_list, images=img, patch_start_idx=patch_start_idx
                    )
        Xii = einops.rearrange(X[0, 0], "h w c -> (h w) c")
        Cii = einops.rearrange(C[0, 0], "h w -> (h w) 1")
        return Xii, Cii, P
    
@torch.inference_mode
def vggt_asymmetric_inference(model, frame_i, frame_j):
    img_i = frame_i.img.unsqueeze(0).unsqueeze(1)
    img_j = frame_j.img.unsqueeze(0).unsqueeze(1)
    imgs = torch.cat([img_i, img_j], dim=1)
    aggregated_tokens_list, patch_start_idx = model.aggregator(imgs)
    # if frame_i.feat is None:
    #     feats = model.feature_extractor(aggregated_tokens_list, imgs, patch_start_idx) 
    #     frame_i.feat = feats[:, 0]

    # if frame_j.feat is None:
    #     feats = model.feature_extractor(aggregated_tokens_list, imgs, patch_start_idx)
    #     frame_i.feat = feats[:, 0]

    P = model.camera_head(aggregated_tokens_list)[-1]
    if config['use_depth']:
        D, C = model.depth_head(
                        aggregated_tokens_list, images=imgs, patch_start_idx=patch_start_idx
                    )
        Dii = D[:, 0]
        Dij = D[:, 1]
        Cii = C[:, 0]
        Cij = C[:, 1]
        return P, Dii, Dij, Cii, Cij
    
    else:
        X, C = model.point_head(
                        aggregated_tokens_list, imgs, patch_start_idx=patch_start_idx
                    )
        Xii = X[:, 0]
        Xij = X[:, 1]
        Cii = C[:, 0]
        Cij = C[:, 1]
        return P, Xii, Xij, Cii, Cij

def vggt_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):
    if config['use_depth']:
        P, Dii, Djj, Cii, Cjj = vggt_asymmetric_inference(model, frame_i, frame_j)
        img_size = frame_i.img.shape[-2:]
        Tij, K = get_extri_intri_from_pose(P, img_size)
        Tji = Tij.inv()
        print(Tij.shape)
        Ki, Kj = K[0], K[1]
        Xii, Cii = depth_to_pointmap(Dii, Cii, Ki)
        Xjj, Cij = depth_to_pointmap(Djj, Cjj, Kj)

        h, w, c = Xjj[0, :].shape
        Xjj = einops.rearrange(Xjj[0, :], "h w c -> (h w) c")
        Xij = Tji.act(Xjj)
        Xij = Xij.view(1, h, w, c)
        idx_i2j, valid_match_j = matching.mymatch_iterative_proj(
            Xii, Xij, P, idx_i_to_j_init=idx_i2j_init
        )

    else:
        P, Xii, Xij, Cii, Cij = vggt_asymmetric_inference(model, frame_i, frame_j)

        idx_i2j, valid_match_j = matching.mymatch_iterative_proj(
            Xii, Xij, P, idx_i_to_j_init=idx_i2j_init
        )

    Xii = einops.rearrange(Xii[0, :], "h w c -> (h w) c")
    Cii = einops.rearrange(Cii[0, :], "h w -> (h w) 1")
    Xij = einops.rearrange(Xij[0, :], "h w c -> (h w) c")
    Cij = einops.rearrange(Cij[0, :], "h w -> (h w) 1")

    return idx_i2j, valid_match_j, P, Xii, Cii, Xij, Cij

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

def resize_img(img, return_transformation=False, mode="crop"):
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")
    
    target_size = 518
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img)
    width0, height0 = img.size

    if mode == "pad":
        if width0 >= height0:
            new_width = target_size
            new_height = round(height0 * (new_width / width0) / 14) * 14 
        else:
            new_height = target_size
            new_width = round(width0 * (new_height / height0) / 14) * 14 
    else:  
        new_width = target_size
        new_height = round(height0 * (new_width / width0) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    width, height = img.size

    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        img = img.crop((0, start_y, width, start_y + target_size))

    if mode == "pad":
        h_padding = target_size - img.shape[1]
        w_padding = target_size - img.shape[2]

        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left

            img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=255)
    
    img = np.asarray(img).astype(np.float32)

    res = dict(
        img=torch.from_numpy(img).permute(2, 0, 1) / 255.0, # (c, h, w)
        true_shape = np.int32(img.shape[:2][::-1]), # (w, h)
        unnormalized_img = img,
    )

    if return_transformation:
        scale_w = width0 / width 
        scale_h = height0 / height
        half_crop_w = (width - img.shape[0]) / 2
        half_crop_h = (height - img.shape[1]) / 2
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)
    
    return res

def match_sim3_scale(T_src: lietorch.Sim3, T_target: lietorch.Sim3) -> lietorch.Sim3:
    log_src = T_src.log()
    log_src[..., 6] = T_target.log()[..., 6]
    return lietorch.Sim3.exp(log_src)

def match_sim3_scale_one(T_src: lietorch.Sim3) -> lietorch.Sim3:
    log_src = T_src.log()
    log_src[..., 6] = 0
    return lietorch.Sim3.exp(log_src)

def get_extri_intri_from_pose(P, img_size):
    extrinsics, intrinsics = pose_encoding_to_extri_intri(P, img_size) 
    if P.shape[1] == 1:
        return None, intrinsics.squeeze(0)
    else:
        return closed_form_sim3(extrinsics[:, 1]), intrinsics.squeeze(0)


def depth_to_pointmap(D, C, K, if_init = False, device='cuda:0'):
    b, h, w = D.shape[:3]
    u = torch.arange(w, device=device).view(1, 1, w).expand(b, h, w)
    v = torch.arange(h, device=device).view(1, h, 1).expand(b, h, w)
    ones = torch.ones_like(u)
    pix = torch.stack((u, v, ones), dim=-1).float() 
    pix = pix.view(b, h, w, 3, 1)
    K_inv = torch.inverse(K).view(1, 1, 1, 3, 3)
    K_inv = K_inv.expand(b, h, w, 3, 3)
    X = torch.matmul(K_inv, pix) * D.unsqueeze(-1)
    if if_init:
        X = einops.rearrange(X[0, :].squeeze(-1), "h w c -> (h w) c")
        C = einops.rearrange(C[0, :], "h w -> (h w) 1")
        return X, C
    else:
        return X.squeeze(-1), C
    

