import sys
import os.path as path
HERE_PATH = path.normpath(path.dirname(__file__))
VGGT_REPO_PATH = path.normpath(path.join(HERE_PATH, '../thirdparty/vggt'))
SALAD_REPO_PATH = path.normpath(path.join(HERE_PATH, '../thirdparty/salad'))
sys.path.insert(0, VGGT_REPO_PATH)
sys.path.insert(0, SALAD_REPO_PATH)

import torch
import einops
import lietorch
from scipy.spatial.transform import Rotation as R_scipy
import mast3r_slam.matching as matching
from PIL import Image
from PIL import ImageOps
from mast3r_slam.config import config

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vpr_model import VPRModel

import numpy as np

def load_vggt(path=None, device="cuda:0"):
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

def load_salad(path=None, device="cuda:0"):
    model = VPRModel(
        backbone_arch='dinov2_vitb14',
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        },
        agg_arch='SALAD',
        agg_config={
            'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
    )

    model.load_state_dict(torch.load(path))
    model = model.eval()
    model = model.to(device)
    return model

@torch.inference_mode
def salad_get_descriptor(model, frame, device= "cuda:0"):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
            img = frame.img
            output = model(img.to(device))
    return output

@torch.inference_mode
def vggt_inference_mono(model, frame):
    img = img_to_imgbschw(frame.img)
    aggregated_tokens_list, patch_start_idx = model.aggregator(img)

    P = model.camera_head(aggregated_tokens_list)[-1]
    _, K = get_extri_intri_from_pose(P, frame.img.shape[-2:])
    if config['use_depth']:
        D, C = model.depth_head(
                    aggregated_tokens_list, images=img, patch_start_idx=patch_start_idx
                )
        Xii = depth_to_pointmap(D[:, 0], K)
        Cii = C[:, 0]
        Xii = einops.rearrange(Xii[0, :], "h w c -> (h w) c")
        Cii = einops.rearrange(Cii[0, :], "h w -> (h w) 1")
        return Xii, Cii, K
    else:
        X, C = model.point_head(
                        aggregated_tokens_list, images=img, patch_start_idx=patch_start_idx
                    )
        Xii = einops.rearrange(X[0, 0], "h w c -> (h w) c")
        Cii = einops.rearrange(C[0, 0], "h w -> (h w) 1")
        return Xii, Cii, K
    
@torch.inference_mode
def vggt_asymmetric_inference(model, frame_i, frame_j):
    img_i = img_to_imgbschw(frame_i.img)
    img_j = img_to_imgbschw(frame_j.img)
    imgs = torch.cat([img_i, img_j], dim=1)
    aggregated_tokens_list, patch_start_idx = model.aggregator(imgs)

    P = model.camera_head(aggregated_tokens_list)[-1]
    if config['use_depth']:
        D, C = model.depth_head(
                    aggregated_tokens_list, images=imgs, patch_start_idx=patch_start_idx
                )
        Dii = D[:, 0] # (b, h, w, 1)
        Djj = D[:, 1] # (b, h, w, 1)
        Cii = C[:, 0] # (b, h, w)
        Cij = C[:, 1] # (b, h, w)
        return P, Dii, Djj, Cii, Cij
    
    else:
        X, C = model.point_head(
                    aggregated_tokens_list, imgs, patch_start_idx=patch_start_idx
                )
        Xii = X[:, 0] # (b, h, w, c)
        Xij = X[:, 1] # (b, h, w, c)
        Cii = C[:, 0] # (b, h, w)
        Cij = C[:, 1] # (b, h, w)
        return P, Xii, Xij, Cii, Cij

def vggt_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):
    if config['use_depth']:
        P, Dii, Djj, Cii, Cjj = vggt_asymmetric_inference(model, frame_i, frame_j)
        img_size = frame_i.img.shape[-2:]
        Tij, K = get_extri_intri_from_pose(P, img_size)
        Tji = Tij.inv()
        Ki, Kj = K[0], K[1] 
        Xii = depth_to_pointmap(Dii, Ki)
        Xjj = depth_to_pointmap(Djj, Kj)
        Cij = Cjj
        Xij = X_tf(Xjj, Tji) # Convert the pointmap from coordinate frame j to frame i
        idx_i2j, valid_match_j = matching.mymatch_iterative_proj(
            Xii, Xij, idx_i_to_j_init=idx_i2j_init
        )

    else:
        P, Xii, Xij, Cii, Cij = vggt_asymmetric_inference(model, frame_i, frame_j)
        idx_i2j, valid_match_j = matching.mymatch_iterative_proj(
            Xii, Xij, idx_i_to_j_init=idx_i2j_init
        )

    Xii = einops.rearrange(Xii[0, :], "h w c -> (h w) c")
    Cii = einops.rearrange(Cii[0, :], "h w -> (h w) 1")
    Xij = einops.rearrange(Xij[0, :], "h w c -> (h w) c")
    Cij = einops.rearrange(Cij[0, :], "h w -> (h w) 1")

    return idx_i2j, valid_match_j, P, Xii, Cii, Xij, Cij


@torch.inference_mode
def vggt_symmmetric_inference(model, frame_ii, frame_jj):
    S = len(frame_ii)
    X, C = [], []
    img_size = frame_ii[0].img.shape[-2:]
    for s in range(S):
        img_i = img_to_imgbschw(frame_ii[s].img)
        img_j = img_to_imgbschw(frame_jj[s].img)
        imgs_ij = torch.cat([img_i, img_j], dim=1)
        imgs_ji = torch.cat([img_j, img_i], dim=1)

        if config['use_depth']:
            aggregated_tokens_list_ij, patch_start_idx_ij = model.aggregator(imgs_ij)
            aggregated_tokens_list_ji, patch_start_idx_ji = model.aggregator(imgs_ji)
            Diijj, Ciijj = model.depth_head(
                        aggregated_tokens_list_ij, imgs_ij, patch_start_idx=patch_start_idx_ij
                    )
            Djjii, Cjjii = model.depth_head(
                    aggregated_tokens_list_ji, imgs_ji, patch_start_idx=patch_start_idx_ji
                )
            Piijj = model.camera_head(aggregated_tokens_list_ij)[-1]
            Pjjii = model.camera_head(aggregated_tokens_list_ji)[-1]
            Tijij, Kiijj = get_extri_intri_from_pose(Piijj, img_size)
            Tjiji, Kjjii = get_extri_intri_from_pose(Pjjii, img_size)
            Tjiij = Tijij.inv()
            Tijji = Tjiji.inv()
            Kiij, Kjij = Kiijj[0], Kjjii[0]
            Kjji, Kiji = Kjjii[1], Kjjii[1]
            Diiij, Djjij, Djjji, Diiji = Diijj[:, 0], Diijj[:, 1], Djjii[:, 0], Djjii[:, 1] # (b, h, w, 1)
            Ciiij, Cjjij, Cjjji, Ciiji = Ciijj[0, 0], Ciijj[0, 1], Cjjii[0, 0], Cjjii[0, 1] # (h, w)
            Xiiij = depth_to_pointmap(Diiij, Kiij)
            Xjjij = depth_to_pointmap(Djjij, Kjij)
            Xjjji = depth_to_pointmap(Djjji, Kjji)
            Xiiji = depth_to_pointmap(Diiji, Kiji)

            Xijij = X_tf(Xjjij, Tjiij)
            Xjiji = X_tf(Xiiji, Tijji)

            X.append(torch.stack([Xiiij[0], Xijij[0], Xjjji[0], Xjiji[0]], dim=0))
            C.append(torch.stack([Ciiij, Cjjij, Cjjji, Ciiji], dim=0))

        else:
            aggregated_tokens_list_ij, patch_start_idx_ij = model.aggregator(imgs_ij)
            aggregated_tokens_list_ji, patch_start_idx_ji = model.aggregator(imgs_ji)
            Xiijj, Ciijj = model.point_head(
                        aggregated_tokens_list_ij, imgs_ij, patch_start_idx=patch_start_idx_ij
                    )
            Xjjii, Cjjii = model.point_head(
                    aggregated_tokens_list_ji, imgs_ji, patch_start_idx=patch_start_idx_ji
                )
            Xii, Xij, Xjj, Xji = Xiijj[0, 0], Xiijj[0, 1], Xjjii[0, 0], Xjjii[0, 1] # (h, w, c)
            Cii, Cij, Cjj, Cji = Ciijj[0, 0], Ciijj[0, 1], Cjjii[0, 0], Cjjii[0, 1] # (h, w)

            X.append(torch.stack([Xii, Xij, Xjj, Xji], dim=0))
            C.append(torch.stack([Cii, Cij, Cjj, Cji], dim=0))

    X = torch.stack(X, dim=1)
    C = torch.stack(C, dim=1)

    return X, C

def vggt_match_symmetric(model, frame_i, frame_j, idx_i2j_init=None):
    X, C = vggt_symmmetric_inference(model, frame_i, frame_j)

    b = X.shape[1]

    Xii, Xij, Xjj, Xji = X[0], X[1], X[2], X[3]
    Cii, Cij, Cjj, Cji = C[0], C[1], C[2], C[3]
    
    X11 = torch.cat([Xii, Xjj], dim=0)
    X21 = torch.cat([Xij, Xji], dim=0)

    idx_1_to_2, valid_match_2 = matching.mymatch_iterative_proj(
        X11, X21
        )
    
    Qii = torch.ones_like(Cii)
    Qji = torch.ones_like(Cii)
    Qjj = torch.ones_like(Cii)
    Qij = torch.ones_like(Cii)

    match_b = X11.shape[0] // 2
    idx_i2j = idx_1_to_2[:match_b]
    idx_j2i = idx_1_to_2[match_b:]
    valid_match_j = valid_match_2[:match_b]
    valid_match_i = valid_match_2[match_b:]

    return (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        Qji.view(b, -1, 1),
        Qij.view(b, -1, 1),
    )

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

def depth_to_pointmap(D, K, device='cuda:0'):
    b, h, w = D.shape[:3]
    u = torch.arange(w, device=device).view(1, 1, w).expand(b, h, w)
    v = torch.arange(h, device=device).view(1, h, 1).expand(b, h, w)
    ones = torch.ones_like(u)
    pix = torch.stack((u, v, ones), dim=-1).float() 
    pix = pix.view(b, h, w, 3, 1)
    K_inv = torch.inverse(K).view(1, 1, 1, 3, 3)
    K_inv = K_inv.expand(b, h, w, 3, 3)
    X = torch.matmul(K_inv, pix) * D.unsqueeze(-1)
    return X.squeeze(-1)
    
def img_to_imgbschw(img):
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    if len(img.shape) == 4:
        img = img.unsqueeze(1)
    return img

def X_tf(X, T):
    b, h, w, c = X.shape
    X_ = einops.rearrange(X[0, :], "h w c -> (h w) c")
    X_ = T.act(X_)
    return X_.view(b, h, w, c)