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
from mast3r_slam.frame import Frame
from vggt.models.vggt import VGGT
from torchvision import transforms as TF
import einops

def load_vggt(path=None, device="cuda"):
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(device)
    return model

@torch.inference_mode
def vggt_inference_mono(model, frame):
    img = frame.img.unsqueeze(1)
    aggregated_tokens_list, patch_start_idx = model.aggregator(img)
    X, C = model.point_head(
                    aggregated_tokens_list, images=img, patch_start_idx=patch_start_idx
                )
    P = model.camera_head(aggregated_tokens_list)
    Xii = einops.rearrange(X.squeeze(1), "b h w c -> b (h w) c")
    Cii = einops.rearrange(C.squeeze(1), "b h w -> b (h w) 1")

    return Xii, Cii
    
@torch.inference_mode
def vggt_asymmetric_inference(model, frame_i, frame_j):
    imgs = torch.stack([frame_i.img, frame_j.img])
    aggregated_tokens_list, patch_start_idx = model.aggregator(imgs)
    P = model.camera_head(aggregated_tokens_list)
    X, C = model.point_head(
                    aggregated_tokens_list, imgs, patch_start_idx=patch_start_idx
                )
    Pji = P[0, 1]

    return Pji