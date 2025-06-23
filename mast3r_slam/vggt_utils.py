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
import lietorch
from scipy.spatial.transform import Rotation as R_scipy

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
    Xii = einops.rearrange(X[0, 0], "h w c -> (h w) c")
    Cii = einops.rearrange(C[0, 0], "h w -> (h w) 1")

    return Xii.squeeze(0), Cii.squeeze(0)
    
@torch.inference_mode
def vggt_asymmetric_inference(model, frame_i, frame_j):
    imgs = torch.stack([frame_i.img, frame_j.img], dim=1)
    aggregated_tokens_list, patch_start_idx = model.aggregator(imgs)
    P = model.camera_head(aggregated_tokens_list)[-1]
    X, C = model.point_head(
                    aggregated_tokens_list, imgs, patch_start_idx=patch_start_idx
                )
    Xii = einops.rearrange(X[0, 0], "h w c -> (h w) c")
    Xij = einops.rearrange(X[0, 1], "h w c -> (h w) c")
    Cii = einops.rearrange(C[0, 0], "h w -> (h w) 1")
    Cij = einops.rearrange(C[0, 1], "h w -> (h w) 1")

    return P, Xii, Xij, Cii, Cij

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

    quats = torch.tensor(quats, device=R.device, dtype=R.dtype)

    t = t.squeeze(-1)
    scale = torch.full((se3.shape[0], 1), scale, dtype=se3.dtype, device=se3.device)

    return lietorch.Sim3(torch.cat([t, quats, scale], dim=-1))
