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
from vggt.heads.dpt_head import DPTHead
from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from torchvision import transforms as TF
import einops
import lietorch
from scipy.spatial.transform import Rotation as R_scipy
import mast3r_slam.matching as matching

class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.feature_extractor = DPTHead(dim_in=2 * embed_dim, output_dim=4, 
                                  features=128, feature_only=True, pos_embed=False)
    
    @staticmethod
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

    @torch.inference_mode
    def vggt_inference_mono(self, frame):
        img = frame.img.unsqueeze(1)
        aggregated_tokens_list, patch_start_idx = self.aggregator(img)
        if frame.feat is None:
            frame.feat = self.feature_extractor(aggregated_tokens_list, frame.img.unsqueeze(1), patch_start_idx)
        X, C = self.point_head(
                        aggregated_tokens_list, images=img, patch_start_idx=patch_start_idx
                    )
        Xii = einops.rearrange(X[0, 0], "h w c -> (h w) c")
        Cii = einops.rearrange(C[0, 0], "h w -> (h w) 1")

        return Xii.squeeze(0), Cii.squeeze(0)
        
    @torch.inference_mode
    def vggt_asymmetric_inference(self, frame_i, frame_j):
        imgs = torch.stack([frame_i.img, frame_j.img], dim=1)
        aggregated_tokens_list, patch_start_idx = self.aggregator(imgs)
        if frame_i.feat is None:
            frame_i.feat = self.feature_extractor(aggregated_tokens_list, frame_i.img.unsqueeze(1), patch_start_idx) # TODO: Dimension Check! 
        if frame_j.feat is None:
            frame_j.feat = self.feature_extractor(aggregated_tokens_list, frame_j.img.unsqueeze(1), patch_start_idx)
        P = self.camera_head(aggregated_tokens_list)[-1]
        X, C = self.point_head(
                        aggregated_tokens_list, imgs, patch_start_idx=patch_start_idx
                    )
        Xii = einops.rearrange(X[0, 0], "h w c -> (h w) c")
        Xij = einops.rearrange(X[0, 1], "h w c -> (h w) c")
        Cii = einops.rearrange(C[0, 0], "h w -> (h w) 1")
        Cij = einops.rearrange(C[0, 1], "h w -> (h w) 1")

        return P, Xii, Xij, Cii, Cij
    
    def vggt_match_asymmetric(self, frame_i, frame_j, idx_i2j_init=None):
        P, Xii, Xij, Cii, Cij = self.vggt_asymmetric_inference(frame_i, frame_j)

        idx_i2j, valid_match_j = matching.mymatch_iterative_proj(
            Xii, Xij, idx_1_to_2_init=idx_i2j_init
        )

        # # How rest of system expects it
        # Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
        # Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
        # Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
        # Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")

        # return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji

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
