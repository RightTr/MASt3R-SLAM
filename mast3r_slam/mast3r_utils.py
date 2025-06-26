import PIL
import numpy as np
import torch
import einops

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import ImgNorm
from mast3r.model import AsymmetricMASt3R
from mast3r_slam.retrieval_database import RetrievalDatabase
from mast3r_slam.config import config
import mast3r_slam.matching as matching
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps


def load_mast3r(path=None, device="cuda"):
    weights_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if path is None
        else path
    )
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    return model


def load_retriever(mast3r_model, retriever_path=None, device="cuda"):
    retriever_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        if retriever_path is None
        else retriever_path
    )
    retriever = RetrievalDatabase(retriever_path, backbone=mast3r_model, device=device)
    return retriever


@torch.inference_mode
def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


def downsample(X, C, D, Q):
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        # C and Q: (...xHxW)
        # X and D: (...xHxWxF)
        X = X[..., ::downsample, ::downsample, :].contiguous()
        C = C[..., ::downsample, ::downsample].contiguous()
        D = D[..., ::downsample, ::downsample, :].contiguous()
        Q = Q[..., ::downsample, ::downsample].contiguous()
    return X, C, D, Q


@torch.inference_mode
def mast3r_symmetric_inference(model, frame_i, frame_j):
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape2, shape1)
    res = [res11, res21, res22, res12]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


# NOTE: Assumes img shape the same
@torch.inference_mode
def mast3r_decode_symmetric_batch(
    model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
):
    B = feat_i.shape[0]
    X, C, D, Q = [], [], [], []
    for b in range(B):
        feat1 = feat_i[b][None]
        feat2 = feat_j[b][None]
        pos1 = pos_i[b][None]
        pos2 = pos_j[b][None]
        res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])
        res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])
        res = [res11, res21, res22, res12]
        Xb, Cb, Db, Qb = zip(
            *[
                (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
                for r in res
            ]
        )
        X.append(torch.stack(Xb, dim=0))
        C.append(torch.stack(Cb, dim=0))
        D.append(torch.stack(Db, dim=0))
        Q.append(torch.stack(Qb, dim=0))

    X, C, D, Q = (
        torch.stack(X, dim=1),
        torch.stack(C, dim=1),
        torch.stack(D, dim=1),
        torch.stack(Q, dim=1),
    )
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


@torch.inference_mode
def mast3r_inference_mono(model, frame):
    if not hasattr(mast3r_inference_mono, "counter"):
        mast3r_inference_mono.counter = 0

    if frame.feat is None:
        frame.feat, frame.pos, _ = model._encode_image(frame.img, frame.img_true_shape)

    feat = frame.feat
    pos = frame.pos
    shape = frame.img_true_shape

    res11, res21 = decoder(model, feat, feat, pos, pos, shape, shape)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)

    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")

    depth_map = X[..., 2]
    depth_map_np = depth_map.detach().cpu().numpy()
    for i in range(depth_map_np.shape[0]):
        depth_min = np.min(depth_map_np[i])
        depth_max = np.max(depth_map_np[i])
        depth_norm = (depth_map_np[i] - depth_min) / (depth_max - depth_min + 1e-8)
        depth_color = plt.cm.jet(depth_norm)[:, :, :3]
        depth_color_uint8 = (depth_color * 255).astype(np.uint8)
        imageio.imwrite(f"/home/pi/Documents/Right/MASt3R-SLAM/temp/init/depth_{mast3r_inference_mono.counter}_{i}.png", depth_color_uint8)
        print(f'Successfully saved depth_mono_{mast3r_inference_mono.counter}_{i}.png')
    mast3r_inference_mono.counter += 1 

    return Xii, Cii

@torch.inference_mode
def mast3r_inference_stereo(model, framepair):
    if not hasattr(mast3r_inference_mono, "counter"):
        mast3r_inference_mono.counter = 0

    if framepair.frame_left.feat is None and framepair.frame_right.feat is None:
        framepair.frame_left.feat, framepair.frame_left.pos, _ = model._encode_image(
            framepair.frame_left.img, framepair.frame_left.img_true_shape)
        framepair.frame_right.feat, framepair.frame_right.pos, _ = model._encode_image(
            framepair.frame_right.img, framepair.frame_right.img_true_shape)

    feat_left = framepair.frame_left.feat
    pos_left = framepair.frame_left.pos
    shape_left = framepair.frame_left.img_true_shape

    feat_right = framepair.frame_right.feat
    pos_right = framepair.frame_right.pos
    shape_right = framepair.frame_right.img_true_shape

    res11, res21 = decoder(model, feat_left, feat_right, pos_left, pos_right, shape_left, shape_right)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)

    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")

    depth_map = X[..., 2]
    depth_map_np = depth_map.detach().cpu().numpy()
    for i in range(depth_map_np.shape[0]):
        depth_min = np.min(depth_map_np[i])
        depth_max = np.max(depth_map_np[i])
        depth_norm = (depth_map_np[i] - depth_min) / (depth_max - depth_min + 1e-8)
        depth_color = plt.cm.jet(depth_norm)[:, :, :3]
        depth_color_uint8 = (depth_color * 255).astype(np.uint8)
        if i == 0:
            imageio.imwrite(f"/home/pi/Documents/Right/MASt3R-SLAM/temp/stereo/left/depth_{mast3r_inference_mono.counter}.png", 
                            depth_color_uint8)
        elif i == 1:
            imageio.imwrite(f"/home/pi/Documents/Right/MASt3R-SLAM/temp/stereo/right/depth_{mast3r_inference_mono.counter}.png", 
                            depth_color_uint8)
        print(f'Successfully saved depth_mono_{mast3r_inference_mono.counter}.png')
    mast3r_inference_mono.counter += 1

    return Xii, Cii


def mast3r_match_symmetric(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
    X, C, D, Q = mast3r_decode_symmetric_batch(
        model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
    )

    # Ordering 4xbxhxwxc
    b = X.shape[1]

    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

    # Always matching both
    X11 = torch.cat((Xii, Xjj), dim=0)
    X21 = torch.cat((Xji, Xij), dim=0)
    D11 = torch.cat((Dii, Djj), dim=0)
    D21 = torch.cat((Dji, Dij), dim=0)

    # tic()
    idx_1_to_2, valid_match_2 = matching.match(X11, X21, D11, D21)
    # toc("Match")

    # TODO: Avoid this
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


@torch.inference_mode
def mast3r_asymmetric_inference(model, frame_i, frame_j):
    if not hasattr(mast3r_asymmetric_inference, "counter"):
        mast3r_asymmetric_inference.counter = 0
        mast3r_match_asymmetric
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    
    depth_map = X[..., 2]
    depth_map_np = depth_map.detach().cpu().numpy()
    for i in range(depth_map_np.shape[0]):
        depth_min = np.min(depth_map_np[i])
        depth_max = np.max(depth_map_np[i])
        depth_norm = (depth_map_np[i] - depth_min) / (depth_max - depth_min + 1e-8)
        depth_color = plt.cm.jet(depth_norm)[:, :, :3]
        depth_color_uint8 = (depth_color * 255).astype(np.uint8)
        imageio.imwrite(f"/home/pi/Documents/Right/MASt3R-SLAM/temp/infer/depth_{mast3r_asymmetric_inference.counter}_{i}.png", depth_color_uint8)
        print(f'Successfully saved depth_infer_{mast3r_asymmetric_inference.counter}_{i}.png')
    mast3r_asymmetric_inference.counter += 1 
    return X, C, D, Q


def mast3r_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):
    X, C, D, Q = mast3r_asymmetric_inference(model, frame_i, frame_j)

    b, h, w = X.shape[:-1]
    # 2 outputs per inference
    b = b // 2

    Xii, Xji = X[:b], X[b:]
    Cii, Cji = C[:b], C[b:]
    Dii, Dji = D[:b], D[b:]
    Qii, Qji = Q[:b], Q[b:]

    idx_i2j, valid_match_j = matching.match(
        Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
    )

    # How rest of system expects it
    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
    Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
    Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")

    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_img(img, return_transformation=False, mode="crop"): #TODO: Resize correctly?
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")
    
    target_size = 518
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img)
    width0, height0 = img.size

    if mode == "pad":
        # Make the largest dimension 518px while maintaining aspect ratio
        if width0 >= height0:
            new_width = target_size
            new_height = round(height0 * (new_width / width0) / 14) * 14  # Make divisible by 14
        else:
            new_height = target_size
            new_width = round(width0 * (new_height / height0) / 14) * 14  # Make divisible by 14
    else:  # mode == "crop"
        # Original behavior: set width to 518px
        new_width = target_size
        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height0 * (new_width / width0) / 14) * 14

    # Resize with new dimensions (width, height)
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    width, height = img.size

    # Center crop height if it's larger than 518 (only in crop mode)
    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        img = img.crop((0, start_y, width, start_y + target_size))

    # For pad mode, pad to make a square of target_size x target_size
    if mode == "pad":
        h_padding = target_size - img.shape[1]
        w_padding = target_size - img.shape[2]

        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left

            # Pad with white (value=1.0)
            img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=255)
    
    img = np.asarray(img).astype(np.float32) # (h, w, c) unnormalized

    res = dict(
        img=torch.from_numpy(img).permute(2, 0, 1) / 255.0, # (b, c, h, w)
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


