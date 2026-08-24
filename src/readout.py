"""Readout Phi: image -> object pose.

This is the piece that makes real, simulated and world-model-generated states
comparable at all. Rather than comparing latents (incomparable across models)
or raw pixels (dominated by appearance rather than physics), every state is
projected onto the same physical quantity: where the object actually is.

Here the object is a saturated red box on a grey floor, so a colour-weighted
centroid is an accurate detector. The pixel -> world map is *fitted* against
simulator ground truth rather than derived from the camera matrix, which also
gives us an honest measure of the readout's own error floor.
"""

import numpy as np
import torch


def redness(frames):
    """frames: uint8 or float [..., H, W, 3] -> [..., H, W] object saliency."""
    if isinstance(frames, torch.Tensor):
        f = frames.float()
        if f.shape[-1] != 3:                       # channels-first -> last
            f = f.movedim(-3, -1)
        if f.max() <= 1.5:                         # [-1,1] -> [0,1]
            f = (f + 1) / 2 if f.min() < -0.01 else f
        else:
            f = f / 255.0
        return (f[..., 0] - 0.5 * (f[..., 1] + f[..., 2])).clamp(min=0)
    f = frames.astype(np.float32)
    if f.shape[-1] != 3:
        f = np.moveaxis(f, -3, -1)
    f = f / 255.0 if f.max() > 1.5 else ((f + 1) / 2 if f.min() < -0.01 else f)
    return np.clip(f[..., 0] - 0.5 * (f[..., 1] + f[..., 2]), 0, None)


def centroid_px(frames, power=3.0):
    """Colour-weighted centroid in pixel coords. Returns [..., 2] as (col, row).

    The weights are raised to a power so that faint, blurry world-model output
    does not drag the centroid toward the background.
    """
    t = isinstance(frames, torch.Tensor)
    r = redness(frames)
    r = r ** power
    H, W = r.shape[-2], r.shape[-1]
    if t:
        rows = torch.arange(H, device=r.device, dtype=r.dtype)
        cols = torch.arange(W, device=r.device, dtype=r.dtype)
        tot = r.sum((-2, -1)).clamp(min=1e-8)
        cr = (r.sum(-1) * rows).sum(-1) / tot
        cc = (r.sum(-2) * cols).sum(-1) / tot
        conf = r.sum((-2, -1))
        return torch.stack([cc, cr], -1), conf
    rows, cols = np.arange(H), np.arange(W)
    tot = np.maximum(r.sum((-2, -1)), 1e-8)
    cr = (r.sum(-1) * rows).sum(-1) / tot
    cc = (r.sum(-2) * cols).sum(-1) / tot
    return np.stack([cc, cr], -1), r.sum((-2, -1))


def fit_pixel_to_world(px, world):
    """Least-squares affine map (col,row) -> (x,y). px:[N,2] world:[N,2]."""
    A = np.concatenate([px, np.ones((len(px), 1))], 1)
    M, *_ = np.linalg.lstsq(A, world, rcond=None)
    pred = A @ M
    err = np.linalg.norm(pred - world, axis=1)
    return M, err


def apply_map(px, M):
    if isinstance(px, torch.Tensor):
        Mt = torch.as_tensor(M, device=px.device, dtype=px.dtype)
        A = torch.cat([px, torch.ones_like(px[..., :1])], -1)
        return A @ Mt
    A = np.concatenate([px, np.ones_like(px[..., :1])], -1)
    return A @ M
