"""The single V-JEPA 2-AC encoder wrapper.

This code used to be copy-pasted into seven scripts; any change to pooling or
preprocessing had to be repeated in each and silently desynchronised the
experiments. Every consumer now goes through here.

Two rules learned the hard way (see the MuJoCo checklist in the project notes):
the encoder must never share a process with MuJoCo's EGL renderer, and frames
are pooled PER FRAME so the temporal structure that carries friction survives.
"""

import sys

from core.paths import VJEPA_DIR, VJEPA_CKPT

CROP = 256

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(VJEPA_DIR))


def build_encoder(device="cuda"):
    """Frozen ViT-g encoder + the DROID eval transform. Returns (encoder, tf)."""
    from src.hub.backbones import _make_vjepa2_ac_model
    from app.vjepa_droid.transforms import make_transforms

    enc, _ = _make_vjepa2_ac_model(model_name="vit_ac_giant", img_size=CROP,
                                   pretrained=False)
    ck = torch.load(VJEPA_CKPT, map_location="cpu", weights_only=False)
    enc.load_state_dict({k.replace("module.", ""): v
                         for k, v in ck["encoder"].items()}, strict=False)
    enc = enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    del ck
    tf = make_transforms(random_horizontal_flip=False,
                         random_resize_aspect_ratio=(1., 1.),
                         random_resize_scale=(1., 1.), reprob=0.,
                         auto_augment=False, motion_shift=False,
                         crop_size=CROP)
    return enc, tf


@torch.no_grad()
def encode_frames(enc, tf, frames, pool="mean"):
    """uint8 frames [T, H, W, 3] -> per-frame pooled features.

    pool: 'mean' -> [T, D];  'mean+std' -> ([T, D], [T, D]);
          'tokens' -> layer-normed token grid [T*N, D].
    Each frame fills a 2-frame tubelet, as in the reference notebook, so the
    encoder sees T independent frames.
    """
    clip = tf(frames).unsqueeze(0).to(next(enc.parameters()).device)
    B, C, T, H, W = clip.size()
    c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = enc(c)
    h = h.view(B, T, -1, h.size(-1)).float()               # [1, T, N, D]
    h = F.layer_norm(h, (h.size(-1),))[0]                  # [T, N, D]
    if pool == "tokens":
        return h.flatten(0, 1).cpu().numpy()
    if pool == "mean+std":
        return h.mean(1).cpu().numpy(), h.std(1).cpu().numpy()
    return h.mean(1).cpu().numpy()


def to256(frames, crop=None, bright=1.0):
    """Real-footage frames (e.g. DROID 320x180) -> centre-cropped 256x256.

    crop=(x0, y0, w, h) overrides the default centre square; bright scales
    intensity. Both exist so nuisance sensitivity can be measured on real
    frames the same way it is in sim.
    """
    out = []
    for f in frames:
        im = Image.fromarray(f)
        if crop is None:
            w, h = im.size
            s = min(w, h)
            box = ((w - s) // 2, 0, (w - s) // 2 + s, s)
        else:
            x0, y0, w, h = crop
            box = (x0, y0, x0 + w, y0 + h)
        im = im.crop(box).resize((CROP, CROP), Image.BILINEAR)
        a = np.asarray(im).astype(np.float32) * bright
        out.append(np.clip(a, 0, 255).astype(np.uint8))
    return np.stack(out)
