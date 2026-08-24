"""Top-down rasteriser for the push scene.

The physics stays MuJoCo's -- real contact, real Coulomb friction. Only the
rasteriser is ours. We swapped it out because EGL on this box is saturated by
another tenant's jobs, and because a fixed top-down camera looking at one box
on a flat floor is something we can draw exactly.

Squares are drawn from a signed distance field with a smooth edge, so the
object's centroid is accurate to well under a pixel. That matters: the readout
must not be the bottleneck when we measure how well a world model tracks
physics.
"""

import numpy as np
import torch

IMG = 96
CAM_HALF = 0.72                     # world half-extent covered by the image
BOX_HALF = 0.05                     # matches the MuJoCo geom
MPP = 2 * CAM_HALF / IMG            # metres per pixel


def world_to_px(xy):
    """(x, y) metres -> (col, row) pixels."""
    c = xy[..., 0] / MPP + (IMG - 1) / 2
    r = (IMG - 1) / 2 - xy[..., 1] / MPP
    return np.stack([c, r], -1)


def _background(device):
    """Static checkerboard floor with a soft vignette, drawn once."""
    yy, xx = torch.meshgrid(torch.arange(IMG, device=device, dtype=torch.float32),
                            torch.arange(IMG, device=device, dtype=torch.float32),
                            indexing="ij")
    wx = (xx - (IMG - 1) / 2) * MPP
    wy = ((IMG - 1) / 2 - yy) * MPP
    chk = (torch.floor(wx / 0.24) + torch.floor(wy / 0.24)) % 2
    base = torch.where(chk > 0.5, 0.34, 0.25)
    rad = torch.sqrt(wx ** 2 + wy ** 2)
    vig = 1.0 - 0.18 * (rad / CAM_HALF).clamp(max=1.0) ** 2
    g = base * vig
    return torch.stack([g * 0.93, g * 1.00, g * 1.10], -1).clamp(0, 1)


_BG_CACHE = {}


def background(device):
    key = str(device)
    if key not in _BG_CACHE:
        _BG_CACHE[key] = _background(device)
    return _BG_CACHE[key]


def render(states, device="cuda", chunk=4096):
    """states: [..., 3] of (x, y, yaw) in world units -> uint8 [..., IMG, IMG, 3].

    Drawn with a rotated-square SDF plus an offset shadow, so the object has an
    unambiguous colour signature (saturated red) against a grey-blue floor.
    """
    s = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
    shape = s.shape[:-1]
    s = s.reshape(-1, 3)
    out = torch.empty((s.shape[0], IMG, IMG, 3), dtype=torch.uint8, device=device)

    yy, xx = torch.meshgrid(torch.arange(IMG, device=device, dtype=torch.float32),
                            torch.arange(IMG, device=device, dtype=torch.float32),
                            indexing="ij")
    wx = (xx - (IMG - 1) / 2) * MPP
    wy = ((IMG - 1) / 2 - yy) * MPP
    bg = background(device)
    edge = 0.6 * MPP                                   # antialias width

    for lo in range(0, s.shape[0], chunk):
        b = s[lo:lo + chunk]
        n = b.shape[0]
        cx, cy, yaw = b[:, 0:1, None], b[:, 1:2, None], b[:, 2:3, None]
        cos, sin = torch.cos(yaw), torch.sin(yaw)

        def cover(ox, oy):
            dx = (wx[None] - cx - ox)
            dy = (wy[None] - cy - oy)
            lx = dx * cos + dy * sin                   # into the box frame
            ly = -dx * sin + dy * cos
            d = torch.maximum(lx.abs(), ly.abs()) - BOX_HALF
            return torch.sigmoid(-d / edge)

        a_box = cover(0.0, 0.0)
        a_sh = cover(0.022, -0.022) * 0.45             # offset drop shadow

        img = bg[None].expand(n, IMG, IMG, 3).clone()
        img = img * (1 - a_sh[..., None] * 0.7)        # darken under the shadow
        col = torch.tensor([0.92, 0.24, 0.19], device=device)
        # slight shading across the face so rotation is visible
        shade = 1.0 + 0.10 * torch.cos(yaw + 0.6)
        face = (col[None, None, None, :] * shade[..., None]).clamp(0, 1)
        img = img * (1 - a_box[..., None]) + face * a_box[..., None]

        out[lo:lo + n] = (img.clamp(0, 1) * 255).round().to(torch.uint8)

    return out.reshape(*shape, IMG, IMG, 3)
