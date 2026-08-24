"""Is the flat landscape the representation's fault, or the distance function's?

Averaging L1 over all 2048 tokens dilutes the signal badly: the cube covers
about 2% of the image, so a physics change perturbs roughly 2% of the tokens
and the mean divides that by fifty. Meanwhile imperceptible uniform noise
touches every token and scores higher than any physics change in the grid.

So before concluding that V-JEPA cannot see the physics, try distances that do
not average the signal away: the largest per-token change, the mean over the
most-affected tokens, and the mean restricted to tokens that overlap the cube.
"""

import sys, os, json
import numpy as np

VJEPA = "/data/pgc/simdroid/stage0/vjepa2"
CKPT = "/data/pgc/simdroid/stage0/vjepa2-ac-vitg.pt"
sys.path.insert(0, VJEPA)
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
from scene import MASS_GRID, MU_GRID, NOMINAL_CELL

CLIPS = os.environ.get("CLIP_DIR", "out/stage0/clips")
OUT = os.environ.get("RES_DIR", "out/stage0")
NT = 256            # tokens per frame at 256 px / patch 16
NF = 8


def main():
    from src.hub.backbones import _make_vjepa2_ac_model
    from app.vjepa_droid.transforms import make_transforms

    enc, _ = _make_vjepa2_ac_model(model_name="vit_ac_giant", img_size=256,
                                   pretrained=False)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    enc.load_state_dict({k.replace("module.", ""): v
                         for k, v in ck["encoder"].items()}, strict=False)
    enc = enc.to("cuda").eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    del ck
    tf = make_transforms(random_horizontal_flip=False,
                         random_resize_aspect_ratio=(1., 1.),
                         random_resize_scale=(1., 1.), reprob=0.,
                         auto_augment=False, motion_shift=False, crop_size=256)

    @torch.no_grad()
    def encode(frames):
        clip = tf(frames).unsqueeze(0).cuda()
        B, C, T, H, W = clip.size()
        c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = enc(c)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2).float()
        return F.layer_norm(h, (h.size(-1),))[0]

    def load(n):
        d = np.load(f"{CLIPS}/{n}.npz")
        return d["clip"], float(d["travel"])

    ref_clip, _ = load("ref")
    z_ref = encode(ref_clip)

    # which tokens does the cube actually occupy? use the pixel change across
    # the whole grid as the mask
    acc = np.zeros(ref_clip.shape[1:3], np.float64)
    for i in range(5):
        for j in range(5):
            c, _ = load(f"grid_{i}_{j}")
            acc += np.abs(c.astype(np.int16) - ref_clip.astype(np.int16)).mean(
                axis=(0, 3))
    p = 16
    tok = acc.reshape(256 // p, p, 256 // p, p).mean(axis=(1, 3)).ravel()
    obj_tokens = np.ascontiguousarray(np.argsort(tok)[::-1][:24])   # top 24 of 256
    print(f"object-bearing tokens: {len(obj_tokens)}/{NT} per frame "
          f"({100*len(obj_tokens)/NT:.1f}%)")

    def dists(z):
        dt = (z - z_ref).abs().mean(-1)            # [T*N] per-token distance
        m = dt.view(NF, NT)
        return {
            "mean_all": float(dt.mean()),
            "max_token": float(dt.max()),
            "top2pct": float(dt.topk(max(1, int(0.02 * dt.numel()))).values.mean()),
            "object_tokens": float(m[:, obj_tokens].mean()),
        }

    keys = ["mean_all", "max_token", "top2pct", "object_tokens"]
    G = {k: np.zeros((5, 5)) for k in keys}
    T = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            c, tr = load(f"grid_{i}_{j}")
            d = dists(encode(c))
            for k in keys:
                G[k][i, j] = d[k]
            T[i, j] = tr

    N = {}
    for f in sorted(os.listdir(CLIPS)):
        if f.startswith("nuis_"):
            c, _ = load(f[:-4])
            N[f[5:-4]] = dists(encode(c))

    ni, nj = NOMINAL_CELL
    print(f"\n{'metric':16s} {'self':>8} {'phys min':>9} {'phys max':>9} "
          f"{'worst nuis':>11} {'signal/nuis':>12} {'corr w/ travel':>15}")
    summary = {}
    for k in keys:
        g = G[k]
        off = [g[i, j] for i in range(5) for j in range(5) if (i, j) != (ni, nj)]
        wn = max(v[k] for v in N.values())
        dtr = np.abs(T - T[ni, nj]).ravel()
        r = float(np.corrcoef(dtr, g.ravel())[0, 1])
        summary[k] = {"self": float(g[ni, nj]), "phys_min": float(min(off)),
                      "phys_max": float(max(off)), "worst_nuisance": float(wn),
                      "signal_over_nuisance": float(min(off) / max(wn, 1e-9)),
                      "corr_with_travel": r, "grid": g.tolist()}
        print(f"{k:16s} {g[ni,nj]:8.4f} {min(off):9.4f} {max(off):9.4f} "
              f"{wn:11.4f} {min(off)/max(wn,1e-9):12.2f} {r:15.3f}")

    json.dump({"summary": summary, "travel": T.tolist(),
               "nuisance": {k: v for k, v in N.items()},
               "obj_tokens": obj_tokens.tolist()},
              open(f"{OUT}/token_analysis.json", "w"), indent=2)
    print(f"\nwrote {OUT}/token_analysis.json")


if __name__ == "__main__":
    main()
