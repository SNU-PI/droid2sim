"""Encode every episode with frozen V-JEPA 2 and store pooled features.

Full token grids would be 11.5 MB per clip and 28 GB overall, so each clip is
reduced to per-frame pooled vectors. Pooling is per FRAME rather than over the
whole clip on purpose: friction only reveals itself in how the motion decays,
so collapsing time would throw away the very thing being looked for.

Two poolings are kept -- the mean over tokens, and the spatial standard
deviation across tokens, which retains something about where in the frame the
activity is.
"""
import sys, os, glob, argparse, time
import numpy as np

VJEPA = "/data/pgc/simdroid/stage0/vjepa2"
CKPT = "/data/pgc/simdroid/stage0/vjepa2-ac-vitg.pt"
sys.path.insert(0, VJEPA)
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

ROOT = os.environ.get("EP_DIR", "/data/pgc/simdroid/episodes")
FEAT = os.environ.get("FEAT_DIR", "/data/pgc/simdroid/features")


def build():
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
    return enc, tf


@torch.no_grad()
def encode(enc, tf, frames):
    clip = tf(frames).unsqueeze(0).cuda()
    B, C, T, H, W = clip.size()
    c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = enc(c)
    h = h.view(B, T, -1, h.size(-1)).float()          # [1, T, N, D]
    h = F.layer_norm(h, (h.size(-1),))[0]             # [T, N, D]
    return h.mean(1).cpu().numpy(), h.std(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    a = ap.parse_args()
    os.makedirs(FEAT, exist_ok=True)
    enc, tf = build()
    for d in a.dirs:
        fam, sub = d.split("/")
        outp = f"{FEAT}/{fam}__{sub}.npz"
        if os.path.exists(outp):
            print(f"skip {d}"); continue
        files = sorted(glob.glob(f"{ROOT}/{d}/*.npz"))
        if not files:
            print(f"empty {d}"); continue
        t0 = time.time()
        M, S, P, X = [], [], [], []
        for f in files:
            z = np.load(f)
            m, s = encode(enc, tf, z["clip"])
            M.append(m.astype(np.float16)); S.append(s.astype(np.float16))
            P.append(z["params"])
            X.append([float(z[k]) for k in ("travel", "vmax", "vend", "zmax",
                                            "zend", "aux")])
        np.savez_compressed(outp, mean=np.stack(M), std=np.stack(S),
                            params=np.stack(P), phys=np.array(X),
                            files=[os.path.basename(f) for f in files])
        print(f"{d}: {len(files)} clips -> {outp}  ({time.time()-t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
