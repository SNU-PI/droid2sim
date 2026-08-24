"""First contact with real footage: run the sim-trained probe on a DROID episode.

There is no ground truth here -- nobody weighed the green block -- so accuracy
cannot be measured. What CAN be measured is whether the probe behaves like a
physical-property estimator when handed real video, or like noise:

  in-distribution   how far the real clip's features sit from the sim training
                    cloud (Mahalanobis in PCA space). Far outside and any
                    prediction is extrapolation.
  motion contrast   the object is pushed only in a short window; predictions
                    on windows where it moves vs. windows where nothing moves.
                    A physics read-out has to react to that.
  stability         predictions across overlapping windows of the same push:
                    the block's mass does not change mid-episode.
  nuisance          same window, re-cropped / re-lit: the invariance measured
                    in sim should carry over, and be the same order.
  plausibility      predicted values in a physically sensible range for a
                    small plastic block on a wooden desk.
"""
import sys, os, json, argparse
import numpy as np

VJEPA = "/data/pgc/simdroid/stage0/vjepa2"
CKPT = "/data/pgc/simdroid/stage0/vjepa2-ac-vitg.pt"
sys.path.insert(0, VJEPA); sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scenes import PARAMS

FEAT = "/data/pgc/simdroid/features"
ALPHAS = np.logspace(-1, 6, 22)
OUT = "out/droid"


def build():
    from src.hub.backbones import _make_vjepa2_ac_model
    from app.vjepa_droid.transforms import make_transforms
    enc, _ = _make_vjepa2_ac_model(model_name="vit_ac_giant", img_size=256, pretrained=False)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    enc.load_state_dict({k.replace("module.", ""): v for k, v in ck["encoder"].items()}, strict=False)
    enc = enc.cuda().eval()
    for p in enc.parameters(): p.requires_grad_(False)
    tf = make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=(1., 1.),
                         random_resize_scale=(1., 1.), reprob=0., auto_augment=False,
                         motion_shift=False, crop_size=256)
    return enc, tf


@torch.no_grad()
def encode(enc, tf, frames):
    clip = tf(frames).unsqueeze(0).cuda()
    B, C, T, H, W = clip.size()
    c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = enc(c)
    h = F.layer_norm(h.view(B, T, -1, h.size(-1)).float(), (h.size(-1),))[0]
    return h.mean(1).cpu().numpy()                    # [T, D]


def to256(frames, crop=None, bright=1.0):
    """DROID frames are 320x180; centre-crop to square-ish then resize, like a
    real deployment would. crop=(x0,y0,w,h) overrides the default centre crop."""
    out = []
    for f in frames:
        im = Image.fromarray(f)
        if crop is None:
            w, h = im.size; s = min(w, h)
            box = ((w - s) // 2, 0, (w - s) // 2 + s, s)
        else:
            x0, y0, w, h = crop; box = (x0, y0, x0 + w, y0 + h)
        im = im.crop(box).resize((256, 256), Image.BILINEAR)
        a = np.asarray(im).astype(np.float32) * bright
        out.append(np.clip(a, 0, 255).astype(np.uint8))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=88)
    ap.add_argument("--family", default="slide")
    ap.add_argument("--aug", action="store_true", help="use camera-augmented probe")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    d = np.load(f"/data/pgc/simdroid/droid/eps/ep{a.ep}.npz", allow_pickle=True)
    frames, sp, grip = d["frames"], d["sp"], d["grip"]
    print(f"ep {a.ep}: {len(frames)} frames @15fps | {str(d['instr'])}")

    # ---- probe from sim --------------------------------------------------------
    names = [n for n, _, _ in PARAMS[a.family]]
    tr = np.load(f"{FEAT}/{a.family}__train_clean.npz")
    Xtr, ytr = tr["mean"].astype(np.float64), np.log(tr["params"])
    if a.aug:
        tc = np.load(f"{FEAT}/{a.family}__train_camera.npz")
        Xtr = np.concatenate([Xtr, tc["mean"].astype(np.float64)]); ytr = np.concatenate([ytr, np.log(tc["params"])])
    T = Xtr.shape[1]
    Xf = Xtr.reshape(len(Xtr), -1)
    sc = StandardScaler().fit(Xf); pca = PCA(n_components=96).fit(sc.transform(Xf))
    Ztr = pca.transform(sc.transform(Xf))
    mdl = RidgeCV(alphas=ALPHAS).fit(Ztr, ytr)
    mu_z, cov_z = Ztr.mean(0), np.cov(Ztr.T) + 1e-6 * np.eye(96)
    icov = np.linalg.inv(cov_z)
    def maha(z): return float(np.sqrt((z - mu_z) @ icov @ (z - mu_z)))
    ref_maha = np.array([maha(z) for z in Ztr])
    print(f"probe: {a.family}, {len(Xtr)} train, T={T} frames; "
          f"sim Mahalanobis median {np.median(ref_maha):.1f}, p95 {np.percentile(ref_maha,95):.1f}")

    enc, tf = build()

    def predict(clip8, crop=None, bright=1.0):
        f = encode(enc, tf, to256(clip8, crop, bright))          # [8, D]
        z = pca.transform(sc.transform(f.reshape(1, -1)))[0]
        p = np.exp(mdl.predict(z[None])[0])
        return p, maha(z)

    # ---- windows: 8 frames sampled over a span of ~1.6 s (matches sim 0.8s@8 frames -> use stride 2)
    n = len(frames); stride = 2; span = (T - 1) * stride + 1
    starts = list(range(0, n - span, 4))
    rows = []
    for s0 in starts:
        idx = list(range(s0, s0 + span, stride))
        p, m = predict(frames[idx])
        rows.append(dict(start=s0, ee_speed=float(sp[idx].mean()), grip=float(grip[idx].mean()),
                         maha=m, **{names[i]: float(p[i]) for i in range(len(names))}))
    R = {"episode": a.ep, "instr": str(d["instr"]), "family": a.family, "aug": a.aug,
         "sim_maha_median": float(np.median(ref_maha)), "sim_maha_p95": float(np.percentile(ref_maha, 95)),
         "windows": rows}

    # ---- motion contrast: windows where the arm moves fastest vs. slowest
    ee = np.array([r["ee_speed"] for r in rows]); mh = np.array([r["maha"] for r in rows])
    hi, lo = np.argsort(ee)[-5:], np.argsort(ee)[:5]
    print(f"\n{'window':>7} {'ee m/s':>7} {'maha':>6} " + " ".join(f"{n:>10}" for n in names))
    for r in rows:
        print(f"{r['start']:7d} {r['ee_speed']:7.3f} {r['maha']:6.1f} " + " ".join(f"{r[n]:10.4f}" for n in names))
    R["motion_contrast"] = {}
    for n_ in names:
        v = np.array([r[n_] for r in rows])
        R["motion_contrast"][n_] = dict(moving_mean=float(v[hi].mean()), still_mean=float(v[lo].mean()),
                                        moving_std=float(v[hi].std()), all_std=float(v.std()),
                                        cv_all=float(v.std() / v.mean()))
        print(f"\n{n_}: moving windows {v[hi].mean():.4f} +- {v[hi].std():.4f} | still windows {v[lo].mean():.4f} "
              f"| CV over all windows {v.std()/v.mean():.3f}")
    print(f"maha: moving {mh[hi].mean():.1f}, still {mh[lo].mean():.1f}, sim p95 {np.percentile(ref_maha,95):.1f}")

    # ---- nuisance on the most-moving window
    s0 = rows[int(hi[-1])]["start"]; idx = list(range(s0, s0 + span, stride)); base = frames[idx]
    p0, _ = predict(base)
    variants = {"crop shift +12px": dict(crop=(70 + 12, 0, 180, 180)),
                "crop shift -12px": dict(crop=(70 - 12, 0, 180, 180)),
                "crop zoom in": dict(crop=(85, 15, 150, 150)),
                "brightness x1.2": dict(bright=1.2), "brightness x0.8": dict(bright=0.8)}
    R["nuisance"] = {"base": {names[i]: float(p0[i]) for i in range(len(names))}}
    print(f"\nnuisance on window {s0}: base " + " ".join(f"{names[i]}={p0[i]:.4f}" for i in range(len(names))))
    for k, kw in variants.items():
        p, m = predict(base, **kw)
        R["nuisance"][k] = {names[i]: float(p[i]) for i in range(len(names))}
        print(f"  {k:18s} " + " ".join(f"{names[i]}={p[i]:.4f} ({(p[i]/p0[i]-1)*100:+5.1f}%)"
                                     for i in range(len(names))) + f"  maha {m:.1f}")

    # ---- temporal stride ablation: same start, different frame spacing
    R["stride"] = {}
    print("\nstride ablation (window covering more/less real time):")
    for st in (1, 2, 3, 4):
        sp_ = (T - 1) * st + 1
        if s0 + sp_ > n: continue
        p, m = predict(frames[list(range(s0, s0 + sp_, st))])
        R["stride"][st] = {names[i]: float(p[i]) for i in range(len(names))}
        print(f"  stride {st} ({sp_/15:.2f}s): " + " ".join(f"{names[i]}={p[i]:.4f}" for i in range(len(names))) + f"  maha {m:.1f}")

    tag = f"ep{a.ep}_{a.family}" + ("_aug" if a.aug else "")
    json.dump(R, open(f"{OUT}/{tag}.json", "w"), indent=2)
    np.savez_compressed(f"{OUT}/{tag}_window.npz", frames=to256(base))
    print(f"\nwrote {OUT}/{tag}.json")


if __name__ == "__main__":
    main()
