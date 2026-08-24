"""Run the sim-trained probe on a real DROID episode.

There is no ground truth on real footage (nobody weighed the object), so
accuracy cannot be scored. What can be measured is whether the probe behaves
like a physical-property estimator or like noise: distance to the sim training
cloud, contrast between windows where the object moves and where it does not,
stability across overlapping windows, and nuisance sensitivity of the same
order as in sim.
"""

import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.encoder import build_encoder, encode_frames, to256
from core.probe import LinearProbe
from core.paths import FEAT_DIR, DROID_EPS
from core.scenes import PARAMS

FEAT = str(FEAT_DIR)
OUT = "out/droid"


def load_probe(family, aug):
    names = [n for n, _, _ in PARAMS[family]]
    tr = np.load(f"{FEAT}/{family}__train_clean.npz")
    Xtr = tr["mean"].astype(np.float64)
    ytr = np.log(tr["params"])
    if aug:
        tc = np.load(f"{FEAT}/{family}__train_camera.npz")
        Xtr = np.concatenate([Xtr, tc["mean"].astype(np.float64)])
        ytr = np.concatenate([ytr, np.log(tc["params"])])
    T = Xtr.shape[1]
    probe = LinearProbe().fit(Xtr.reshape(len(Xtr), -1), ytr)
    return names, probe, T, len(Xtr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=88)
    ap.add_argument("--family", default="slide")
    ap.add_argument("--aug", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    d = np.load(f"{DROID_EPS}/ep{a.ep}.npz", allow_pickle=True)
    frames, sp, grip = d["frames"], d["sp"], d["grip"]
    print(f"ep {a.ep}: {len(frames)} frames @15fps | {str(d['instr'])}")

    names, probe, T, ntr = load_probe(a.family, a.aug)
    print(f"probe: {a.family}, {ntr} train, T={T} frames; sim maha p95 {probe.ref_p95:.1f}")

    enc, tf = build_encoder()

    def predict(clip, crop=None, bright=1.0):
        f = encode_frames(enc, tf, to256(clip, crop, bright))
        x = f.reshape(1, -1)
        return np.exp(probe.predict(x)[0]), float(probe.maha(x)[0])

    # 8 frames at stride 2 cover ~1 s, matching the sim episodes
    n = len(frames)
    stride = 2
    span = (T - 1) * stride + 1
    rows = []
    for s0 in range(0, n - span, 4):
        idx = list(range(s0, s0 + span, stride))
        p, m = predict(frames[idx])
        rows.append(dict(start=s0, ee_speed=float(sp[idx].mean()),
                         grip=float(grip[idx].mean()), maha=m,
                         **{names[i]: float(p[i]) for i in range(len(names))}))
    R = {"episode": a.ep, "instr": str(d["instr"]), "family": a.family,
         "aug": a.aug, "sim_maha_p95": probe.ref_p95, "windows": rows}

    ee = np.array([r["ee_speed"] for r in rows])
    mh = np.array([r["maha"] for r in rows])
    hi, lo = np.argsort(ee)[-5:], np.argsort(ee)[:5]
    print(f"\n{'window':>7} {'ee m/s':>7} {'maha':>6} " + " ".join(f"{n_:>10}" for n_ in names))
    for r in rows:
        print(f"{r['start']:7d} {r['ee_speed']:7.3f} {r['maha']:6.1f} "
              + " ".join(f"{r[n_]:10.4f}" for n_ in names))
    R["motion_contrast"] = {}
    for n_ in names:
        v = np.array([r[n_] for r in rows])
        R["motion_contrast"][n_] = dict(
            moving_mean=float(v[hi].mean()), still_mean=float(v[lo].mean()),
            cv_all=float(v.std() / v.mean()))
        print(f"\n{n_}: moving {v[hi].mean():.4f} | still {v[lo].mean():.4f} "
              f"| CV {v.std()/v.mean():.3f}")
    print(f"maha: moving {mh[hi].mean():.1f}, still {mh[lo].mean():.1f}, "
          f"sim p95 {probe.ref_p95:.1f}")

    # nuisance sensitivity on the most-moving window
    s0 = rows[int(hi[-1])]["start"]
    base = frames[list(range(s0, s0 + span, stride))]
    p0, _ = predict(base)
    variants = {"crop shift +12px": dict(crop=(82, 0, 180, 180)),
                "crop shift -12px": dict(crop=(58, 0, 180, 180)),
                "crop zoom in": dict(crop=(85, 15, 150, 150)),
                "brightness x1.2": dict(bright=1.2),
                "brightness x0.8": dict(bright=0.8)}
    R["nuisance"] = {"base": {names[i]: float(p0[i]) for i in range(len(names))}}
    print(f"\nnuisance on window {s0}: base "
          + " ".join(f"{names[i]}={p0[i]:.4f}" for i in range(len(names))))
    for k, kw in variants.items():
        p, m = predict(base, **kw)
        R["nuisance"][k] = {names[i]: float(p[i]) for i in range(len(names))}
        print(f"  {k:18s} " + " ".join(
            f"{names[i]}={p[i]:.4f} ({(p[i]/p0[i]-1)*100:+5.1f}%)"
            for i in range(len(names))) + f"  maha {m:.1f}")

    # window covering more or less real time
    R["stride"] = {}
    print("\nstride ablation:")
    for st in (1, 2, 3, 4):
        sp_ = (T - 1) * st + 1
        if s0 + sp_ > n:
            continue
        p, m = predict(frames[list(range(s0, s0 + sp_, st))])
        R["stride"][st] = {names[i]: float(p[i]) for i in range(len(names))}
        print(f"  stride {st} ({sp_/15:.2f}s): " + " ".join(
            f"{names[i]}={p[i]:.4f}" for i in range(len(names))) + f"  maha {m:.1f}")

    tag = f"ep{a.ep}_{a.family}" + ("_aug" if a.aug else "")
    json.dump(R, open(f"{OUT}/{tag}.json", "w"), indent=2)
    print(f"\nwrote {OUT}/{tag}.json")


if __name__ == "__main__":
    main()
