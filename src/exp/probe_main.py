"""Can the physical parameters be read back out of the representation?

A linear probe on frozen features separates two explanations of the stage-0
failure: the information is absent, or it is present and the hand-written
distance simply could not expose it.

Two baselines carry the argument:

    physics   the true trajectory summary from the simulator, an upper bound
              on what any visual system could recover from the episode
    pixels    downsampled raw frames, what you get with no world model at all

Scoring is on held-out episodes, and on the same episodes re-rendered with the
camera moved and the lights changed. A probe that only works on the clean
render is reading the scene, not the physics.
"""

import sys, os, glob, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import r2_score

from core.probe import LinearProbe
from core.paths import EP_DIR, FEAT_DIR
from core.scenes import PARAMS

FEAT, ROOT = str(FEAT_DIR), str(EP_DIR)


def pixel_feats(fam, sub):
    fs = sorted(glob.glob(f"{ROOT}/{fam}/{sub}/*.npz"))
    out = []
    for f in fs:
        c = np.load(f)["clip"].astype(np.float32) / 255.0
        t, h, w, _ = c.shape
        g = c.reshape(t, 16, h // 16, 16, w // 16, 3).mean(axis=(2, 4))
        out.append(g.reshape(-1))
    return np.stack(out)


def build_features(fam, sub, kind):
    d = np.load(f"{FEAT}/{fam}__{sub}.npz")
    if kind == "physics":
        return d["phys"].astype(np.float64)
    if kind == "pixels":
        return pixel_feats(fam, sub).astype(np.float64)
    m = d["mean"].astype(np.float32)
    if kind == "vjepa":
        return m.reshape(len(m), -1).astype(np.float64)
    if kind == "vjepa+d":
        dm = np.diff(m, axis=1)
        return np.concatenate([m.reshape(len(m), -1),
                               dm.reshape(len(m), -1)], 1).astype(np.float64)
    raise ValueError(kind)


def run_family(fam, kinds):
    names = [n for n, _, _ in PARAMS[fam]]
    ytr = np.log(np.load(f"{FEAT}/{fam}__train_clean.npz")["params"])
    res = {}
    for kind in kinds:
        try:
            Xtr = build_features(fam, "train_clean", kind)
        except FileNotFoundError:
            continue
        probe = LinearProbe().fit(Xtr, ytr)
        row = {}
        for sub in ("test_clean", "test_camera", "test_light"):
            try:
                Xt = build_features(fam, sub, kind)
            except FileNotFoundError:
                continue
            yt = np.log(np.load(f"{FEAT}/{fam}__{sub}.npz")["params"])
            pr = probe.predict(Xt)
            row[sub] = {names[i]: float(r2_score(yt[:, i], pr[:, i]))
                        for i in range(len(names))}
        res[kind] = row
    return names, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+",
                    default=["slide", "roll", "bounce", "collide", "incline"])
    ap.add_argument("--kinds", nargs="+",
                    default=["physics", "pixels", "vjepa", "vjepa+d"])
    ap.add_argument("--out", default="out/multi/probe.json")
    a = ap.parse_args()

    allres = {}
    for fam in a.families:
        if not os.path.exists(f"{FEAT}/{fam}__train_clean.npz"):
            print(f"skip {fam} (no features)")
            continue
        names, res = run_family(fam, a.kinds)
        allres[fam] = res
        print(f"\n=== {fam}  ({' , '.join(names)}) ===")
        print(f"{'features':10s} " + "".join(
            f"{s.replace('test_', ''):>26s}" for s in
            ("test_clean", "test_camera", "test_light")))
        for kind, row in res.items():
            cells = []
            for sub in ("test_clean", "test_camera", "test_light"):
                cells.append("  ".join(f"{n}={row[sub][n]:+.3f}" for n in names)
                             if sub in row else "-")
            print(f"{kind:10s} " + "".join(f"{c:>26s}" for c in cells))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(allres, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
