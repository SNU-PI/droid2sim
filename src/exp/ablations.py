"""Three follow-ups to the probe result.

sample efficiency  how many labelled episodes does the read-out actually need?
                   If a few dozen suffice, calibrating a new scene is cheap.

frame count        the probe is given only the first k frames. Mass shows up in
                   how fast the object leaves; friction only in how that speed
                   decays, so the two should become readable at different
                   points, and the earlier context-length story predicts the
                   ordering.

camera augmentation the probe's one weakness was a 1 cm camera shift. If
                   training on shifted renders repairs it, viewpoint is a
                   nuisance you can train away rather than a hard limit.
"""
import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import r2_score

from core.probe import LinearProbe
from core.paths import FEAT_DIR
from core.scenes import PARAMS

FEAT = str(FEAT_DIR)


def load(fam, sub):
    d = np.load(f"{FEAT}/{fam}__{sub}.npz")
    return d["mean"].astype(np.float32), np.log(d["params"])


def fit_eval(Xtr, ytr, tests, npca=96):
    probe = LinearProbe(npca=npca).fit(Xtr, ytr)
    out = {}
    for name, (Xt, yt) in tests.items():
        pr = probe.predict(Xt)
        out[name] = [float(r2_score(yt[:, i], pr[:, i])) for i in range(yt.shape[1])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--out", default="out/multi/ablations.json")
    a = ap.parse_args()
    R = {}
    for fam in a.families:
        if not os.path.exists(f"{FEAT}/{fam}__train_clean.npz"):
            continue
        names = [n for n, _, _ in PARAMS[fam]]
        Mtr, ytr = load(fam, "train_clean")            # [N, T, D]
        raw = {}
        for sub in ("test_clean", "test_camera", "test_light"):
            if os.path.exists(f"{FEAT}/{fam}__{sub}.npz"):
                raw[sub] = load(fam, sub)               # keep [N, T, D]
        flat = {nm: (M.reshape(len(M), -1), y) for nm, (M, y) in raw.items()}
        F = {"names": names}

        # --- sample efficiency
        F["nsamples"] = {}
        for n in (25, 50, 100, 175, len(Mtr)):
            n = min(n, len(Mtr))
            idx = np.random.default_rng(0).permutation(len(Mtr))[:n]
            F["nsamples"][str(n)] = fit_eval(
                Mtr[idx].reshape(n, -1), ytr[idx], flat)

        # --- frame count: probe sees only the first k frames, train AND test
        F["nframes"] = {}
        T = Mtr.shape[1]
        for k in range(1, T + 1):
            tk = {nm: (M[:, :k].reshape(len(M), -1), y)
                  for nm, (M, y) in raw.items()}
            F["nframes"][str(k)] = fit_eval(
                Mtr[:, :k].reshape(len(Mtr), -1), ytr, tk)

        # --- camera augmentation
        if os.path.exists(f"{FEAT}/{fam}__train_camera.npz"):
            Mc, yc = load(fam, "train_camera")
            Xa = np.concatenate([Mtr.reshape(len(Mtr), -1),
                                 Mc.reshape(len(Mc), -1)], 0)
            ya = np.concatenate([ytr, yc], 0)
            F["augmented"] = fit_eval(Xa, ya, flat)
        R[fam] = F
        print(f"{fam}: done", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
