"""Motion-only probe: does dropping the appearance channel fix sim-to-real?

droid_gap.py showed the real clip is far out of distribution in appearance but
inside it in the motion channel. If the physics lives in how the
representation changes over time, a probe that never sees appearance should
keep its sim accuracy and stop being out-of-distribution on real video.

Feature variants come from core.probe.feature_variants; each is scored on sim
held-out sets and on windows of a real DROID episode.
"""

import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import r2_score

from core.encoder import build_encoder, encode_frames, to256
from core.probe import LinearProbe, feature_variants
from core.paths import FEAT_DIR, DROID_EPS
from core.scenes import PARAMS

FEAT = str(FEAT_DIR)
OUT = "out/droid"


def fit(fam, kind, aug=False):
    tr = np.load(f"{FEAT}/{fam}__train_clean.npz")
    Xtr, ytr = tr["mean"].astype(np.float64), np.log(tr["params"])
    if aug:
        tc = np.load(f"{FEAT}/{fam}__train_camera.npz")
        Xtr = np.concatenate([Xtr, tc["mean"].astype(np.float64)])
        ytr = np.concatenate([ytr, np.log(tc["params"])])
    probe = LinearProbe().fit(feature_variants(Xtr, kind), ytr)

    def run(M):
        F = feature_variants(M, kind)
        return probe.predict(F), probe.maha(F)

    return run, probe.ref_p95


def main():
    fam = "slide"
    names = [n for n, _, _ in PARAMS[fam]]
    os.makedirs(OUT, exist_ok=True)
    enc, tf = build_encoder()
    d = np.load(f"{DROID_EPS}/ep88.npz", allow_pickle=True)
    fr, sp = d["frames"], d["sp"]

    n = len(fr)
    span = 15
    starts = list(range(0, n - span, 4))
    real = np.stack([encode_frames(enc, tf, to256(fr[list(range(s, s + span, 2))]))
                     for s in starts])
    ee = np.array([sp[s:s + span].mean() for s in starts])
    hi, lo = np.argsort(ee)[-5:], np.argsort(ee)[:5]

    R = {}
    print(f"{'variant':16s} | sim clean R2 | sim cam R2 | real maha/p95 "
          f"| mu moving vs still")
    for kind in ("full", "diff", "diff-nomean"):
        for aug in (False, True):
            run, p95 = fit(fam, kind, aug)
            row = {}
            for sub in ("test_clean", "test_camera"):
                te = np.load(f"{FEAT}/{fam}__{sub}.npz")
                pr, _ = run(te["mean"].astype(np.float64))
                yt = np.log(te["params"])
                row[sub] = [float(r2_score(yt[:, i], pr[:, i])) for i in range(2)]
            pr, mh = run(real)
            pr = np.exp(pr)
            row["real"] = dict(
                maha_med=float(np.median(mh)), p95=p95,
                mass_moving=float(pr[hi, 0].mean()), mass_still=float(pr[lo, 0].mean()),
                mu_moving=float(pr[hi, 1].mean()), mu_still=float(pr[lo, 1].mean()),
                corr_mu_ee=float(np.corrcoef(ee, pr[:, 1])[0, 1]),
                preds=pr.tolist(), ee=ee.tolist())
            key = f"{kind}{'+aug' if aug else ''}"
            R[key] = row
            r = row["real"]
            print(f"{key:16s} | {row['test_clean'][0]:+.2f} {row['test_clean'][1]:+.2f}"
                  f"  | {row['test_camera'][0]:+.2f} {row['test_camera'][1]:+.2f}"
                  f" | {r['maha_med']:5.1f}/{p95:4.1f} | {r['mu_moving']:.3f} vs "
                  f"{r['mu_still']:.3f}  corr(ee,mu)={r['corr_mu_ee']:+.2f}")
    json.dump(R, open(f"{OUT}/motion_probe.json", "w"), indent=2)
    print(f"\nwrote {OUT}/motion_probe.json")


if __name__ == "__main__":
    main()
