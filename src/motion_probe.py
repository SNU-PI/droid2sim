"""Motion-only probe: train the read-out on frame-difference features only.

The gap analysis showed the real DROID clip is 19x out of distribution in the
appearance channel but INSIDE the sim distribution in the motion channel
(frame-to-frame feature differences). If the physics really lives in how the
representation changes over time, a probe that never sees the appearance
channel should (a) still work in sim, and (b) stop being blind on real video.

Three feature variants, each scored in sim (held-out) and on the DROID window:
    full        [T, D] as before                       (appearance + motion)
    diff        [T-1, D] frame differences only        (motion)
    diff-nomean diff, additionally centred per clip    (motion, appearance-free)
"""
import sys, os, json
import numpy as np
VJEPA = "/data/pgc/simdroid/stage0/vjepa2"
sys.path.insert(0, VJEPA); sys.path.insert(0, os.path.dirname(__file__))
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from droid_probe import build, encode, to256
from scenes import PARAMS

FEAT = "/data/pgc/simdroid/features"; OUT = "out/droid"; ALPHAS = np.logspace(-1, 6, 22)


def feats(M, kind):
    if kind == "full": return M.reshape(len(M), -1)
    D = np.diff(M, axis=1)
    if kind == "diff": return D.reshape(len(M), -1)
    if kind == "diff-nomean":
        D = D - D.mean(axis=(1,), keepdims=True)         # remove per-clip mean drift
        return D.reshape(len(M), -1)
    raise ValueError(kind)


def fit(fam, kind, aug=False):
    tr = np.load(f"{FEAT}/{fam}__train_clean.npz"); Xtr = tr["mean"].astype(np.float64); ytr = np.log(tr["params"])
    if aug:
        tc = np.load(f"{FEAT}/{fam}__train_camera.npz")
        Xtr = np.concatenate([Xtr, tc["mean"].astype(np.float64)]); ytr = np.concatenate([ytr, np.log(tc["params"])])
    F = feats(Xtr, kind)
    sc = StandardScaler().fit(F); pca = PCA(n_components=96).fit(sc.transform(F))
    Z = pca.transform(sc.transform(F)); mdl = RidgeCV(alphas=ALPHAS).fit(Z, ytr)
    mu = Z.mean(0); icov = np.linalg.inv(np.cov(Z.T) + 1e-6 * np.eye(96))
    ref = np.sqrt(np.einsum("ij,jk,ik->i", Z - mu, icov, Z - mu))
    def run(M):
        z = pca.transform(sc.transform(feats(M, kind)))
        return mdl.predict(z), np.sqrt(np.einsum("ij,jk,ik->i", z - mu, icov, z - mu))
    return run, float(np.percentile(ref, 95))


def main():
    fam = "slide"; names = [n for n, _, _ in PARAMS[fam]]
    enc, tf = build()
    d = np.load("/data/pgc/simdroid/droid/eps/ep88.npz", allow_pickle=True); fr = d["frames"]; sp = d["sp"]
    # real windows: pre-encode all
    n = len(fr); span = 15; starts = list(range(0, n - span, 4))
    real_feats = np.stack([encode(enc, tf, to256(fr[list(range(s, s + span, 2))])) for s in starts])
    ee = np.array([sp[s:s+span].mean() for s in starts])
    hi, lo = np.argsort(ee)[-5:], np.argsort(ee)[:5]
    R = {}
    print(f"{'variant':14s} | sim clean R2 (mass, mu) | sim cam R2 | real maha/p95 | real mass moving vs still | real mu moving vs still")
    for kind in ("full", "diff", "diff-nomean"):
        for aug in (False, True):
            run, p95 = fit(fam, kind, aug)
            row = {}
            for sub in ("test_clean", "test_camera"):
                te = np.load(f"{FEAT}/{fam}__{sub}.npz"); Xt = te["mean"].astype(np.float64); yt = np.log(te["params"])
                pr, _ = run(Xt); row[sub] = [float(r2_score(yt[:, i], pr[:, i])) for i in range(2)]
            pr, mh = run(real_feats); pr = np.exp(pr)
            row["real"] = dict(maha_med=float(np.median(mh)), p95=p95,
                               mass_moving=float(pr[hi, 0].mean()), mass_still=float(pr[lo, 0].mean()),
                               mu_moving=float(pr[hi, 1].mean()), mu_still=float(pr[lo, 1].mean()),
                               mass_cv=float(pr[:, 0].std() / pr[:, 0].mean()),
                               corr_mass_ee=float(np.corrcoef(ee, pr[:, 0])[0, 1]),
                               corr_mu_ee=float(np.corrcoef(ee, pr[:, 1])[0, 1]),
                               preds=pr.tolist(), ee=ee.tolist())
            R[f"{kind}{'+aug' if aug else ''}"] = row
            r = row["real"]
            print(f"{kind+('+aug' if aug else ''):14s} | {row['test_clean'][0]:+.2f} {row['test_clean'][1]:+.2f}"
                  f"           | {row['test_camera'][0]:+.2f} {row['test_camera'][1]:+.2f} "
                  f"| {r['maha_med']:5.1f}/{p95:4.1f} | {r['mass_moving']:.3f} vs {r['mass_still']:.3f}"
                  f"  | {r['mu_moving']:.3f} vs {r['mu_still']:.3f}   corr(ee,mu)={r['corr_mu_ee']:+.2f}")
    json.dump(R, open(f"{OUT}/motion_probe.json", "w"), indent=2)
    print(f"\nwrote {OUT}/motion_probe.json")


if __name__ == "__main__":
    main()
