"""Cross-family transfer: is friction encoded abstractly, or per-scene?

slide, collide and incline sweep the same physical quantity (object-surface
sliding friction) over nearly the same range in three visually different
interactions. A probe trained on one family and tested on another tells us
whether the latent carries friction as a reusable variable.

Result (2026-08-19): absolute values do not transfer (R2 collapses) but the
ordering largely does (Spearman ~0.85 into incline), so a new scene needs a
small recalibration set rather than a full retrain.
"""

import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import r2_score

from core.probe import LinearProbe
from core.paths import FEAT_DIR

FEAT = str(FEAT_DIR)
MU_COL = {"slide": 1, "collide": 1, "incline": 1}
FAMS = ["slide", "collide", "incline"]


def load(fam, sub):
    d = np.load(f"{FEAT}/{fam}__{sub}.npz")
    X = d["mean"].astype(np.float64).reshape(len(d["mean"]), -1)
    return X, np.log(d["params"][:, MU_COL[fam]])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    out = {}
    hdr = "train/test"
    print(f"{hdr:12s} " + "".join(f"{f:>14s}" for f in FAMS))
    for ftr in FAMS:
        Xtr, ytr = load(ftr, "train_clean")
        probe = LinearProbe().fit(Xtr, ytr[:, None])
        row, cells = {}, []
        for fte in FAMS:
            Xte, yte = load(fte, "test_clean")
            pr = probe.predict(Xte)[:, 0]
            row[fte] = {"r2": float(r2_score(yte, pr)),
                        "spearman": spearman(yte, pr)}
            cells.append(f"{row[fte]['r2']:+.2f}/{row[fte]['spearman']:+.2f}")
        out[ftr] = row
        print(f"{ftr:12s} " + "".join(f"{c:>14s}" for c in cells))
    print("\n(cell = R2 / Spearman; diagonal = within-family)")
    os.makedirs("out/multi", exist_ok=True)
    json.dump(out, open("out/multi/transfer.json", "w"), indent=2)


if __name__ == "__main__":
    main()
