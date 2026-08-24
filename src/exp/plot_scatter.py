"""Truth-vs-prediction scatter for every parameter of every family."""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from core.probe import LinearProbe
from core.paths import FEAT_DIR
from core.scenes import PARAMS

FEAT = str(FEAT_DIR)
FAMS = ["slide", "roll", "bounce", "collide", "incline"]
UNOBS = {("bounce", "mass"), ("incline", "mass")}
UNITS = {"mass": "kg", "mass2": "kg"}
C_OK, C_NO = "#1b4965", "#9aa5b1"


def load(fam, sub):
    d = np.load(f"{FEAT}/{fam}__{sub}.npz")
    return (d["mean"].astype(np.float64).reshape(len(d["mean"]), -1),
            np.log(d["params"]))


def main():
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.2))
    fig.patch.set_facecolor("white")
    for j, fam in enumerate(FAMS):
        names = [n for n, _, _ in PARAMS[fam]]
        Xtr, ytr = load(fam, "train_clean")
        Xte, yte = load(fam, "test_clean")
        pr = LinearProbe().fit(Xtr, ytr).predict(Xte)
        for i, n in enumerate(names):
            ax = axes[i, j]
            y, p = np.exp(yte[:, i]), np.exp(pr[:, i])
            unobs = (fam, n) in UNOBS
            ax.scatter(y, p, s=22, color=C_NO if unobs else C_OK,
                       alpha=0.75, edgecolors="none")
            lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
            ax.plot([lo, hi], [lo, hi], "--", color="#c1121f", lw=1.3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"true {n} {UNITS.get(n, '')}".strip(), fontsize=9.5)
            if j == 0:
                ax.set_ylabel("V-JEPA probe prediction", fontsize=9.5)
            title = f"{fam} · {n}\nR² = {r2_score(yte[:, i], pr[:, i]):+.3f}"
            if unobs:
                title += "   (unobservable)"
            ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left",
                         color="#666" if unobs else "#161a20")
            ax.grid(alpha=0.25, which="both", lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
    fig.suptitle("Frozen V-JEPA 2 -> Ridge probe: held-out prediction vs truth",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs("out/multi", exist_ok=True)
    fig.savefig("out/multi/probe_scatter.png", dpi=150, bbox_inches="tight")
    print("wrote out/multi/probe_scatter.png")


if __name__ == "__main__":
    main()
