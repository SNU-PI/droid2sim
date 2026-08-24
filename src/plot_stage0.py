"""Stage 0 figure: does the frozen V-JEPA 2 latent respond to mass and friction?"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
OUT = "out/stage0"
C_PHYS, C_NUIS, C_REF, C_CAL = "#1b4965", "#c1121f", "#f0a202", "#2a9d8f"


def main():
    ta = json.load(open(f"{OUT}/token_analysis.json"))
    cal = json.load(open(f"{OUT}/calibration.json"))
    es = json.load(open(f"{OUT}/energy_summary.json"))
    from scene import MASS_GRID, MU_GRID, NOMINAL_CELL
    mass, mu = MASS_GRID, MU_GRID
    ni, nj = NOMINAL_CELL

    best = "object_tokens"
    G = np.array(ta["summary"][best]["grid"])
    travel = np.array(ta["travel"])
    nuis = {k: v[best] for k, v in ta["nuisance"].items()}

    fig = plt.figure(figsize=(15, 8.6))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34,
                          height_ratios=[1.1, 1.0])

    # 1. the landscape
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(G, cmap="viridis", origin="upper")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f"{v:.3f}" for v in mu], rotation=45, fontsize=8.5)
    ax.set_yticklabels([f"{v:.3f}" for v in mass], fontsize=8.5)
    ax.set_xlabel("friction  mu"); ax.set_ylabel("mass (kg)")
    ax.scatter([nj], [ni], s=260, facecolors="none", edgecolors=C_REF, lw=3)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{G[i,j]:.3f}", ha="center", va="center", fontsize=7.6,
                    color="white" if G[i, j] < G.max() * 0.65 else "black")
    ax.set_title("latent distance to the reference\n(gold ring = reference physics)",
                 loc="left", fontweight="bold", fontsize=11)
    ax.grid(False); plt.colorbar(im, ax=ax, fraction=0.046)

    # 2. travel, for scale
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(travel, cmap="magma", origin="upper")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f"{v:.3f}" for v in mu], rotation=45, fontsize=8.5)
    ax.set_yticklabels([f"{v:.3f}" for v in mass], fontsize=8.5)
    ax.set_xlabel("friction  mu"); ax.set_ylabel("mass (kg)")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{travel[i,j]:.1f}", ha="center", va="center",
                    fontsize=7.6,
                    color="white" if travel[i, j] < travel.max() * 0.6 else "black")
    ax.set_title("the physics is clearly different\n(cube travel, cm)",
                 loc="left", fontweight="bold", fontsize=11)
    ax.grid(False); plt.colorbar(im, ax=ax, fraction=0.046)

    # 3. THE decisive plot: response curve
    ax = fig.add_subplot(gs[0, 2])
    pd = [0.0] + [c["pixdiff"] for c in cal if c["name"].startswith("noise")]
    en = [0.0] + [c["energy"] for c in cal if c["name"].startswith("noise")]
    ax.plot(pd, en, "o-", color=C_CAL, lw=2.2, ms=7, label="added pixel noise")
    off = [G[i, j] for i in range(5) for j in range(5) if (i, j) != (ni, nj)]
    ppix = np.array(es["pixdiff"]).ravel()
    ax.scatter(ppix, G.ravel(), s=44, color=C_PHYS, zorder=5,
               label="physics changes")
    ax.axhspan(min(off), max(off), color=C_PHYS, alpha=0.12)
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_xlabel("mean pixel change (grey levels)")
    ax.set_ylabel("latent distance")
    ax.set_title("the distance saturates almost immediately", loc="left",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

    # 4. physics vs nuisance
    ax = fig.add_subplot(gs[1, 0])
    names = sorted(nuis, key=lambda k: nuis[k])
    ax.barh(np.arange(len(names)), [nuis[k] for k in names], color=C_NUIS,
            height=0.62)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8.5); ax.invert_yaxis()
    ax.axvspan(min(off), max(off), color=C_PHYS, alpha=0.18)
    ax.axvline(min(off), color=C_PHYS, lw=2)
    ax.axvline(max(off), color=C_PHYS, lw=2)
    ax.text(min(off), -0.75, "physics band", color=C_PHYS, fontsize=9.5,
            fontweight="bold")
    ax.set_xlabel("latent distance to the reference")
    ax.set_title("irrelevant changes (bars) reach into\nthe physics band",
                 loc="left", fontweight="bold", fontsize=11)
    ax.grid(alpha=0.25, axis="x"); ax.spines[["top", "right"]].set_visible(False)

    # 5. metric comparison
    ax = fig.add_subplot(gs[1, 1])
    ks = ["mean_all", "top2pct", "max_token", "object_tokens"]
    s = ta["summary"]
    x = np.arange(len(ks))
    ax.bar(x - 0.2, [s[k]["signal_over_nuisance"] for k in ks], 0.38,
           color=C_PHYS, label="signal / nuisance")
    ax.bar(x + 0.2, [s[k]["corr_with_travel"] for k in ks], 0.38,
           color=C_NUIS, label="corr with physical difference")
    ax.axhline(1.0, color="#333", ls="--", lw=1.6)
    ax.text(3.45, 1.03, "needs to be\nwell above 1", fontsize=8.5, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(ks, rotation=18, fontsize=9)
    ax.set_title("no choice of distance rescues it", loc="left",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(alpha=0.25, axis="y"); ax.spines[["top", "right"]].set_visible(False)

    # 6. latent vs physical difference
    ax = fig.add_subplot(gs[1, 2])
    dtr = np.abs(travel - travel[ni, nj]).ravel()
    ax.scatter(dtr, G.ravel(), s=52, color=C_PHYS, alpha=0.8, edgecolors="none")
    r = s[best]["corr_with_travel"]
    ax.set_xlabel("difference in cube travel (cm)")
    ax.set_ylabel("latent distance")
    ax.set_title(f"a bigger physical change does not reliably\nmean a bigger "
                 f"latent change   (r = {r:.2f})", loc="left",
                 fontweight="bold", fontsize=11)
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Stage 0 — frozen V-JEPA 2 (ViT-g, DROID-post-trained) vs a "
                 "controlled mass/friction sweep",
                 fontsize=14, fontweight="bold", y=0.985)
    fig.savefig(f"{OUT}/stage0_result.png", dpi=145, bbox_inches="tight")
    print(f"wrote {OUT}/stage0_result.png")


if __name__ == "__main__":
    main()
