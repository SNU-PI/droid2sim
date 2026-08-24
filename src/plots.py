"""Figures for the pilot."""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import WorldModel, N_PRED
from train import load_split, norm_params, get_batch, E_EVAL
from readout import centroid_px, apply_map
from sim import MASSES, FRICTIONS, NOMINAL
import raster

OUT = "out"
NOM_C = NOMINAL[0] * len(FRICTIONS) + NOMINAL[1]
ORDER = ["k1", "k2", "k4", "k8", "k16"]
C_SIM, C_WM, C_ORA = "#1b4965", "#c1121f", "#2a9d8f"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})


def fig_ablation(res):
    ks = [res[t]["k"] for t in ORDER]
    dcorr = [res[t]["delta_pearson"] for t in ORDER]
    gain = [res[t]["delta_gain"] for t in ORDER]
    aerr = [res[t]["abs_err_m"] * 1000 for t in ORDER]
    o = res["oracle"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].plot(ks, aerr, "o-", color="#6c757d", lw=2, ms=7)
    ax[0].axhline(o["abs_err_m"] * 1000, ls="--", color=C_ORA, lw=2,
                  label=f"oracle (told the physics): {o['abs_err_m']*1000:.1f} mm")
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(ks)
    ax[0].set_xticklabels([str(k) for k in ks])
    ax[0].set_xlabel("context length k (frames)")
    ax[0].set_ylabel("absolute position error (mm)")
    ax[0].set_title("The obvious metric\nabsolute error", loc="left", fontweight="bold")
    ax[0].legend(fontsize=9, frameon=False)

    ax[1].plot(ks, dcorr, "o-", color=C_WM, lw=2, ms=7, label="physics sensitivity  r")
    ax[1].plot(ks, gain, "s--", color="#e07a5f", lw=1.6, ms=6, label="sensitivity gain")
    ax[1].axhline(o["delta_pearson"], ls="--", color=C_ORA, lw=2,
                  label=f"oracle: r = {o['delta_pearson']:.3f}")
    ax[1].axhline(0, color="#adb5bd", lw=1)
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(ks)
    ax[1].set_xticklabels([str(k) for k in ks])
    ax[1].set_ylim(-0.08, 1.05)
    ax[1].set_xlabel("context length k (frames)")
    ax[1].set_ylabel("agreement with the simulator's response")
    ax[1].set_title("The metric that separates them\ndifference-of-differences",
                    loc="left", fontweight="bold")
    ax[1].legend(fontsize=9, frameon=False, loc="lower right")

    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_ablation.png", dpi=150)
    plt.close(fig)


def fig_scatter(tab, res):
    show = [("k1", "k = 1 frame"), ("k4", "k = 4 frames"),
            ("k16", "k = 16 frames"), ("oracle", "oracle (given m, mu)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharex=True, sharey=True)
    keep = np.arange(25) != NOM_C
    for ax, (t, lbl) in zip(axes, show):
        P, T = tab[f"{t}__P"], tab[f"{t}__T"]
        dP = (P - P[:, NOM_C:NOM_C + 1])[:, keep].reshape(-1) * 1000
        dT = (T - T[:, NOM_C:NOM_C + 1])[:, keep].reshape(-1) * 1000
        s = np.random.default_rng(0).choice(len(dT), min(4000, len(dT)), replace=False)
        ax.scatter(dT[s], dP[s], s=3, alpha=0.18,
                   color=C_ORA if t == "oracle" else C_WM, edgecolors="none")
        lim = np.abs(dT).max() * 1.05
        ax.plot([-lim, lim], [-lim, lim], color="#495057", lw=1.2, ls="--")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"{lbl}\nr = {res[t]['delta_pearson']:.3f},  "
                     f"slope = {res[t]['delta_gain']:.2f}", fontsize=10.5)
        ax.set_xlabel("simulator's change (mm)")
    axes[0].set_ylabel("world model's change (mm)")
    fig.suptitle("Same episode, physics changed: does the model's answer move the way the simulator's does?",
                 fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_heat(tab):
    show = [("simulator", None), ("k1", "k = 1"), ("k16", "k = 16"),
            ("oracle", "oracle")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    T = tab["k1__T"]
    truth = np.linalg.norm(T, axis=2).mean(0).reshape(5, 5) * 1000
    vmax = truth.max()
    for ax, (t, lbl) in zip(axes, show):
        if t == "simulator":
            grid, title = truth, "simulator (ground truth)"
        else:
            P = tab[f"{t}__P"]
            grid = np.linalg.norm(P, axis=2).mean(0).reshape(5, 5) * 1000
            title = f"world model, {lbl}"
        im = ax.imshow(grid, cmap="magma", vmin=0, vmax=vmax, origin="upper")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels([f"{v:.3f}" for v in FRICTIONS], fontsize=8, rotation=45)
        ax.set_yticklabels([f"{v:.2f}" for v in MASSES], fontsize=8)
        ax.set_xlabel("friction"); ax.grid(False)
        ax.set_title(title, fontsize=10.5)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{grid[i,j]:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white" if grid[i, j] < vmax * .6 else "black")
    axes[0].set_ylabel("mass")
    fig.suptitle("Predicted travel distance (mm) across the physics grid",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    fig.savefig(f"{OUT}/fig3_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_params(res):
    labels = [t.replace("k", "k=") for t in ORDER] + ["oracle"]
    mass = [res[t]["mass_pearson"] for t in ORDER] + [res["oracle"]["mass_pearson"]]
    fric = [res[t]["friction_pearson"] for t in ORDER] + [res["oracle"]["friction_pearson"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 4))
    ax.bar(x - 0.19, mass, 0.36, label="mass  (shows up in early speed)", color="#3d5a80")
    ax.bar(x + 0.19, fric, 0.36, label="friction  (shows up in decay)", color="#ee6c4d")
    ax.axhline(0, color="#adb5bd", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("sensitivity agreement  r")
    ax.set_title("Which parameter does the model pick up, and when?",
                 loc="left", fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_params.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def fig_qualitative(dev="cuda"):
    ev = load_split("data/eval", dev)
    seeds = ev["seed"].cpu().numpy()
    s0 = np.unique(seeds)[3]
    cells = {"light + slippery": 0, "nominal": NOM_C, "heavy + rough": 24}
    tabs = np.load(f"{OUT}/tables.npz")
    M = tabs["M"]

    models = {}
    for t in ["k1", "k16", "oracle"]:
        ck = torch.load(f"ckpt/{t}.pt", map_location=dev, weights_only=False)
        n = WorldModel(ck["k"], cond_params=ck["oracle"], width=ck["width"]).to(dev)
        n.load_state_dict(ck["sd"]); n.eval()
        models[t] = (n, ck["k"], ck["oracle"])

    rows, labels = [], []
    pick = [0, 3, 5, 7]                        # which predicted frames to show
    for cname, c in cells.items():
        i = int(np.where((seeds == s0) &
                         (ev["pidx"].cpu().numpy()[:, 0] * 5 +
                          ev["pidx"].cpu().numpy()[:, 1] == c))[0][0])
        idx = torch.tensor([i], device=dev)
        ctx, tgt = get_batch(ev, idx, E_EVAL, 1, dev)
        rows.append(np.concatenate(
            [((tgt[0, p].permute(1, 2, 0).cpu().numpy() + 1) / 2) for p in pick], 1))
        labels.append(f"SIM  {cname}")
        for t, (n, k, orc) in models.items():
            c2, _ = get_batch(ev, idx, E_EVAL, k, dev)
            p = norm_params(ev["params"][idx]) if orc else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = n(c2, ev["actions"][idx], p).float()
            rows.append(np.concatenate(
                [((out[0, q].permute(1, 2, 0).cpu().numpy() + 1) / 2) for q in pick], 1))
            labels.append(f"WM {t}  {cname}")

    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 1.15 * len(rows)))
    for ax, r, l in zip(axes, rows, labels):
        ax.imshow(np.clip(r, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        ax.set_ylabel(l, rotation=0, ha="right", va="center", fontsize=8.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_title("Predicted frames  (t = +1, +4, +6, +8 after the context)",
                      fontsize=11, fontweight="bold", loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_rollouts.png", dpi=150,
                                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    res = json.load(open(f"{OUT}/results.json"))
    tab = np.load(f"{OUT}/tables.npz")
    fig_ablation(res); print("fig1")
    fig_scatter(tab, res); print("fig2")
    fig_heat(tab); print("fig3")
    fig_params(res); print("fig4")
    try:
        fig_qualitative(); print("fig5")
    except Exception as e:
        print("fig5 skipped:", e)
