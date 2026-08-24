import sys, os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
from scene import MASS_GRID, MU_GRID, NOMINAL_CELL
import imageio.v2 as iio

C_P, C_N, C_R = "#1b4965", "#c1121f", "#f0a202"
ni, nj = NOMINAL_CELL
fig = plt.figure(figsize=(15, 9)); fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.32, height_ratios=[1, 1])

D = {}
for tag in ("stage0", "stage0_franka"):
    es = json.load(open(f"out/{tag}/energy_summary.json"))
    ta = json.load(open(f"out/{tag}/token_analysis.json"))
    D[tag] = dict(E=np.array(ta["summary"]["object_tokens"]["grid"]),
                  pix=np.array(es["pixdiff"]),
                  npix=es["nuisance_pixdiff"],
                  nE={k: v["object_tokens"] for k, v in ta["nuisance"].items()},
                  s=ta["summary"])

# scene look
ax = fig.add_subplot(gs[0, 0])
img = iio.imread("out/stage0_franka/clips/ref.npz.png") if os.path.exists(
    "out/stage0_franka/clips/ref.npz.png") else None
c = np.load("out/stage0_franka/clips/ref.npz")["clip"]
ax.imshow(np.concatenate([c[0], c[4]], 1)); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("the scene with the arm added\n(static geoms, collisions off)",
             loc="left", fontweight="bold", fontsize=11); ax.grid(False)

for k, (tag, lab) in enumerate((("stage0", "no arm"), ("stage0_franka", "with arm"))):
    ax = fig.add_subplot(gs[0, k + 1])
    G = D[tag]["E"]
    im = ax.imshow(G, cmap="viridis", origin="upper")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f"{v:.2f}" for v in MU_GRID], fontsize=8)
    ax.set_yticklabels([f"{v:.2f}" for v in MASS_GRID], fontsize=8)
    ax.set_xlabel("friction"); ax.set_ylabel("mass (kg)")
    ax.scatter([nj], [ni], s=200, facecolors="none", edgecolors=C_R, lw=2.5)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{G[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if G[i, j] < G.max()*0.65 else "black")
    ax.set_title(f"energy landscape — {lab}", loc="left", fontweight="bold",
                 fontsize=11); ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046)

# THE panel: matched magnitude
ax = fig.add_subplot(gs[1, 0:2])
for tag, mk, lab in (("stage0", "o", "no arm"), ("stage0_franka", "s", "with arm")):
    d = D[tag]
    pix, E = d["pix"].ravel(), d["E"].ravel()
    keep = pix > 1e-6
    ax.scatter(pix[keep], E[keep], marker=mk, s=52, color=C_P, alpha=0.75,
               label=f"physics changes ({lab})",
               facecolors="none" if tag == "stage0_franka" else C_P, linewidths=1.6)
    nx = [d["npix"][k] for k in d["nE"]]; ny = [d["nE"][k] for k in d["nE"]]
    ax.scatter(nx, ny, marker=mk, s=70, color=C_N, alpha=0.85,
               label=f"irrelevant changes ({lab})",
               facecolors="none" if tag == "stage0_franka" else C_N, linewidths=1.6)
ax.set_xscale("log")
ax.set_xlabel("how much the image changed (mean grey levels)")
ax.set_ylabel("latent distance")
ax.set_title("At the same amount of image change, a physics change and an "
             "irrelevant change\nmove the representation by the same amount",
             loc="left", fontweight="bold", fontsize=12)
ax.legend(fontsize=8.5, frameon=False, ncol=2, loc="upper left")
ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

# S/N bars
ax = fig.add_subplot(gs[1, 2])
ks = ["mean_all", "top2pct", "max_token", "object_tokens"]
x = np.arange(len(ks))
ax.bar(x - 0.2, [D["stage0"]["s"][k]["signal_over_nuisance"] for k in ks], 0.38,
       color=C_P, label="no arm")
ax.bar(x + 0.2, [D["stage0_franka"]["s"][k]["signal_over_nuisance"] for k in ks],
       0.38, color="#7fb3d5", label="with arm")
ax.axhline(1.0, color="#333", ls="--", lw=1.6)
ax.text(-0.4, 1.04, "needs to be well above 1", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(ks, rotation=20, fontsize=8.5)
ax.set_ylabel("signal / nuisance"); ax.set_ylim(0, 1.25)
ax.set_title("adding the arm did not help", loc="left", fontweight="bold",
             fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.25, axis="y"); ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Stage 0 continued — ruling out the DROID domain gap",
             fontsize=14, fontweight="bold", y=0.98)
fig.savefig("out/stage0_franka/franka_result.png", dpi=145, bbox_inches="tight")
print("wrote out/stage0_franka/franka_result.png")
