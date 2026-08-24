import sys, json, glob, itertools
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
from scene import MASS_GRID, MU_GRID

C_P, C_N, C_R, C_A = "#1b4965", "#c1121f", "#f0a202", "#2a9d8f"
fig = plt.figure(figsize=(15.5, 9.6)); fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 3, hspace=0.44, wspace=0.34)

# --- 1 scene
ax = fig.add_subplot(gs[0, 0])
c = np.load("out/stage0_zoom/clips/ref.npz")["clip"]
ax.imshow(np.concatenate([c[0], c[3], c[7]], 1))
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
ax.set_title("final scene: arm in frame, camera closed in",
             loc="left", fontweight="bold", fontsize=11)

# --- 2 encoder landscape
ta = json.load(open("out/stage0_zoom/token_analysis.json"))
G = np.array(ta["summary"]["object_tokens"]["grid"])
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(G, cmap="viridis"); ax.grid(False)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels([f"{v:.2f}" for v in MU_GRID], fontsize=8)
ax.set_yticklabels([f"{v:.2f}" for v in MASS_GRID], fontsize=8)
ax.set_xlabel("friction"); ax.set_ylabel("mass (kg)")
ax.scatter([2], [2], s=210, facecolors="none", edgecolors=C_R, lw=2.5)
ax.set_title("encoder distance\n(compares images already rendered)",
             loc="left", fontweight="bold", fontsize=10.5)
plt.colorbar(im, ax=ax, fraction=0.046)

# --- 3 predictor landscape for the same true cell
d = json.load(open("out/stage0_zoom/ac_predictor_22.json"))
E = np.array(d["E"])
ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(E, cmap="viridis"); ax.grid(False)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels([f"{v:.2f}" for v in MU_GRID], fontsize=8)
ax.set_yticklabels([f"{v:.2f}" for v in MASS_GRID], fontsize=8)
ax.set_xlabel("friction"); ax.set_ylabel("mass (kg)")
ax.scatter([2], [2], s=210, facecolors="none", edgecolors=C_R, lw=2.5, label="truth")
am = d["argmin"]; ax.scatter([am[1]], [am[0]], marker="x", s=150, color=C_N, lw=3,
                             label="model's pick")
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.set_title("AC predictor energy\n(model predicts, then we search physics)",
             loc="left", fontweight="bold", fontsize=10.5)
plt.colorbar(im, ax=ax, fraction=0.046)

# --- 4 rank histogram
z = np.load("out/stage0_zoom/ac_summary.npz")
ranks = z["ranks"]
ax = fig.add_subplot(gs[1, 0])
ax.hist(ranks, bins=np.arange(-0.5, 25.5, 2), color=C_A, edgecolor="white")
ax.axvline(ranks.mean(), color=C_P, lw=2.5, label=f"mean {ranks.mean():.1f}")
ax.axvline(12, color=C_N, lw=2.5, ls="--", label="chance 12.0")
ax.set_xlabel("rank of the TRUE physics among 25 cells (0 = best)")
ax.set_ylabel("count")
ax.set_title("the predictor ranks the right physics\nbetter than chance — but rarely first",
             loc="left", fontweight="bold", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

# --- 5 signal / nuisance summary
ax = fig.add_subplot(gs[1, 1])
runs = [("stage0", "no arm\nwide"), ("stage0_franka", "arm\nwide"),
        ("stage0_zoom", "arm\nzoom")]
allv, nocamv = [], []
for t, _ in runs:
    tt = json.load(open(f"out/{t}/token_analysis.json"))
    g = np.array(tt["summary"]["object_tokens"]["grid"])
    off = [g[i, j] for i in range(5) for j in range(5) if (i, j) != (2, 2)]
    nu = {k: v["object_tokens"] for k, v in tt["nuisance"].items()}
    allv.append(min(off) / max(nu.values()))
    nocamv.append(min(off) / max(v for k, v in nu.items() if not k.startswith("camera")))
x = np.arange(3)
ax.bar(x - 0.2, allv, 0.38, color=C_N, label="all nuisances")
ax.bar(x + 0.2, nocamv, 0.38, color=C_P, label="camera fixed")
ax.axhline(1.0, color="#333", ls="--", lw=1.8)
ax.text(-0.45, 1.05, "usable above this line", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in runs], fontsize=9)
ax.set_ylabel("signal / nuisance")
ax.set_title("only with a fixed camera does the\nphysics signal clear the noise",
             loc="left", fontweight="bold", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.25, axis="y"); ax.spines[["top", "right"]].set_visible(False)

# --- 6 landscape similarity
L = z["landscapes"]
cs = [np.corrcoef(L[a].ravel(), L[b].ravel())[0, 1]
      for a, b in itertools.combinations(range(len(L)), 2)]
ax = fig.add_subplot(gs[1, 2])
ax.hist(cs, bins=22, color="#8d99ae", edgecolor="white")
ax.axvline(np.mean(cs), color=C_N, lw=2.5, label=f"mean {np.mean(cs):.2f}")
ax.set_xlabel("correlation between energy landscapes\nfrom different true physics")
ax.set_ylabel("count")
ax.set_title("its forecast is largely the same\nwhatever physics produced the context",
             loc="left", fontweight="bold", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Stage 0 complete — V-JEPA 2-AC as a source of physical-parameter labels",
             fontsize=14.5, fontweight="bold", y=0.985)
fig.savefig("out/stage0_zoom/final_result.png", dpi=145, bbox_inches="tight")
print("wrote out/stage0_zoom/final_result.png")
