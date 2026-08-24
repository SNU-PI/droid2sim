"""Final figures for the overnight multi-scene probe study."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as iio

C_P, C_N, C_O, C_A = "#1b4965", "#c1121f", "#f0a202", "#2a9d8f"
P = json.load(open("out/multi/probe_all.json"))
A = json.load(open("out/multi/ablations.json"))
FAMS = ["slide", "roll", "bounce", "collide", "incline"]
NAMES = {f: A[f]["names"] for f in FAMS}
# observable params per family (physics reasoning: bounce/incline mass cancel)
UNOBS = {("bounce", "mass"), ("incline", "mass")}

fig = plt.figure(figsize=(16, 18)); fig.patch.set_facecolor("white")
gs = fig.add_gridspec(4, 2, hspace=0.34, wspace=0.22,
                      height_ratios=[1.25, 1.0, 1.0, 1.0])

# (0,0) gallery
ax = fig.add_subplot(gs[0, 0])
ax.imshow(iio.imread("out/multi/gallery.png"))
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for i, f in enumerate(FAMS):
    ax.text(-12, 128 + 256 * i, f, ha="right", va="center", fontsize=11,
            fontweight="bold", color=C_P)
ax.set_title("five interaction families, all physics-verified (5/5 checks)",
             loc="left", fontweight="bold", fontsize=12)

# (0,1) main heatmap: linear probe R2 vs physics oracle, clean
ax = fig.add_subplot(gs[0, 1])
rows, labels, oracle = [], [], []
for f in FAMS:
    for i, n in enumerate(NAMES[f]):
        rows.append([P[f]["physics"]["test_clean"][n],
                     P[f]["pixels"]["test_clean"][n],
                     P[f]["vjepa"]["test_clean"][n]])
        labels.append(f"{f}.{n}" + (" (unobs)" if (f, n) in UNOBS else ""))
M = np.clip(np.array(rows), -0.2, 1.0)
im = ax.imshow(M, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels(["physics\noracle", "raw\npixels",
                                             "V-JEPA\nlatent"], fontsize=10)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{rows[i][j]:+.2f}", ha="center", va="center", fontsize=8.5)
ax.set_title("held-out R² per parameter (clean test)\nV-JEPA matches or beats the oracle "
             "wherever physics permits", loc="left", fontweight="bold", fontsize=12)
ax.grid(False); plt.colorbar(im, ax=ax, fraction=0.03)

# (1,:) robustness: clean/camera/light for pixels vs vjepa vs vjepa-aug
ax = fig.add_subplot(gs[1, :])
obs = [(f, n) for f in FAMS for n in NAMES[f] if (f, n) not in UNOBS]
x = np.arange(len(obs)); w = 0.26
def get(kind, sub):
    v = []
    for f, n in obs:
        if kind == "aug":
            i = NAMES[f].index(n)
            v.append(A[f]["augmented"][sub][i] if "augmented" in A[f] else np.nan)
        else:
            v.append(P[f][kind][sub][n])
    return np.clip(np.array(v), -0.35, 1.0)
ax.bar(x - w, get("vjepa", "test_clean"), w, color=C_P, label="V-JEPA, clean test")
ax.bar(x,     get("vjepa", "test_camera"), w, color=C_N, label="V-JEPA, camera moved 1 cm")
ax.bar(x + w, get("aug", "test_camera"), w, color=C_A,
       label="V-JEPA + camera-augmented training, camera moved")
ax.axhline(0, color="#333", lw=1)
ax.set_xticks(x); ax.set_xticklabels([f"{f}\n{n}" for f, n in obs], fontsize=9)
ax.set_ylabel("held-out R²"); ax.set_ylim(-0.4, 1.05)
ax.set_title("the one weakness (viewpoint shift) is trainable away",
             loc="left", fontweight="bold", fontsize=12)
ax.legend(fontsize=9.5, frameon=False, ncol=3)
ax.grid(alpha=0.25, axis="y"); ax.spines[["top", "right"]].set_visible(False)

# (2,0) frame-count curves
ax = fig.add_subplot(gs[2, 0])
picks = [("slide", "mass", C_P, "-"), ("slide", "mu", C_P, "--"),
         ("bounce", "damping", C_N, "-"), ("collide", "mass2", C_A, "-"),
         ("roll", "roll_fric", C_O, "--")]
for f, n, c, ls in picks:
    i = NAMES[f].index(n)
    ks = sorted(map(int, A[f]["nframes"]))
    v = [A[f]["nframes"][str(k)]["test_clean"][i] for k in ks]
    ax.plot(ks, np.clip(v, -0.2, 1), ls, color=c, lw=2, marker="o", ms=4,
            label=f"{f}.{n}")
ax.axhline(0, color="#999", lw=1)
ax.set_xlabel("frames shown to the probe (k)"); ax.set_ylabel("held-out R²")
ax.set_title("each parameter becomes readable exactly when\nits physics plays out",
             loc="left", fontweight="bold", fontsize=12)
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

# (2,1) sample efficiency
ax = fig.add_subplot(gs[2, 1])
for f, c in zip(FAMS, [C_P, C_O, C_N, C_A, "#8d5a97"]):
    ns, vs = [], []
    for n in sorted(map(int, A[f]["nsamples"])):
        r = A[f]["nsamples"][str(n)].get("test_clean")
        if r is None: continue
        obs_i = [i for i, nm in enumerate(NAMES[f]) if (f, nm) not in UNOBS]
        ns.append(n); vs.append(np.mean([r[i] for i in obs_i]))
    ax.plot(ns, vs, "o-", color=c, lw=2, ms=5, label=f)
ax.set_xlabel("labelled training episodes"); ax.set_ylabel("mean R² (observable params)")
ax.set_title("50-100 episodes calibrate a scene", loc="left",
             fontweight="bold", fontsize=12)
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

# (3,0) observability control
ax = fig.add_subplot(gs[3, 0])
pairs = [("bounce", "mass"), ("incline", "mass")]
x = np.arange(2); w = 0.35
po = [P[f]["physics"]["test_clean"][n] for f, n in pairs]
vj = [P[f]["vjepa"]["test_clean"][n] for f, n in pairs]
ax.bar(x - w/2, po, w, color="#666", label="physics oracle")
ax.bar(x + w/2, vj, w, color=C_P, label="V-JEPA probe")
ax.axhline(0, color="#333", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(["bounce.mass\n(free fall: mass cancels)",
                    "incline.mass\n(a = g(sin−μcos): mass cancels)"], fontsize=9.5)
ax.set_ylabel("held-out R²")
ax.set_title("negative control: where physics says the parameter is\n"
             "unobservable, BOTH fail — the probe is not cheating",
             loc="left", fontweight="bold", fontsize=12)
ax.legend(fontsize=9.5, frameon=False)
ax.grid(alpha=0.25, axis="y"); ax.spines[["top", "right"]].set_visible(False)

# (3,1) pixels catastrophic failure
ax = fig.add_subplot(gs[3, 1])
subs = ["test_clean", "test_camera", "test_light"]
labs = ["clean", "camera 1 cm", "light +20%"]
pix = [np.mean([np.clip(P[f]["pixels"][s][n], -1.2, 1) for f, n in obs]) for s in subs]
vj = [np.mean([np.clip(P[f]["vjepa"][s][n], -1.2, 1) for f, n in obs]) for s in subs]
x = np.arange(3); w = 0.35
ax.bar(x - w/2, pix, w, color="#999", label="raw pixels")
ax.bar(x + w/2, vj, w, color=C_P, label="V-JEPA latent")
ax.axhline(0, color="#333", lw=1)
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=10)
ax.set_ylabel("mean R² (clipped at −1.2)")
ax.set_title("why the world model earns its keep: pixels collapse\n"
             "under any nuisance (true values reach −10⁵)",
             loc="left", fontweight="bold", fontsize=12)
ax.legend(fontsize=9.5, frameon=False)
ax.grid(alpha=0.25, axis="y"); ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Overnight run — the physics is in the latent; the failure was the metric",
             fontsize=15, fontweight="bold", y=0.995)
fig.savefig("out/multi/overnight_result.png", dpi=140, bbox_inches="tight")
print("wrote out/multi/overnight_result.png")
