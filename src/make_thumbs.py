"""Still-image and GIF versions of the two episodes, for viewers that will not
play mp4. Frames are annotated so the difference is readable without motion."""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "2")

import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from scene import (PushScene, MASS_GRID, MU_GRID, NOMINAL_MASS, NOMINAL_MU,
                   CTRL_DT, N_FRAMES)

OUT = "out/stage0"
PICK = [0, 10, 14, 18, 24, 32, 39]

RUNS = [("normal",   NOMINAL_MASS,  NOMINAL_MU,  "#2d6a9f"),
        ("abnormal", MASS_GRID[-1], MU_GRID[-1], "#c1121f")]


def main():
    os.makedirs(OUT, exist_ok=True)
    sc = PushScene()
    data = []
    for name, m, mu, col in RUNS:
        st, fr, tr = sc.rollout(m, mu, render=True, fine=True)
        data.append(dict(name=name, m=m, mu=mu, col=col, st=st, fr=fr, tr=tr))
        iio.mimwrite(f"{OUT}/{name}.gif", fr, duration=0.05, loop=0)
        print(f"   {OUT}/{name}.gif")

    # side-by-side gif
    pad = np.full((N_FRAMES, data[0]["fr"].shape[1], 6, 3), 25, np.uint8)
    iio.mimwrite(f"{OUT}/compare.gif",
                 np.concatenate([data[0]["fr"], pad, data[1]["fr"]], axis=2),
                 duration=0.05, loop=0)
    print(f"   {OUT}/compare.gif")

    # annotated contact sheet
    nr, nc = 2, len(PICK)
    fig, axes = plt.subplots(nr, nc, figsize=(2.05 * nc, 2.35 * nr))
    fig.patch.set_facecolor("white")
    for r, d in enumerate(data):
        y0 = d["st"][0, 1]
        for c, fi in enumerate(PICK):
            ax = axes[r, c]
            ax.imshow(d["fr"][fi]); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(d["col"]); s.set_linewidth(2)
            trav = (d["st"][fi, 1] - y0) * 100
            if r == 0:
                ax.set_title(f"t = {fi*CTRL_DT:.2f} s", fontsize=10, pad=5)
            ax.set_xlabel(f"{trav:+.1f} cm", fontsize=9.5, color=d["col"],
                          labelpad=2)
        axes[r, 0].set_ylabel(
            f"{d['name'].upper()}\nmass {d['m']:.3f} kg\n$\\mu$ {d['mu']:.3f}"
            .replace("$\\mu$", "mu"),
            rotation=0, ha="right", va="center", fontsize=10.5,
            color=d["col"], labelpad=14)
    fig.suptitle("Same pusher, same start, same camera — only mass and friction differ",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005, "numbers under each frame: distance the cube has travelled",
             ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.955])
    fig.savefig(f"{OUT}/thumbnails.png", dpi=140)
    print(f"   {OUT}/thumbnails.png")

    # single hero frame comparison at the moment of largest divergence
    fi = 39
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.6))
    for k, d in enumerate(data):
        ax[k].imshow(d["fr"][fi]); ax[k].set_xticks([]); ax[k].set_yticks([])
        trav = (d["st"][fi, 1] - d["st"][0, 1]) * 100
        ax[k].set_title(f"{d['name'].upper()}   mass {d['m']:.3f} kg, mu {d['mu']:.3f}\n"
                        f"travelled {trav:.1f} cm",
                        fontsize=11.5, color=d["col"], fontweight="bold")
        for s in ax[k].spines.values():
            s.set_color(d["col"]); s.set_linewidth(2.5)
    fig.suptitle(f"final frame  (t = {fi*CTRL_DT:.2f} s)", fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(f"{OUT}/final_frame.png", dpi=150)
    print(f"   {OUT}/final_frame.png")


if __name__ == "__main__":
    main()
