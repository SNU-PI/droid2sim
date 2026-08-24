"""Render the normal and abnormal episodes and the physics behind them.

Exactly two things differ between the two runs, and nothing else: the object's
mass and its sliding friction against the table. Same pusher, same rail speed,
same starting pose, same camera, same lighting, same seed.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "2")

import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scene import (PushScene, MASS_GRID, MU_GRID, NOMINAL_MASS, NOMINAL_MU,
                   CTRL_DT, N_FRAMES, PUSHER_MASS, PUSHER_V0)

OUT = "out/stage0"
FPS = 25          # 20 ms frames -> 50 Hz; play at 25 fps = half speed

NORMAL = dict(mass=NOMINAL_MASS, mu=NOMINAL_MU, name="normal")
ABNORMAL = dict(mass=MASS_GRID[-1], mu=MU_GRID[-1], name="abnormal")


def label(frames, lines, colour=(255, 255, 255)):
    """Burn a few lines of text into the top-left of every frame."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frames
    out = []
    for f in frames:
        im = Image.fromarray(f)
        d = ImageDraw.Draw(im)
        for i, t in enumerate(lines):
            d.text((6, 5 + 12 * i), t, fill=colour)
        out.append(np.asarray(im))
    return np.stack(out)


def write(path, frames, fps=FPS):
    try:
        iio.mimwrite(path, frames, fps=fps, quality=9,
                     macro_block_size=1)
        return path
    except Exception as e:
        alt = path.rsplit(".", 1)[0] + ".gif"
        iio.mimwrite(alt, frames, duration=1.0 / fps)
        print(f"   (mp4 failed: {type(e).__name__}; wrote {alt})")
        return alt


def main():
    os.makedirs(OUT, exist_ok=True)
    sc = PushScene()
    runs = {}

    print("rendering; only mass and friction differ between the two runs")
    for cfg in (NORMAL, ABNORMAL):
        st, fr, tr = sc.rollout(cfg["mass"], cfg["mu"], render=True, fine=True)
        runs[cfg["name"]] = dict(st=st, fr=fr, tr=tr, **cfg)
        travel = float(np.linalg.norm(st[-1, :2] - st[0, :2]))
        print(f"   {cfg['name']:9s} mass={cfg['mass']:.3f} kg  mu={cfg['mu']:.3f}"
              f"   peak {tr[:,0].max():.3f} m/s   travel {travel*100:.1f} cm")

        tagged = label(fr, [f"{cfg['name'].upper()}",
                            f"mass  {cfg['mass']:.3f} kg",
                            f"mu    {cfg['mu']:.3f}"])
        p = write(f"{OUT}/{cfg['name']}.mp4", tagged)
        print(f"   -> {p}")

    # side-by-side
    a, b = runs["normal"], runs["abnormal"]
    pad = np.full((N_FRAMES, a["fr"].shape[1], 4, 3), 30, np.uint8)
    sbs = np.concatenate([
        label(a["fr"], ["NORMAL", f"mass  {a['mass']:.3f} kg", f"mu    {a['mu']:.3f}"]),
        pad,
        label(b["fr"], ["ABNORMAL", f"mass  {b['mass']:.3f} kg", f"mu    {b['mu']:.3f}"],
              colour=(255, 210, 120)),
    ], axis=2)
    p = write(f"{OUT}/compare.mp4", sbs)
    print(f"   -> {p}")

    # frame strip
    pick = [0, 8, 12, 16, 22, 30, 39]
    strip = np.concatenate([
        np.concatenate([a["fr"][i] for i in pick], 1),
        np.concatenate([b["fr"][i] for i in pick], 1)], 0)
    iio.imwrite(f"{OUT}/strip.png", strip)
    print(f"   -> {OUT}/strip.png")

    # the physics behind the two videos
    dt = sc.model.opt.timestep
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for r, c in ((a, "#1b4965"), (b, "#c1121f")):
        t = np.arange(len(r["tr"])) * dt
        ax[0].plot(t, r["tr"][:, 0], color=c, lw=2,
                   label=f"{r['name']}: m={r['mass']:.3f} kg, mu={r['mu']:.3f}")
        ax[0].plot(t, r["tr"][:, 1], color=c, lw=1, ls=":", alpha=0.6)
        tf = np.arange(N_FRAMES) * CTRL_DT
        ax[1].plot(tf, (r["st"][:, 1] - r["st"][0, 1]) * 100, color=c, lw=2,
                   label=r["name"])
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("speed (m/s)")
    ax[0].set_title("object speed (solid) and pusher (dotted)", loc="left",
                    fontsize=11, fontweight="bold")
    ax[0].legend(fontsize=8.5, frameon=False)
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("travel (cm)")
    ax[1].set_title("distance travelled", loc="left", fontsize=11, fontweight="bold")
    ax[1].legend(fontsize=9, frameon=False)
    for x in ax:
        x.grid(alpha=0.25); x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/physics.png", dpi=150)
    print(f"   -> {OUT}/physics.png")

    print(f"\nwhat changed: mass {a['mass']:.3f} -> {b['mass']:.3f} kg "
          f"({b['mass']/a['mass']:.2f}x),  mu {a['mu']:.3f} -> {b['mu']:.3f} "
          f"({b['mu']/a['mu']:.2f}x)")
    print(f"what did NOT change: pusher {PUSHER_MASS} kg at {PUSHER_V0} m/s, "
          f"start pose, camera, lighting, solver settings")


if __name__ == "__main__":
    main()
