"""Where does the sim-to-real gap live in feature space?

The real DROID clip sits far outside the sim feature cloud. Before deciding
what to do about that, decompose the distance: is it the appearance channel
(time-mean features) or the motion channel (frame differences)? Is it already
present in a single static frame? Does compositing the sim object onto the
real background close it?

Finding (2026-08-19): appearance is ~19x out of distribution, motion is inside
it. That is what motivates the motion-only probe in motion_probe.py.
"""

import sys, os, json, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imageio.v2 as iio

from core.encoder import build_encoder, encode_frames, to256
from core.probe import GaussianCloud
from core.paths import FEAT_DIR, EP_DIR, DROID_EPS

FEAT = str(FEAT_DIR)
FAMS = ["slide", "roll", "bounce", "collide", "incline"]
OUT = "out/droid"


def fit_cloud(X, npca=96):
    return GaussianCloud(npca).fit(X)


def main():
    os.makedirs(OUT, exist_ok=True)
    enc, tf = build_encoder()
    d = np.load(f"{DROID_EPS}/ep88.npz", allow_pickle=True)
    fr = d["frames"]
    idx = list(range(20, 35, 2))
    real = to256(fr[idx])
    f_real = encode_frames(enc, tf, real)

    sim = np.load(f"{FEAT}/slide__train_clean.npz")["mean"].astype(np.float64)
    R = {}

    print("=== 1. distance of the real clip to each sim cloud ===")
    R["by_family"] = {}
    for fam in FAMS:
        X = np.load(f"{FEAT}/{fam}__train_clean.npz")["mean"].astype(np.float64)
        c = fit_cloud(X.reshape(len(X), -1))
        m = float(c.maha(f_real.reshape(1, -1))[0])
        R["by_family"][fam] = (m, c.ref_p95)
        print(f"  {fam:8s} maha {m:6.1f}   (sim p95 {c.ref_p95:5.1f})   "
              f"ratio {m/c.ref_p95:4.1f}x")
    Xa = np.concatenate([np.load(f"{FEAT}/slide__{s}.npz")["mean"].astype(np.float64)
                         for s in ("train_clean", "train_camera")])
    c = fit_cloud(Xa.reshape(len(Xa), -1))
    m = float(c.maha(f_real.reshape(1, -1))[0])
    R["slide_aug"] = (m, c.ref_p95)
    print(f"  slide + camera-aug         maha {m:6.1f}   (p95 {c.ref_p95:5.1f})")

    print("\n=== 2. which channel is out of distribution? ===")
    app = fit_cloud(sim.mean(1), npca=64)
    mot = fit_cloud(np.diff(sim, axis=1).reshape(len(sim), -1), npca=64)
    ma = float(app.maha(f_real.mean(0)[None])[0])
    mm = float(mot.maha(np.diff(f_real, axis=0).reshape(1, -1))[0])
    print(f"  appearance   maha {ma:6.1f}  (p95 {app.ref_p95:5.1f})  ratio {ma/app.ref_p95:4.1f}x")
    print(f"  motion       maha {mm:6.1f}  (p95 {mot.ref_p95:5.1f})  ratio {mm/mot.ref_p95:4.1f}x")
    R["channels"] = {"appearance": (ma, app.ref_p95), "motion": (mm, mot.ref_p95)}

    print("\n=== 3. a single frame repeated 8x (zero motion) ===")
    full = fit_cloud(sim.reshape(len(sim), -1))
    f_still = encode_frames(enc, tf, np.repeat(real[:1], 8, axis=0))
    simclip = np.load(sorted(glob.glob(f"{EP_DIR}/slide/train_clean/*.npz"))[3])["clip"]
    f_simstill = encode_frames(enc, tf, np.repeat(simclip[:1], 8, axis=0))
    m_r = float(full.maha(f_still.reshape(1, -1))[0])
    m_s = float(full.maha(f_simstill.reshape(1, -1))[0])
    print(f"  real frame x8  maha {m_r:6.1f}  |  sim frame x8  maha {m_s:6.1f}  "
          f"(p95 {full.ref_p95:5.1f})")
    R["static"] = {"real": m_r, "sim": m_s, "p95": full.ref_p95}

    print("\n=== 4. sim object composited onto the real background ===")
    sc = simclip.astype(np.int16)
    mask = (sc[..., 0] - sc[..., 1] > 60) & (sc[..., 0] - sc[..., 2] > 60)
    comp = np.repeat(real[:1], 8, axis=0).astype(np.int16)
    comp[mask] = sc[mask]
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    m_c = float(full.maha(encode_frames(enc, tf, comp).reshape(1, -1))[0])
    m_sim = float(full.maha(encode_frames(enc, tf, simclip).reshape(1, -1))[0])
    m_real = float(full.maha(f_real.reshape(1, -1))[0])
    print(f"  composite maha {m_c:6.1f}   (pure sim {m_sim:5.1f}, real {m_real:5.1f})")
    R["composite"] = m_c
    iio.imwrite(f"{OUT}/composite_check.png",
                np.concatenate([real[3], simclip[3], comp[3]], 1))

    print("\n=== 5. how much of the gap is outside the workspace? ===")
    for frac in (0.6, 0.4):
        h = int(256 * frac)
        o = (256 - h) // 2
        masked = np.full_like(real, 110)
        masked[:, o:o+h, o:o+h] = real[:, o:o+h, o:o+h]
        m = float(full.maha(encode_frames(enc, tf, masked).reshape(1, -1))[0])
        print(f"  centre {int(frac*100)}% kept   maha {m:6.1f}")
        R[f"mask_{frac}"] = m

    json.dump(R, open(f"{OUT}/gap_analysis.json", "w"), indent=2)
    print(f"\nwrote {OUT}/gap_analysis.json")


if __name__ == "__main__":
    main()
