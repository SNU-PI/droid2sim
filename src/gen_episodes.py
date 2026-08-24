"""Generate episodes with continuously sampled physics, for probe training.

Grid sweeps answered "can a hand-written distance find the right cell". The
question now is different: is the information about the physical parameters
present in the representation at all, in a form anything could read? That needs
many episodes with continuous parameters, not 25 cells.

The test split is also rendered under nuisance perturbations. A probe that
scores well on clean frames but collapses when the camera shifts 1 cm is
reading the scene, not the physics, and would be useless as a labeller.

Rendering integrity is checked at both ends of every batch. Mid-process
corruption is real here -- see render_clips.py -- and a batch whose closing
reference no longer reproduces is discarded whole.
"""
import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "2")

import mujoco
from scenes import Scene, FAMILIES, PARAMS, FRAME_IDX, sample_params

ROOT = os.environ.get("EP_DIR", "/data/pgc/simdroid/episodes")
REF_TOL = 0.05
NUIS = {"clean": None, "camera": ("camera", 0.010), "light": ("light", 0.20)}


def ref_params(family):
    """A fixed mid-range setting used only as a rendering integrity probe."""
    return {n: float(np.exp(0.5 * (np.log(lo) + np.log(hi))))
            for n, lo, hi in PARAMS[family]}


def apply_nuisance(sc, kind, amt):
    m = sc.model
    if kind == "camera":
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, sc.cam)
        m.cam_pos[cid, 0] += amt
    elif kind == "light":
        m.light_diffuse[0] *= (1.0 + amt)


def render_ref(sc, family):
    _, fr = sc.rollout(ref_params(family), render=True, frames_at=FRAME_IDX[family])
    return fr


def summarize(st, family):
    d = np.linalg.norm(st[-1, :2] - st[0, :2])
    return dict(travel=float(d), vmax=float(st[:, 6].max()),
                vend=float(st[-1, 6]), zmax=float(st[:, 2].max()),
                zend=float(st[-1, 2]), aux=float(st[-1, 7]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--variant", default="clean")
    ap.add_argument("--total", type=int, default=250)
    ap.add_argument("--batch", type=int, default=5)
    a = ap.parse_args()

    out = f"{ROOT}/{a.family}/{a.split}_{a.variant}"
    os.makedirs(out, exist_ok=True)
    base = 0 if a.split == "train" else 10000
    todo = [i for i in range(a.total) if not os.path.exists(f"{out}/{i:05d}.npz")]
    if not todo:
        print("ALL PRESENT"); return 0
    todo = todo[:a.batch]

    sc = Scene(a.family, cam="close")
    r0 = render_ref(sc, a.family)
    canon = f"{ROOT}/{a.family}/_ref.npy"
    if os.path.exists(canon):
        d = float(np.abs(r0.astype(np.int16) - np.load(canon).astype(np.int16)).mean())
        if d > REF_TOL:
            print(f"  REF MISMATCH {d:.4f}; dropping batch"); return 2
    else:
        np.save(canon, r0)

    nu = NUIS[a.variant]
    if nu:
        apply_nuisance(sc, *nu)

    staged = []
    for i in todo:
        rng = np.random.default_rng(base + i)
        p = sample_params(a.family, rng)
        st, fr = sc.rollout(p, render=True, frames_at=FRAME_IDX[a.family])
        if fr is None or fr.mean() < 25 or fr.std() < 15:
            print(f"  REJECT {i}"); return 2
        staged.append((i, fr, p, summarize(st, a.family)))

    if not nu:                      # closing integrity check on the clean camera
        r1 = render_ref(sc, a.family)
        d = float(np.abs(r1.astype(np.int16) - r0.astype(np.int16)).mean())
        if d > REF_TOL:
            print(f"  DRIFT {d:.4f} during batch; dropping {len(staged)} clips")
            return 2

    for i, fr, p, s in staged:
        np.savez_compressed(f"{out}/{i:05d}.npz", clip=fr,
                            params=np.array([p[n] for n, _, _ in PARAMS[a.family]]),
                            **s)
    print(f"  {a.family}/{a.split}_{a.variant}: +{len(staged)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
