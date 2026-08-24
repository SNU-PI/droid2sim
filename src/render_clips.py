"""Phase 1 only: render clips to disk. No torch, no CUDA.

Two hard-won constraints shape this file.

Long render loops die. A single process rendering the whole grid was killed
silently around the twentieth rollout, so each process renders only a handful
and exits; the driver loop re-runs until every clip exists.

Rebuilding the renderer inside a process yields BLANK frames. The obvious fix
for the above -- recycling the scene every N rollouts -- created a second EGL
context mid-process and most clips came back solid black. They were saved
without complaint and only showed up later as nonsense encoder distances. So:
exactly one scene per process, and every clip is validated before it is
written.
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "2")

from scene import PushScene, MASS_GRID, MU_GRID, NOMINAL_MASS, NOMINAL_MU
import mujoco

OUT = os.environ.get("CLIP_DIR", "out/stage0/clips")
FRANKA = os.environ.get("WITH_FRANKA", "0") == "1"
CAM = os.environ.get("CAM", "droidcam")
FRAME_IDX = [8, 12, 16, 20, 24, 28, 32, 36]

NUISANCES = [("camera", 0.005), ("camera", 0.010), ("camera", 0.020),
             ("light", 0.10), ("light", 0.25),
             ("cube_hue", 0.10), ("cube_hue", 0.25),
             ("floor_hue", 0.10), ("floor_hue", 0.25)]

# Every process re-renders the nominal episode first and compares it against
# the canonical reference clip. Renders coming out of a fresh EGL context are
# occasionally wrong in ways that are not blank -- a missing shadow shifts the
# whole image by ~15 grey levels, which is thirty times the physics signal and
# looks entirely plausible on inspection. Validating shape and brightness is
# not enough; the only reliable test is whether a known episode reproduces.
REF_TOL = 0.05


def apply_nuisance(sc, kind, amount):
    m = sc.model
    # must perturb the camera actually being rendered, not always the wide one
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAM)
    if kind == "camera":
        m.cam_pos[cam, 0] += amount
    elif kind == "light":
        m.light_diffuse[0] *= (1.0 + amount)
    elif kind == "cube_hue":
        mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "objmat")
        m.mat_rgba[mid, 0] = np.clip(m.mat_rgba[mid, 0] - amount, 0, 1)
        m.mat_rgba[mid, 1] = np.clip(m.mat_rgba[mid, 1] + amount, 0, 1)
    elif kind == "floor_hue":
        # the floor material is plain rgba; the table's is textured, and
        # tinting a textured material barely changes the render at all
        mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "floor")
        m.mat_rgba[mid, 2] = np.clip(m.mat_rgba[mid, 2] + amount, 0, 1)
        m.mat_rgba[mid, 0] = np.clip(m.mat_rgba[mid, 0] - amount * 0.5, 0, 1)
    else:
        raise ValueError(kind)


def render_twice(sc, mass, mu, name):
    """Render the same episode twice and require the two to agree.

    Corruption is not confined to the first render of a process: a context that
    validated cleanly at startup still produced wrong frames later on, off by
    enough to swamp the physics. Since the simulation is bitwise deterministic,
    two renders of one episode must be identical, and any disagreement is the
    renderer misbehaving. This is the only per-clip check that actually catches
    it.
    """
    st, fr1 = sc.rollout(mass, mu, render=True)
    _, fr2 = sc.rollout(mass, mu, render=True)
    c1, c2 = fr1[FRAME_IDX], fr2[FRAME_IDX]
    d = float(np.abs(c1.astype(np.int16) - c2.astype(np.int16)).mean())
    if d > 0.02:
        print(f"  UNSTABLE {name}: two renders differ by {d:.4f}", flush=True)
        return None, None
    if not validate(c1, name):
        return None, None
    return c1, st


def check_reference(sc):
    """Re-render the nominal episode and require it to match the stored one."""
    path = f"{OUT}/ref.npz"
    _, fr = sc.rollout(NOMINAL_MASS, NOMINAL_MU, render=True)
    clip = fr[FRAME_IDX]
    if not os.path.exists(path):
        return clip, True                    # first process defines the reference
    ref = np.load(path)["clip"]
    d = float(np.abs(clip.astype(np.int16) - ref.astype(np.int16)).mean())
    if d > REF_TOL:
        print(f"  REF MISMATCH {d:.4f} > {REF_TOL} - discarding this process",
              flush=True)
        return clip, False
    return clip, True


def validate(clip, name):
    """Reject blank or frozen renders instead of saving them silently."""
    m, s = float(clip.mean()), float(clip.std())
    inter = float(np.abs(np.diff(clip.astype(np.int16), axis=0)).mean())
    ok = (m > 25.0) and (s > 15.0) and (inter > 0.02)
    if not ok:
        print(f"  REJECT {name}: mean={m:.2f} std={s:.2f} interframe={inter:.4f}",
              flush=True)
    return ok


def grid_jobs():
    js = [("ref", NOMINAL_MASS, NOMINAL_MU)]
    for i, m in enumerate(MASS_GRID):
        for j, mu in enumerate(MU_GRID):
            js.append((f"grid_{i}_{j}", m, mu))
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--nuis", default=None, help="index into the nuisance list")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.nuis is not None:
        kind, amt = NUISANCES[int(a.nuis)]
        name = f"nuis_{kind}_{amt}"
        if os.path.exists(f"{OUT}/{name}.npz"):
            print("ALREADY PRESENT"); return 0
        sc = PushScene(franka=FRANKA, cam=CAM)
        _, ok = check_reference(sc)          # validate BEFORE perturbing
        if not ok:
            return 2
        apply_nuisance(sc, kind, amt)
        clip, st = render_twice(sc, NOMINAL_MASS, NOMINAL_MU, name)
        if clip is None:
            return 2
        np.savez_compressed(f"{OUT}/{name}.npz", clip=clip,
                            travel=float(np.linalg.norm(st[-1, :2] - st[0, :2]) * 100),
                            mass=NOMINAL_MASS, mu=NOMINAL_MU)
        print(f"  {name:24s} ok", flush=True)
        return 0

    todo = [j for j in grid_jobs() if not os.path.exists(f"{OUT}/{j[0]}.npz")]
    if not todo:
        print("ALL GRID CLIPS PRESENT"); return 0
    todo = todo[:a.limit]
    sc = PushScene(franka=FRANKA, cam=CAM)                      # exactly one scene for this process
    ref_clip, ok = check_reference(sc)
    if not ok:
        return 2
    if not os.path.exists(f"{OUT}/ref.npz"):
        np.savez_compressed(f"{OUT}/ref.npz", clip=ref_clip, travel=0.0,
                            mass=NOMINAL_MASS, mu=NOMINAL_MU)
        print("  wrote canonical ref", flush=True)
        todo = [t for t in todo if t[0] != "ref"]
    for name, mass, mu in todo:
        clip, st = render_twice(sc, mass, mu, name)
        if clip is None:
            return 2
        travel = float(np.linalg.norm(st[-1, :2] - st[0, :2]) * 100)
        np.savez_compressed(f"{OUT}/{name}.npz", clip=clip, travel=travel,
                            mass=mass, mu=mu)
        print(f"  {name:24s} travel={travel:5.1f}cm", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
