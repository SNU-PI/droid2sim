"""Generate the paired dataset.

Every episode seed is replayed under all 25 (mass, friction) cells with an
identical initial condition and identical action. That pairing is what makes
the difference-of-differences metric possible: for a fixed seed, the only thing
that changes between two cells is the physics.

Physics comes from MuJoCo; frames come from our own top-down rasteriser
(see raster.py for why).
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from sim import PushEnv, sample_episode_params, MASSES, FRICTIONS, N_FRAMES
import raster

NCELL = len(MASSES) * len(FRICTIONS)


def rollout_states(seeds):
    """MuJoCo physics only -- no rendering, so this is fast."""
    env = PushEnv()
    n = len(seeds) * NCELL
    states = np.zeros((n, N_FRAMES, 5), np.float32)
    actions = np.zeros((n, 2), np.float32)
    params = np.zeros((n, 2), np.float32)
    pidx = np.zeros((n, 2), np.int8)
    epseed = np.zeros(n, np.int32)
    c = 0
    for si, sd in enumerate(seeds):
        rng = np.random.default_rng(sd)
        ic, pd, imp = sample_episode_params(rng)
        for i, m in enumerate(MASSES):
            for j, mu in enumerate(FRICTIONS):
                _, st, ac = env.rollout(m, mu, ic, pd, imp, render=False)
                states[c], actions[c] = st, ac
                params[c], pidx[c], epseed[c] = (m, mu), (i, j), sd
                c += 1
        if (si + 1) % 100 == 0:
            print(f"  physics {si+1}/{len(seeds)} seeds", flush=True)
    return states, actions, params, pidx, epseed


def generate(seeds, out_prefix, device):
    states, actions, params, pidx, epseed = rollout_states(seeds)
    n = states.shape[0]
    frames = np.lib.format.open_memmap(
        f"{out_prefix}_frames.npy", mode="w+", dtype=np.uint8,
        shape=(n, N_FRAMES, raster.IMG, raster.IMG, 3))

    step = 64
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        img = raster.render(states[lo:hi, :, :3], device=device)
        frames[lo:hi] = img.cpu().numpy()
        if (lo // step) % 40 == 0:
            print(f"  raster {hi}/{n}", flush=True)
    frames.flush()

    np.savez(f"{out_prefix}_meta.npz", states=states, actions=actions,
             params=params, pidx=pidx, seed=epseed,
             masses=MASSES, frictions=FRICTIONS)
    print(f"wrote {out_prefix}_frames.npy  {frames.shape}  "
          f"{frames.nbytes/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-seeds", type=int, default=400)
    p.add_argument("--eval-seeds", type=int, default=80)
    p.add_argument("--outdir", default="data")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    print(f"train: {a.train_seeds} seeds x {NCELL} cells", flush=True)
    generate(list(range(a.train_seeds)), f"{a.outdir}/train", a.device)
    print(f"eval: {a.eval_seeds} held-out seeds x {NCELL} cells", flush=True)
    generate(list(range(10_000, 10_000 + a.eval_seeds)), f"{a.outdir}/eval", a.device)
