# vjepa-physics-probe

Reading physical parameters (mass, friction, restitution) out of frozen
V-JEPA 2 features, to label object physics for real2sim.

**In sim:** linear probe reaches R² 0.91–0.99 on observable parameters across
five verified MuJoCo interactions; fails exactly where physics forbids.
**Sim-to-real:** motion-only features close the distribution gap on DROID
clips, but real-footage physics is not read yet.

## Layout

```
src/core/   shared library: scenes.py (5 verified families), encoder.py
            (single V-JEPA wrapper), probe.py (single fit/eval), paths.py
src/gen/    verify_scenes.py (analytic physics checks, must pass 5/5),
            gen_episodes.py (data generation with render-integrity checks)
src/exp/    experiment entry points: encode_all, probe_main, ablations,
            transfer, droid_probe / droid_gap / motion_probe, plotting
data/       symlink to bulk storage (episodes, features, checkpoints) — not in git
out/        results and figures — not in git
```

## Reproduce

Environment: Python 3.11, torch 2.6+cu124, mujoco 3.11, scikit-learn, timm, einops.
Paths default to `data/` inside the repo; override with `VJEPA_DIR`,
`VJEPA_CKPT`, `EP_DIR`, `FEAT_DIR`, `DROID_EPS` if your layout differs.
The checkpoint is `vjepa2-ac-vitg.pt` from the official V-JEPA 2 release.

```bash
python src/gen/verify_scenes.py            # physics checks — require 5/5
./run_gen.sh 0 slide roll bounce &         # generate episodes (arg 1 = EGL GPU)
./run_gen.sh 1 collide incline
./run_gen_aug.sh 0 slide roll bounce collide incline   # camera-augmented train set
python src/exp/encode_all.py --dirs slide/train_clean ...   # all family/split pairs
python src/exp/probe_main.py               # main result table
python src/exp/ablations.py --families slide roll bounce collide incline
```

## Pitfalls encoded in this code

- `mj_setConst()` is mandatory after changing `body_mass`; otherwise the mass
  matrix keeps its compile-time value and every mass sweep is silently wrong.
- EGL rendering corrupts silently (blank frames, missing shadows, segfaults).
  Generation renders each episode twice and requires bitwise agreement, plus a
  reference-reproduction check per batch.
- MuJoCo EGL and PyTorch CUDA must never share a process; rendering and
  encoding are separate stages.
