# cosmos-vwm-physics-thresholds

Controlled intuitive-physics evaluation for **Cosmos Predict2 Video2World**:
from pixels only, predict which side of a physical threshold a MuJoCo scene
lands on. Inference only; no weights are trained.

Six environments separate static configuration reading from dynamic state
estimation:

| regime | environment | threshold outcome |
|---|---|---|
| static | Seesaw | tips left / right from visible torque balance |
| static | Lean | holds / slides from rod angle and fixed contact friction |
| static | Tower | stands / falls from cumulative center of mass |
| dynamic | Hill | returns / crosses from visible velocity and hill height |
| dynamic | Collision | separates / bounces after impact |
| dynamic | Domino | propagation stops / reaches the final domino |

Official 480p/16 FPS `Cosmos-Predict2-2B-Video2World`. Static scenes get one
image, dynamic scenes five strictly pre-outcome frames; endpoints are compared
with MuJoCo at a shared +0.3125 s horizon.

## Layout

```text
src/core/
  threshold/                          six MuJoCo environments and shared rollout base

src/gen/
  render.py                            shared EGL/render/video/preview helpers
  sweep_spec.py                        sweep ranges, cameras, prompts, and margins
  make_sweep.py                        seven margins × six environments
  make_diagnostics.py                  native 832×480, exact-16fps Tower/Hill set
  make_gifs.py                         visual physics smoke tests
  make_policy_inputs.py                pixel-conditioning previews

src/exp/
  cosmos_v2w_sweep.py                  six-environment Predict2 V2W inference
  collect_v2w_diagnostics.py           resumable multi-seed corrected collection
  analyze_physics_sweeps.py            GT-calibrated outcome decoding
  analyze_corrected_diagnostics.py     threshold, seed, and intervention statistics
  vae_reconstruction_diagnostics.py    frozen-VAE reconstruction ceiling
  verify_diagnostic_collection.py      MP4/PNG/metadata integrity checks
  prioritize_diagnostic_jobs.py        job ordering for the time-limited run
  render_v2w_triplet_gifs.py           LAST / PHYSICS GT / COSMOS GIFs
  render_diagnostic_comparison_gif.py  corrected Tower/Hill GIFs
  render_intuitive_v2w_gifs.py         sweep-wide qualitative GIFs

artifacts/                              generated data and media; never committed
```

`cosmos_policy_pixels.py` and `cosmos_policy_sweep.py` retain the initial
Cosmos-Policy comparison baseline. They are not the main VWM experiment.

## Environment

Python 3.11+, MuJoCo 3.11, PyTorch/CUDA, Diffusers with
`Cosmos2VideoToWorldPipeline`, NumPy, Pillow, imageio/ffmpeg, matplotlib, and
the Cosmos Predict2 2B Video2World checkpoint.

Run MuJoCo EGL rendering and the PyTorch runner in **separate processes**;
sharing one corrupts the encoder output silently. Data goes to `artifacts/`.

## Reproduce

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src \
python src/gen/make_sweep.py \
  --stats /path/to/libero_dataset_statistics.json \
  --output-dir artifacts/physics_sweep

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src \
python src/gen/make_gifs.py \
  --output-dir artifacts/threshold_gifs
```

Run the model:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python src/exp/cosmos_v2w_sweep.py \
  --model-dir /path/to/Cosmos-Predict2-2B-Video2World \
  --original-checkpoint /path/to/model-480p-16fps.pt \
  --manifest artifacts/physics_sweep/manifest.jsonl \
  --sweep-root artifacts/physics_sweep \
  --output-dir artifacts/physics_sweep/cosmos_v2w
```

Analyze and render:

```bash
PYTHONPATH=src python src/exp/analyze_physics_sweeps.py \
  --sweep-root artifacts/physics_sweep

MUJOCO_GL=egl PYTHONPATH=src python src/exp/render_v2w_triplet_gifs.py \
  --sweep-root artifacts/physics_sweep
```

The root wrappers `run_sweep.sh` and `run_vwm.sh` provide the same entry
points with fewer arguments.

## Result

The corrected run removes the earlier timing and aspect confounds: native
832×480, exact 16 FPS sampling, five pre-event Hill frames, paired
base/oracle prompts, a Hill image-only ablation, and frozen-VAE plus file
integrity checks. Three hours produced 535 rollouts (213 base, 213 oracle,
109 image-only), every video decoding as 832×480, 16 FPS, 21 frames.

| condition | accuracy |
|---|---|
| majority baseline | 57.1% |
| base Tower | 44.2% |
| base Hill | 46.8% |
| base → oracle prompt (paired) | 45.5% → 54.0%, McNemar p=0.004 |
| base → Hill image-only | 90.9% of decisions flip, p=0.34 |

Base Tower nearly always answers **stable**, base Hill **crosses**, image-only
Hill **fails**. Conditioning moves the generated outcome, but the model does
not recover the analytic threshold.

Outcome decoders are calibrated on MuJoCo endpoints only and then frozen.
Tower is read from red/blue/grey block displacement, Hill from red-ball
horizontal displacement; object loss and morphing count as validity failures
rather than physics decisions.
