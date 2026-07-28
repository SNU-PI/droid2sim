# droid2sim

Reproduction of [PolaRiS](https://github.com/arhanjain/polaris) Real-to-Sim
evaluations for the DROID generalist policy (`pi05_droid_jointpos_polaris`),
rendered at QHD (2560x1440 per camera).

A static comparison page (real DROID demo vs. sim rollout) is in
[`index.html`](./index.html). Cloning this repo plus the steps below is
sufficient to reproduce the rollouts.

## Repository layout

```
.
|-- index.html                       Rendered comparison grid (open in browser)
|-- polaris/                         Vendored arhanjain/polaris source (patched)
|-- assets/
|   |-- real/                        6 real-world DROID rollouts (7.6s)
|   `-- sim/<env>/episode_0_mid.mp4  6 sim rollouts at QHD, 2x slow (7.5s)
|-- results/<env>/eval_results.csv   Eval scores per environment (5 rollouts)
`-- scripts/build_pages.py           Regenerate index.html from assets/+results/
```

## Upstream repositories

| Upstream | Status | How it appears here |
|---|---|---|
| [arhanjain/polaris](https://github.com/arhanjain/polaris) | **vendored + modified** | `polaris/` (see `git log polaris/`) |
| [arhanjain/sim-evals](https://github.com/arhanjain/sim-evals) | **unused** | Not present. A lighter zero-shot eval path; this repo uses the full PolaRiS pipeline. |

The two commits `308c215` (vendor upstream) and `b55b668` (apply local
modifications) isolate every line we changed.

| File under `polaris/` | Change |
|---|---|
| `pyproject.toml` | Pin `setuptools<70` as an extra build dep for `flatdict` (fixes `pkg_resources` ImportError under modern setuptools) |
| `src/polaris/environments/droid_cfg.py` | Raise native render resolution from 1280x720 to 2560x1440 on `wrist_cam`, `external_cam`, and scene-derived cameras |
| `scripts/eval.py` | Save native QHD frames from `obs["splat"]` instead of the policy-input `viz` (224x224) |

## Prerequisites

- Linux with an NVIDIA GPU supporting Vulkan ray tracing (tested on RTX PRO 6000)
- NVIDIA driver with the matching Vulkan and EGL ICDs installed:
  - `/usr/share/vulkan/icd.d/nvidia_icd.json`
  - `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
- CUDA toolkit (`nvcc`) matching the torch wheels (default: CU13; switch in
  `polaris/pyproject.toml` if needed)
- `ffmpeg` (for saving rollout videos): `sudo apt install ffmpeg`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

### 1. Sync the polaris environment

```bash
cd polaris
uv sync
```

This pulls Isaac Sim (`isaaclab[all,isaacsim]==2.3.0`) and the CUDA-built
submodules (`diff-surfel-rasterization`, `simple-knn`). First sync is heavy
(>15GB cache, several minutes).

### 2. Sync openpi (for the policy server)

```bash
cd polaris/third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 3. Download the PolaRiS-Hub data

```bash
cd polaris
uvx hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir PolaRiS-Hub
```

This is ~1.7GB containing the six scanned scenes. `eval.py` resolves
`./PolaRiS-Hub/<env>/scene.usda` relative to the working directory.

## Running the evaluation

Two processes are needed: a policy server (openpi) and the eval client
(polaris/isaaclab). They must use different GPUs - Omniverse and JAX
otherwise contend for the same renderer.

### Policy server (e.g. GPU 0)

```bash
cd polaris/third_party/openpi
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
  uv run scripts/serve_policy.py \
    --port 8001 policy:checkpoint \
    --policy.config pi05_droid_jointpos_polaris \
    --policy.dir gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris
```

First launch downloads the 11.6GB checkpoint into `~/.cache/openpi`
(anonymous gs:// via `fsspec[gcs]`, no `gcloud` CLI required). Wait until
the log shows `server listening on 0.0.0.0:8001`.

### Eval client (e.g. GPU 1, separate process)

```bash
cd polaris
unset DISPLAY
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
  uv run scripts/eval.py \
    --environment DROID-FoodBussing \
    --policy.port 8001 \
    --run-folder runs/food_bussing \
    --rollouts 5
```

Repeat with the other five environments (`DROID-BlockStackKitchen`,
`DROID-PanClean`, `DROID-MoveLatteCup`, `DROID-OrganizeTools`,
`DROID-TapeIntoContainer`). Each run produces `eval_results.csv` and one
`.mp4` per episode under `--run-folder`.

### Why the env vars matter

- `unset DISPLAY` forces Isaac Sim down the EGL path; running with an
  Xvfb-provided X display produces `GLXBadFBConfig` because Xvfb only
  exposes software GLX.
- `VK_ICD_FILENAMES`, `__EGL_VENDOR_LIBRARY_FILENAMES`,
  `__GLX_VENDOR_LIBRARY_NAME` pin the NVIDIA Vulkan and EGL ICDs.
  Without them Isaac Sim may fail at `omni.gpu_foundation_factory`
  with "Failed to create any GPU devices".
- `OMNI_KIT_ACCEPT_EULA=YES` bypasses the interactive EULA prompt that
  otherwise blocks headless runs at first launch.
- `CUDA_VISIBLE_DEVICES` must point each process at a distinct GPU.
  Sharing one GPU with the policy server can trip Omniverse renderer
  initialization.

## Results (pi05, 5 rollouts per environment)

| Env | avg progress | success |
|---|---:|---:|
| FoodBussing       | 0.57 | 1/5 |
| BlockStackKitchen | 0.34 | 0/5 |
| PanClean          | 0.80 | 2/5 |
| MoveLatteCup      | 0.00 | 0/5 |
| OrganizeTools     | 0.13 | 0/5 |
| TapeIntoContainer | 0.07 | 0/5 |
| **overall**       | **0.32** | **3/30** |

Per-episode rows are in `results/<env>/eval_results.csv`.

## Viewing the comparison page

Open `index.html` locally, or enable GitHub Pages on this repo (Settings >
Pages > Source: `main`/root). The page references only files in `assets/`
and is fully static.

## Regenerating the page

After editing a CSV or replacing a video in `assets/`:

```bash
python3 scripts/build_pages.py
```
