# droid2sim

Reproduction of [PolaRiS](https://github.com/arhanjain/polaris) Real-to-Sim evaluations
for the DROID generalist policy (pi05), with QHD (2560x1440 per camera) sim renders.

A static rendered grid comparing real-world DROID demos against sim rollouts is in
[`index.html`](./index.html).

## What's in this repo

```
.
|-- index.html                    Rendered comparison grid (open in browser)
|-- assets/
|   |-- real/                     6 real-world DROID rollouts (time-warped to 7.6s)
|   `-- sim/<env>/                6 sim rollouts (episode 0, QHD, 2x slow = 7.5s)
|-- results/<env>/eval_results.csv  Eval scores (success/progress per episode)
|-- patches/polaris.patch         Local modifications to arhanjain/polaris
`-- scripts/
    |-- setup_polaris.sh          Clone + patch polaris for reproducing rollouts
    `-- build_pages.py            Regenerate index.html from assets/ + results/
```

## Upstream repositories

This work uses two upstream repositories from the PolaRiS authors. Neither is forked
or vendored here; modifications to `polaris` are tracked as a single patch file.

| Upstream | Status | How it appears here |
|---|---|---|
| [arhanjain/polaris](https://github.com/arhanjain/polaris) | **modified** | `patches/polaris.patch` (camera resolution + native frame capture + build fix) |
| [arhanjain/sim-evals](https://github.com/arhanjain/sim-evals) | **unused** | Not present. Considered as a lighter zero-shot alternative; the QHD eval pipeline here uses the full `polaris` repo. |

The patch modifies four files in `arhanjain/polaris`:

| File | Change |
|---|---|
| `pyproject.toml` | Add `[tool.uv.extra-build-dependencies]` pinning `setuptools<70` for the `flatdict` build (fixes `pkg_resources` ImportError on modern setuptools) |
| `src/polaris/environments/droid_cfg.py` | Raise camera native resolution from 1280x720 to 2560x1440 (wrist + external + dynamic scene cameras) |
| `scripts/eval.py` | Replace the policy-input-sized `viz` (224x224 per camera) with the native QHD frames pulled from `obs["splat"]` for video output |
| `uv.lock` | Mechanical lock-file regeneration triggered by the above |

## Reproducing the sim rollouts

1. Clone and patch `arhanjain/polaris` (outputs into `./external/polaris` by default):

   ```bash
   ./scripts/setup_polaris.sh
   ```

2. Follow the upstream README at https://github.com/arhanjain/polaris for environment
   setup (`uv sync`, openpi sync, HF `PolaRiS-Hub` download, ffmpeg).

3. Launch the policy server (in `external/polaris/third_party/openpi`):

   ```bash
   CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 uv run scripts/serve_policy.py \
     --port 8001 policy:checkpoint \
     --policy.config pi05_droid_jointpos_polaris \
     --policy.dir gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris
   ```

4. Run eval (in `external/polaris`, headless server needs nvidia ICDs explicit):

   ```bash
   unset DISPLAY
   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
   __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
   __GLX_VENDOR_LIBRARY_NAME=nvidia \
   CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \
   uv run scripts/eval.py \
     --environment DROID-FoodBussing --policy.port 8001 \
     --run-folder runs/foodbussing --rollouts 5
   ```

   Repeat for the other five environments (`DROID-BlockStackKitchen`, `DROID-PanClean`,
   `DROID-MoveLatteCup`, `DROID-OrganizeTools`, `DROID-TapeIntoContainer`).

## Results summary (pi05, 5 rollouts each)

| Env | avg progress | success |
|---|---:|---:|
| FoodBussing       | 0.57 | 1/5 |
| BlockStackKitchen | 0.34 | 0/5 |
| PanClean          | 0.80 | 2/5 |
| MoveLatteCup      | 0.00 | 0/5 |
| OrganizeTools     | 0.13 | 0/5 |
| TapeIntoContainer | 0.07 | 0/5 |
| **overall**       | **0.32** | **3/30** |

## Viewing the page

Either open `index.html` locally, or enable GitHub Pages on this repository
(Settings -> Pages -> Source: `main`/root). The page is fully static and uses
only files in `assets/`.

## Regenerating the page

After updating an `eval_results.csv` or replacing a video under `assets/`:

```bash
python3 scripts/build_pages.py
```
