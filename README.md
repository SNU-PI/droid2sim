# droid2sim

Docker setup to render it headless or view it interactively in a browser.

Scene assets live on Hugging Face
([`Parkprogrammer/droid2sim`](https://huggingface.co/datasets/Parkprogrammer/droid2sim));
this repo holds the code and Docker setup to run them.

## Reproduce

```bash
# 1. clone
git clone git@github.com:SNU-PI/droid2sim.git
cd droid2sim

# 2. download the scene into data/panclean/
huggingface-cli download Parkprogrammer/droid2sim --repo-type dataset \
  --include "scenes/sample/panclean/*" \
  --local-dir /tmp/hf-droid2sim
mkdir -p data
mv /tmp/hf-droid2sim/scenes/sample/panclean data/panclean

# 3. start Isaac Sim (pick a free GPU; opens WebRTC + fixes data/ perms)
./scripts/docker/launch.sh -g 0
```

The container mounts this repo at `/work`, so `data/panclean` is
`/work/data/panclean` inside it regardless of where you cloned.

### Render headless → PNG

```bash
docker exec -e SCENE_DIR=/work/data/panclean isim0-isaac-sim-1 \
  /isaac-sim/python.sh /work/scripts/docker_render/render_views.py \
  --view match --axis-convert --out kitchen.png
docker cp isim0-isaac-sim-1:/tmp/kitchen.png ./kitchen.png   # OUT_DIR defaults to /tmp
```

### View interactively (browser)

Open `http://<server-ip>:8210/` (port is `8210 + GPU`). Connect to the server
IP directly — an SSH tunnel will not work (WebRTC media is UDP). Details and
live-scripting in [`docs/interactive_gui.md`](docs/interactive_gui.md).

Stop with `./scripts/docker/down.sh -g 0`.

## Layout

```
scripts/
├── docker/           launch.sh / down.sh / stream.sh   (per-GPU container control)
├── docker_render/    render_views / inspect_scene / measure_alignment / gui_send
├── glb_to_usd.py     glTF → USD conversion (NVIDIA omni.kit.asset_converter)
└── build_alchera_real.py   assemble a scene.usda placed in the robot frame
docker/               docker-compose.yml + web-viewer (WebRTC)
docs/
├── interactive_gui.md        browser GUI, live control, the teal-material gotcha
└── polaris_reproduction.md   earlier PolaRiS real-to-sim eval reproduction
```

## Notes

- Source assets are **Y-up, real scale** (countertop ~1.1 m); the DROID robot
  frame is **Z-up** with the surface ~0.05 m from the base. `render_views.py
  --axis-convert` and `build_alchera_real.py` handle the conversion.
- If materials render **teal/cyan**, texture files are unreadable by the
  container user — `launch.sh` fixes this (`chmod a+rX data/`); see
  [`docs/interactive_gui.md`](docs/interactive_gui.md).
