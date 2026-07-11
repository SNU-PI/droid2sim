# Interactive Isaac Sim GUI (WebRTC) + live scripting

How to view a scene interactively in the browser, drive the running GUI from
Python, and the gotchas that cost us hours.

## 1. Bring up the container

```bash
./scripts/docker/launch.sh -g <GPU>     # e.g. -g 2
```

This starts two containers for that GPU (project `isim<GPU>`):

- `isim<GPU>-isaac-sim-1`   — Isaac Sim, running the WebRTC **streaming** app
- `isim<GPU>-web-viewer-1`  — the browser front-end (vite) on port `8210+GPU`

Ports are derived from the GPU index:

| service      | port           |
|--------------|----------------|
| web-viewer   | `8210 + GPU`   |
| signal (TCP) | `49100 + GPU`  |
| stream (UDP) | `47998 - GPU`  |
| jupyter code | `8227` (fixed) |

Wait until `docker ps` shows the isaac-sim container as `healthy` (~1-3 min).

## 2. Connect the browser (WebRTC)

**Connect directly to the server IP — not via an SSH tunnel.**

- `docker/.env` sets `ISAACSIM_HOST=147.46.219.233` (the machine's IP). The
  web-viewer is built with this baked in as its `signalingServer`.
- Open `http://147.46.219.233:<8210+GPU>/` in a browser.

Why not SSH tunnel: `ssh -L` forwards **TCP only**. WebRTC media + STUN are
**UDP**, so a tunnel gives you `StreamerNoStunResponsesReceived` /
`WAITING FOR STREAM…` forever. Direct-IP works because the firewall already
allows these ports.

If you rebuild the web-viewer for a different host, force it:

```bash
ISAACSIM_HOST=147.46.219.233 GPU_DEVICE=<GPU> \
ISAACSIM_SIGNAL_PORT=$((49100+GPU)) ISAACSIM_STREAM_PORT=$((47998-GPU)) WEB_VIEWER_PORT=$((8210+GPU)) \
  docker compose -p isim<GPU> -f docker/docker-compose.yml up -d --build --no-deps web-viewer
```

## 3. Drive the running GUI from Python

The compose enables `isaacsim.code_editor.jupyter`, which listens on TCP `8227`.
Send `<token> + <python source>` and it runs in the live app; the reply is JSON
`{"status": "...", "output": "..."}`. Helper:

```bash
docker exec isim<GPU>-isaac-sim-1 \
  /isaac-sim/kit/python/bin/python3 /work/scripts/docker_render/gui_send.py <code-file>
```

`gui_send.py` reads the token from
`/isaac-sim/exts/isaacsim.code_editor.jupyter/data/launchers/token.txt`.

Example — open a scene in the live GUI:

```python
import omni.usd
omni.usd.get_context().open_stage("/work/data/alchera_nvidia/asset.usd")
```

## 4. Capture what the GUI is showing

To see the viewport pixels yourself (the browser stream isn't scrape-able):

```python
import omni.usd, omni.kit.viewport.utility as vpu
vpu.capture_viewport_to_file(vpu.get_active_viewport(), file_path="/tmp/cap.png")
```

Write to **`/tmp` inside the container**, not `/work` — the repo mount is owned
by the host user (uid 1020) and the container runs as uid 1234, so `/work` is
read-only to it. Then copy out:

```bash
docker cp isim<GPU>-isaac-sim-1:/tmp/cap.png ./runs/cap.png
```

The capture is async; give the streaming loop a few seconds before copying.

## 5. Gotcha: everything renders teal (cyan)

**Symptom:** textured materials show as flat cyan in RTX (both viewport and
headless); flat-color materials (e.g. ketchup) render fine.

**Cause:** the texture files are mode `600` (owner-only). The container user
(uid 1234) can't read them, UJITSO fails to build the texture, and RTX falls
back to teal. The log shows:

```
[omni.ujitso] failed to load local file for .../wall_Normal.png
[omni.rtx]    Failed to request UJITSO build result for: .../wall_Normal.png
```

**Fix:** make the data readable (no sudo needed — you own the files):

```bash
chmod -R a+rX data/
```

`launch.sh` does this automatically for `data/` on every start.

## 6. Coordinate frame

Source assets from the glTF→USD conversion are **Y-up, real scale**
(countertop ~1.1 m). The DROID robot frame is **Z-up** with the working surface
~0.05 m in front of the base. To view/align, wrap the asset in an Xform rotated
+90° about X and translated down, e.g.:

```python
xf.AddTranslateOp().Set(Gf.Vec3d(0.166, 0.078, -1.068))
xf.AddOrientOp(...).Set(Gf.Quatf(0.70710678, 0.70710678, 0, 0))  # +90° about X
```

`render_views.py --axis-convert` applies this for headless renders.
