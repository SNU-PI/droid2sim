import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
args, _ = ap.parse_known_args()

_a = argparse.ArgumentParser().parse_known_args()[0]
_a.enable_cameras = True
_a.headless = True
_app = AppLauncher(_a); _sim = _app.app

import omni.usd  # noqa: E402
import omni.kit.app  # noqa: E402
from pxr import Sdf, UsdGeom, UsdLux, Gf  # noqa: E402

for ext in ("omni.replicator.core", "omni.kit.viewport.utility"):
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(ext, True)

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

alchera = UsdGeom.Xform.Define(stage, "/World/alchera")
alchera.GetPrim().GetPayloads().AddPayload(str(args.usd))

dome = UsdLux.DomeLight.Define(stage, "/World/dome"); dome.CreateIntensityAttr(1500.0)

cam = UsdGeom.Camera.Define(stage, "/World/cam")
xform_api = UsdGeom.XformCommonAPI(cam)
xform_api.SetTranslate(Gf.Vec3d(2.0, -2.0, 2.0))
xform_api.SetRotate(Gf.Vec3f(-135.0, 0.0, 45.0))
cam.CreateFocalLengthAttr(24.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

for _ in range(60):
    _sim.update()

import omni.replicator.core as rep  # noqa: E402

rp = rep.create.render_product(str(cam.GetPath()), (1280, 720))
annot = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach(rp)

for _ in range(30):
    _sim.update()

rep.orchestrator.step(rt_subframes=8)
_sim.update()

import numpy as np  # noqa: E402
import mediapy  # noqa: E402

rgb = annot.get_data()
if rgb.ndim == 3 and rgb.shape[-1] == 4:
    rgb = rgb[..., :3]
if rgb.dtype != np.uint8:
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)

args.out.parent.mkdir(parents=True, exist_ok=True)
mediapy.write_image(str(args.out), rgb)
print(f"wrote {args.out}  shape={rgb.shape}  min={rgb.min()} max={rgb.max()}")

_sim.close()
