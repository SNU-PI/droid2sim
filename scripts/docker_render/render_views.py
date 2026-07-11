import argparse
import os

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

# Omniverse/USD modules must be imported after SimulationApp is instantiated.
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from pxr import UsdGeom, UsdLux, Gf  # noqa: E402
from PIL import Image  # noqa: E402

SCENE_DIR = os.environ.get("SCENE_DIR", os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.environ.get("OUT_DIR", os.getcwd())

# axis conversion (Y-up -> Z-up) + height offset used for the raw source scene
CONVERT_ORIENT = Gf.Quatf(0.70710678, 0.70710678, 0.0, 0.0)   # +90 deg about X
CONVERT_TRANSLATE = Gf.Vec3d(0.166, 0.078, -1.068)

# camera presets: (eye, target, up, focal, resolution)
VIEWS = {
    "iso":     (Gf.Vec3d(1.35, -1.05, 1.05), Gf.Vec3d(0.5, 0.02, 0.05), Gf.Vec3d(0, 0, 1), 22.0, (1100, 1100)),
    "topdown": (Gf.Vec3d(0.4, 0.0, 2.4),     Gf.Vec3d(0.4, 0.0, 0.05),  Gf.Vec3d(1, 0, 0), 18.0, (1000, 1000)),
    "match":   (Gf.Vec3d(0.15, -0.55, 0.5),  Gf.Vec3d(0.55, 0.05, 0.03), Gf.Vec3d(0, 0, 1), 15.0, (1350, 760)), }

ap = argparse.ArgumentParser()
ap.add_argument("--usd", default=f"{SCENE_DIR}/asset.usd", help="scene USD to render")
ap.add_argument("--view", default="iso", choices=list(VIEWS))
ap.add_argument("--out", default="render.png", help="output png (relative to OUT_DIR)")
ap.add_argument("--axis-convert", action="store_true", help="rotate Y-up source into the robot frame")
ap.add_argument("--robot", action="store_true", help="load the DROID robot at the origin")
ap.add_argument("--dome-intensity", type=float, default=1400.0)
args, _ = ap.parse_known_args()

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

if args.axis_convert:
    reo = UsdGeom.Xform.Define(stage, "/World/reorient")
    rxf = UsdGeom.Xform(reo.GetPrim())
    rxf.AddTranslateOp().Set(CONVERT_TRANSLATE)
    rxf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(CONVERT_ORIENT)
    holder = UsdGeom.Xform.Define(stage, "/World/reorient/scene")
else:
    holder = UsdGeom.Xform.Define(stage, "/World/scene")
holder.GetPrim().GetReferences().AddReference(args.usd, "/World")

if args.robot:
    rob = UsdGeom.Xform.Define(stage, "/World/robot")
    rob.GetPrim().GetReferences().AddReference(f"{SCENE_DIR}/robot/noninstanceable.usd")

UsdLux.DomeLight.Define(stage, "/World/dome").CreateIntensityAttr(args.dome_intensity)

eye, tgt, up, focal, res = VIEWS[args.view]
cam = UsdGeom.Camera.Define(stage, "/World/cam")
mat = Gf.Matrix4d(1).SetLookAt(eye, tgt, up).GetInverse()
UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(mat)
cam.CreateFocalLengthAttr(focal)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))

for _ in range(90): app.update()
rp = rep.create.render_product("/World/cam", res)
annot = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach(rp)
for _ in range(30): app.update()
rep.orchestrator.step(rt_subframes=12)
for _ in range(6): app.update()

rgb = annot.get_data()
if rgb.ndim == 3 and rgb.shape[-1] == 4: rgb = rgb[..., :3]
if rgb.dtype != np.uint8: rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)

out = os.path.join(OUT_DIR, args.out)
Image.fromarray(rgb).save(out)
print(f"WROTE {out}  shape={rgb.shape}", flush=True)

app.close()