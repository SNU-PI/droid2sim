# Sent to the live Isaac Sim GUI (via gui_send.py) to assemble the PanClean
# scene and frame it through a saved camera, so the browser shows the kitchen
# immediately after launch instead of an empty viewport.
#
# Builds the stage in code from asset.usd (Y-up, real scale): wraps it in the
# Z-up robot frame, adds the dome + key light, and a 15mm camera matching the
# render_views "match" preset. No scene file needs to ship — only asset.usd.
import os
import omni.usd
import omni.kit.viewport.utility as vpu
from pxr import UsdGeom, UsdLux, Gf

SCENE_DIR = os.environ.get("SCENE_DIR", "/work/data/panclean")
ASSET = os.environ.get("PANCLEAN_ASSET", f"{SCENE_DIR}/asset.usd")

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

# asset.usd is Y-up real scale; +90 deg about X and translate down puts the
# countertop in the DROID robot frame (~0.05 m in front of the base).
kitchen = UsdGeom.Xform.Define(stage, "/World/kitchen")
kx = UsdGeom.Xformable(kitchen)
kx.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.166, 0.078, -1.068))
kx.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(0.70710677, 0.70710677, 0, 0))
kitchen.GetPrim().GetReferences().AddReference(ASSET, "/World")

dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(1000.0)

key = UsdLux.DistantLight.Define(stage, "/World/key")
key.CreateIntensityAttr(3000.0)
key.CreateAngleAttr(0.53)
UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45, 10, 0))

cam = UsdGeom.Camera.Define(stage, "/World/viewcam")
cam.CreateFocalLengthAttr(15.0)
# near clip must be well under the ~0.6 m eye-to-scene distance, or the default
# 1.0 m near plane clips the whole kitchen and the viewport renders black.
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))
m = Gf.Matrix4d(
    0.8320502943378436, -0.554700196225229, 0, 0,
    0.30288403343499937, 0.45432605015249916, 0.8377643477989348, 0,
    -0.46470804811457017, -0.6970620721718555, 0.5460319565346199, 0,
    0.15, -0.55, 0.5, 1,
)
UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(m)

vp = vpu.get_active_viewport()
vp.set_active_camera("/World/viewcam")
print("assembled scene from %s; active camera %s" % (ASSET, vp.get_active_camera()))
