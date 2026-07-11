
import argparse
import os
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

# Omniverse/USD modules must be imported after SimulationApp is instantiated.
from pxr import Usd, UsdGeom, Gf  # noqa: E402

SCENE_DIR = os.environ.get("SCENE_DIR", os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--usd", default=f"{SCENE_DIR}/asset.usd", help="source scene (Y-up)")
ap.add_argument("--prims", nargs="*", default=["strove", "sponge", "frying_pan", "floor", "sink",
                         "cup", "PepperRed", "mustard", "ketchup", "Coca_Cola"])
ap.add_argument("--reference", default=None, help="optional target scene USD (e.g. polaris_scene/scene.usda)")
ap.add_argument("--reference-prim", default="g60_stovetop_zed", help="reference prim in --reference to compare against")
args, _ = ap.parse_known_args()
cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])

def zup(p):
    """Y-up -> Z-up: (x, y, z) -> (x, -z, y)."""
    return Gf.Vec3d(p[0], -p[2], p[1])

def converted_center(prim):
    b = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = b.GetMin(), b.GetMax()
    corners = [Gf.Vec3d(x, y, z) for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])]
    c = [zup(v) for v in corners]
    xs, ys, zs = [v[0] for v in c], [v[1] for v in c], [v[2] for v in c]
    center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)
    return center, max(zs), min(zs)

stage = Usd.Stage.Open(args.usd)
print("=== source objects (Z-up converted) ===", flush=True)
for name in args.prims:
    p = stage.GetPrimAtPath(f"/World/{name}")
    if not p.IsValid(): continue
    c, ztop, zbot = converted_center(p)
    print(f"{name:12} center=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})  ztop={ztop:.3f}  zbot={zbot:.3f}", flush=True)

if args.reference:
    ref = Usd.Stage.Open(args.reference)
    rp = ref.GetPrimAtPath(f"/World/{args.reference_prim}")
    if rp.IsValid():
        b = cache.ComputeWorldBound(rp).ComputeAlignedRange()
        mn, mx = b.GetMin(), b.GetMax()
        print("\n=== reference (target frame) ===", flush=True)
        print(f"{args.reference_prim} center=("f"{(mn[0]+mx[0])/2:.3f},{(mn[1]+mx[1])/2:.3f},{(mn[2]+mx[2])/2:.3f})  ztop={mx[2]:.3f}", flush=True)

app.close()