import argparse
import os

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

# Omniverse/USD modules must be imported after SimulationApp is instantiated.
from pxr import Usd, UsdGeom  # noqa: E402

SCENE_DIR = os.environ.get("SCENE_DIR", os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--usd", default=f"{SCENE_DIR}/asset.usd", help="USD file to inspect")
ap.add_argument("--mode", default="all", choices=["tree", "bounds", "xform", "all"])
ap.add_argument("--prims", nargs="*", default=["strove", "sponge", "frying_pan", "floor", "wall"],                help="prim names (under /World) for bounds/xform")
ap.add_argument("--max-depth", type=int, default=3, help="tree depth limit")
args, _ = ap.parse_known_args()
stage = Usd.Stage.Open(args.usd)


def show_tree():
    print("=== TREE ===", flush=True)
    for p in stage.Traverse():
        if p.GetPath().pathString.count("/") <= args.max_depth:
            print(f"{p.GetTypeName():8} {p.GetPath()}", flush=True)

def show_bounds():
    print("=== BOUNDS ===", flush=True)
    print(f"upAxis={UsdGeom.GetStageUpAxis(stage)} metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)}", flush=True)
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    for name in args.prims:
        p = stage.GetPrimAtPath(f"/World/{name}")
        if not p.IsValid(): continue
        b = cache.ComputeWorldBound(p).ComputeAlignedRange()
        mn, mx = b.GetMin(), b.GetMax()
        print(f"{name:12} min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f})  "
              f"max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f})", flush=True)

def show_xform():
    print("=== XFORM ===", flush=True)
    for name in args.prims:
        p = stage.GetPrimAtPath(f"/World/{name}")
        if not p.IsValid(): continue
        t, o = p.GetAttribute("xformOp:translate"), p.GetAttribute("xformOp:orient")
        print(f"{name:12} translate={t.Get() if t else None}  orient={o.Get() if o else None}", flush=True)

if args.mode in ("tree", "all"):
    show_tree()
if args.mode in ("bounds", "all"):
    show_bounds()
if args.mode in ("xform", "all"):
    show_xform()

app.close()