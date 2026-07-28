import argparse
import asyncio
import omni.kit.app  # noqa: E402
import omni.kit.asset_converter as ac  # noqa: E402
from pathlib import Path
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--in",  dest="src", required=True, type=Path)
ap.add_argument("--out", dest="dst", required=True, type=Path)
args, _ = ap.parse_known_args()

_a = argparse.ArgumentParser().parse_known_args()[0]
_a.enable_cameras = True
_a.headless = True # NOTE(jehyun): Headless Generation
_app = AppLauncher(_a); _sim = _app.app

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "omni.kit.asset_converter", True ) 

ctx = ac.AssetConverterContext()
ctx.ignore_materials = True
ctx.use_meter_as_world_unit = True
ctx.convert_stage_up_z = True

async def _run():
    task = ac.get_instance().create_converter_task(
        str(args.src), str(args.dst), None, ctx,
    )
    ok = await task.wait_until_finished()
    print(f"converter status: {task.get_status()!r}  ok={ok}")
    if not ok:
        print(f"  error: {task.get_error_message()}")

asyncio.get_event_loop().run_until_complete(_run())
print(f"wrote {args.dst}  ({args.dst.stat().st_size/1e6:.1f} MB)"
      if args.dst.exists() else "output missing")

_sim.close()
