from pathlib import Path
import json

ROOT = Path("/home/parkprogrammer/jehyun/droid2sim/data/PolaRiS-Hub")

def build_scene(
    src_dir: Path,
    dst_dir: Path,
    payload_rel: str = "full_scene.usd",
    target_stove_top_z: float = 0.056,
    stove_half_height: float = 0.504,
    stove_center_target: tuple = (0.55, 0.0),
    camera=None,
    lights=None,
) -> Path:
    manifest = {m["folder"]: m["world_translate"]
                for m in json.loads((src_dir / "manifest.json").read_text())}
    sx, sy, sz = manifest["stove"]
    dx = stove_center_target[0] - sx
    dy = stove_center_target[1] - sy
    dz = (target_stove_top_z - stove_half_height) - sz

    cam = camera or dict(
        translate=(0.113286, 0.482930, 0.404185),
        orient=(-0.421182, -0.233667, 0.425148, 0.766325),
        focal=1.0476, hap=2.5452, vap=1.4721,
    )
    lit = lights or dict(dome_intensity=4000, key_intensity=8000)

    import os
    dst_dir.mkdir(parents=True, exist_ok=True)
    for link, target in [
        (dst_dir / "assets",                  src_dir / "assets"),
        (dst_dir / payload_rel,               src_dir / Path(payload_rel).name),
        (dst_dir / "initial_conditions.json", ROOT / "pan_clean/initial_conditions.json"),
    ]:
        if link.is_symlink() or link.exists(): os.remove(link)
        link.symlink_to(target.resolve())

    scene = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Xform "alchera_scene" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
        prepend payload = @./{payload_rel}@
    )
    {{
        bool physics:kinematicEnabled = 1
        bool physics:rigidBodyEnabled = 1
        double3 xformOp:translate = ({dx:.4f}, {dy:.4f}, {dz:.4f})
        quatd xformOp:orient = (1, 0, 0, 0)
        double3 xformOp:scale = (1, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}

    def Camera "external_cam" ( kind = "model" )
    {{
        float2 clippingRange = (0.01, 10000000)
        float focalLength = {cam["focal"]}
        float horizontalAperture = {cam["hap"]}
        float verticalAperture = {cam["vap"]}
        quatd xformOp:orient = {tuple(cam["orient"])}
        double3 xformOp:translate = {tuple(cam["translate"])}
        double3 xformOp:scale = (1, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}

    def DomeLight "envLight" ( prepend apiSchemas = ["ShapingAPI"] )
    {{
        float inputs:intensity = {lit["dome_intensity"]}
        color3f inputs:color = (1, 1, 1)
    }}

    def DistantLight "keyLight" ( prepend apiSchemas = ["ShapingAPI"] )
    {{
        float inputs:angle = 3
        float inputs:intensity = {lit["key_intensity"]}
        quatd xformOp:orient = (0.65328, 0.27060, 0.27060, 0.65328)
    }}
}}
"""
    (dst_dir / "scene.usda").write_text(scene)
    print(f"wrote {dst_dir/'scene.usda'}  offset=({dx:+.3f},{dy:+.3f},{dz:+.3f})")
    return dst_dir / "scene.usda"


if __name__ == "__main__":
    build_scene(
        src_dir=ROOT / "pan_clean_alchera",
        dst_dir=ROOT / "pan_clean_alcheraA1",
    )
