"""Generate scene.usda for DROID-PanCleanAlcheraReal.

Background = NVIDIA-converted asset.usd, referenced whole, with a single axis
conversion (Y-up -> Z-up, +90 deg about X) and a height offset so the stovetop
lands at robot height. The asset's own sponge/pan are hidden. Manipulable
sponge/pan are the proven individual meshes at PolaRiS-reachable coordinates.
"""
from pathlib import Path
import os

ROOT = Path("/home/parkprogrammer/jehyun/droid2sim/data/PolaRiS-Hub")
DST_DIR = ROOT / "pan_clean_alcheraReal"

KITCHEN_ORIENT = (0.70710678, 0.70710678, 0.0, 0.0)   # +90 deg about X
KITCHEN_TRANSLATE = (0.166, 0.078, -1.068)

SPONGE_T = (0.5294183970681007, -0.022660877717386577, 0.06433805102708207)
PAN_T = (0.5631938515485317, 0.21159406034871397, 0.05608738411494982)
PAN_ORIENT = (0.19625393, 0.0, 0.0, -0.9805531)


def build():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    links = {
        "asset.usd": Path("/home/parkprogrammer/jehyun/droid2sim/data/alchera_nvidia/asset.usd"),
        "textures": Path("/home/parkprogrammer/jehyun/droid2sim/data/alchera_nvidia/textures"),
        "assets": ROOT / "pan_clean_alchera/assets",
        "initial_conditions.json": ROOT / "pan_clean/initial_conditions.json",
    }
    for name, target in links.items():
        link = DST_DIR / name
        if link.is_symlink() or link.exists():
            os.remove(link)
        link.symlink_to(target.resolve())

    scene = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Xform "kitchen" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
        prepend references = @./asset.usd@</World>
    )
    {{
        bool physics:kinematicEnabled = 1
        bool physics:rigidBodyEnabled = 1
        quatf xformOp:orient = {KITCHEN_ORIENT}
        double3 xformOp:translate = {KITCHEN_TRANSLATE}
        double3 xformOp:scale = (1, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

        over "sponge"
        {{
            token visibility = "invisible"
        }}

        over "frying_pan"
        {{
            token visibility = "invisible"
        }}

        over "strove"
        {{
            over "Cube_008" (
                prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
            )
            {{
                uniform token physics:approximation = "convexHull"
                bool physics:collisionEnabled = 1
            }}
        }}
    }}

    def "sponge" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
        prepend references = @./assets/sponge/mesh.usda@
    )
    {{
        bool physics:kinematicEnabled = 0
        bool physics:rigidBodyEnabled = 1
        double3 xformOp:translate = {SPONGE_T}
        quatf xformOp:orient = (1, 0, 0, 0)
        float3 xformOp:scale = (1, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}

    def "pan" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
        prepend references = @./assets/frying_pan/mesh.usda@
    )
    {{
        bool physics:kinematicEnabled = 1
        bool physics:rigidBodyEnabled = 1
        double3 xformOp:translate = {PAN_T}
        quatf xformOp:orient = {PAN_ORIENT}
        float3 xformOp:scale = (1, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}

    def Camera "external_cam" ( kind = "model" )
    {{
        float2 clippingRange = (0.01, 10000000)
        float focalLength = 1.0476
        float focusDistance = 400
        float horizontalAperture = 2.5452
        float verticalAperture = 1.4721
        quatd xformOp:orient = (-0.4211824302119951, -0.2336669335513614, 0.4251479824119222, 0.766325203615602)
        double3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0.1132860923533613, 0.48292959665334745, 0.404185346166213)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}
}}
"""
    (DST_DIR / "scene.usda").write_text(scene)
    print(f"wrote {DST_DIR / 'scene.usda'}")


if __name__ == "__main__":
    build()
