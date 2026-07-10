"""Blender headless: split Alchera asset.glb into per-object USD payloads.

Run with:
    blender --background --python scripts/alchera_extract.py

Output:
    data/PolaRiS-Hub/pan_clean_alchera/assets/{sponge, frying_pan, ...}/mesh.usd
    data/PolaRiS-Hub/pan_clean_alchera/full_scene.usd
"""
import bpy
import os
from pathlib import Path

ROOT = Path("/home/parkprogrammer/jehyun/droid2sim")
GLB  = ROOT / "droid_sample/new_sample/asset/asset.glb"
OUT  = ROOT / "data/PolaRiS-Hub/pan_clean_alchera"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "assets").mkdir(exist_ok=True)

# Node-name -> asset folder mapping (matches PolaRiS layout when possible)
TARGETS = {
    "sponge":     "sponge",
    "frying pan": "frying_pan",
    "sushi":      "sushi",
    "PepperRed":  "bell_pepper",
    "mustard":    "mustard",
    "ketchup":    "ketchup",
    "cup":        "mug",
    "Coca Cola":  "coke",
    "strove":     "stove",
    "desk":       "desk",
    "sink":       "sink",
    "wall":       "wall",
    "floor":      "floor",
}

# Wipe default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

print(f"[import] {GLB}")
bpy.ops.import_scene.gltf(filepath=str(GLB), merge_vertices=False)
print(f"[import] {len(bpy.data.objects)} objects in scene")

# glTF Blender importer places objects Z-up already (converts Y-up on import).
# So current scene orientation is USD-friendly (Z-up).

all_names = [o.name for o in bpy.data.objects]
print("scene objects:", all_names)

def _select_only(name):
    bpy.ops.object.select_all(action='DESELECT')
    obj = bpy.data.objects.get(name)
    if obj is None:
        return None
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return obj

def _apply_transforms(obj):
    """Bake current world transform into vertices, center origin on geometry,
    then move object to world origin so exported USD is drop-in ready."""
    _select_only(obj.name)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0.0, 0.0, 0.0)

def _export_usd(dst, selected_only=True, materials=True):
    dst = Path(dst)
    dst_dir = dst.parent
    # Blender writes texture paths relative to its cwd; hop into the target dir
    # so relative paths become @./textures/... (Isaac-Sim-resolvable).
    cwd0 = os.getcwd()
    os.chdir(dst_dir)
    kwargs = dict(
        filepath=str(dst),
        selected_objects_only=selected_only,
        export_animation=False,
        export_uvmaps=True,
        export_normals=True,
        export_materials=materials,
        use_instancing=False,
    )
    for k, v in dict(
        export_textures=materials,
        overwrite_textures=True,
        generate_preview_surface=True,
        relative_paths=True,
    ).items():
        try:
            bpy.ops.wm.usd_export.get_rna_type().properties[k]
            kwargs[k] = v
        except Exception:
            pass
    bpy.ops.wm.usd_export(**kwargs)
    os.chdir(cwd0)
    print(f"[usd] wrote {dst}  ({os.path.getsize(dst)/1e6:.2f} MB)")

# Per-object export
manifest = []
for node_name, folder in TARGETS.items():
    obj = bpy.data.objects.get(node_name)
    if obj is None:
        print(f"[skip] {node_name!r} not in scene")
        continue
    dst_dir = OUT / "assets" / folder
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Store original world translation before applying transforms (for scene.usda placement)
    world_t = tuple(obj.matrix_world.translation)

    _apply_transforms(obj)
    _select_only(node_name)
    # Force ASCII (.usda) so we can post-process texture paths
    _export_usd(dst_dir / "mesh.usda", selected_only=True, materials=True)
    manifest.append({"folder": folder, "node": node_name, "world_translate": world_t})

# Full-scene export
bpy.ops.object.select_all(action='SELECT')
_export_usd(OUT / "full_scene.usd", selected_only=False, materials=True)

import json
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"[done] manifest -> {OUT / 'manifest.json'}")
