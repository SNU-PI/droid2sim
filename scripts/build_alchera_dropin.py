"""Convert copied scene.usda into an Alchera drop-in variant.

- Redirects non-sponge/non-pan payloads to ../pan_clean/assets/...
- Swaps sponge and frying_pan payloads to local alchera .usd
- Rescales sponge/pan Xform to (1,1,1) since alchera meshes are real-scale
- Rewrites collision `over` statements to match alchera mesh prim paths
"""
from pathlib import Path
import re

SRC = Path("/home/parkprogrammer/jehyun/droid2sim/data/PolaRiS-Hub/pan_clean_alchera/scene.usda")
text = SRC.read_text()

# Alchera USD internal mesh paths (from bounding-box probe)
ALCHERA_MESH_PATH = {
    "sponge":     ("Cube_001",  "Cube_001"),
    "frying_pan": ("Cylinder",  "Cylinder"),
}

# --- 1. redirect non-alchera payloads to original pan_clean ---
non_alchera = ["g60_stovetop_zed", "mustard", "ketchup",
               "bell_pepper", "sushi", "coke", "blue_mug"]
for name in non_alchera:
    text = text.replace(
        f"@./assets/{name}/mesh.usdz@",
        f"@../pan_clean/assets/{name}/mesh.usdz@",
    )

# --- 2. swap alchera payloads (sponge + frying_pan) to local .usd ---
text = text.replace("@./assets/frying_pan/mesh.usdz@", "@./assets/frying_pan/mesh.usd@")
text = text.replace("@./assets/sponge/mesh.usdz@",     "@./assets/sponge/mesh.usd@")

# --- 3. rescale sponge/pan Xform to (1,1,1) ---
# pan block: replace scale (0.35,0.35,0.35) -> (1,1,1)
text = text.replace("float3 xformOp:scale = (0.35, 0.35, 0.35)",
                    "float3 xformOp:scale = (1, 1, 1)", 1)
# sponge block: match a specific scale line (there are multiple scale lines; only sponge's is (0.09,0.07,0.09))
text = text.replace("float3 xformOp:scale = (0.09, 0.07, 0.09)",
                    "float3 xformOp:scale = (1, 1, 1)", 1)

# --- 4. Fix collision over-statements for sponge/pan ---
# PolaRiS had `over "mesh" { over "mesh" ( apis... ) { ... } }`
# For alchera we need `over "root" { over "<xform>" { over "<mesh>" ( apis... ) { ... } } }`
# Simplest: rewrite the mesh-collision block for each.
def replace_over_mesh(text, obj_name, xform_name, mesh_name):
    # Match the entire outer 'over "mesh"' block and replace prim names.
    # The pattern is fragile w.r.t. exact whitespace — leverage that all such
    # blocks are formatted uniformly in scene.usda.
    pattern = re.compile(
        r'(        over "mesh"\n'
        r'        \{\n'
        r'            over "mesh" \((?P<apis>[^\)]*?)\)\n'
        r'            \{\n'
        r'                uniform token physics:approximation = "(?P<approx>[^"]+)"\n'
        r'                bool physics:collisionEnabled = 1\n'
        r'            \}\n'
        r'        \})',
        re.DOTALL)

    def _sub(m):
        return (
            f'        over "root"\n'
            f'        {{\n'
            f'            over "{xform_name}"\n'
            f'            {{\n'
            f'                over "{mesh_name}" ({m.group("apis")})\n'
            f'                {{\n'
            f'                    uniform token physics:approximation = "{m.group("approx")}"\n'
            f'                    bool physics:collisionEnabled = 1\n'
            f'                }}\n'
            f'            }}\n'
            f'        }}'
        )
    # Only one such block per object; sub once.
    return pattern.sub(_sub, text, count=1)

# There are many `over "mesh"` blocks (one per object). We must limit
# the scope to just the pan and sponge sections. Trick: process by
# slicing the file per named block.

def rewrite_object_collision(text, def_name, xform_name, mesh_name):
    # locate `def "<def_name>"` header + first `}` closing the object block
    hdr = f'def "{def_name}"'
    i0 = text.find(hdr)
    if i0 < 0:
        return text
    # Find end of block: naive brace balancing scan
    depth = 0
    j = text.find("{", i0)
    end = None
    while j != -1:
        ch = text[j]
        if ch == '{': depth += 1
        elif ch == '}': depth -= 1
        if depth == 0:
            end = j + 1
            break
        j = text.find("}", j+1) if depth < 0 else min(
            [x for x in (text.find("{", j+1), text.find("}", j+1)) if x>=0]) if any(x>=0 for x in (text.find("{", j+1), text.find("}", j+1))) else -1
    if end is None:
        end = i0 + 5000  # fallback
    block = text[i0:end]
    new_block = replace_over_mesh(block, def_name, xform_name, mesh_name)
    return text[:i0] + new_block + text[end:]

text = rewrite_object_collision(text, "pan",    *ALCHERA_MESH_PATH["frying_pan"])
text = rewrite_object_collision(text, "sponge", *ALCHERA_MESH_PATH["sponge"])

SRC.write_text(text)
print(f"patched scene.usda -> {SRC}")
