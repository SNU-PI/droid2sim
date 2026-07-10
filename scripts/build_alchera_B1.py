from pathlib import Path
import re

SRC = Path("/home/parkprogrammer/jehyun/droid2sim/data/PolaRiS-Hub/pan_clean/scene.usda")
DST = Path("/home/parkprogrammer/jehyun/droid2sim/data/PolaRiS-Hub/pan_clean_alcheraB1/scene.usda")

ALCHERA = [
    ("pan",     "frying_pan",  "Cylinder"),
    ("sponge",  "sponge",      "Cube_001"),
    ("mustard", "mustard",     "Cylinder_011"),
    ("ketchup", "ketchup",     "Cylinder_003"),
    ("pepper",  "bell_pepper", "mesh_010"),
    ("sushi",   "sushi",       "mesh_012"),
    ("coke",    "coke",        "Cylinder_005"),
    ("mug",     "mug",         "Cylinder_002"),
]
BG = ["g60_stovetop_zed"]

def block_span(t, name):
    i = t.find(f'def "{name}"')
    if i < 0: raise KeyError(name)
    j = t.find("{", i); d = 0
    while j < len(t):
        d += (t[j] == "{") - (t[j] == "}")
        if d == 0: return i, j + 1
        j += 1
    raise ValueError(f"unbalanced braces in block {name!r}")

def patch(block, folder, prim):
    block = re.sub(r"prepend payload = @[^@]*@",
                   f"prepend payload = @./assets/{folder}/mesh.usda@", block, count=1)
    block = re.sub(r"float3 xformOp:scale = \([^)]+\)",
                   "float3 xformOp:scale = (1, 1, 1)", block)
    block = re.sub(r"bool physics:kinematicEnabled = 0",
                   "bool physics:kinematicEnabled = 1", block)
    return re.sub(
        r'over "mesh"\s*\{\s*over "mesh" \((?P<apis>[^)]*?)\)\s*\{\s*'
        r'uniform token physics:approximation = "(?P<approx>[^"]+)"\s*'
        r'bool physics:collisionEnabled = 1[\s\S]*?\}\s*\}',
        lambda m: (
            f'over "root"\n'
            f'        {{\n'
            f'            over "{prim}"\n'
            f'            {{\n'
            f'                over "{prim}" ({m["apis"]})\n'
            f'                {{\n'
            f'                    uniform token physics:approximation = "{m["approx"]}"\n'
            f'                    bool physics:collisionEnabled = 1\n'
            f'                }}\n'
            f'            }}\n'
            f'        }}'
        ),
        block, count=1,
    )

DST.parent.mkdir(parents=True, exist_ok=True)
text = SRC.read_text()

for n in BG:
    text = text.replace(f"@./assets/{n}/mesh.usdz@", f"@../pan_clean/assets/{n}/mesh.usdz@")

for name, folder, prim in ALCHERA:
    a, b = block_span(text, name)
    text = text[:a] + patch(text[a:b], folder, prim) + text[b:]

DST.write_text(text)
print(f"wrote {DST} ({len(text.splitlines())} lines)")
