"""What to sweep, and how each family is framed.

Pure data plus the analytic margin lookup: which parameter is varied, over
what range, which cameras stand in for the robot's primary and wrist views,
and the text prompt handed to the video model. Kept apart from the rendering
code so the experimental design can be read and edited on its own.
"""

from __future__ import annotations

import numpy as np

from core.threshold import FAMILIES, Seesaw, Lean, Tower, Hill

N_POINTS = 7
SIM_FPS = 50
FUTURE_OFFSET = 16  # 16 * 0.02 s = 0.32 s
DYNAMIC_CONDITION_INDICES = (0, 2, 4, 6, 8)

STATIC = ("seesaw", "lean", "tower")
DYNAMIC = ("hill", "collide", "domino")

SWEPT_KEY = {"seesaw": "d2", "lean": "theta", "tower": "o2",
             "hill": "v0", "collide": "damp", "domino": "s"}

DOMINO_THRESHOLD = 0.064   # measured by bisection; no closed form

PRIMARY_CAM = {
    "seesaw": "close",
    "lean": "side",
    "tower": "close",
    "hill": "side",
    "collide": "side",
    "domino": "close",
}

WRIST_CAM = {
    "seesaw": "side",
    "lean": "close",
    "tower": "side",
    "hill": "close",
    # The close camera points above the floor in this scene, so retain a
    # second visible observation instead of feeding an empty wrist image.
    "collide": "side",
    "domino": "side",
}

PROMPTS = {
    "seesaw": (
        "A fixed camera observes two red cubes resting at different distances on a hinged seesaw. "
        "Gravity acts naturally and the seesaw moves according to physical balance."
    ),
    "lean": (
        "A fixed camera observes a red rod leaning between a vertical wall and a tabletop. "
        "The rod moves naturally under gravity and contact friction."
    ),
    "tower": (
        "A fixed camera observes a stack of three colored cubes. "
        "The blocks move naturally under gravity and contact."
    ),
    "hill": (
        "A fixed camera observes a red ball rolling from left to right toward a smooth gray hill. "
        "The ball continues according to momentum, rolling contact, and gravity."
    ),
    "collide": (
        "A fixed camera observes a red ball moving right toward a larger blue ball at rest on a smooth surface. "
        "The balls collide and continue with realistic contact physics."
    ),
    "domino": (
        "A fixed camera observes five dominoes in a line. The leftmost domino is falling to the right, "
        "and the dominoes continue with realistic contact physics."
    ),
}


def sweep_specs() -> dict[str, list[dict[str, float]]]:
    """Return one-dimensional, visibility-preserving sweeps around thresholds."""
    seesaw_d2 = np.linspace(0.19, 0.27, N_POINTS)
    lean_theta = np.linspace(0.65, 0.92, N_POINTS)
    tower_o2 = np.linspace(0.015, 0.045, N_POINTS)
    hill_v0 = np.linspace(0.65, 1.10, N_POINTS)
    collide_damp = np.geomspace(8.0, 240.0, N_POINTS)
    domino_spacing = np.linspace(0.048, 0.080, N_POINTS)
    return {
        "seesaw": [dict(s1=0.025, d1=0.18, s2=0.023, d2=float(x)) for x in seesaw_d2],
        "lean": [dict(theta=float(x)) for x in lean_theta],
        "tower": [dict(o1=0.010, o2=float(x)) for x in tower_o2],
        "hill": [dict(v0=float(x), h=0.055) for x in hill_v0],
        "collide": [dict(damp=float(x)) for x in collide_damp],
        "domino": [dict(s=float(x)) for x in domino_spacing],
    }


def sweep_value(family: str, params: dict[str, float]) -> float:
    return params[SWEPT_KEY[family]]


def expected_margin(family: str, params: dict[str, float]) -> float | None:
    """Analytic margin where available; collide is identified by simulation."""
    if family == "seesaw":
        return Seesaw.margin_of(params)
    if family == "lean":
        return Lean.margin_of(params)
    if family == "tower":
        return Tower.margin_of(params)
    if family == "hill":
        return Hill.margin_of(params)
    if family == "domino":
        return DOMINO_THRESHOLD - params["s"]
    return None
