"""Shared experimental design for the energy-conservation scene family."""

from __future__ import annotations

import numpy as np

from core.threshold import ENERGY_FAMILIES


PRINCIPLE = "energy_conservation"
NORMALIZED_MARGINS = np.asarray(
    (-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30),
    dtype=np.float64,
)
CONDITION_INDICES = (0, 1, 2, 3, 4)
FUTURE_OFFSET = 5              # 5 / 16 s = 0.3125 s
CAMERA = {"hill": "side", "ramp": "side", "pendulum": "pendulum_side"}

PROMPTS = {
    "hill": (
        "A fixed camera observes a red bead moving from left to right along a smooth grey hill. "
        "The bead continues under gravity while remaining on the track and keeping its shape."
    ),
    "ramp": (
        "A fixed camera observes a red slider moving up a straight grey ramp toward a blue gate. "
        "The slider continues under gravity while remaining on the ramp and keeping its shape."
    ),
    "pendulum": (
        "A fixed camera observes a red pendulum bob moving from its lowest point. "
        "The rigid pendulum continues under gravity while keeping its shape and arm length."
    ),
}


def energy_specs():
    """Return matched configurations on a shared dimensionless margin axis."""
    return {
        name: [scene_cls.params_for_margin(float(m)) for m in NORMALIZED_MARGINS]
        for name, scene_cls in ENERGY_FAMILIES.items()
    }
