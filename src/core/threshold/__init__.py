"""Threshold scenes: three static, three dynamic.

Static scenes carry their label in the configuration, so a single settled
frame is in principle enough to read the outcome:

    seesaw   hinged plank, two cubes        net torque sign
    lean     rod against a wall             tan(theta) vs 1/(2 mu)
    tower    three-cube stack               partial centre of mass over support

Dynamic scenes need motion; the outcome is a discrete event:

    hill     ball at a ridge                (7/10) v0^2 vs g h, mass cancels
    collide  two balls, one moving          restitution vs mass ratio
    domino   five dominoes, spacing swept   chain propagates or dies

Every scene reports a continuous margin (signed distance to its threshold), a
binary outcome from simulation, and the frame at which the outcome starts to
become visible, so probes can be restricted to strictly pre-outcome frames.
"""

import numpy as np

from core.threshold.base import Base, G, TABLE_Z, IMG, CTRL_DT
from core.threshold.seesaw import Seesaw
from core.threshold.lean import Lean
from core.threshold.tower import Tower
from core.threshold.hill import Hill
from core.threshold.energy_hill import EnergyHill
from core.threshold.ramp import Ramp
from core.threshold.pendulum import Pendulum
from core.threshold.collide import Collide
from core.threshold.domino import Domino

FAMILIES = {"seesaw": Seesaw, "lean": Lean, "tower": Tower,
            "hill": Hill, "collide": Collide, "domino": Domino}

STATIC = ("seesaw", "lean", "tower")
DYNAMIC = ("hill", "collide", "domino")

# Kept separate from FAMILIES so the original six-scene sweep and its stored
# artifacts remain stable.  These scenes share a dimensionless energy margin.
ENERGY_FAMILIES = {"hill": EnergyHill, "ramp": Ramp, "pendulum": Pendulum}


def sample_params(family, rng):
    """Draw one episode's parameters, spread across the family's threshold."""
    return FAMILIES[family].sample(rng)


__all__ = ["FAMILIES", "STATIC", "DYNAMIC", "sample_params", "Base",
           "Seesaw", "Lean", "Tower", "Hill", "Collide", "Domino",
           "EnergyHill", "Ramp", "Pendulum", "ENERGY_FAMILIES",
           "G", "TABLE_Z", "IMG", "CTRL_DT"]
