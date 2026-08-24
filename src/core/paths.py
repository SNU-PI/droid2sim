"""Project paths in one place, all relative to the repo root.

The repo keeps large artefacts under data/, which is a symlink onto the RAID
(home quota is 512 GiB). Every default below is therefore relative; an
environment variable overrides it for machines with a different layout.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _p(env, rel):
    return Path(os.environ.get(env, ROOT / rel))


VJEPA_DIR = _p("VJEPA_DIR", "data/stage0/vjepa2")
VJEPA_CKPT = _p("VJEPA_CKPT", "data/stage0/vjepa2-ac-vitg.pt")
EP_DIR = _p("EP_DIR", "data/episodes")
FEAT_DIR = _p("FEAT_DIR", "data/features")
DROID_EPS = _p("DROID_EPS", "data/droid/eps")
MENAGERIE_PANDA = _p("MENAGERIE_PANDA",
                     "data/stage0/mujoco_menagerie/franka_emika_panda")
OUT_DIR = _p("OUT_DIR", "out")
