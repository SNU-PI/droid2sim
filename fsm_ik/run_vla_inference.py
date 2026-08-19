#!/usr/bin/env python3
"""Launch rendered, closed-loop SmolVLA inference inside the Isaac environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_GEN_SCRIPT = PROJECT_ROOT / "run_data_gen.py"


def main() -> None:
    default_arguments = [
        str(DATA_GEN_SCRIPT),
        "--vla-inference",
        "--headless",
        "--enable_cameras",
        "--width",
        "256",
        "--height",
        "256",
        "--camera-fps",
        "10",
        "--physics-dt",
        "0.01",
        "--validation-index",
        "0",
        "--output-dir",
        str(PROJECT_ROOT / "outputs" / "vla_inference"),
        "--task-description",
        "Pick up the sponge and place it in the frying pan.",
    ]
    # User arguments come last, so argparse accepts explicit overrides such as
    # --validation-index, --max-runtime, --policy-port, or --output-dir.
    os.execv(sys.executable, [sys.executable, *default_arguments, *sys.argv[1:]])


if __name__ == "__main__":
    main()
