"""Render paired camera observations for Cosmos-Policy OOD pixel tests."""

import argparse
import json
from pathlib import Path

import numpy as np

# Import the EGL bootstrap before make_sweep/core.threshold imports MuJoCo.
from gen.render import SWEEP_CFG, roll, write_preview
from core.threshold import Hill, Tower
from gen.make_sweep import neutral_proprio


CASES = {
    "tower_falls": (Tower, dict(o1=0.020, o2=0.028), 0, 16),
    "tower_stands": (Tower, dict(o1=0.005, o2=0.010), 0, 16),
    "hill_returns": (Hill, dict(v0=0.65, h=0.055), 8, 24),
    "hill_crosses": (Hill, dict(v0=1.05, h=0.055), 8, 24),
}


def render_view(scene_cls, params, camera):
    frames, labels = roll(scene_cls, params, camera, SWEEP_CFG)
    return {"frames": frames, **labels}


def preview(path, arrays):
    write_preview(path, list(arrays.values()), list(arrays))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cosmos_policy/inputs"))
    parser.add_argument("--cases", nargs="*", choices=tuple(CASES), default=tuple(CASES))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    proprio = neutral_proprio(args.stats)

    for name in args.cases:
        scene_cls, params, current_idx, future_idx = CASES[name]
        primary = render_view(scene_cls, params, "close")
        wrist = render_view(scene_cls, params, "side")
        current_primary = primary["frames"][current_idx]
        current_wrist = wrist["frames"][current_idx]
        gt_future_primary = primary["frames"][future_idx]
        gt_future_wrist = wrist["frames"][future_idx]
        np.savez_compressed(
            args.output_dir / f"{name}.npz",
            name=name,
            primary_image=current_primary,
            wrist_image=current_wrist,
            proprio=proprio,
            gt_future_primary=gt_future_primary,
            gt_future_wrist=gt_future_wrist,
            margin=primary["margin"],
            outcome=primary["outcome"],
        )
        preview(
            args.output_dir / f"{name}.png",
            {
                "current_primary": current_primary,
                "current_wrist": current_wrist,
                "gt_t+16_primary": gt_future_primary,
            },
        )
        print(name, f"margin={primary['margin']:+.4f}", f"outcome={primary['outcome']}")


if __name__ == "__main__":
    main()
