"""Export direct Last / Physics GT / Cosmos GIFs for the energy sweep.

Each selected sample is written as three plain, equally timed GIFs with no
page wrapper or in-frame annotation.  The sequence starts at the last observed
frame and ends seven 16-FPS steps later, so GT and V2W share the same horizon.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

SELECTED = {
    "below": 1,  # normalized margin -0.20, GT failure
    "above": 6,  # normalized margin +0.20, GT success
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-root", type=Path, default=Path("artifacts/energy_conservation_sweep")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/energy_conservation_sweep/analysis/triplet_gifs"),
    )
    parser.add_argument(
        "--physics-gif-root",
        type=Path,
        default=Path("artifacts/energy_conservation_gifs"),
        help="Exact 12-frame below/above GT GIFs produced by make_energy_gifs.py.",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def square(frame, size=320):
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return np.asarray(
        image.crop((left, top, left + side, top + side)).resize(
            (size, size), Image.Resampling.LANCZOS
        )
    )


def save_gif(path: Path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for index, frame in enumerate(frames):
        array = np.asarray(frame, dtype=np.uint8).copy()
        # Avoid delta-frame disposal glitches in desktop/browser GIF viewers.
        if index % 2:
            array = np.where(array < 255, array + 1, array - 1).astype(np.uint8)
        images.append(Image.fromarray(array).convert("RGB"))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=300,
        loop=0,
        disposal=1,
        optimize=False,
    )


def main():
    args = parse_args()
    root = args.sweep_root.resolve()
    output = args.output_dir.resolve()
    records = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (root / "manifest.jsonl").read_text().splitlines()
            if line.strip()
        )
    }

    for family in ("hill", "ramp", "pendulum"):
        # make_energy_gifs.py stores the exact -0.20/+0.20 rollouts as two
        # 416x240 panels below a 42-pixel caption strip. Reusing those frames
        # avoids running the identical MuJoCo episodes a second time.
        gt_pair = imageio.mimread(
            args.physics_gif_root.resolve() / f"energy_{family}.gif",
            memtest=False,
        )
        for side, sweep_index in SELECTED.items():
            sample_id = f"energy_{family}_{sweep_index:02d}"
            record = records[sample_id]
            current_idx = int(record["current_idx"])
            panel = 0 if side == "below" else 1
            gt_frames = [
                square(frame[42:282, panel * 416:(panel + 1) * 416])
                for frame in gt_pair[current_idx:]
            ]
            last_frames = [gt_frames[0].copy() for _ in gt_frames]

            cosmos_path = (
                root / "cosmos_v2w" / f"seed_{args.seed:02d}" / family
                / sample_id / "rollout.mp4"
            )
            cosmos_all = imageio.mimread(cosmos_path, memtest=False)
            cosmos_frames = [
                square(cosmos_all[index])
                for index in range(current_idx, current_idx + len(gt_frames))
            ]

            sample_dir = output / family / side
            save_gif(sample_dir / "last.gif", last_frames)
            save_gif(sample_dir / "physics_gt.gif", gt_frames)
            save_gif(sample_dir / "cosmos.gif", cosmos_frames)
            print(
                f"{family}/{side}: {sample_id} "
                f"m={record['normalized_margin']:+.2f} y={record['outcome']}"
            )


if __name__ == "__main__":
    main()
