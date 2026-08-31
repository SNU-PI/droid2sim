"""Export separate LAST OBSERVED, PHYSICS GT, and COSMOS GIFs.

No labels, borders, or composite canvas are written into the GIFs.  Each
triplet uses six equally timed frames and ends at the shared evaluation
horizon: +0.32 s for MuJoCo and +0.31 s for Cosmos Predict2 V2W.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")

from core.threshold import FAMILIES  # noqa: E402
from gen.sweep_spec import PRIMARY_CAM  # noqa: E402


SELECTED = {
    "seesaw": "seesaw_06",      # GT: tips right
    "lean": "lean_00",          # GT: slides
    "tower": "tower_06",        # GT: falls
    "hill": "hill_06",          # GT: crosses
    "collide": "collide_00",    # GT: bounces
    "domino": "domino_00",      # GT: propagates
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=Path("artifacts/physics_sweep"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/physics_sweep/analysis/triplet_gifs"),
    )
    return parser.parse_args()


def square(frame, size=256):
    image = Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")
    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return np.asarray(
        image.crop((left, top, left + side, top + side)).resize(
            (size, size), Image.Resampling.LANCZOS
        )
    )


def find_rollout(root: Path, sample_id: str):
    v2w_root = root / "cosmos_v2w"
    matches = []
    if (v2w_root / sample_id / "rollout.mp4").exists():
        matches.append(v2w_root / sample_id / "rollout.mp4")
    matches.extend(v2w_root.glob(f"*/{sample_id}/rollout.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Cosmos rollout for {sample_id}, got {matches}")
    return matches[0]


def save_gif(path: Path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Six frames x 450 ms = 2.7 s.  The intentionally slow playback makes the
    # short, shared physics horizon readable without inserting blank frames.
    pil_frames = []
    for index, frame in enumerate(frames):
        array = np.asarray(frame).astype(np.uint8).copy()
        if index % 2:
            # An imperceptible one-level shift makes every GIF frame a full
            # frame.  This prevents some preview renderers from briefly
            # exposing an unpainted background between delta frames.
            array = np.where(array < 255, array + 1, array - 1).astype(np.uint8)
        pil_frames.append(Image.fromarray(array).convert("RGB"))
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=450,
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

    for family, sample_id in SELECTED.items():
        record = records[sample_id]
        scene = FAMILIES[family]()
        scene.cam = PRIMARY_CAM[family]
        rollout = scene.run(record["params"], render=True)
        if scene._r is not None:
            scene._r.close()

        current_idx = int(record["current_idx"])
        future_idx = int(record["future_idx"])
        gt_indices = np.rint(np.linspace(current_idx, future_idx, 6)).astype(int)
        gt_frames = [square(rollout["frames"][index]) for index in gt_indices]
        last_frames = [gt_frames[0].copy() for _ in range(6)]

        cosmos_all = imageio.mimread(find_rollout(root, sample_id), memtest=False)
        conditioning = int(record["condition_pixel_frames"])
        # Last conditioning frame plus five generated frames.
        cosmos_indices = range(conditioning - 1, conditioning + 5)
        cosmos_frames = [square(cosmos_all[index]) for index in cosmos_indices]

        family_dir = output / family
        save_gif(family_dir / "last.gif", last_frames)
        save_gif(family_dir / "physics_gt.gif", gt_frames)
        save_gif(family_dir / "cosmos.gif", cosmos_frames)
        print(f"{family}: {sample_id} -> {family_dir}")


if __name__ == "__main__":
    main()
