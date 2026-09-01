"""Render the exact visual prefix supplied to Cosmos for displayed samples.

Static threshold scenes are exported as an explicitly named single-frame GIF.
Dynamic threshold and energy scenes replay all five conditioning frames, making
the model input distinguishable from the frozen ``last.gif`` comparison card.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


THRESHOLD_SELECTED = {
    "seesaw": "seesaw_06",
    "lean": "lean_00",
    "tower": "tower_06",
    "hill": "hill_06",
    "collide": "collide_00",
    "domino": "domino_00",
}
ENERGY_SELECTED = {"below": 1, "above": 6}


def square(frame, size):
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return np.asarray(
        image.crop((left, top, left + side, top + side)).resize(
            (size, size), Image.Resampling.LANCZOS
        )
    )


def save_gif(path, frames, duration=450):
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for index, frame in enumerate(frames):
        array = np.asarray(frame, dtype=np.uint8).copy()
        if index % 2:
            array = np.where(array < 255, array + 1, array - 1).astype(np.uint8)
        images.append(Image.fromarray(array).convert("RGB"))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        disposal=1,
        optimize=False,
    )


def records(path):
    return {
        row["id"]: row
        for row in (
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    }


def render_threshold(root):
    manifest = records(root / "manifest.jsonl")
    output = root / "analysis" / "triplet_gifs"
    for family, sample_id in THRESHOLD_SELECTED.items():
        record = manifest[sample_id]
        with np.load(root / record["input_npz"], allow_pickle=False) as data:
            condition = np.asarray(data["condition_primary"])
        frames = [square(frame, 256) for frame in condition]
        if len(frames) == 1:
            filename = "observed_single_frame.gif"
            frames = frames * 5
        else:
            filename = f"observed_history_{len(frames)}frames.gif"
        save_gif(output / family / filename, frames)
        print(f"threshold/{family}: {len(condition)} input frame(s)")


def render_energy(root):
    manifest = records(root / "manifest.jsonl")
    output = root / "analysis" / "triplet_gifs"
    for family in ("hill", "ramp", "pendulum"):
        for side, index in ENERGY_SELECTED.items():
            sample_id = f"energy_{family}_{index:02d}"
            record = manifest[sample_id]
            with np.load(root / record["input_npz"], allow_pickle=False) as data:
                condition = np.asarray(data["condition_primary"])
            frames = [square(frame, 320) for frame in condition]
            save_gif(
                output / family / side / f"observed_history_{len(frames)}frames.gif",
                frames,
            )
            print(f"energy/{family}/{side}: {len(condition)} input frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    artifacts = args.artifacts_root.resolve()
    render_threshold(artifacts / "physics_sweep")
    render_energy(artifacts / "energy_conservation_sweep")


if __name__ == "__main__":
    main()
