"""Render below/above-threshold GIFs for the energy scene family."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from gen.render import DIAG_CFG, font, roll
from core.threshold import ENERGY_FAMILIES
from gen.energy_spec import CAMERA


DISPLAY_FRAMES = {name: 12 for name in ENERGY_FAMILIES}


def compose(left, right, left_outcome, right_outcome, frame_index):
    width, height = 416, 240
    header = 42
    canvas = Image.new("RGB", (2 * width, height + header), (23, 26, 30))
    draw = ImageDraw.Draw(canvas)
    fnt = font(14)
    for col, (array, margin, outcome) in enumerate(
        ((left, -0.20, left_outcome), (right, 0.20, right_outcome))
    ):
        image = Image.fromarray(array).resize((width, height), Image.Resampling.BILINEAR)
        canvas.paste(image, (col * width, header))
        label = f"m={margin:+.2f}  GT={'SUCCESS' if outcome else 'FAILURE'}"
        draw.text((col * width + 12, 12), label, fill="white", font=fnt)
    draw.text((width - 42, 12), f"{frame_index / 16:.2f}s",
              fill=(190, 198, 210), font=fnt)
    return np.asarray(canvas)


def render_family(family, output_dir: Path):
    scene_cls = ENERGY_FAMILIES[family]
    below, below_labels = roll(
        scene_cls, scene_cls.params_for_margin(-0.20), CAMERA[family], DIAG_CFG
    )
    above, above_labels = roll(
        scene_cls, scene_cls.params_for_margin(0.20), CAMERA[family], DIAG_CFG
    )
    count = min(DISPLAY_FRAMES[family], len(below), len(above))
    frames = [
        compose(below[i], above[i], below_labels["outcome"],
                above_labels["outcome"], i)
        for i in range(count)
    ]
    path = output_dir / f"energy_{family}.gif"
    imageio.mimsave(path, frames, duration=90, loop=0, palettesize=128)
    print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/energy_conservation_gifs")
    )
    parser.add_argument(
        "--families", nargs="*", choices=tuple(ENERGY_FAMILIES),
        default=tuple(ENERGY_FAMILIES),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for family in args.families:
        render_family(family, args.output_dir)


if __name__ == "__main__":
    main()
