"""Render one side-by-side GIF for each threshold-scene family.

Each GIF shows representative configurations on opposite sides of the
family's threshold.  Run from the repository root with, for example:

    MUJOCO_GL=egl PYTHONPATH=src python src/gen/make_gifs.py
"""

from pathlib import Path
import argparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

# Import the EGL bootstrap before MuJoCo chooses a rendering backend.
from gen.render import font as _font_shared
from core.threshold import FAMILIES


CASES = {
    "seesaw": (
        (dict(s1=0.026, d1=0.18, s2=0.022, d2=0.22), "LEFT HEAVIER"),
        (dict(s1=0.022, d1=0.18, s2=0.026, d2=0.22), "RIGHT HEAVIER"),
    ),
    "lean": (
        (dict(theta=0.68), "BELOW: SLIDES"),
        (dict(theta=0.90), "ABOVE: HOLDS"),
    ),
    "tower": (
        (dict(o1=0.020, o2=0.028), "BELOW: FALLS"),
        (dict(o1=0.005, o2=0.010), "ABOVE: STANDS"),
    ),
    "hill": (
        (dict(v0=0.65, h=0.055), "BELOW: RETURNS"),
        (dict(v0=1.05, h=0.055), "ABOVE: CROSSES"),
    ),
    "collide": (
        (dict(damp=180.0), "BELOW: CONTINUES"),
        (dict(damp=8.0), "ABOVE: BOUNCES"),
    ),
    "domino": (
        (dict(s=0.072), "BELOW: STOPS"),
        (dict(s=0.055), "ABOVE: PROPAGATES"),
    ),
}

# Keep the hill outcome on screen: the full rollout continues long enough for
# labelling, but after frame 49 both outcomes soon leave the camera frustum.
DISPLAY_FRAMES = {"hill": 50}



def _text_center(draw, xy, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    draw.text((xy[0] - width / 2, xy[1]), text, font=font, fill=fill)


def compose(name, left, right, left_label, right_label, left_margin, right_margin):
    title_h = 46
    gap = 4
    h, w = left.shape[:2]
    canvas = Image.new("RGB", (2 * w + gap, h + title_h), (25, 27, 31))
    canvas.paste(Image.fromarray(left), (0, title_h))
    canvas.paste(Image.fromarray(right), (w + gap, title_h))
    draw = ImageDraw.Draw(canvas)
    title_font = _font_shared(15)
    detail_font = _font_shared(11)
    _text_center(draw, (w / 2, 5), left_label, title_font, (242, 242, 242))
    _text_center(draw, (w + gap + w / 2, 5), right_label, title_font, (242, 242, 242))
    _text_center(draw, (w / 2, 25), f"margin {left_margin:+.3f}", detail_font, (190, 195, 205))
    _text_center(draw, (w + gap + w / 2, 25), f"margin {right_margin:+.3f}", detail_font, (190, 195, 205))
    draw.rectangle((w, title_h, w + gap - 1, h + title_h), fill=(25, 27, 31))
    return np.asarray(canvas)


def render_family(name, output_dir):
    scene = FAMILIES[name]()
    (left_p, left_label), (right_p, right_label) = CASES[name]
    left = scene.run(left_p, render=True)
    right = scene.run(right_p, render=True)
    display_frames = DISPLAY_FRAMES.get(name, len(left["frames"]))
    frames = [
        compose(
            name,
            a,
            b,
            left_label,
            right_label,
            left["margin"],
            right["margin"],
        )
        for a, b in zip(left["frames"][:display_frames], right["frames"][:display_frames])
    ]
    # Hold the endpoints briefly so the starting configuration and outcome are
    # both easy to inspect without slowing the simulated motion itself.
    frames = [frames[0]] * 8 + frames + [frames[-1]] * 16
    path = output_dir / f"{name}.gif"
    imageio.mimsave(path, frames, duration=40, loop=0, palettesize=128)
    print(
        f"{name:8s} -> {path} | "
        f"outcomes {left['outcome']}/{right['outcome']} | "
        f"margins {left['margin']:+.4f}/{right['margin']:+.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("out/threshold_gifs"))
    parser.add_argument("--families", nargs="*", choices=tuple(FAMILIES), default=tuple(FAMILIES))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.families:
        render_family(name, args.output_dir)


if __name__ == "__main__":
    main()
