"""Render a two-row LAST OBSERVED / PHYSICS GT / COSMOS comparison GIF."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("MUJOCO_GL", "egl")

from core.threshold import Hill, Tower  # noqa: E402


ROWS = (
    {"family": "tower", "sample_id": "tower_06", "seed": 4, "title": "Tower — GT: FALLS"},
    {"family": "hill", "sample_id": "hill_06", "seed": 1, "title": "Hill — GT: CROSSES"},
)
HEADERS = ("LAST OBSERVED", "PHYSICS GT", "COSMOS")


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size)


def card(image, size, radius=14):
    image = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    image = image.resize(size, Image.Resampling.LANCZOS)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    background = Image.new("RGB", size, (54, 56, 60))
    background.paste(image, (0, 0), mask)
    border = ImageDraw.Draw(background)
    border.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, outline=(65, 67, 71), width=2)
    return background


def simulate(record):
    scene = Tower() if record["family"] == "tower" else Hill()
    scene.cam = "close" if record["family"] == "tower" else "side"
    scene.capture_dt = 1 / 16
    scene.render_height = 480
    scene.render_width = 832
    rollout = scene.run(record["params"], render=True)
    if scene._r is not None:
        scene._r.close()
    return rollout["frames"]


def build_row(root, records, spec):
    record = records[spec["sample_id"]]
    gt_all = simulate(record)
    start = int(record["current_idx"])
    cosmos_path = (
        root / "rollouts" / "base" / spec["family"] / spec["sample_id"]
        / f"seed_{spec['seed']:02d}" / "rollout.mp4"
    )
    cosmos_all = imageio.mimread(cosmos_path, memtest=False)
    cosmos_start = int(record["condition_pixel_frames"]) - 1
    count = min(17, len(gt_all) - start, len(cosmos_all) - cosmos_start)
    gt = [gt_all[start + index] for index in range(count)]
    cosmos = [cosmos_all[cosmos_start + index] for index in range(count)]
    last = [gt[0]] * count
    return {"title": spec["title"], "last": last, "gt": gt, "cosmos": cosmos, "count": count}


def compose(rows, frame_index):
    width, height = 1500, 980
    canvas = Image.new("RGB", (width, height), (23, 23, 23))
    draw = ImageDraw.Draw(canvas)
    title_font, header_font = font(30, True), font(23, True)
    card_w, card_h = 430, 248
    xs = (50, 535, 1020)
    row_tops = (38, 515)
    for row, top in zip(rows, row_tops):
        draw.text((50, top), row["title"], fill=(224, 224, 226), font=title_font)
        header_y = top + 72
        for x, header in zip(xs, HEADERS):
            draw.text((x, header_y), header, fill=(220, 220, 222), font=header_font)
        divider_y = header_y + 45
        draw.line((50, divider_y, 1450, divider_y), fill=(66, 66, 68), width=2)
        card_y = divider_y + 38
        images = (row["last"][frame_index], row["gt"][frame_index], row["cosmos"][frame_index])
        for x, source in zip(xs, images):
            canvas.paste(card(source, (card_w, card_h)), (x, card_y))
    return np.asarray(canvas)


def save_full_frame_gif(path, frames, duration_ms=250):
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, frame in enumerate(frames):
        array = np.asarray(frame, dtype=np.uint8).copy()
        # Defeat delta-frame optimization in previewers that flash black between frames.
        if index % 2:
            array = np.where(array < 255, array + 1, array - 1).astype(np.uint8)
        prepared.append(Image.fromarray(array).convert("RGB"))
    prepared[0].save(
        path,
        save_all=True,
        append_images=prepared[1:],
        duration=duration_ms,
        loop=0,
        disposal=1,
        optimize=False,
    )


def save_row_triplets(output_dir, rows):
    """Write raw, separately embeddable GIFs with no surrounding page UI."""
    for row, spec in zip(rows, ROWS):
        row_dir = output_dir / spec["family"]
        for key, filename in (
            ("last", "last_observed.gif"),
            ("gt", "physics_gt.gif"),
            ("cosmos", "cosmos.gif"),
        ):
            frames = [
                np.asarray(
                    Image.fromarray(np.asarray(frame).astype(np.uint8)).resize(
                        (416, 240), Image.Resampling.LANCZOS
                    )
                )
                for frame in row[key]
            ]
            save_full_frame_gif(row_dir / filename, frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/physics_diagnostics_3h"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "analysis" / "comparison_gifs" / "tower_hill_triptych.gif"
    records = {
        row["id"]: row
        for row in (
            json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()
        )
    }
    rows = [build_row(root, records, spec) for spec in ROWS]
    count = min(row["count"] for row in rows)
    frames = [compose(rows, index) for index in range(count)]
    save_full_frame_gif(output.resolve(), frames)
    save_row_triplets(output.resolve().parent / "separate", rows)
    Image.fromarray(frames[-1]).save(output.resolve().with_suffix(".png"))
    print(json.dumps({
        "output": str(output.resolve()),
        "separate_output": str((output.resolve().parent / "separate")),
        "frames": count,
        "duration_seconds": count * 0.25,
    }, indent=2))


if __name__ == "__main__":
    main()
