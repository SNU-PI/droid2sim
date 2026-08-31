"""Render human-readable Physics GT vs Cosmos V2W comparison GIFs.

The original sweep GIFs are useful for checking all seven samples at once,
but their 112-pixel-high, seven-column layout is not a presentation view.
This script selects one clear example from either side of each threshold and
renders two rows with explicit LAST OBSERVED, PHYSICS GT, and COSMOS V2W
columns.  The V2W column is the only animated/generated column.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence


FAMILIES = ("seesaw", "lean", "tower", "hill", "collide", "domino")

# Use the extremes of each sweep so the physical outcomes are visually clear.
SELECTED = {
    "seesaw": ("seesaw_00", "seesaw_06"),
    "lean": ("lean_00", "lean_06"),
    "tower": ("tower_06", "tower_00"),
    "hill": ("hill_00", "hill_06"),
    "collide": ("collide_06", "collide_00"),
    "domino": ("domino_06", "domino_00"),
}

OUTCOME_NAMES = {
    "seesaw": {0: "TIPS LEFT", 1: "TIPS RIGHT"},
    "lean": {0: "SLIDES", 1: "HOLDS"},
    "tower": {0: "FALLS", 1: "STANDS"},
    "hill": {0: "RETURNS", 1: "CROSSES"},
    "collide": {0: "CONTINUES", 1: "BOUNCES"},
    "domino": {0: "STOPS", 1: "PROPAGATES"},
}

FAMILY_TITLES = {
    "seesaw": "SEESAW TORQUE THRESHOLD",
    "lean": "LEANING ROD FRICTION THRESHOLD",
    "tower": "BLOCK TOWER STABILITY THRESHOLD",
    "hill": "HILL-CROSSING ENERGY THRESHOLD",
    "collide": "COLLISION BOUNCE THRESHOLD",
    "domino": "DOMINO PROPAGATION THRESHOLD",
}

BG = (18, 20, 24)
PANEL_BG = (30, 33, 39)
TEXT = (244, 246, 250)
MUTED = (174, 181, 194)
GT = (72, 196, 255)
GENERATED = (239, 91, 166)
CORRECT = (72, 205, 133)
WRONG = (255, 96, 96)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=Path("artifacts/physics_sweep"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/physics_sweep/analysis/intuitive_v2w"),
    )
    return parser.parse_args()


def font(size: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def centered(draw, center_x, y, text, *, fill, text_font):
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((center_x - (box[2] - box[0]) / 2, y), text, fill=fill, font=text_font)


def fit_square(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.Resampling.LANCZOS
    )


def read_manifest(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["id"]: row for row in rows}


def read_predictions(path: Path):
    with path.open(newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def read_v2w_cells(path: Path):
    """Recover the seven V2W videos from the legacy seven-column GIF."""
    gif = Image.open(path)
    columns = [[] for _ in range(7)]
    durations = []
    for frame in ImageSequence.Iterator(gif):
        rgb = frame.convert("RGB")
        durations.append(int(frame.info.get("duration", 63)))
        for col in range(7):
            # Legacy layout: 7 x (192 x 112), with a 42-pixel header.
            cell = rgb.crop((col * 192, 42, (col + 1) * 192, 154))
            columns[col].append(cell)
    return columns, durations


def draw_panel(canvas, image, x, y, size, border):
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x - 2, y - 2, x + size + 1, y + size + 1), outline=border, width=3)


def compose_frame(family, rows, cosmos_frames, frame_index, root):
    panel = 270
    label_w = 190
    gap = 18
    x_positions = [label_w, label_w + panel + gap, label_w + 2 * (panel + gap)]
    top = 116
    row_pitch = 352
    width = label_w + 3 * panel + 2 * gap + 24
    height = top + 2 * row_pitch + 18
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    centered(draw, width / 2, 12, FAMILY_TITLES[family], fill=TEXT, text_font=font(24))
    centered(
        draw,
        width / 2,
        44,
        "Only the magenta column is model-generated; cyan is simulator ground truth",
        fill=MUTED,
        text_font=font(14),
    )

    headers = (
        ("LAST OBSERVED", MUTED),
        ("PHYSICS GT  +0.32s", GT),
        ("COSMOS PREDICT2 V2W  (ANIMATED)", GENERATED),
    )
    for x, (label, color) in zip(x_positions, headers):
        centered(draw, x + panel / 2, 78, label, fill=color, text_font=font(15))

    for row_index, row in enumerate(rows):
        record, prediction, current, gt = row
        y = top + row_index * row_pitch
        outcome = int(record["outcome"])
        predicted = int(prediction["v2w_endpoint_prediction"])
        truth_name = OUTCOME_NAMES[family][outcome]
        predicted_name = OUTCOME_NAMES[family][predicted]
        correct = predicted == outcome
        conditioning = int(record["condition_pixel_frames"])
        phase_generated = frame_index >= conditioning

        draw.rounded_rectangle((10, y - 2, label_w - 18, y + panel + 2), radius=12, fill=PANEL_BG)
        draw.text((27, y + 28), "GROUND TRUTH", fill=GT, font=font(12))
        outcome_font = font(19 if len(truth_name) > 10 else 23)
        predicted_font = font(17 if len(predicted_name) > 10 else 19)
        draw.text((27, y + 53), truth_name, fill=TEXT, font=outcome_font)
        draw.text((27, y + 101), f"margin {float(record['margin']):+.3f}", fill=MUTED, font=font(13))
        draw.text((27, y + 126), record["id"], fill=MUTED, font=font(12))
        readout_color = CORRECT if correct else WRONG
        draw.text((27, y + 179), "V2W @ +0.31s", fill=GENERATED, font=font(12))
        draw.text((27, y + 204), predicted_name, fill=readout_color, font=predicted_font)
        draw.text((27, y + 237), "CORRECT" if correct else "WRONG", fill=readout_color, font=font(13))

        draw_panel(canvas, current, x_positions[0], y, panel, MUTED)
        draw_panel(canvas, gt, x_positions[1], y, panel, GT)
        cosmos = fit_square(cosmos_frames[row_index][frame_index], panel)
        draw_panel(canvas, cosmos, x_positions[2], y, panel, GENERATED if phase_generated else MUTED)

        phase = "GENERATED" if phase_generated else "CONDITIONING"
        phase_color = GENERATED if phase_generated else MUTED
        tag = f"{phase}   frame {frame_index:02d}"
        box = draw.textbbox((0, 0), tag, font=font(12))
        tag_w = box[2] - box[0] + 20
        tx = x_positions[2] + panel - tag_w - 8
        ty = y + 9
        draw.rounded_rectangle((tx, ty, tx + tag_w, ty + 25), radius=7, fill=(18, 20, 24))
        draw.text((tx + 10, ty + 5), tag, fill=phase_color, font=font(12))

        verdict = f"ENDPOINT:  GT {truth_name}   |   V2W {predicted_name}   {'OK' if correct else 'X'}"
        centered(
            draw,
            x_positions[2] + panel / 2,
            y + panel + 12,
            verdict,
            fill=readout_color,
            text_font=font(13),
        )

    return canvas


def render_family(family, root, output_dir, manifest, predictions):
    selected_ids = SELECTED[family]
    legacy_columns, durations = read_v2w_cells(
        root / "analysis" / "v2w_rollouts" / f"{family}.gif"
    )
    rows = []
    videos = []
    for sample_id in selected_ids:
        record = manifest[sample_id]
        with np.load(root / record["input_npz"], allow_pickle=False) as data:
            current = fit_square(Image.fromarray(data["primary_image"]), 270)
            gt = fit_square(Image.fromarray(data["gt_future_primary"]), 270)
        rows.append((record, predictions[sample_id], current, gt))
        videos.append(legacy_columns[int(record["sweep_index"])])

    # Evaluate exactly five generated frames after the last conditioning
    # frame: index 5 for static inputs, index 9 for five-frame dynamic inputs.
    # Do not continue to the unrelated 1.25 s final frame in this comparison.
    endpoint_frames = int(rows[0][0]["condition_pixel_frames"]) + 5
    n_frames = min(endpoint_frames, *(len(video) for video in videos))
    frames = [compose_frame(family, rows, videos, index, root) for index in range(n_frames)]
    # Let viewers inspect both the observed starting point and the generated endpoint.
    frames = [frames[0]] * 8 + frames + [frames[-1]] * 18
    frame_durations = [80] * 8 + durations[:n_frames] + [100] * 18
    path = output_dir / f"{family}.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_durations,
        loop=0,
        optimize=False,
    )
    print(path)


def main():
    args = parse_args()
    root = args.sweep_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(root / "manifest.jsonl")
    predictions = read_predictions(root / "analysis" / "samples.csv")
    for family in FAMILIES:
        render_family(family, root, output_dir, manifest, predictions)


if __name__ == "__main__":
    main()
