"""Generate a native-480p matched-margin energy-conservation benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Select EGL before importing MuJoCo through the scene package.
from gen.render import DIAG_CFG, font, roll, write_video
from core.threshold import ENERGY_FAMILIES
from gen.energy_spec import (CAMERA, CONDITION_INDICES, FUTURE_OFFSET,
                             NORMALIZED_MARGINS, PRINCIPLE, PROMPTS,
                             energy_specs)


def family_preview(path: Path, records, root: Path):
    """Current / fixed-horizon GT / final GT across the common margin axis."""
    cell_w, cell_h, header, label_w = 260, 150, 42, 88
    canvas = Image.new(
        "RGB",
        (label_w + len(records) * cell_w, header + 3 * cell_h),
        (23, 26, 30),
    )
    draw = ImageDraw.Draw(canvas)
    fnt = font(12)
    for row, label in enumerate(("CURRENT", "GT +0.31s", "GT FINAL")):
        draw.text((8, header + row * cell_h + cell_h // 2 - 7), label,
                  fill="white", font=fnt)
    for col, record in enumerate(records):
        with np.load(root / record["input_npz"]) as data:
            images = (
                data["current_primary"],
                data["gt_future_primary"],
                data["gt_final_primary"],
            )
        x = label_w + col * cell_w
        title = f"m={record['normalized_margin']:+.2f}  y={record['outcome']}"
        draw.text((x + 8, 13), title, fill="white", font=fnt)
        for row, array in enumerate(images):
            image = Image.fromarray(array).resize(
                (cell_w, cell_h), Image.Resampling.BILINEAR
            )
            canvas.paste(image, (x, header + row * cell_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def build_sample(family, index, target_margin, params, root: Path):
    scene_cls = ENERGY_FAMILIES[family]
    frames, labels = roll(scene_cls, params, CAMERA[family], DIAG_CFG)
    current_idx = CONDITION_INDICES[-1]
    future_idx = current_idx + FUTURE_OFFSET
    condition = frames[list(CONDITION_INDICES)]

    measured_margin = float(labels["margin"])
    expected_outcome = int(target_margin > 0)
    if not np.isclose(measured_margin, target_margin, atol=1e-9):
        raise RuntimeError(
            f"{family}_{index:02d}: target/calculated margin mismatch "
            f"({target_margin} vs {measured_margin})"
        )
    if int(labels["outcome"]) != expected_outcome:
        raise RuntimeError(
            f"{family}_{index:02d}: analytic/simulation outcome mismatch"
        )
    if expected_outcome and int(labels["event_frame"]) <= current_idx:
        raise RuntimeError(
            f"{family}_{index:02d}: conditioning reaches the outcome event"
        )
    if future_idx >= len(frames):
        raise RuntimeError(f"{family}_{index:02d}: future frame out of range")

    sample_id = f"energy_{family}_{index:02d}"
    input_rel = Path("inputs") / f"{sample_id}.npz"
    condition_rel = Path("conditions") / f"{sample_id}.mp4"
    prompt_rel = Path("conditions") / f"{sample_id}.txt"
    np.savez_compressed(
        root / input_rel,
        name=sample_id,
        principle=PRINCIPLE,
        family=family,
        condition_primary=condition,
        current_primary=frames[current_idx],
        gt_future_primary=frames[future_idx],
        gt_final_primary=frames[-1],
        normalized_margin=measured_margin,
        outcome=int(labels["outcome"]),
        event_frame=int(labels["event_frame"]),
        condition_indices=np.asarray(CONDITION_INDICES),
        current_idx=current_idx,
        future_idx=future_idx,
        capture_fps=DIAG_CFG.fps,
    )
    write_video(root / condition_rel, condition, DIAG_CFG)
    (root / prompt_rel).write_text(PROMPTS[family])

    return {
        "id": sample_id,
        "principle": PRINCIPLE,
        "family": family,
        "sweep_index": index,
        "normalized_margin": measured_margin,
        "params": params,
        "outcome": int(labels["outcome"]),
        "event_frame": int(labels["event_frame"]),
        "condition_indices": list(CONDITION_INDICES),
        "current_idx": current_idx,
        "future_idx": future_idx,
        "capture_fps": DIAG_CFG.fps,
        "render_size": [DIAG_CFG.height, DIAG_CFG.width],
        "input_npz": str(input_rel),
        "condition_path": str(condition_rel),
        "prompt": PROMPTS[family],
        "prompt_file": str(prompt_rel),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/energy_conservation_sweep"),
    )
    parser.add_argument(
        "--families", nargs="*", choices=tuple(ENERGY_FAMILIES),
        default=tuple(ENERGY_FAMILIES),
    )
    args = parser.parse_args()
    root = args.output_dir.resolve()
    for sub in ("inputs", "conditions", "previews"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    specs = energy_specs()
    records = []
    for family in args.families:
        family_records = []
        for index, (margin, params) in enumerate(
            zip(NORMALIZED_MARGINS, specs[family])
        ):
            record = build_sample(family, index, float(margin), params, root)
            records.append(record)
            family_records.append(record)
            print(
                f"{record['id']}: m={record['normalized_margin']:+.2f} "
                f"y={record['outcome']} event={record['event_frame']}"
            )
        family_preview(root / "previews" / f"{family}.png", family_records, root)

    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    summary = {
        "principle": PRINCIPLE,
        "samples": len(records),
        "families": list(args.families),
        "normalized_margins": NORMALIZED_MARGINS.tolist(),
        "render_size": [DIAG_CFG.height, DIAG_CFG.width],
        "capture_fps": DIAG_CFG.fps,
        "conditioning_frames": len(CONDITION_INDICES),
        "evaluation_horizon_seconds": FUTURE_OFFSET / DIAG_CFG.fps,
        "manifest": str(manifest),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
