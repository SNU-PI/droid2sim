"""Generate the six-family threshold sweep for pixel world models.

One sweep feeds two deliberately different interfaces:

* Cosmos-Policy gets the current primary and wrist frames plus neutral LIBERO
  proprio, matching its robot-policy API.
* Cosmos Predict2 Video2World gets a single still for the static families or
  five pre-outcome frames for the dynamic ones, encoded as a 16 fps mp4.

Every sample also carries a MuJoCo future frame 0.32 s after the last
conditioning frame, so endpoint comparisons share one physical horizon.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Import the rendering bootstrap before MuJoCo.  mujoco.gl_context chooses its
# backend at import time, so setting MUJOCO_GL afterwards leaves GLFW selected.
from gen.render import SWEEP_CFG, font, roll, write_video
from core.threshold import FAMILIES
from gen.sweep_spec import (DYNAMIC, DYNAMIC_CONDITION_INDICES, FUTURE_OFFSET,
                            N_POINTS, PRIMARY_CAM, PROMPTS, SIM_FPS, WRIST_CAM,
                            expected_margin, sweep_specs, sweep_value)


def render_views(family, params):
    primary, labels = roll(FAMILIES[family], params, PRIMARY_CAM[family], SWEEP_CFG)
    wrist, _ = roll(FAMILIES[family], params, WRIST_CAM[family], SWEEP_CFG)
    return primary, wrist, labels


def family_preview(path: Path, records, root: Path):
    """One column per sweep point: current frame above, ground-truth future below."""
    cell, header = 256, 52
    canvas = Image.new("RGB", (len(records) * cell, 2 * cell + header), (23, 26, 30))
    draw = ImageDraw.Draw(canvas)
    fnt = font(11)
    for col, rec in enumerate(records):
        with np.load(root / rec["input_npz"]) as data:
            current = Image.fromarray(data["primary_image"])
            future = Image.fromarray(data["gt_future_primary"])
        x = col * cell
        canvas.paste(current, (x, header))
        canvas.paste(future, (x, header + cell))
        for row, text, fill in ((8, f"m={rec['margin']:+.3f} y={rec['outcome']}", "white"),
                                (27, f"p={rec['sweep_value']:.3g}", (190, 198, 210))):
            box = draw.textbbox((0, 0), text, font=fnt)
            draw.text((x + (cell - box[2] + box[0]) / 2, row), text, fill=fill, font=fnt)
    draw.text((5, header + 5), "CURRENT", fill="white", font=fnt)
    draw.text((5, header + cell + 5), "GT +0.32s", fill="white", font=fnt)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def neutral_proprio(stats_path: Path) -> np.ndarray:
    stats = json.loads(stats_path.read_text())
    lo = np.asarray(stats["proprio_min"], dtype=np.float64)
    hi = np.asarray(stats["proprio_max"], dtype=np.float64)
    return (lo + hi) / 2


def build_sample(family, index, params, proprio, root: Path):
    """Render one sweep point and write its npz, conditioning input and prompt."""
    sample_id = f"{family}_{index:02d}"
    is_dynamic = family in DYNAMIC
    cond_idx = DYNAMIC_CONDITION_INDICES if is_dynamic else (0,)
    current_idx = cond_idx[-1]
    future_idx = current_idx + FUTURE_OFFSET

    primary, wrist, lab = render_views(family, params)
    assert future_idx < len(primary), (family, future_idx, len(primary))

    analytic = expected_margin(family, params)
    if analytic is not None and not np.isclose(lab["margin"], analytic, atol=1e-7):
        raise RuntimeError(f"{sample_id}: analytic/sim margin mismatch")

    input_rel = Path("inputs") / f"{sample_id}.npz"
    video_rel = Path("v2w_inputs") / f"{sample_id}.mp4"
    prompt_rel = Path("v2w_inputs") / f"{sample_id}.txt"
    cond_frames = primary[list(cond_idx)]

    np.savez_compressed(
        root / input_rel, name=sample_id, family=family,
        primary_image=primary[current_idx], wrist_image=wrist[current_idx],
        proprio=proprio, condition_primary=cond_frames,
        gt_future_primary=primary[future_idx], gt_future_wrist=wrist[future_idx],
        gt_final_primary=primary[-1], gt_final_wrist=wrist[-1],
        margin=lab["margin"], outcome=lab["outcome"],
        event_frame=lab["event_frame"], current_idx=current_idx,
        future_idx=future_idx)

    if is_dynamic:
        write_video(root / video_rel, cond_frames, SWEEP_CFG)
    else:
        (root / video_rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cond_frames[0]).save((root / video_rel).with_suffix(".png"))
        video_rel = video_rel.with_suffix(".png")
    (root / prompt_rel).write_text(PROMPTS[family])

    return {
        "id": sample_id, "family": family,
        "kind": "dynamic" if is_dynamic else "static",
        "sweep_index": index, "sweep_value": sweep_value(family, params),
        "params": params, "margin": float(lab["margin"]),
        "outcome": int(lab["outcome"]), "event_frame": int(lab["event_frame"]),
        "current_idx": current_idx, "future_idx": future_idx,
        "condition_indices": list(cond_idx),
        "condition_pixel_frames": len(cond_idx),
        "prompt": PROMPTS[family], "input_npz": str(input_rel),
        "v2w_input": str(video_rel), "prompt_file": str(prompt_rel),
    }


def merge_manifest(manifest: Path, records):
    """Re-running one family must not drop the others already on disk."""
    if manifest.exists():
        existing = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        merged = {r["id"]: r for r in existing}
        merged.update({r["id"]: r for r in records})
        order = {name: i for i, name in enumerate(FAMILIES)}
        records = sorted(merged.values(),
                         key=lambda r: (order[r["family"]], r["sweep_index"]))
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/physics_sweep"))
    ap.add_argument("--families", nargs="*", choices=tuple(FAMILIES),
                    default=tuple(FAMILIES))
    args = ap.parse_args()

    root = args.output_dir.resolve()
    for sub in ("inputs", "v2w_inputs", "previews"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    proprio = neutral_proprio(args.stats)
    specs = sweep_specs()
    records = []

    for family in args.families:
        fam_records = []
        for index, params in enumerate(specs[family]):
            rec = build_sample(family, index, params, proprio, root)
            records.append(rec)
            fam_records.append(rec)
            print(f"{rec['id']}: p={rec['sweep_value']:.5g} "
                  f"margin={rec['margin']:+.5f} outcome={rec['outcome']} "
                  f"event={rec['event_frame']} cond={rec['condition_indices']}")
        family_preview(root / "previews" / f"{family}.png", fam_records, root)

    manifest = root / "manifest.jsonl"
    records = merge_manifest(manifest, records)
    summary = {
        "samples": len(records),
        "families": list(dict.fromkeys(r["family"] for r in records)),
        "points_per_family": N_POINTS,
        "sim_fps": SIM_FPS,
        "v2w_conditioning_fps": SWEEP_CFG.fps,
        "future_horizon_seconds": FUTURE_OFFSET * 0.02,
        "manifest": str(manifest),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
