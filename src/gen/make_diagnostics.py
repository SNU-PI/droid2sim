"""Generate native-480p, exact-16fps Tower/Hill diagnostics and GPU jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# gen.render selects EGL before importing MuJoCo through core.threshold.
from gen.render import DIAG_CFG, roll, write_video
from core.threshold import Hill, Tower

FPS = DIAG_CFG.fps
HEIGHT = DIAG_CFG.height
WIDTH = DIAG_CFG.width
BASE_SEEDS = tuple(range(1, 97))
ORACLE_SEEDS = tuple(range(1, 25))
IMAGE_ONLY_SEEDS = tuple(range(1, 25))

BASE_PROMPTS = {
    "tower": (
        "A fixed camera observes a stack of three rigid colored cubes. "
        "Gravity acts naturally, contacts are solid, and every cube keeps its shape and identity."
    ),
    "hill": (
        "A fixed camera observes a rigid red ball rolling from left to right toward a smooth gray hill. "
        "The ball follows rolling contact, momentum, and gravity while keeping its shape and identity."
    ),
}

ORACLE_PROMPTS = {
    "tower": {
        0: (
            "The offset stack is unstable. Under gravity the rigid cubes topple and the top gray cube "
            "falls off the stack. Every cube keeps its shape and identity."
        ),
        1: (
            "The offset stack is stable. Under gravity all three rigid cubes remain standing in the same "
            "stack. Every cube keeps its shape and identity."
        ),
    },
    "hill": {
        0: (
            "The rigid red ball lacks enough kinetic energy to cross the hill. It rolls uphill, slows, "
            "then returns to the left under gravity while keeping its shape."
        ),
        1: (
            "The rigid red ball has enough kinetic energy to cross the hill. It rolls over the crest and "
            "continues to the right under gravity while keeping its shape."
        ),
    },
}


def render(scene_cls, params, camera):
    """Full-resolution capture at exactly one frame per model timestep."""
    frames, labels = roll(scene_cls, params, camera, DIAG_CFG)
    return {"frames": frames, **labels}


def specs():
    return {
        "tower": [dict(o1=0.010, o2=float(x)) for x in np.linspace(0.015, 0.045, 7)],
        "hill": [dict(v0=float(x), h=0.055, x0=-0.55) for x in np.linspace(0.65, 1.10, 7)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/physics_diagnostics_3h"))
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "conditions").mkdir(parents=True, exist_ok=True)
    (root / "jobs").mkdir(parents=True, exist_ok=True)

    records = []
    for family, family_specs in specs().items():
        for index, params in enumerate(family_specs):
            sample_id = f"{family}_{index:02d}"
            scene_cls = Tower if family == "tower" else Hill
            camera = "close" if family == "tower" else "side"
            result = render(scene_cls, params, camera)
            condition_indices = [0] if family == "tower" else [0, 1, 2, 3, 4]
            current_idx = condition_indices[-1]
            future_idx = current_idx + 5
            condition = result["frames"][condition_indices]
            assert future_idx < len(result["frames"])

            npz_rel = Path("inputs") / f"{sample_id}.npz"
            np.savez_compressed(
                root / npz_rel,
                family=family,
                condition_primary=condition,
                current_primary=result["frames"][current_idx],
                gt_future_primary=result["frames"][future_idx],
                gt_final_primary=result["frames"][-1],
                margin=result["margin"],
                outcome=result["outcome"],
                event_frame=result["event_frame"],
                capture_fps=FPS,
                condition_indices=np.asarray(condition_indices),
                current_idx=current_idx,
                future_idx=future_idx,
            )
            if family == "tower":
                cond_rel = Path("conditions") / f"{sample_id}.png"
                Image.fromarray(condition[0]).save(root / cond_rel)
            else:
                cond_rel = Path("conditions") / f"{sample_id}.mp4"
                write_video(root / cond_rel, condition, DIAG_CFG)

            record = {
                "id": sample_id,
                "family": family,
                "kind": "static" if family == "tower" else "dynamic",
                "sweep_index": index,
                "params": params,
                "margin": float(result["margin"]),
                "outcome": int(result["outcome"]),
                "event_frame": int(result["event_frame"]),
                "condition_indices": condition_indices,
                "condition_pixel_frames": len(condition_indices),
                "current_idx": current_idx,
                "future_idx": future_idx,
                "input_npz": str(npz_rel),
                "condition_path": str(cond_rel),
                "base_prompt": BASE_PROMPTS[family],
                "oracle_prompt": ORACLE_PROMPTS[family][int(result["outcome"])],
                "render_size": [HEIGHT, WIDTH],
                "capture_fps": FPS,
            }
            records.append(record)
            print(
                f"{sample_id}: margin={record['margin']:+.4f} y={record['outcome']} "
                f"event={record['event_frame']} cond={condition_indices}"
            )

    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))

    jobs = []
    for seed in BASE_SEEDS:
        for record in records:
            jobs.append(
                {
                    "sample_id": record["id"],
                    "seed": seed,
                    "run_variant": "base",
                    "prompt_variant": "base",
                }
            )
    for seed in ORACLE_SEEDS:
        for record in records:
            jobs.append(
                {
                    "sample_id": record["id"],
                    "seed": seed,
                    "run_variant": "oracle",
                    "prompt_variant": "oracle",
                }
            )
    for seed in IMAGE_ONLY_SEEDS:
        for record in records:
            if record["family"] == "hill":
                jobs.append(
                    {
                        "sample_id": record["id"],
                        "seed": seed,
                        "run_variant": "image_only",
                        "prompt_variant": "base",
                    }
                )

    shards = [[] for _ in range(args.num_shards)]
    for index, job in enumerate(jobs):
        shards[index % args.num_shards].append(job)
    for index, shard in enumerate(shards):
        (root / "jobs" / f"shard_{index}.jsonl").write_text(
            "".join(json.dumps(job) + "\n" for job in shard)
        )

    summary = {
        "records": len(records),
        "base_rollouts": len(records) * len(BASE_SEEDS),
        "oracle_rollouts": len(records) * len(ORACLE_SEEDS),
        "hill_image_only_rollouts": 7 * len(IMAGE_ONLY_SEEDS),
        "total_rollouts": len(jobs),
        "base_seeds": list(BASE_SEEDS),
        "oracle_seeds": list(ORACLE_SEEDS),
        "image_only_seeds": list(IMAGE_ONLY_SEEDS),
        "num_shards": args.num_shards,
        "render_size": [HEIGHT, WIDTH],
        "fps": FPS,
        "condition_timestamps_seconds": {
            "tower": [0.0],
            "hill": [index / FPS for index in range(5)],
        },
        "evaluation_horizon_seconds": 5 / FPS,
    }
    (root / "collection_spec.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
