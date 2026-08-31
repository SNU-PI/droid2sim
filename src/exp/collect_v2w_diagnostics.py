"""Collect inference-only Cosmos Predict2 diagnostic rollouts from a job shard."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as cosmos_pipeline_module
from diffusers import Cosmos2VideoToWorldPipeline
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
from PIL import Image


NEGATIVE_PROMPT = (
    "blurry, low resolution, camera motion, camera shake, morphing objects, disappearing objects, "
    "duplicate objects, changing object count, unrealistic motion, implausible contact, text, watermark"
)


class DisabledSyntheticOnlySafetyChecker:
    def to(self, *args, **kwargs):
        return self

    def check_text_safety(self, prompt):
        return True

    def check_video_safety(self, video):
        return video


cosmos_pipeline_module.CosmosSafetyChecker = DisabledSyntheticOnlySafetyChecker


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mse(a, b):
    a = np.asarray(a, dtype=np.float32) / 255.0
    b = np.asarray(b, dtype=np.float32) / 255.0
    return float(np.mean((a - b) ** 2))


def motion_metrics(current, gt, prediction):
    current = np.asarray(current, dtype=np.float32) / 255.0
    gt = np.asarray(gt, dtype=np.float32) / 255.0
    prediction = np.asarray(prediction, dtype=np.float32) / 255.0
    target = (gt - current).reshape(-1)
    predicted = (prediction - current).reshape(-1)
    nt = float(np.linalg.norm(target))
    npred = float(np.linalg.norm(predicted))
    return {
        "motion_delta_cosine": float(np.dot(target, predicted) / max(nt * npred, 1e-12)),
        "motion_magnitude_ratio": npred / max(nt, 1e-12),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--job-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    records = {row["id"]: row for row in load_jsonl(args.data_root / "manifest.jsonl")}
    jobs = load_jsonl(args.job_file)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    transformer = CosmosTransformer3DModel.from_single_file(
        str(args.original_checkpoint),
        config=str(args.model_dir / "transformer"),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
        str(args.model_dir),
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    shard_results = args.output_root / "shards" / f"{args.job_file.stem}.jsonl"
    shard_results.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if shard_results.exists():
        completed = {
            (row["sample_id"], row["seed"], row["run_variant"]): row
            for row in load_jsonl(shard_results)
        }

    for position, job in enumerate(jobs, 1):
        record = records[job["sample_id"]]
        key = (record["id"], int(job["seed"]), job["run_variant"])
        sample_dir = (
            args.output_root
            / job["run_variant"]
            / record["family"]
            / record["id"]
            / f"seed_{int(job['seed']):02d}"
        )
        if key in completed and (sample_dir / "rollout.mp4").exists():
            print(f"[{position:03d}/{len(jobs):03d}] skip {key}", flush=True)
            continue

        with np.load(args.data_root / record["input_npz"], allow_pickle=False) as data:
            conditions = [Image.fromarray(frame).convert("RGB") for frame in data["condition_primary"]]
            current = np.asarray(data["current_primary"])
            gt = np.asarray(data["gt_future_primary"])
        prompt = record[f"{job['prompt_variant']}_prompt"]
        call_args = {
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": 480,
            "width": 832,
            "num_frames": 21,
            "num_inference_steps": args.steps,
            "guidance_scale": 7.0,
            "fps": 16,
            "generator": torch.Generator(device="cuda").manual_seed(int(job["seed"])),
            "output_type": "pil",
        }
        if record["kind"] == "dynamic" and job["run_variant"] != "image_only":
            call_args["video"] = conditions
        else:
            call_args["image"] = conditions[-1]

        started = time.time()
        generated = pipe(**call_args).frames[0]
        frames = np.stack([np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in generated])
        elapsed = time.time() - started
        last_condition = record["condition_pixel_frames"] - 1
        eval_index = last_condition + 5
        endpoint = frames[eval_index]
        sample_dir.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(sample_dir / "rollout.mp4", frames, fps=16, codec="libx264", quality=9)
        Image.fromarray(endpoint).save(sample_dir / "endpoint.png")
        Image.fromarray(frames[-1]).save(sample_dir / "final.png")
        result = {
            "sample_id": record["id"],
            "family": record["family"],
            "margin": record["margin"],
            "outcome": record["outcome"],
            "seed": int(job["seed"]),
            "run_variant": job["run_variant"],
            "prompt_variant": job["prompt_variant"],
            "prompt": prompt,
            "num_frames": len(frames),
            "eval_index": eval_index,
            "elapsed_seconds": elapsed,
            "mse_endpoint_gt": mse(endpoint, gt),
            "mse_current_gt": mse(current, gt),
            "mse_endpoint_current": mse(endpoint, current),
            **motion_metrics(current, gt, endpoint),
            "output_dir": str(sample_dir),
        }
        (sample_dir / "metadata.json").write_text(json.dumps(result, indent=2))
        completed[key] = result
        shard_results.write_text(
            "".join(json.dumps(row) + "\n" for row in completed.values())
        )
        print(
            f"[{position:03d}/{len(jobs):03d}] {key} {elapsed:.1f}s "
            f"mse={result['mse_endpoint_gt']:.5f} cos={result['motion_delta_cosine']:+.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
