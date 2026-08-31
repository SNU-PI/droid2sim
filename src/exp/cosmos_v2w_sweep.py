"""Generate controlled physics rollouts with Cosmos Predict2 Video2World.

This runner uses the official 480p/16 FPS monolithic checkpoint through
Diffusers' original-Cosmos converter. Static families condition on one image;
dynamic families condition on the five pre-outcome frames stored in the shared
sweep manifest. The model is loaded once and reused for all selected samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as cosmos_pipeline_module
from diffusers import Cosmos2VideoToWorldPipeline
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
from PIL import Image, ImageDraw, ImageFont


NEGATIVE_PROMPT = (
    "blurry, low resolution, camera motion, camera shake, morphing objects, disappearing objects, "
    "duplicate objects, unrealistic motion, implausible contact, text, watermark"
)


class DisabledSyntheticOnlySafetyChecker:
    """No-op replacement for the unavailable separately gated guardrail.

    NVIDIA's official CLI exposes ``--disable_guardrail``. This experiment is
    limited to locally rendered geometric primitives, so mirror that option
    without modifying the installed Diffusers package.
    """

    def to(self, *args, **kwargs):
        return self

    def check_text_safety(self, prompt):
        return True

    def check_video_safety(self, video):
        return video


cosmos_pipeline_module.CosmosSafetyChecker = DisabledSyntheticOnlySafetyChecker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="*")
    parser.add_argument("--ids", nargs="*", help="Optional exact sample ids for smoke tests or resuming subsets.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--num-frames", type=int, default=21, help="Must be 4k+1; 21 covers 1.25 s at 16 FPS.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    return parser.parse_args()


def load_records(path: Path, families: list[str] | None, ids: list[str] | None):
    selected = set(families or [])
    selected_ids = set(ids or [])
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if selected:
        records = [record for record in records if record["family"] in selected]
    if selected_ids:
        records = [record for record in records if record["id"] in selected_ids]
    return records


def letterbox(array: np.ndarray, height: int, width: int) -> Image.Image:
    image = Image.fromarray(np.asarray(array).astype(np.uint8)).convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    # Match the renderer's neutral sky tone instead of introducing black bars.
    canvas = Image.new("RGB", (width, height), tuple(int(x) for x in np.asarray(array)[0, 0]))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def to_uint8(frames) -> np.ndarray:
    if isinstance(frames, list):
        return np.stack([np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames])
    array = np.asarray(frames)
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0 if array.max() <= 1.5 else array, 0, 255).astype(np.uint8)
    return array


def mse(a, b):
    a = np.asarray(a, dtype=np.float32) / 255.0
    b = np.asarray(b, dtype=np.float32) / 255.0
    return float(np.mean((a - b) ** 2))


def motion_metrics(current, gt, prediction):
    current = np.asarray(current, dtype=np.float32) / 255.0
    gt = np.asarray(gt, dtype=np.float32) / 255.0
    prediction = np.asarray(prediction, dtype=np.float32) / 255.0
    target_delta = (gt - current).reshape(-1)
    pred_delta = (prediction - current).reshape(-1)
    target_norm = float(np.linalg.norm(target_delta))
    pred_norm = float(np.linalg.norm(pred_delta))
    cosine = float(np.dot(target_delta, pred_delta) / max(target_norm * pred_norm, 1e-12))
    return {
        "motion_delta_cosine": cosine,
        "motion_magnitude_ratio": pred_norm / max(target_norm, 1e-12),
    }


def comparison(path: Path, images: dict[str, Image.Image]):
    size = 224
    header = 34
    canvas = Image.new("RGB", (size * len(images), size + header), (24, 27, 31))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for index, (label, image) in enumerate(images.items()):
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        x = index * size
        canvas.paste(image, (x, header))
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (size - box[2] + box[0]) / 2, 9), label, fill="white", font=font)
    canvas.save(path)


def main():
    args = parse_args()
    if args.num_frames % 4 != 1:
        raise ValueError("--num-frames must have the form 4k+1 for the causal video tokenizer")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.manifest, args.families, args.ids)

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
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    results_path = args.output_dir / "results.jsonl"
    if results_path.exists():
        existing = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
        results = {item["id"]: item for item in existing}
    else:
        results = {}

    for position, record in enumerate(records, 1):
        with np.load(args.sweep_root / record["input_npz"], allow_pickle=False) as data:
            current_raw = np.asarray(data["primary_image"])
            gt_raw = np.asarray(data["gt_future_primary"])
            condition_raw = np.asarray(data["condition_primary"])
        current = letterbox(current_raw, args.height, args.width)
        gt = letterbox(gt_raw, args.height, args.width)
        conditions = [letterbox(frame, args.height, args.width) for frame in condition_raw]
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        call_args = {
            "prompt": record["prompt"],
            "negative_prompt": NEGATIVE_PROMPT,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.steps,
            "guidance_scale": 7.0,
            "fps": 16,
            "generator": generator,
            "output_type": "pil",
        }
        if record["kind"] == "dynamic":
            call_args["video"] = conditions
        else:
            call_args["image"] = conditions[0]
        generated = pipe(**call_args).frames[0]
        frames = to_uint8(generated)

        # Five 16 FPS steps are 0.3125 s. Dynamic outputs include the five
        # conditioning frames, while static outputs include the initial image.
        last_condition = record["condition_pixel_frames"] - 1
        eval_index = min(last_condition + 5, len(frames) - 1)
        future = frames[eval_index]
        sample_dir = args.output_dir / record["id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(sample_dir / "rollout.mp4", frames, fps=16, codec="libx264", quality=8)
        Image.fromarray(future).save(sample_dir / "future_primary.png")
        Image.fromarray(frames[-1]).save(sample_dir / "last_primary.png")
        comparison(
            sample_dir / "comparison.png",
            {
                "current": current,
                "physics GT +0.32s": gt,
                "V2W +0.31s": Image.fromarray(future),
                "V2W final": Image.fromarray(frames[-1]),
            },
        )
        current_array = np.asarray(current)
        gt_array = np.asarray(gt)
        metrics = {
            **record,
            "seed": args.seed,
            "denoising_steps": args.steps,
            "guardrail_disabled_for_synthetic_inputs": True,
            "num_generated_frames": len(frames),
            "prediction_eval_index": eval_index,
            "mse_prediction_gt": mse(future, gt_array),
            "mse_current_gt": mse(current_array, gt_array),
            "mse_prediction_current": mse(future, current_array),
            **motion_metrics(current_array, gt_array, future),
            "output_dir": str(sample_dir),
        }
        (sample_dir / "metadata.json").write_text(json.dumps(metrics, indent=2))
        results[record["id"]] = metrics
        results_path.write_text(
            "".join(json.dumps(item) + "\n" for item in results.values())
        )
        print(
            f"[{position:02d}/{len(records):02d}] {record['id']} "
            f"mse(gt)={metrics['mse_prediction_gt']:.5f} "
            f"motion_cos={metrics['motion_delta_cosine']:+.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
