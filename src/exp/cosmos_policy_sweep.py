"""Run the official Cosmos-Policy LIBERO checkpoint over a physics manifest.

The model is loaded once per process and all selected samples are evaluated in
sequence. Launch separate processes with disjoint ``--families`` selections to
use multiple GPUs without reloading the checkpoint for every sweep point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)


DEFAULT_TASK = "push the plate to the front of the stove"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="*")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    return parser.parse_args()


def load_records(path: Path, families: list[str] | None):
    selected = set(families or [])
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if selected:
        records = [record for record in records if record["family"] in selected]
    return records


def save_image(path: Path, array):
    array = np.asarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def resize_array(array, shape):
    h, w = shape
    return np.asarray(Image.fromarray(np.asarray(array).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR))


def mse(a, b):
    a = np.asarray(a, dtype=np.float32) / 255.0
    b = np.asarray(b, dtype=np.float32) / 255.0
    return float(np.mean((a - b) ** 2))


def comparison(path: Path, images: dict[str, np.ndarray]):
    size = 224
    header = 34
    canvas = Image.new("RGB", (size * len(images), size + header), (24, 27, 31))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for index, (label, array) in enumerate(images.items()):
        image = Image.fromarray(np.asarray(array).astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR)
        x = index * size
        canvas.paste(image, (x, header))
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (size - box[2] + box[0]) / 2, 9), label, fill="white", font=font)
    canvas.save(path)


def make_config(args):
    checkpoint = args.checkpoint_dir
    return SimpleNamespace(
        suite="libero",
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=str(checkpoint / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=str(checkpoint / "libero_dataset_statistics.json"),
        t5_text_embeddings_path=str(checkpoint / "libero_t5_embeddings.pkl"),
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        chunk_size=16,
        num_open_loop_steps=16,
        trained_with_image_aug=True,
        use_jpeg_compression=True,
        flip_images=False,
        num_denoising_steps_action=args.steps,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        use_third_person_image=True,
        num_third_person_images=1,
        num_wrist_images=1,
        use_variance_scale=False,
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.manifest, args.families)
    config = make_config(args)
    dataset_stats = load_dataset_stats(config.dataset_stats_path)
    init_t5_text_embeddings_cache(config.t5_text_embeddings_path)
    model, _ = get_model(config)
    results_path = args.output_dir / "results.jsonl"
    if results_path.exists():
        existing = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
        results = {item["id"]: item for item in existing}
    else:
        results = {}

    for position, record in enumerate(records, 1):
        with np.load(args.sweep_root / record["input_npz"], allow_pickle=False) as data:
            observation = {
                "primary_image": data["primary_image"],
                "wrist_image": data["wrist_image"],
                "proprio": data["proprio"],
            }
            gt = np.asarray(data["gt_future_primary"])
        result = get_action(
            config,
            model,
            dataset_stats,
            observation,
            args.task,
            seed=args.seed,
            num_denoising_steps_action=args.steps,
            generate_future_state_and_value_in_parallel=True,
        )
        prediction = result["future_image_predictions"]
        future = np.asarray(prediction["future_image"])
        future_wrist = np.asarray(prediction["future_wrist_image"])
        sample_dir = args.output_dir / record["id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_image(sample_dir / "future_primary.png", future)
        save_image(sample_dir / "future_wrist.png", future_wrist)
        comparison(
            sample_dir / "comparison.png",
            {
                "current": observation["primary_image"],
                "physics GT +0.32s": gt,
                "Policy +16": future,
            },
        )
        gt_resized = resize_array(gt, future.shape[:2])
        current_resized = resize_array(observation["primary_image"], future.shape[:2])
        metrics = {
            **record,
            "task": args.task,
            "seed": args.seed,
            "denoising_steps": args.steps,
            "value_prediction": float(result["value_prediction"]),
            "mse_prediction_gt": mse(future, gt_resized),
            "mse_current_gt": mse(current_resized, gt_resized),
            "mse_prediction_current": mse(future, current_resized),
            "actions": np.asarray(result["actions"]).tolist(),
            "output_dir": str(sample_dir),
        }
        (sample_dir / "metadata.json").write_text(json.dumps(metrics, indent=2))
        results[record["id"]] = metrics
        results_path.write_text(
            "".join(json.dumps(item) + "\n" for item in results.values())
        )
        print(
            f"[{position:02d}/{len(records):02d}] {record['id']} "
            f"value={metrics['value_prediction']:.4f} "
            f"mse(gt)={metrics['mse_prediction_gt']:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
