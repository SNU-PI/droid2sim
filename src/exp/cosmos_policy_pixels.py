"""Generate future pixels with the official Cosmos-Policy LIBERO checkpoint.

This script is meant to run inside the upstream NVlabs/cosmos-policy Docker
environment.  It first supports the repository's bundled LIBERO observation;
an optional NPZ input can then replace the two camera images and proprio state
without changing the inference path.
"""

import argparse
import json
import pickle
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


DEFAULT_TASK = "put both the alphabet soup and the tomato sauce in the basket"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--sample-observation", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--no-flip", action="store_true")
    return parser.parse_args()


def load_observation(args):
    if args.input_npz is None:
        with args.sample_observation.open("rb") as handle:
            return pickle.load(handle), "official_libero_sample"

    with np.load(args.input_npz) as data:
        observation = {
            "primary_image": data["primary_image"],
            "wrist_image": data["wrist_image"],
            "proprio": data["proprio"],
        }
        name = str(data["name"].item()) if "name" in data else args.input_npz.stem
    return observation, name


def save_image(path, array):
    array = np.asarray(array)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def make_comparison(path, images):
    labels = list(images)
    pil_images = [Image.fromarray(np.asarray(images[label]).astype(np.uint8)) for label in labels]
    target_h = max(image.height for image in pil_images)
    resized = [
        image.resize((round(image.width * target_h / image.height), target_h), Image.Resampling.BILINEAR)
        for image in pil_images
    ]
    header = 34
    canvas = Image.new("RGB", (sum(image.width for image in resized), target_h + header), (24, 26, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    x = 0
    for label, image in zip(labels, resized):
        canvas.paste(image, (x, header))
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (image.width - (box[2] - box[0])) / 2, 9), label, fill="white", font=font)
        x += image.width
    canvas.save(path)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observation, input_name = load_observation(args)

    checkpoint = args.checkpoint_dir
    # Importing upstream PolicyEvalConfig also imports LIBERO, whose package
    # initializer prompts interactively for a dataset directory.  The inference
    # utilities use a structural config object, so keep this pixel-only runner
    # independent of the simulator with the same required fields.
    config = SimpleNamespace(
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
        flip_images=not args.no_flip,
        num_denoising_steps_action=args.steps,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        use_third_person_image=True,
        num_third_person_images=1,
        num_wrist_images=1,
        use_variance_scale=False,
    )

    dataset_stats = load_dataset_stats(config.dataset_stats_path)
    init_t5_text_embeddings_cache(config.t5_text_embeddings_path)
    model, _ = get_model(config)
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

    future = result["future_image_predictions"]
    arrays = {
        "current_primary": observation["primary_image"],
        "future_primary": future["future_image"],
        "current_wrist": observation["wrist_image"],
        "future_wrist": future["future_wrist_image"],
    }
    for label, array in arrays.items():
        save_image(args.output_dir / f"{label}.png", array)
    make_comparison(args.output_dir / "comparison.png", arrays)

    metadata = {
        "input_name": input_name,
        "task": args.task,
        "seed": args.seed,
        "denoising_steps": args.steps,
        "value_prediction": float(result["value_prediction"]),
        "actions": np.asarray(result["actions"]).tolist(),
        "current_primary_shape": list(np.asarray(observation["primary_image"]).shape),
        "future_primary_shape": list(np.asarray(future["future_image"]).shape),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
