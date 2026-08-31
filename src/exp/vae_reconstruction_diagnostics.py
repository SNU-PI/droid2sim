"""Measure the frozen Cosmos VAE reconstruction ceiling; no weight updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from PIL import Image


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metrics(reference, reconstruction):
    a = np.asarray(reference, dtype=np.float32) / 255.0
    b = np.asarray(reconstruction, dtype=np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return {"mse": mse, "psnr": float(-10.0 * np.log10(max(mse, 1e-12)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vae = AutoencoderKLWan.from_pretrained(
        str(args.model_dir),
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    vae.eval()
    processor = VideoProcessor(vae_scale_factor=8)
    results = []

    def reconstruct(frames):
        images = [Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB") for frame in frames]
        if len(images) == 1:
            video = processor.preprocess(images[0], 480, 832).unsqueeze(2)
        else:
            video = processor.preprocess_video(images, 480, 832)
        video = video.to("cuda", dtype=vae.dtype)
        with torch.inference_mode():
            latents = vae.encode(video).latent_dist.mode()
            decoded = vae.decode(latents, return_dict=False)[0]
        output = processor.postprocess_video(decoded, output_type="np")
        output = np.clip(np.asarray(output) * 255.0, 0, 255).astype(np.uint8)
        return output[0]

    for record in load_jsonl(args.data_root / "manifest.jsonl"):
        sample_dir = args.output_dir / record["family"] / record["id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        with np.load(args.data_root / record["input_npz"], allow_pickle=False) as data:
            named = {
                "current": np.asarray(data["current_primary"]),
                "endpoint": np.asarray(data["gt_future_primary"]),
                "final": np.asarray(data["gt_final_primary"]),
            }
            condition = np.asarray(data["condition_primary"])

        row = {
            "sample_id": record["id"],
            "family": record["family"],
            "margin": record["margin"],
            "outcome": record["outcome"],
        }
        for name, frame in named.items():
            reconstruction = reconstruct([frame])[0]
            Image.fromarray(reconstruction).save(sample_dir / f"{name}_reconstruction.png")
            for key, value in metrics(frame, reconstruction).items():
                row[f"{name}_{key}"] = value

        condition_reconstruction = reconstruct(condition)
        imageio.mimsave(
            sample_dir / "condition_reconstruction.mp4",
            condition_reconstruction,
            fps=16,
            codec="libx264",
            quality=9,
        )
        condition_metric = metrics(condition, condition_reconstruction)
        row.update({f"condition_{key}": value for key, value in condition_metric.items()})
        (sample_dir / "metadata.json").write_text(json.dumps(row, indent=2))
        results.append(row)
        print(record["id"], json.dumps(row), flush=True)

    (args.output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results)
    )


if __name__ == "__main__":
    main()
