"""Aggregate Cosmos-Policy and Cosmos V2W threshold sweeps.

Outcome decoders are deliberately small, scene-specific image measurements.
Their thresholds and directions are calibrated only on MuJoCo frames, then
frozen before application to model predictions. The reported GT decoder
ceiling makes cases whose outcome is not yet visible at +0.32 s explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FAMILIES = ("seesaw", "lean", "tower", "hill", "collide", "domino")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=Path("artifacts/physics_sweep"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/physics_sweep/analysis"))
    return parser.parse_args()


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_sharded_results(root: Path):
    rows = []
    paths = []
    if (root / "results.jsonl").exists():
        paths.append(root / "results.jsonl")
    paths.extend(sorted(root.glob("*/results.jsonl")))
    for path in paths:
        rows.extend(load_jsonl(path))
    return {row["id"]: row for row in rows}


def output_dir_for(root: Path, sample_id: str):
    matches = []
    if (root / sample_id).is_dir():
        matches.append(root / sample_id)
    matches.extend(root.glob(f"*/{sample_id}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one output for {sample_id} under {root}, got {matches}")
    return matches[0]


def square(array, size=256):
    image = Image.fromarray(np.asarray(array).astype(np.uint8)).convert("RGB")
    if image.width != image.height:
        side = min(image.width, image.height)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR))


def color_mask(image, color):
    x = np.asarray(image, dtype=np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    if color == "red":
        return (r > 105) & (r > 1.22 * g) & (r > 1.12 * b)
    if color == "blue":
        return (b > 80) & (b > 1.16 * r) & (b > 1.04 * g)
    raise ValueError(color)


def centroid(mask):
    y, x = np.nonzero(mask)
    if len(x) < 12:
        return None
    return np.array([x.mean(), y.mean()], dtype=np.float64)


def principal_axis(mask):
    y, x = np.nonzero(mask)
    if len(x) < 12:
        return None
    points = np.stack([x - x.mean(), y - y.mean()], axis=1)
    _, vectors = np.linalg.eigh(points.T @ points)
    axis = vectors[:, -1]
    return axis / max(np.linalg.norm(axis), 1e-12)


def proxy(family, image, current):
    red = color_mask(image, "red")
    blue = color_mask(image, "blue")
    if family == "seesaw":
        def tilt(mask):
            y, x = np.nonzero(mask)
            if len(x) < 24:
                return np.nan
            mid = (x.min() + x.max()) / 2
            left, right = y[x < mid], y[x >= mid]
            return float(right.mean() - left.mean()) if len(left) > 5 and len(right) > 5 else np.nan

        return tilt(red) - tilt(color_mask(current, "red"))
    if family == "lean":
        axis = principal_axis(red)
        base = principal_axis(color_mask(current, "red"))
        return float(abs(axis[1]) - abs(base[1])) if axis is not None and base is not None else np.nan
    if family == "tower":
        centers = [centroid(red), centroid(blue)]
        base = [centroid(color_mask(current, "red")), centroid(color_mask(current, "blue"))]
        if any(value is None for value in centers + base):
            return np.nan
        return float(np.linalg.norm(np.concatenate(centers) - np.concatenate(base)))
    if family in ("hill", "collide"):
        center = centroid(red)
        base = centroid(color_mask(current, "red"))
        return float(center[0] - base[0]) if center is not None and base is not None else np.nan
    if family == "domino":
        axis = principal_axis(blue)
        base = principal_axis(color_mask(current, "blue"))
        return float(abs(axis[0]) - abs(base[0])) if axis is not None and base is not None else np.nan
    raise ValueError(family)


def best_rule(values, labels, directions=(1, -1)):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    finite = np.isfinite(values)
    unique = np.unique(values[finite])
    if not len(unique):
        return {"threshold": 0.0, "direction": 1, "accuracy": 0.0}
    candidates = [unique[0] - 1e-6, unique[-1] + 1e-6]
    candidates += list((unique[:-1] + unique[1:]) / 2)
    best = None
    for direction in directions:
        for threshold in candidates:
            pred = values >= threshold if direction == 1 else values <= threshold
            pred[~finite] = False
            accuracy = float(np.mean(pred.astype(int) == labels))
            candidate = {"threshold": float(threshold), "direction": direction, "accuracy": accuracy}
            if best is None or accuracy > best["accuracy"]:
                best = candidate
    return best


def apply_rule(value, rule):
    if not np.isfinite(value):
        return 0
    return int(value >= rule["threshold"] if rule["direction"] == 1 else value <= rule["threshold"])


def motion_metrics(current, gt, prediction):
    current = current.astype(np.float32) / 255.0
    gt = gt.astype(np.float32) / 255.0
    prediction = prediction.astype(np.float32) / 255.0
    target = (gt - current).reshape(-1)
    predicted = (prediction - current).reshape(-1)
    nt, npred = float(np.linalg.norm(target)), float(np.linalg.norm(predicted))
    return {
        "mse": float(np.mean((prediction - gt) ** 2)),
        "current_mse": float(np.mean((current - gt) ** 2)),
        "motion_cosine": float(np.dot(target, predicted) / max(nt * npred, 1e-12)),
        "motion_ratio": npred / max(nt, 1e-12),
    }


def rankdata(values):
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        indices = np.flatnonzero(values == value)
        ranks[indices] = ranks[indices].mean()
    return ranks


def spearman(a, b):
    a, b = rankdata(np.asarray(a)), rankdata(np.asarray(b))
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def contact_sheet(path, family, rows):
    cell = 192
    left = 112
    top = 50
    image_rows = [
        ("CURRENT", "current"),
        ("MUJOCO +0.32s", "gt"),
    ]
    if "policy" in rows[0]:
        image_rows.append(("COSMOS-POLICY", "policy"))
    image_rows.extend([
        ("PREDICT2 V2W +0.31s", "v2w"),
        ("PREDICT2 V2W FINAL", "v2w_final"),
    ])
    canvas = Image.new("RGB", (left + 7 * cell, top + len(image_rows) * cell), (23, 26, 30))
    draw = ImageDraw.Draw(canvas)
    f = font(11)
    for row_index, (label, _) in enumerate(image_rows):
        draw.text((7, top + row_index * cell + cell / 2 - 6), label, fill="white", font=f)
    for col, row in enumerate(rows):
        label = f"m={row['margin']:+.3f} y={row['outcome']}"
        box = draw.textbbox((0, 0), label, font=f)
        draw.text((left + col * cell + (cell - box[2] + box[0]) / 2, 17), label, fill="white", font=f)
        for row_index, (_, key) in enumerate(image_rows):
            image = Image.fromarray(row[key]).resize((cell, cell), Image.Resampling.BILINEAR)
            canvas.paste(image, (left + col * cell, top + row_index * cell))
    canvas.save(path, quality=92)


def rollout_gif(path, family_rows):
    cell_w, cell_h, header = 192, 112, 42
    f = font(10)
    videos = [imageio.mimread(row["v2w_video"]) for row in family_rows]
    n_frames = min(len(video) for video in videos)
    output = []
    for frame_index in range(n_frames):
        canvas = Image.new("RGB", (7 * cell_w, header + cell_h), (23, 26, 30))
        draw = ImageDraw.Draw(canvas)
        for col, (row, video) in enumerate(zip(family_rows, videos)):
            image = Image.fromarray(np.asarray(video[frame_index])).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
            canvas.paste(image, (col * cell_w, header))
            label = f"m={row['margin']:+.3f} y={row['outcome']}"
            box = draw.textbbox((0, 0), label, font=f)
            draw.text((col * cell_w + (cell_w - box[2] + box[0]) / 2, 7), label, fill="white", font=f)
        draw.text((7, 25), f"{frame_index / 16:.2f}s", fill=(205, 210, 220), font=f)
        output.append(np.asarray(canvas))
    imageio.mimsave(path, output, duration=62.5, loop=0, palettesize=128)


def main():
    args = parse_args()
    root = args.sweep_root.resolve()
    out = args.output_dir.resolve()
    (out / "contact_sheets").mkdir(parents=True, exist_ok=True)
    (out / "v2w_rollouts").mkdir(parents=True, exist_ok=True)
    manifest = load_jsonl(root / "manifest.jsonl")
    policy_results = load_sharded_results(root / "cosmos_policy")
    v2w_results = load_sharded_results(root / "cosmos_v2w")
    policy_available = len(policy_results) == len(manifest)
    if len(v2w_results) != len(manifest):
        raise RuntimeError(("incomplete V2W outputs", len(manifest), len(v2w_results)))
    if policy_results and not policy_available:
        raise RuntimeError(("partial Cosmos-Policy outputs", len(manifest), len(policy_results)))

    prepared = []
    for record in manifest:
        sample_id = record["id"]
        with np.load(root / record["input_npz"], allow_pickle=False) as data:
            current = square(data["primary_image"])
            gt = square(data["gt_future_primary"])
            gt_final = square(data["gt_final_primary"])
        v2w_dir = output_dir_for(root / "cosmos_v2w", sample_id)
        v2w = square(np.asarray(Image.open(v2w_dir / "future_primary.png")))
        v2w_final = square(np.asarray(Image.open(v2w_dir / "last_primary.png")))
        row = {
            **record,
            "current": current,
            "gt": gt,
            "gt_final": gt_final,
            "v2w": v2w,
            "v2w_final": v2w_final,
            "v2w_video": v2w_dir / "rollout.mp4",
        }
        if policy_available:
            policy_dir = output_dir_for(root / "cosmos_policy", sample_id)
            policy = square(np.asarray(Image.open(policy_dir / "future_primary.png")))
            row["policy"] = policy
            row["policy_value"] = policy_results[sample_id]["value_prediction"]
            row["policy_motion"] = motion_metrics(current, gt, policy)
        row["v2w_motion"] = motion_metrics(current, gt, v2w)
        prepared.append(row)

    summaries = {}
    flat_rows = []
    for family in FAMILIES:
        rows = [row for row in prepared if row["family"] == family]
        labels = [row["outcome"] for row in rows]
        gt_proxy = [proxy(family, row["gt"], row["current"]) for row in rows]
        future_rule = best_rule(gt_proxy, labels)
        final_proxy = [proxy(family, row["gt_final"], row["current"]) for row in rows]
        final_rule = best_rule(final_proxy, labels)
        v2w_proxy = [proxy(family, row["v2w"], row["current"]) for row in rows]
        v2w_final_proxy = [proxy(family, row["v2w_final"], row["current"]) for row in rows]
        v2w_pred = [apply_rule(value, future_rule) for value in v2w_proxy]
        v2w_final_pred = [apply_rule(value, final_rule) for value in v2w_final_proxy]
        moving_rows = [row for row in rows if row["v2w_motion"]["current_mse"] > 1e-5]
        summary = {
            "family": family,
            "gt_endpoint_decoder_ceiling": future_rule["accuracy"],
            "gt_final_decoder_ceiling": final_rule["accuracy"],
            "v2w_endpoint_outcome_accuracy": float(np.mean(np.asarray(v2w_pred) == labels)),
            "v2w_final_outcome_accuracy": float(np.mean(np.asarray(v2w_final_pred) == labels)),
            "moving_samples": len(moving_rows),
            "v2w_motion_cosine_mean": float(
                np.mean([row["v2w_motion"]["motion_cosine"] for row in moving_rows])
            ) if moving_rows else 0.0,
            "v2w_mse_mean": float(np.mean([row["v2w_motion"]["mse"] for row in rows])),
            "endpoint_proxy_rule": future_rule,
            "final_proxy_rule": final_rule,
        }
        if policy_available:
            policy_proxy = [proxy(family, row["policy"], row["current"]) for row in rows]
            policy_pred = [apply_rule(value, future_rule) for value in policy_proxy]
            # Policy value is nominally a success probability; its direction is fixed.
            value_rule = best_rule([row["policy_value"] for row in rows], labels, directions=(1,))
            summary.update({
                "policy_pixel_outcome_accuracy": float(np.mean(np.asarray(policy_pred) == labels)),
                "policy_value_fixed_direction_accuracy": value_rule["accuracy"],
                "policy_value_margin_spearman": spearman(
                    [row["policy_value"] for row in rows], [row["margin"] for row in rows]
                ),
                "policy_motion_cosine_mean": float(
                    np.mean([row["policy_motion"]["motion_cosine"] for row in moving_rows])
                ) if moving_rows else 0.0,
                "policy_mse_mean": float(np.mean([row["policy_motion"]["mse"] for row in rows])),
            })
        summaries[family] = summary
        for index, row in enumerate(rows):
            flat_row = {
                    "id": row["id"],
                    "family": family,
                    "margin": row["margin"],
                    "outcome": row["outcome"],
                    "v2w_endpoint_prediction": v2w_pred[index],
                    "v2w_final_prediction": v2w_final_pred[index],
                    "v2w_motion_cosine": row["v2w_motion"]["motion_cosine"],
                    "v2w_mse": row["v2w_motion"]["mse"],
            }
            if policy_available:
                flat_row.update({
                    "policy_value": row["policy_value"],
                    "policy_pixel_prediction": policy_pred[index],
                    "policy_motion_cosine": row["policy_motion"]["motion_cosine"],
                    "policy_mse": row["policy_motion"]["mse"],
                })
            flat_rows.append(flat_row)
        contact_sheet(out / "contact_sheets" / f"{family}.jpg", family, rows)
        rollout_gif(out / "v2w_rollouts" / f"{family}.gif", rows)

    aggregate_keys = [
        "v2w_endpoint_outcome_accuracy",
        "v2w_final_outcome_accuracy",
        "v2w_motion_cosine_mean",
        "v2w_mse_mean",
    ]
    if policy_available:
        aggregate_keys += [
            "policy_pixel_outcome_accuracy",
            "policy_value_fixed_direction_accuracy",
            "policy_motion_cosine_mean",
            "policy_mse_mean",
        ]
    aggregate = {key: float(np.mean([summaries[f][key] for f in FAMILIES])) for key in aggregate_keys}
    payload = {"families": summaries, "macro_average": aggregate, "samples": len(flat_rows)}
    (out / "summary.json").write_text(json.dumps(payload, indent=2))
    with (out / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, family in zip(axes.flat, FAMILIES):
        rows = [row for row in prepared if row["family"] == family]
        margin = np.array([row["margin"] for row in rows])
        order = np.argsort(margin)
        if policy_available:
            axis.plot(margin[order], np.array([row["policy_motion"]["motion_cosine"] for row in rows])[order], "o-", label="Policy")
        axis.plot(margin[order], np.array([row["v2w_motion"]["motion_cosine"] for row in rows])[order], "o-", label="V2W")
        axis.axvline(0, color="0.5", linestyle="--", linewidth=1)
        axis.axhline(0, color="0.8", linewidth=1)
        axis.set_title(family)
        axis.set_xlabel("signed physics margin")
        axis.set_ylabel("motion-delta cosine")
        axis.set_ylim(-0.2, 1.0)
    axes[0, 0].legend()
    fig.savefig(out / "motion_alignment.png", dpi=180)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
