"""Analyze the corrected, inference-only Cosmos Predict2 physics collection.

The outcome decoder is calibrated exclusively on MuJoCo endpoints.  It uses
scene-specific visible state: colored-block displacement for Tower and red-ball
horizontal displacement for Hill.  Generative validity is reported separately
so object loss/morphing is not silently counted as a physics decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


FAMILIES = ("tower", "hill")
VARIANTS = ("base", "oracle", "image_only")


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def color_mask(image, color):
    x = np.asarray(image, dtype=np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    if color == "red":
        return (r > 105) & (r > 1.22 * g) & (r > 1.12 * b)
    if color == "blue":
        return (b > 80) & (b > 1.16 * r) & (b > 1.04 * g)
    if color == "grey":
        mask = (np.minimum(np.minimum(r, g), b) > 125) & ((np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) < 38)
        roi = np.zeros(mask.shape, dtype=bool)
        h, w = mask.shape
        roi[int(0.12 * h) : int(0.78 * h), int(0.25 * w) : int(0.70 * w)] = True
        return mask & roi
    raise ValueError(color)


def centroid(mask):
    y, x = np.nonzero(mask)
    if len(x) < 12:
        return None
    return np.array([x.mean(), y.mean()], dtype=np.float64)


def visible_proxy(family, image, current):
    if family == "hill":
        center = centroid(color_mask(image, "red"))
        base = centroid(color_mask(current, "red"))
        return float(center[0] - base[0]) if center is not None and base is not None else np.nan
    if family == "tower":
        displacements = []
        for color in ("red", "blue", "grey"):
            center = centroid(color_mask(image, color))
            base = centroid(color_mask(current, color))
            if center is None or base is None:
                return np.nan
            displacements.append(np.linalg.norm(center - base))
        # Stable is outcome=1, hence negate motion to orient both families alike.
        return -float(np.linalg.norm(displacements))
    raise ValueError(family)


def validity(family, image, current):
    colors = ("red",) if family == "hill" else ("red", "blue", "grey")
    ratios = []
    missing = []
    for color in colors:
        reference_area = int(color_mask(current, color).sum())
        predicted_area = int(color_mask(image, color).sum())
        missing.append(reference_area < 12 or predicted_area < 12)
        ratios.append(predicted_area / max(reference_area, 1))
    valid = not any(missing) and all(0.35 <= ratio <= 2.8 for ratio in ratios)
    return valid, float(min(ratios)), float(max(ratios))


def best_rule(values, labels):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    finite = np.isfinite(values)
    unique = np.unique(values[finite])
    if not len(unique):
        return {"threshold": 0.0, "accuracy": 0.0}
    candidates = [unique[0] - 1e-6, unique[-1] + 1e-6]
    candidates += list((unique[:-1] + unique[1:]) / 2)
    best = None
    for threshold in candidates:
        prediction = (values >= threshold).astype(int)
        prediction[~finite] = 0
        accuracy = float(np.mean(prediction == labels))
        candidate = {"threshold": float(threshold), "accuracy": accuracy}
        if best is None or candidate["accuracy"] > best["accuracy"]:
            best = candidate
    return best


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        index = np.flatnonzero(values == value)
        ranks[index] = ranks[index].mean()
    return ranks


def spearman(a, b):
    if len(a) < 2:
        return np.nan
    a, b = rankdata(a), rankdata(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def mean(values):
    values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else np.nan


def wilson_interval(correct, total, z=1.959963984540054):
    if total == 0:
        return [np.nan, np.nan]
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [float(center - radius), float(center + radius)]


def exact_mcnemar_p(toward, away):
    discordant = toward + away
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(toward, away) + 1)) / (2**discordant)
    return float(min(1.0, 2 * tail))


def aggregate(rows):
    result = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["run_variant"], row["family"])].append(row)
    for (variant, family), group in sorted(grouped.items()):
        valid = [row for row in group if row["valid"]]
        condition_groups = defaultdict(list)
        for row in group:
            condition_groups[row["sample_id"]].append(row)
        conditions = []
        for sample_id, sample_rows in sorted(condition_groups.items()):
            probability = mean([row["prediction"] for row in sample_rows])
            conditions.append({
                "sample_id": sample_id,
                "margin": sample_rows[0]["margin"],
                "outcome": sample_rows[0]["outcome"],
                "n": len(sample_rows),
                "probability_outcome_1": probability,
                "proxy_mean": mean([row["proxy"] for row in sample_rows]),
                "proxy_std": float(np.nanstd([row["proxy"] for row in sample_rows])),
                "valid_rate": mean([row["valid"] for row in sample_rows]),
            })
        conditions.sort(key=lambda item: item["margin"])
        probabilities = [item["probability_outcome_1"] for item in conditions]
        labels = [item["outcome"] for item in conditions]
        violations = sum(right + 1e-12 < left for left, right in zip(probabilities, probabilities[1:]))
        correct = sum(row["prediction"] == row["outcome"] for row in group)
        result[f"{variant}/{family}"] = {
            "n": len(group),
            "accuracy": correct / len(group),
            "accuracy_wilson_95": wilson_interval(correct, len(group)),
            "valid_rate": mean([row["valid"] for row in group]),
            "valid_only_accuracy": mean([row["prediction"] == row["outcome"] for row in valid]),
            "positive_rate": mean([row["prediction"] for row in group]),
            "condition_brier": mean([(p - y) ** 2 for p, y in zip(probabilities, labels)]),
            "margin_probability_spearman": spearman([item["margin"] for item in conditions], probabilities),
            "monotonic_violations": int(violations),
            "mean_endpoint_mse": mean([row["mse_endpoint_gt"] for row in group]),
            "mean_motion_cosine": mean([row["motion_delta_cosine"] for row in group]),
            "mean_motion_ratio": mean([row["motion_magnitude_ratio"] for row in group]),
            "conditions": conditions,
        }
    return result


def paired_effect(rows, left_variant, right_variant, family=None):
    table = {(row["run_variant"], row["sample_id"], row["seed"]): row for row in rows}
    pairs = []
    for (variant, sample_id, seed), left in table.items():
        if variant != left_variant or (family and left["family"] != family):
            continue
        right = table.get((right_variant, sample_id, seed))
        if right is not None:
            pairs.append((left, right))
    toward = sum(left["prediction"] != left["outcome"] and right["prediction"] == right["outcome"] for left, right in pairs)
    away = sum(left["prediction"] == left["outcome"] and right["prediction"] != right["outcome"] for left, right in pairs)
    return {
        "n_pairs": len(pairs),
        "left_accuracy": mean([left["prediction"] == left["outcome"] for left, _ in pairs]),
        "right_accuracy": mean([right["prediction"] == right["outcome"] for _, right in pairs]),
        "toward_gt": toward,
        "away_from_gt": away,
        "exact_mcnemar_p": exact_mcnemar_p(toward, away),
        "prediction_flip_rate": mean([left["prediction"] != right["prediction"] for left, right in pairs]),
        "proxy_shift": mean([right["proxy"] - left["proxy"] for left, right in pairs]),
    }


def save_csv(path, rows):
    fields = [
        "run_variant", "family", "sample_id", "seed", "margin", "outcome", "proxy",
        "prediction", "correct", "valid", "area_ratio_min", "area_ratio_max",
        "mse_endpoint_gt", "mse_current_gt", "mse_endpoint_current",
        "motion_delta_cosine", "motion_magnitude_ratio", "elapsed_seconds", "output_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_probabilities(path, summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, family in zip(axes, FAMILIES):
        for variant, marker in zip(VARIANTS, ("o", "s", "^")):
            item = summary.get(f"{variant}/{family}")
            if not item:
                continue
            conditions = item["conditions"]
            axis.plot(
                [row["margin"] for row in conditions],
                [row["probability_outcome_1"] for row in conditions],
                marker=marker, label=variant,
            )
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set(title=family.title(), xlabel="analytic physics margin", ylabel="P(decoded outcome=1)", ylim=(-0.05, 1.05))
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_report(path, total_jobs, rows, rules, summary, comparisons, vae, collection_complete):
    lines = [
        "# Cosmos Predict2 corrected physics diagnostics",
        "",
        "Inference only. No checkpoint, optimizer state, or model weight was updated.",
        "",
        f"- Three-hour collection complete: **{bool(collection_complete)}**",
        f"- Persisted rollouts: **{len(rows)}** from a planned pool of {total_jobs}",
        f"- Tower GT decoder ceiling: **{rules['tower']['accuracy']:.1%}**",
        f"- Hill GT decoder ceiling: **{rules['hill']['accuracy']:.1%}**",
        "",
        "## Aggregate results",
        "",
        "A majority-class baseline is 57.1% in both seven-condition families.",
        "",
        "| variant/family | n | accuracy (Wilson 95%) | valid | condition Brier | margin Spearman | motion cosine |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in summary.items():
        lines.append(
            f"| {key} | {item['n']} | {item['accuracy']:.1%} "
            f"[{item['accuracy_wilson_95'][0]:.1%}, {item['accuracy_wilson_95'][1]:.1%}] | {item['valid_rate']:.1%} | "
            f"{item['condition_brier']:.3f} | {item['margin_probability_spearman']:+.3f} | {item['mean_motion_cosine']:+.3f} |"
        )
    lines += ["", "## Paired interventions", ""]
    for name, item in comparisons.items():
        lines.append(
            f"- **{name}**: {item['n_pairs']} pairs; left {item['left_accuracy']:.1%} → right "
            f"{item['right_accuracy']:.1%}; toward GT {item['toward_gt']}, away {item['away_from_gt']}, "
            f"flip rate {item['prediction_flip_rate']:.1%}, exact McNemar p={item['exact_mcnemar_p']:.4f}."
        )
    if vae:
        lines += [
            "",
            "## Frozen VAE reconstruction upper bound",
            "",
            f"- Conditions: {len(vae)}",
            f"- Current-frame PSNR: {mean([row['current_psnr'] for row in vae]):.2f} dB",
            f"- Endpoint PSNR: {mean([row['endpoint_psnr'] for row in vae]):.2f} dB",
        ]
    lines += [
        "",
        "## Decoder protocol",
        "",
        "Tower is decoded by displacement of the visible red, blue, and grey blocks; Hill by red-ball horizontal displacement. "
        "Both thresholds are calibrated once on MuJoCo endpoints only. Object presence/area validity is reported independently.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/physics_diagnostics_3h"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = load_jsonl(root / "manifest.jsonl")
    records = {row["id"]: row for row in manifest}
    current_images, gt_images = {}, {}
    for record in manifest:
        with np.load(root / record["input_npz"], allow_pickle=False) as data:
            current_images[record["id"]] = np.asarray(data["current_primary"])
            gt_images[record["id"]] = np.asarray(data["gt_future_primary"])

    rules = {}
    for family in FAMILIES:
        family_records = [row for row in manifest if row["family"] == family]
        values = [visible_proxy(family, gt_images[row["id"]], current_images[row["id"]]) for row in family_records]
        labels = [row["outcome"] for row in family_records]
        rules[family] = {**best_rule(values, labels), "gt_proxies": values, "gt_labels": labels}

    metadata_paths = sorted((root / "rollouts").glob("*/*/*/seed_*/metadata.json"))
    rows = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        record = records[metadata["sample_id"]]
        endpoint = np.asarray(Image.open(metadata_path.parent / "endpoint.png").convert("RGB"))
        current = current_images[record["id"]]
        value = visible_proxy(record["family"], endpoint, current)
        valid, ratio_min, ratio_max = validity(record["family"], endpoint, current)
        prediction = int(np.isfinite(value) and value >= rules[record["family"]]["threshold"])
        rows.append({
            **metadata,
            "proxy": value,
            "prediction": prediction,
            "correct": int(prediction == record["outcome"]),
            "valid": int(valid),
            "area_ratio_min": ratio_min,
            "area_ratio_max": ratio_max,
        })

    summary = aggregate(rows)
    comparisons = {
        "base→oracle (all)": paired_effect(rows, "base", "oracle"),
        "base→image_only (Hill)": paired_effect(rows, "base", "image_only", family="hill"),
    }
    vae = load_jsonl(root / "vae_reconstruction" / "results.jsonl")
    spec = json.loads((root / "collection_spec.json").read_text())
    total_jobs = int(spec["total_rollouts"])
    completion_path = root / "COLLECTION_COMPLETE.json"
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
    time_limited_complete = bool(completion.get("time_limited_collection_complete", False))
    result = {
        "completed_rollouts": len(rows),
        "planned_job_pool": total_jobs,
        "planned_job_pool_exhausted": len(rows) == total_jobs,
        "time_limited_collection_complete": time_limited_complete,
        "decoder_rules": rules,
        "aggregates": summary,
        "comparisons": comparisons,
        "vae_reconstruction": {
            "n": len(vae),
            "current_psnr_mean": mean([row["current_psnr"] for row in vae]),
            "endpoint_psnr_mean": mean([row["endpoint_psnr"] for row in vae]),
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=True))
    save_csv(output / "rollouts.csv", rows)
    plot_probabilities(output / "condition_probabilities.png", summary)
    markdown_report(output / "REPORT.md", total_jobs, rows, rules, summary, comparisons, vae, time_limited_complete)
    print(json.dumps({"completed": len(rows), "total": total_jobs, "rules": rules, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
