"""Verify persisted Cosmos diagnostic rollouts and write an integrity report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/physics_diagnostics_3h"))
    args = parser.parse_args()
    root = args.root.resolve()
    expected = []
    for path in sorted((root / "jobs").glob("shard_*.jsonl")):
        expected.extend(load_jsonl(path))
    expected_keys = {
        (job["run_variant"], job["sample_id"], int(job["seed"])) for job in expected
    }

    issues = []
    keys = []
    counts = Counter()
    video_bytes = 0
    metadata_paths = sorted((root / "rollouts").glob("*/*/*/seed_*/metadata.json"))
    for position, path in enumerate(metadata_paths, 1):
        try:
            row = json.loads(path.read_text())
            key = (row["run_variant"], row["sample_id"], int(row["seed"]))
            keys.append(key)
            counts[row["run_variant"]] += 1
            if key not in expected_keys:
                issues.append(f"unexpected key: {key}")
            endpoint_path, final_path, video_path = path.parent / "endpoint.png", path.parent / "final.png", path.parent / "rollout.mp4"
            for image_path in (endpoint_path, final_path):
                with Image.open(image_path) as image:
                    if image.size != (832, 480):
                        issues.append(f"bad image size {image_path}: {image.size}")
            reader = imageio.get_reader(video_path)
            meta = reader.get_meta_data()
            frame_count = reader.count_frames()
            reader.close()
            if tuple(meta["size"]) != (832, 480) or float(meta["fps"]) != 16.0 or frame_count != 21:
                issues.append(f"bad video {video_path}: size={meta['size']} fps={meta['fps']} frames={frame_count}")
            video_bytes += video_path.stat().st_size
        except Exception as error:
            issues.append(f"{path}: {type(error).__name__}: {error}")
        if position % 100 == 0:
            print(f"checked {position}/{len(metadata_paths)}", flush=True)

    duplicates = [list(key) for key, count in Counter(keys).items() if count > 1]
    missing = expected_keys - set(keys)
    completion_path = root / "COLLECTION_COMPLETE.json"
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
    report = {
        "completed": len(metadata_paths),
        "planned_job_pool": len(expected_keys),
        "planned_job_pool_exhausted": not missing,
        "time_limited_collection_complete": bool(completion.get("time_limited_collection_complete", False)),
        "counts_by_variant": dict(sorted(counts.items())),
        "duplicate_keys": duplicates,
        "missing_count": len(missing),
        "video_bytes": video_bytes,
        "issues": issues,
        "valid_completed_files": not duplicates and not issues,
    }
    output = root / "analysis" / "integrity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
