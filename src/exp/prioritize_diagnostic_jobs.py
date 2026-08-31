"""Reorder existing shards so paired interventions finish before extra seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANT_ORDER = {"base": 0, "oracle": 1, "image_only": 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=Path)
    args = parser.parse_args()
    for path in sorted(args.job_dir.glob("shard_*.jsonl")):
        jobs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        def key(job):
            paired = int(job["seed"]) <= 24
            # Paired base/oracle/image-only seeds first, then additional base seeds.
            group = 0 if paired else 1
            return (
                group,
                int(job["seed"]),
                job["sample_id"],
                VARIANT_ORDER[job["run_variant"]],
            )

        jobs.sort(key=key)
        path.write_text("".join(json.dumps(job) + "\n" for job in jobs))
        print(path, len(jobs), jobs[0], jobs[-1])


if __name__ == "__main__":
    main()
