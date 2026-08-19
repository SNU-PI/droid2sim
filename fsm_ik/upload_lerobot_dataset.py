#!/usr/bin/env python3
"""Upload the local sponge pick-place LeRobot dataset to a private HF repo."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# Uploading a dataset does not need CUDA. Hide all GPUs before importing torch
# through LeRobot so this process cannot reserve VRAM on a shared machine.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROJECT_ROOT = Path(__file__).resolve().parent
LEROBOT_SRC = PROJECT_ROOT / "lerobot" / "src"
if LEROBOT_SRC.is_dir():
    sys.path.insert(0, str(LEROBOT_SRC))

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:
    raise SystemExit(
        "Missing LeRobot/Hugging Face dependencies. Run this script in the "
        "'lerobot' Conda environment.\n"
        "Example: conda run --no-capture-output -n lerobot python "
        f"{Path(__file__).resolve()}"
    ) from exc


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "lerobot_data" / "sponge_pick_place_50"
DEFAULT_REPO_ID = "ssangjunpark/droid_sim_sponge_pick_place_50"
DEFAULT_TAGS = [
    "lerobot",
    "smolvla",
    "isaac-sim",
    "franka",
    "imitation-learning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a local LeRobot dataset, including camera videos, to a "
            "private Hugging Face dataset repository."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Local LeRobot dataset directory (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Destination in OWNER/DATASET form (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Dataset-card tag. Repeat to supply several tags.",
    )
    parser.add_argument(
        "--license",
        default=None,
        help="Optional Hugging Face license identifier, for example apache-2.0.",
    )
    parser.add_argument(
        "--large-folder",
        action="store_true",
        help="Use the resumable large-folder uploader (unnecessary for the current 143 MB dataset).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate authentication and local files without creating or uploading the repository.",
    )
    return parser.parse_args()


def validate_local_dataset(root: Path) -> tuple[int, int]:
    required = [
        root / "meta" / "info.json",
        root / "meta" / "stats.json",
        root / "meta" / "tasks.parquet",
        root / "data",
        root / "videos",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Dataset is incomplete; missing:\n{formatted}")

    parquet_count = sum(1 for _ in (root / "data").rglob("*.parquet"))
    video_count = sum(1 for _ in (root / "videos").rglob("*.mp4"))
    if parquet_count == 0 or video_count == 0:
        raise RuntimeError(
            f"Expected episode Parquet files and MP4 videos, found "
            f"{parquet_count} Parquet files and {video_count} videos."
        )
    return parquet_count, video_count


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()

    if "/" not in args.repo_id or args.repo_id.startswith("/") or args.repo_id.endswith("/"):
        raise ValueError("--repo-id must use OWNER/DATASET form.")

    parquet_count, video_count = validate_local_dataset(dataset_root)

    api = HfApi()
    try:
        account = api.whoami()
    except HfHubHTTPError as exc:
        raise RuntimeError(
            "Hugging Face authentication failed. Run 'hf auth login' with a "
            "write-capable token, then retry."
        ) from exc

    account_name = account.get("name", "unknown")
    print(f"[hf-upload] Authenticated as: {account_name}", flush=True)
    print(f"[hf-upload] Local dataset: {dataset_root}", flush=True)
    print(
        f"[hf-upload] Found {parquet_count} episode Parquet files and "
        f"{video_count} camera videos.",
        flush=True,
    )

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        download_videos=False,
        video_backend="pyav",
    )
    print(
        f"[hf-upload] Dataset metadata: {dataset.num_episodes} episodes, "
        f"{dataset.num_frames} frames.",
        flush=True,
    )

    if args.dry_run:
        print("[hf-upload] Dry run passed; nothing was uploaded.", flush=True)
        return 0

    print(f"[hf-upload] Destination: {args.repo_id} (private)", flush=True)

    # Protect the data before uploading. create_repo handles a new destination;
    # update_repo_settings also makes an existing destination private first.
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    api.update_repo_settings(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=True,
    )

    dataset.push_to_hub(
        private=True,
        push_videos=True,
        tags=args.tags or DEFAULT_TAGS,
        license=args.license,
        upload_large_folder=args.large_folder,
    )

    info = api.dataset_info(args.repo_id)
    if info.private is not True:
        raise RuntimeError(
            f"Upload finished, but {args.repo_id} was not reported as private."
        )

    hub_file_count = len(info.siblings or [])
    print("[hf-upload] Upload completed successfully.", flush=True)
    print(f"[hf-upload] Private repository: {info.private}", flush=True)
    print(f"[hf-upload] Hub files: {hub_file_count}", flush=True)
    print(
        f"[hf-upload] URL: https://huggingface.co/datasets/{args.repo_id}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[hf-upload] Interrupted by user. It is safe to rerun.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[hf-upload] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
