#!/usr/bin/env python3
"""Convert synchronized Isaac rollouts into an appendable LeRobot v3 dataset.

Run this in the Python 3.12 lerobot Conda environment after Isaac exits.
Conversion, video encoding, and validation are CPU-only. Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# Set this before LeRobot imports torch. The exporter must never reserve VRAM.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_LEROBOT_SRC = SCRIPT_DIR / "lerobot" / "src"
if not LOCAL_LEROBOT_SRC.is_dir():
    raise RuntimeError(f"Local LeRobot source tree is missing: {LOCAL_LEROBOT_SRC}")
sys.path.insert(0, str(LOCAL_LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

MAIN_IMAGE_KEY = "observation.images.image"
WRIST_IMAGE_KEY = "observation.images.image2"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
MANIFEST_NAME = "droid_sim_manifest.json"
DEFAULT_TASK = "Pick up the sponge and place it in the frying pan."


def log(message: str) -> None:
    print(f"[lerobot-export] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=str,
        nargs="+",
        required=True,
        help="Raw rollout directory/directories, or the literal latest.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=SCRIPT_DIR / "lerobot_data" / "sponge_pick_place",
        help="Local LeRobot dataset directory. Existing datasets are resumed.",
    )
    parser.add_argument(
        "--repo-id",
        default="ssangjunpark/droid_sim_sponge_pick_place",
        help="LeRobot repo id stored in metadata; this command never uploads.",
    )
    parser.add_argument("--task", default=None, help="Override the rollout language instruction.")
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Dataset rate. Raw synchronized frames are deterministically downsampled.",
    )
    parser.add_argument("--width", type=int, default=256, help="Output video width.")
    parser.add_argument("--height", type=int, default=256, help="Output video height.")
    parser.add_argument(
        "--allow-failed",
        action="store_true",
        help="Allow conversion when raw metadata does not say success=true.",
    )
    parser.add_argument(
        "--allow-world-frame-fallback",
        action="store_true",
        help="Allow old rollouts without tcp_pose_b. Not recommended for deployment data.",
    )
    parser.add_argument(
        "--skip-decode-check",
        action="store_true",
        help="Skip post-write decoding of the first and last training samples.",
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_id(run_dir: Path) -> str:
    digest = hashlib.sha256(run_dir.name.encode("utf-8"))
    for name in ("metadata.json", "trajectory.npz"):
        digest.update(sha256_file(run_dir / name).encode("ascii"))
    return digest.hexdigest()


def load_manifest(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "meta" / MANIFEST_NAME
    if not path.is_file():
        return {"schema_version": 1, "episodes": []}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(dataset_root: Path, manifest: dict[str, Any]) -> None:
    path = dataset_root / "meta" / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resolve_run_dirs(values: list[str]) -> list[Path]:
    available = sorted(
        (path for path in (SCRIPT_DIR / "outputs").glob("run_*") if path.is_dir()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    resolved: list[Path] = []
    for value in values:
        if value == "latest":
            if not available:
                raise FileNotFoundError(f"No raw rollouts found below {SCRIPT_DIR / 'outputs'}")
            path = available[-1]
        else:
            path = Path(value).expanduser()
            path = (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()
        if path not in resolved:
            resolved.append(path)
    return resolved


def normalized_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norms = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("Encountered a zero-length TCP quaternion.")
    return quaternion / norms

def continuous_quaternions_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Choose one temporally continuous sign for equivalent q/-q rotations."""
    quaternion = normalized_quaternion_wxyz(quaternion).copy()
    if quaternion[0, 0] < 0.0:
        quaternion[0] *= -1.0
    for index in range(1, quaternion.shape[0]):
        if float(np.dot(quaternion[index - 1], quaternion[index])) < 0.0:
            quaternion[index] *= -1.0
    return quaternion


def quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def relative_rotation_vectors(current_wxyz: np.ndarray, next_wxyz: np.ndarray) -> np.ndarray:
    current = normalized_quaternion_wxyz(current_wxyz)
    following = normalized_quaternion_wxyz(next_wxyz)
    inverse = current.copy()
    inverse[..., 1:] *= -1.0
    relative = normalized_quaternion_wxyz(quaternion_multiply_wxyz(following, inverse))
    relative = np.where(relative[..., :1] < 0.0, -relative, relative)
    vector_norm = np.linalg.norm(relative[..., 1:], axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[..., 0], 0.0, 1.0))
    scale = np.divide(
        angle,
        vector_norm,
        out=np.full_like(angle, 2.0),
        where=vector_norm > 1.0e-8,
    )
    return relative[..., 1:] * scale[..., None]


def sampling_indices(sim_time: np.ndarray, target_fps: int) -> np.ndarray:
    times = np.asarray(sim_time, dtype=np.float64)
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("sim_time must be strictly increasing and contain at least two frames.")
    source_period = float(np.median(np.diff(times)))
    source_fps = 1.0 / source_period
    if target_fps > source_fps * (1.0 + 1.0e-3):
        raise ValueError(f"Requested {target_fps} fps exceeds raw rate {source_fps:.6g} fps.")

    count = int(math.floor((times[-1] - times[0]) * target_fps + 1.0e-6)) + 1
    targets = times[0] + np.arange(count, dtype=np.float64) / target_fps
    right = np.clip(np.searchsorted(times, targets, side="left"), 0, times.size - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(
        np.abs(times[left] - targets) <= np.abs(times[right] - targets),
        left,
        right,
    ).astype(np.int64)
    if np.unique(indices).size != indices.size:
        raise ValueError("Requested rate maps multiple dataset frames to one raw frame.")
    max_error = float(np.max(np.abs(times[indices] - targets)))
    if max_error > 0.51 * source_period:
        raise ValueError(
            f"Raw timestamps cannot align to {target_fps} fps; max error is {max_error:.6f}s."
        )
    return indices


def resize_rgb(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def validate_raw_run(
    run_dir: Path,
    metadata: dict[str, Any],
    trajectory: Any,
    allow_failed: bool,
    allow_world_frame_fallback: bool,
) -> tuple[str, np.ndarray]:
    if not allow_failed and metadata.get("success") is not True:
        raise ValueError(
            f"Refusing unsuccessful rollout {run_dir}; use --allow-failed only if intentional."
        )
    required_arrays = {"sim_time", "gripper_position", "gripper_action"}
    missing = sorted(required_arrays.difference(trajectory.files))
    if missing:
        raise ValueError(f"{run_dir} is missing trajectory arrays: {missing}")

    if "tcp_pose_b" in trajectory.files:
        pose_key = "tcp_pose_b"
    elif allow_world_frame_fallback and "tcp_pose_w" in trajectory.files:
        pose_key = "tcp_pose_w"
        log(f"WARNING: {run_dir.name} has no base pose; using explicit world-frame fallback.")
    else:
        raise ValueError(
            f"{run_dir.name} predates base-frame recording. Re-record it, or explicitly pass "
            "--allow-world-frame-fallback (not recommended for deployment data)."
        )

    poses = np.asarray(trajectory[pose_key])
    if poses.ndim != 2 or poses.shape[1] != 7 or not np.all(np.isfinite(poses)):
        raise ValueError(f"{pose_key} must be a finite [frames, 7] array.")
    frame_count = poses.shape[0]
    for key in required_arrays:
        if np.asarray(trajectory[key]).shape[0] != frame_count:
            raise ValueError(f"Trajectory field {key} is not synchronized with {pose_key}.")
    for image_dir in (run_dir / "rgb", run_dir / "rgb_wrist"):
        count = len(list(image_dir.glob("*.png")))
        if count != frame_count:
            raise ValueError(f"Expected {frame_count} PNGs in {image_dir}, found {count}.")
    return pose_key, poses


def make_features(width: int, height: int) -> dict[str, dict[str, Any]]:
    video = {
        "dtype": "video",
        "shape": (height, width, 3),
        "names": ["height", "width", "channel"],
    }
    return {
        MAIN_IMAGE_KEY: dict(video),
        WRIST_IMAGE_KEY: dict(video),
        STATE_KEY: {"dtype": "float32", "shape": (8,), "names": ["state"]},
        ACTION_KEY: {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }


def require_matching_dataset(
    dataset: LeRobotDataset,
    repo_id: str,
    fps: int,
    features: dict[str, dict[str, Any]],
) -> None:
    if dataset.meta.repo_id != repo_id:
        raise ValueError(
            f"Existing dataset repo_id is {dataset.meta.repo_id!r}, not {repo_id!r}."
        )
    if not math.isclose(float(dataset.meta.fps), float(fps)):
        raise ValueError(f"Existing dataset fps is {dataset.meta.fps}, not {fps}.")
    for key, expected in features.items():
        actual = dataset.meta.features.get(key)
        if actual is None:
            raise ValueError(f"Existing dataset is missing feature {key}.")
        compatible = (
            actual["dtype"] == expected["dtype"]
            and tuple(actual["shape"]) == tuple(expected["shape"])
        )
        if not compatible:
            raise ValueError(f"Existing feature {key} is incompatible: {actual}")


def open_writer(
    dataset_root: Path,
    repo_id: str,
    fps: int,
    features: dict[str, dict[str, Any]],
) -> LeRobotDataset:
    encoder_threads = max(1, min(8, os.cpu_count() or 1))
    if (dataset_root / "meta" / "info.json").is_file():
        dataset = LeRobotDataset.resume(
            repo_id=repo_id,
            root=dataset_root,
            video_backend="pyav",
            encoder_threads=encoder_threads,
        )
        require_matching_dataset(dataset, repo_id, fps, features)
        log(f"Resuming {dataset_root} at episode {dataset.meta.total_episodes}.")
        return dataset
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise ValueError(f"Dataset root is nonempty but not a LeRobot dataset: {dataset_root}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        robot_type="panda",
        fps=fps,
        features=features,
        use_videos=True,
        video_backend="pyav",
        encoder_threads=encoder_threads,
    )
    log(f"Created local LeRobot v3 dataset at {dataset_root}.")
    return dataset


def build_training_arrays(
    poses_wxyz: np.ndarray,
    gripper_position: np.ndarray,
    gripper_action: np.ndarray,
    open_position: float,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(poses_wxyz[indices], dtype=np.float64)
    quaternions = continuous_quaternions_wxyz(poses[:, 3:7])
    opening = np.clip(
        np.asarray(gripper_position, dtype=np.float64)[indices] / open_position,
        0.0,
        1.0,
    )
    # State: measured base-frame xyz, quaternion xyzw, measured opening fraction.
    states = np.concatenate(
        (poses[:, :3], quaternions[:, [1, 2, 3, 0]], opening[:, None]),
        axis=1,
    )

    # Frame t predicts the feasible motion actually observed at frame t+1.
    translation_delta = np.zeros((indices.size, 3), dtype=np.float64)
    rotation_delta = np.zeros((indices.size, 3), dtype=np.float64)
    translation_delta[:-1] = poses[1:, :3] - poses[:-1, :3]
    rotation_delta[:-1] = relative_rotation_vectors(
        quaternions[:-1],
        quaternions[1:],
    )
    commands = np.asarray(gripper_action, dtype=np.float64)[indices]
    if not np.all(np.isin(commands, (-1.0, 1.0))):
        raise ValueError("gripper_action must contain only -1 (close) and +1 (open).")
    actions = np.concatenate(
        (translation_delta, rotation_delta, commands[:, None]),
        axis=1,
    )
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise ValueError("Generated state/action contains NaN or infinity.")
    return states.astype(np.float32), actions.astype(np.float32)


def append_run(
    dataset: LeRobotDataset,
    run_dir: Path,
    metadata: dict[str, Any],
    trajectory: Any,
    pose_key: str,
    poses: np.ndarray,
    indices: np.ndarray,
    task_override: str | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    open_position = float(metadata.get("gripper_open_position", 0.04))
    if open_position <= 0.0:
        raise ValueError("gripper_open_position must be positive.")
    states, actions = build_training_arrays(
        poses,
        trajectory["gripper_position"],
        trajectory["gripper_action"],
        open_position,
        indices,
    )
    task = (task_override or metadata.get("task_description") or DEFAULT_TASK).strip()
    if not task:
        raise ValueError("Task description cannot be empty.")

    main_paths = sorted((run_dir / "rgb").glob("*.png"))
    wrist_paths = sorted((run_dir / "rgb_wrist").glob("*.png"))
    episode_index = dataset.meta.total_episodes
    for output_frame, raw_index in enumerate(indices):
        dataset.add_frame(
            {
                MAIN_IMAGE_KEY: resize_rgb(main_paths[int(raw_index)], width, height),
                WRIST_IMAGE_KEY: resize_rgb(wrist_paths[int(raw_index)], width, height),
                STATE_KEY: states[output_frame],
                ACTION_KEY: actions[output_frame],
                "task": task,
            }
        )
        written = output_frame + 1
        if written == 1 or written % 50 == 0 or written == indices.size:
            log(f"episode={episode_index} frames={written}/{indices.size}")
    dataset.save_episode(parallel_encoding=True)
    log(f"Saved episode {episode_index}; encoded both camera streams.")
    episode_record = {
        "episode_index": episode_index,
        "source_run": run_dir.name,
        "source_id": source_id(run_dir),
        "source_pose_field": pose_key,
        "source_fps": float(metadata["camera_fps"]),
        "dataset_fps": float(dataset.meta.fps),
        "source_frames": int(poses.shape[0]),
        "dataset_frames": int(indices.size),
        "task": task,
    }
    # Keep the dataset episode index separate from the requested source-position
    # index. If one of the 50 positions fails, later successful positions are
    # still packed contiguously as LeRobot episodes while remaining auditable.
    if "episode_index" in metadata:
        episode_record["source_episode_index"] = int(metadata["episode_index"])
    if "initial_sponge_position_w_xyz" in metadata:
        episode_record["initial_sponge_position_w_xyz"] = [
            float(value) for value in metadata["initial_sponge_position_w_xyz"]
        ]
    return episode_record


def validate_dataset(dataset_root: Path, repo_id: str, decode: bool) -> LeRobotDataset:
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )
    if len(dataset) != dataset.meta.total_frames:
        raise RuntimeError("Readable frame count disagrees with LeRobot metadata.")
    if dataset.num_episodes != dataset.meta.total_episodes:
        raise RuntimeError("Readable episode count disagrees with LeRobot metadata.")
    expected = {MAIN_IMAGE_KEY, WRIST_IMAGE_KEY, STATE_KEY, ACTION_KEY}
    if not expected.issubset(dataset.features):
        raise RuntimeError(
            f"Dataset is missing training features: {sorted(expected - set(dataset.features))}"
        )

    if decode:
        expected_image_shape = (
            3,
            dataset.meta.features[MAIN_IMAGE_KEY]["shape"][0],
            dataset.meta.features[MAIN_IMAGE_KEY]["shape"][1],
        )
        for index in sorted({0, len(dataset) - 1}):
            sample = dataset[index]
            if tuple(sample[STATE_KEY].shape) != (8,):
                raise RuntimeError(f"Unexpected state shape at frame {index}.")
            if tuple(sample[ACTION_KEY].shape) != (7,):
                raise RuntimeError(f"Unexpected action shape at frame {index}.")
            for key in (MAIN_IMAGE_KEY, WRIST_IMAGE_KEY):
                if tuple(sample[key].shape) != expected_image_shape:
                    raise RuntimeError(
                        f"Unexpected decoded shape {tuple(sample[key].shape)} for {key}."
                    )
        log("Decoded first and last synchronized samples with LeRobot/PyAV.")
    return dataset


def main() -> int:
    args = build_parser().parse_args()
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("fps, width, and height must be positive.")

    dataset_root = args.dataset_root.expanduser().resolve()
    features = make_features(args.width, args.height)
    manifest = load_manifest(dataset_root)
    prior_source_ids = {
        episode["source_id"] for episode in manifest.get("episodes", [])
    }

    prepared: list[tuple[Path, dict[str, Any], Any, str, np.ndarray, np.ndarray]] = []
    for run_dir in resolve_run_dirs(args.run_dir):
        metadata_path = run_dir / "metadata.json"
        trajectory_path = run_dir / "trajectory.npz"
        if not metadata_path.is_file() or not trajectory_path.is_file():
            raise FileNotFoundError(f"Not a complete raw rollout directory: {run_dir}")
        identifier = source_id(run_dir)
        if identifier in prior_source_ids:
            log(f"Skipping already exported source {run_dir.name}.")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        trajectory = np.load(trajectory_path, allow_pickle=False)
        pose_key, poses = validate_raw_run(
            run_dir,
            metadata,
            trajectory,
            args.allow_failed,
            args.allow_world_frame_fallback,
        )
        indices = sampling_indices(trajectory["sim_time"], args.fps)
        prepared.append((run_dir, metadata, trajectory, pose_key, poses, indices))
        log(
            f"Prepared {run_dir.name}: {poses.shape[0]} raw frames -> "
            f"{indices.size} frames at {args.fps} fps."
        )

    if not prepared:
        log("Nothing new to export.")
        return 0

    dataset = open_writer(dataset_root, args.repo_id, args.fps, features)
    exported: list[dict[str, Any]] = []
    try:
        for item in prepared:
            exported.append(
                append_run(dataset, *item, args.task, args.width, args.height)
            )
    finally:
        for item in prepared:
            item[2].close()
        dataset.finalize()
    log("Finalized Parquet footers, episode metadata, statistics, and videos.")

    validated = validate_dataset(
        dataset_root,
        args.repo_id,
        decode=not args.skip_decode_check,
    )
    manifest.update(
        {
            "schema_version": 1,
            "format": "LeRobot v3.0",
            "repo_id": args.repo_id,
            "robot_type": "panda",
            "camera_mapping": {
                MAIN_IMAGE_KEY: "main",
                WRIST_IMAGE_KEY: "wrist",
            },
            "state": {
                "key": STATE_KEY,
                "layout": [
                    "x_b",
                    "y_b",
                    "z_b",
                    "qx_b",
                    "qy_b",
                    "qz_b",
                    "qw_b",
                    "gripper_open_fraction",
                ],
                "semantics": (
                    "measured TCP pose in the recorded control frame plus measured "
                    "per-finger opening normalized to [0,1]"
                ),
            },
            "action": {
                "key": ACTION_KEY,
                "layout": [
                    "delta_x_b",
                    "delta_y_b",
                    "delta_z_b",
                    "rotation_vector_x_b",
                    "rotation_vector_y_b",
                    "rotation_vector_z_b",
                    "gripper",
                ],
                "semantics": (
                    "executed TCP delta from observation t to t+1 in the recorded "
                    "control frame; terminal arm delta is zero"
                ),
                "gripper": {"open": 1.0, "close": -1.0},
            },
            "episodes": [*manifest.get("episodes", []), *exported],
        }
    )
    write_manifest(dataset_root, manifest)
    log(
        f"VALID: {validated.num_episodes} episode(s), {len(validated)} frames, "
        f"two {args.width}x{args.height} video streams. Local only: {dataset_root}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"ERROR: {type(exc).__name__}: {exc}")
        raise
