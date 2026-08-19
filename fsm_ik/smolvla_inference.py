#!/usr/bin/env python3
"""Load a local SmolVLA checkpoint for offline or localhost inference.

The ``dataset`` mode is a non-actuating checkpoint smoke test. The ``server``
mode keeps LeRobot in its own Conda environment and exposes a small, localhost-
only NumPy protocol for an Isaac Sim process running in another environment.
"""

from __future__ import annotations

import argparse
import io
import os
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
LEROBOT_SRC = PROJECT_ROOT / "lerobot" / "src"
if LEROBOT_SRC.is_dir():
    sys.path.insert(0, str(LEROBOT_SRC))

try:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
except ImportError as exc:
    raise SystemExit(
        "Run this script from the fresh LeRobot environment. Example:\n"
        "  conda run --no-capture-output -n lerobot2 python "
        f"{Path(__file__).resolve()} --help"
    ) from exc


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "lerobot"
    / "outputs"
    / "train"
    / "droid_sim_smolvla_smoke_camfix"
    / "checkpoints"
    / "000050"
    / "pretrained_model"
)
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "lerobot_data" / "sponge_pick_place_50"
DEFAULT_REPO_ID = "ssangjunpark/droid_sim_sponge_pick_place_50"
DEFAULT_TASK = "Pick up the sponge and place it in the frying pan."

MAIN_IMAGE_KEY = "observation.images.image"
WRIST_IMAGE_KEY = "observation.images.image2"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

PROTOCOL_VERSION = 2
LENGTH_STRUCT = struct.Struct("!Q")
MAX_PACKET_BYTES = 64 * 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("dataset", "server"),
        default="dataset",
        help="Run a local dataset-frame check or serve live observations on localhost.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint pretrained_model directory (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument("--device", default="cuda", help="PyTorch device, normally cuda or cpu.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Local LeRobot dataset used by dataset mode.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--num-actions",
        type=int,
        default=1,
        help="Number of consecutive dataset observations to query in dataset mode.",
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=50,
        help="Execute this many actions before replanning; 50 matches the checkpoint's trained chunk.",
    )
    return parser


def validate_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    missing = [path / name for name in required if not (path / name).is_file()]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"Checkpoint is incomplete; missing:\n{formatted}")
    return path


def as_chw_float_tensor(image: Any) -> torch.Tensor:
    tensor = image.detach().cpu() if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(f"Expected one 3D RGB image, got shape {tuple(tensor.shape)}.")
    if tensor.shape[-1] in (3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1)
    elif tensor.shape[0] == 4:
        tensor = tensor[:3]
    elif tensor.shape[0] != 3:
        raise ValueError(f"Cannot determine RGB channel axis for shape {tuple(tensor.shape)}.")
    tensor = tensor.contiguous()
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32).div_(255.0)
    else:
        tensor = tensor.to(torch.float32)
        if not torch.isfinite(tensor).all():
            raise ValueError("RGB input contains NaN or infinity.")
        if tensor.max().item() > 1.5:
            tensor = tensor.div(255.0)
    return tensor.clamp_(0.0, 1.0)


class SmolVLARunner:
    """Own a policy and its saved pre/postprocessing pipelines."""

    def __init__(self, checkpoint: Path, device: str, replan_steps: int) -> None:
        self.checkpoint = validate_checkpoint(checkpoint)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        if replan_steps < 0:
            raise ValueError("--replan-steps cannot be negative.")
        self.actions_since_plan = 0

        print(f"[smolvla] Loading checkpoint: {self.checkpoint}", flush=True)
        config = PreTrainedConfig.from_pretrained(str(self.checkpoint))
        config.device = str(self.device)
        policy_class = get_policy_class(config.type)
        self.policy = policy_class.from_pretrained(str(self.checkpoint), config=config)
        self.policy = self.policy.to(self.device).eval()
        self.replan_steps = replan_steps or int(self.policy.config.n_action_steps)
        if self.replan_steps <= 0:
            raise ValueError("The effective action-chunk execution length must be positive.")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.reset()

        parameter_count = sum(parameter.numel() for parameter in self.policy.parameters())
        print(
            f"[smolvla] Ready on {self.device}: {parameter_count:,} parameters, "
            f"chunk={self.policy.config.chunk_size}, execute/replan={self.replan_steps}.",
            flush=True,
        )

    def reset(self) -> None:
        self.policy.reset()
        self.actions_since_plan = 0

    def infer(
        self,
        main_rgb: Any,
        wrist_rgb: Any,
        state: Any,
        task: str,
    ) -> tuple[np.ndarray, float, int]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).flatten()
        if state_tensor.shape != (8,):
            raise ValueError(
                "Expected the training state [xyz_b, quaternion_xyzw_b, gripper_open_fraction] "
                f"with shape (8,), got {tuple(state_tensor.shape)}."
            )
        if not torch.isfinite(state_tensor).all():
            raise ValueError("State contains NaN or infinity.")
        if not task.strip():
            raise ValueError("Task instruction cannot be empty.")

        if self.actions_since_plan >= self.replan_steps:
            self.policy.reset()
            self.actions_since_plan = 0
        plan_step = self.actions_since_plan

        raw_frame = {
            MAIN_IMAGE_KEY: as_chw_float_tensor(main_rgb),
            WRIST_IMAGE_KEY: as_chw_float_tensor(wrist_rgb),
            STATE_KEY: state_tensor,
            "task": task.strip(),
        }
        start = time.perf_counter()
        with torch.inference_mode():
            processed = self.preprocessor(raw_frame)
            action = self.policy.select_action(processed)
            action = self.postprocessor(action)
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action)
        action_np = action.detach().to("cpu", dtype=torch.float32).numpy().reshape(-1)
        latency_s = time.perf_counter() - start
        if action_np.shape != (7,):
            raise RuntimeError(f"Policy returned action shape {action_np.shape}; expected (7,).")
        if not np.all(np.isfinite(action_np)):
            raise RuntimeError("Policy returned NaN or infinity.")
        self.actions_since_plan += 1
        return action_np, latency_s, plan_step


def run_dataset_mode(args: argparse.Namespace, runner: SmolVLARunner) -> int:
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root.expanduser().resolve(),
        download_videos=False,
        video_backend="pyav",
    )
    if args.frame_index < 0 or args.frame_index >= len(dataset):
        raise IndexError(f"--frame-index must be in [0, {len(dataset) - 1}].")
    if args.num_actions <= 0:
        raise ValueError("--num-actions must be positive.")

    final_index = min(len(dataset), args.frame_index + args.num_actions)
    print(
        f"[smolvla] Dataset check: frames {args.frame_index}..{final_index - 1} "
        f"from {args.dataset_root}.",
        flush=True,
    )
    for index in range(args.frame_index, final_index):
        sample = dataset[index]
        predicted, latency_s, plan_step = runner.infer(
            sample[MAIN_IMAGE_KEY],
            sample[WRIST_IMAGE_KEY],
            sample[STATE_KEY],
            str(sample.get("task", args.task)),
        )
        target = torch.as_tensor(sample[ACTION_KEY]).detach().cpu().numpy().reshape(-1)
        print(
            f"[smolvla] frame={index} plan_step={plan_step:02d} latency={latency_s:.3f}s "
            f"pred={np.array2string(predicted, precision=5, floatmode='fixed')} "
            f"target={np.array2string(target, precision=5, floatmode='fixed')}",
            flush=True,
        )
    return 0


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("Peer disconnected while a packet was being received.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_packet(connection: socket.socket) -> dict[str, np.ndarray]:
    (size,) = LENGTH_STRUCT.unpack(_recv_exact(connection, LENGTH_STRUCT.size))
    if size <= 0 or size > MAX_PACKET_BYTES:
        raise ValueError(f"Invalid packet size {size} bytes.")
    payload = _recv_exact(connection, size)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def send_packet(connection: socket.socket, **arrays: Any) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError(f"Outgoing packet is too large: {len(payload)} bytes.")
    connection.sendall(LENGTH_STRUCT.pack(len(payload)))
    connection.sendall(payload)


def scalar_string(value: np.ndarray, name: str) -> str:
    if value.size != 1:
        raise ValueError(f"{name} must contain one string.")
    return str(value.reshape(()).item())


def serve_connection(
    connection: socket.socket,
    address: tuple[str, int],
    runner: SmolVLARunner,
    default_task: str,
) -> bool:
    print(f"[smolvla-server] Client connected: {address[0]}:{address[1]}", flush=True)
    while True:
        try:
            request = recv_packet(connection)
        except EOFError:
            print("[smolvla-server] Client disconnected.", flush=True)
            return False

        command = scalar_string(request.get("command", np.asarray("infer")), "command")
        if command == "reset":
            runner.reset()
            send_packet(connection, ok=np.asarray(True), protocol=np.asarray(PROTOCOL_VERSION))
            continue
        if command == "shutdown":
            send_packet(connection, ok=np.asarray(True), protocol=np.asarray(PROTOCOL_VERSION))
            return True
        if command != "infer":
            send_packet(connection, ok=np.asarray(False), error=np.asarray(f"Unknown command: {command}"))
            continue

        try:
            task = scalar_string(request.get("task", np.asarray(default_task)), "task")
            action, latency_s, plan_step = runner.infer(
                request["main_rgb"],
                request["wrist_rgb"],
                request["state"],
                task,
            )
            send_packet(
                connection,
                ok=np.asarray(True),
                protocol=np.asarray(PROTOCOL_VERSION),
                action=action,
                latency_s=np.asarray(latency_s, dtype=np.float64),
                plan_step=np.asarray(plan_step, dtype=np.int64),
            )
        except Exception as exc:
            send_packet(
                connection,
                ok=np.asarray(False),
                error=np.asarray(f"{type(exc).__name__}: {exc}"),
            )


def run_server_mode(args: argparse.Namespace, runner: SmolVLARunner) -> int:
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("The policy server is intentionally restricted to localhost.")
    if not (1 <= args.port <= 65535):
        raise ValueError("--port must be in [1, 65535].")

    family = socket.AF_INET6 if args.host == "::1" else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(
            f"[smolvla-server] READY host={args.host} port={args.port} "
            f"protocol={PROTOCOL_VERSION}",
            flush=True,
        )
        while True:
            connection, address = server.accept()
            with connection:
                if serve_connection(connection, address, runner, args.task):
                    break
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and "CUDA_VISIBLE_DEVICES" not in os.environ:
        print(
            "[smolvla] WARNING: CUDA_VISIBLE_DEVICES is unset; set it to one physical GPU before launch.",
            file=sys.stderr,
        )
    runner = SmolVLARunner(args.checkpoint, args.device, args.replan_steps)
    if args.mode == "dataset":
        return run_dataset_mode(args, runner)
    return run_server_mode(args, runner)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[smolvla] Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[smolvla] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
