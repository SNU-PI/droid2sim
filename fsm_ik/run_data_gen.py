#!/usr/bin/env python3
"""Headless Franka pick-and-place rollout for ``data_gen.usda``.

The script binds to the assets already authored in the stage:

* ``/World/franka``
* ``/World/sponge_manipulation``
* ``/World/frying_pan``
* ``/World/main``
* ``/World/franka/panda_hand/wrist``

It resets the Panda to Isaac Lab's neutral joint pose, creates task-space
waypoint Xforms under ``/World/DataGenWaypoints``, moves the sponge to the pan
with adaptive damped-least-squares differential IK, null-space posture
stabilization, and records camera-synchronized trajectory data.

Typical usage (the script forces headless camera rendering even if the flags
are omitted)::

    conda run --no-capture-output -n env_isaaclab python \
        /path/to/run_data_gen.py --headless --enable_cameras

The existing camera poses and optics are preserved. Synchronized images and
MP4s (when an FFmpeg backend is available), metadata, and trajectory arrays
are written to a timestamped directory below ``outputs/``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import socket
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# This must be set before AppLauncher imports/starts Kit. Respect a caller's
# single physical-GPU selection and default to physical GPU 0. Whichever GPU
# is exposed becomes this process's only device and is addressed as cuda:0.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
visible_cuda_device = os.environ["CUDA_VISIBLE_DEVICES"]
if not visible_cuda_device.isdigit():
    raise RuntimeError(
        "CUDA_VISIBLE_DEVICES must select exactly one numeric physical GPU."
    )
physical_gpu_id = int(visible_cuda_device)

from isaaclab.app import AppLauncher


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        type=Path,
        default=SCRIPT_DIR / "data_gen.usda",
        help="USD stage to open. Defaults to data_gen.usda beside this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
        help="Parent directory for timestamped rollout outputs.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Hard-coded sponge-position episode index in [0, 49].",
    )
    parser.add_argument(
        "--validation-index",
        type=int,
        default=None,
        help="Inference-only held-out sponge position index in [0, 9]; overrides episode-index.",
    )
    parser.add_argument(
        "--preview-sponge-positions",
        action="store_true",
        help="Render one main/wrist image for all 50 sponge positions, then exit without a rollout.",
    )
    parser.add_argument(
        "--preview-output-dir",
        type=Path,
        default=SCRIPT_DIR / "tmp" / "sponge_position_previews",
        help="Parent directory for timestamped 50-position preview outputs.",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default="Pick up the sponge and place it in the frying pan.",
        help="Language instruction stored with the rollout for VLA training.",
    )
    parser.add_argument("--width", type=int, default=640, help="Rendered RGB width.")
    parser.add_argument("--height", type=int, default=480, help="Rendered RGB height.")
    parser.add_argument("--camera-fps", type=float, default=20.0, help="RGB and trajectory capture rate.")
    parser.add_argument("--physics-dt", type=float, default=0.01, help="Physics time step in seconds.")
    parser.add_argument("--max-runtime", type=float, default=90.0, help="Maximum rollout duration in seconds.")
    parser.add_argument("--state-timeout", type=float, default=25.0, help="Timeout for each motion state.")
    parser.add_argument("--neutral-settle-time", type=float, default=1.0, help="Neutral-pose settling time in s.")
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="Seconds between progress printouts; set to 0 to disable periodic status.",
    )
    parser.add_argument("--settle-time", type=float, default=0.25, help="Required in-tolerance time per waypoint.")
    parser.add_argument("--position-tolerance", type=float, default=0.025, help="Waypoint position tolerance in m.")
    parser.add_argument(
        "--pre-grasp-position-tolerance",
        type=float,
        default=0.010,
        help="Position tolerance for PRE_GRASP in m.",
    )
    parser.add_argument(
        "--grasp-position-tolerance",
        type=float,
        default=0.005,
        help="Strict TCP position tolerance for GRASP and CLOSE in m.",
    )
    parser.add_argument(
        "--orientation-tolerance",
        type=float,
        default=math.radians(10.0),
        help="Waypoint orientation tolerance in radians.",
    )
    parser.add_argument(
        "--grasp-orientation-tolerance",
        type=float,
        default=math.radians(3.0),
        help="Strict TCP orientation tolerance for PRE_GRASP, GRASP, and CLOSE in radians.",
    )
    parser.add_argument("--pre-grasp-height", type=float, default=0.10, help="Pre-grasp clearance in m.")
    parser.add_argument(
        "--grasp-descent-speed",
        type=float,
        default=0.025,
        help="Average Cartesian speed for the final descent to the sponge in m/s.",
    )
    parser.add_argument("--lift-speed", type=float, default=0.025, help="Average Cartesian lift speed in m/s.")
    parser.add_argument(
        "--place-descent-speed",
        type=float,
        default=0.025,
        help="Average Cartesian speed while lowering to the pan in m/s.",
    )
    parser.add_argument(
        "--retreat-speed",
        type=float,
        default=0.025,
        help="Average Cartesian speed while rising from the pan in m/s.",
    )
    parser.add_argument(
        "--transport-speed",
        type=float,
        default=0.05,
        help="Average TCP speed along the smooth LIFT-to-pan transport curve in m/s.",
    )
    parser.add_argument(
        "--trajectory-seed",
        type=int,
        default=20260817,
        help="Base random seed; episode-index is added to produce a reproducible episode path.",
    )
    parser.add_argument(
        "--transport-noise-std",
        type=float,
        nargs=3,
        default=(0.008, 0.003, 0.008),
        metavar=("X", "Y", "Z"),
        help="Gaussian XYZ std in meters for each of two transport-curve control points.",
    )
    parser.add_argument(
        "--release-noise-std",
        type=float,
        nargs=3,
        default=(0.002, 0.0005, 0.002),
        metavar=("X", "Y", "Z"),
        help="Small Gaussian XYZ std in meters shared by ABOVE_PAN, PLACE, and RETREAT.",
    )
    parser.add_argument(
        "--noise-clip-sigma",
        type=float,
        default=2.0,
        help="Clip all sampled Cartesian noise to this many standard deviations.",
    )
    parser.add_argument("--lift-height", type=float, default=0.15, help="Lift height above the sponge in m.")
    parser.add_argument("--place-clearance", type=float, default=0.05, help="TCP clearance above the pan in m.")
    parser.add_argument(
        "--grasp-height-bias",
        type=float,
        default=0.0,
        help="Vertical offset added to the sponge bounding-box center in m.",
    )
    parser.add_argument("--tcp-offset", type=float, default=0.107, help="panda_hand to grasp TCP offset in m.")
    parser.add_argument("--open-position", type=float, default=0.04, help="Target for each open finger in m.")
    parser.add_argument(
        "--closed-position",
        type=float,
        default=0.0,
        help="Fully-closed command for each finger in m; rigid contact stops the actual fingers.",
    )
    parser.add_argument(
        "--gripper-effort-limit",
        type=float,
        default=40.0,
        help="Per-finger PhysX drive effort limit in N.",
    )
    parser.add_argument(
        "--gripper-velocity-limit",
        type=float,
        default=0.03,
        help="Per-finger PhysX linear-drive velocity limit in m/s.",
    )
    parser.add_argument(
        "--gripper-position-tolerance",
        type=float,
        default=0.002,
        help="Position tolerance used to detect completed open/contact motion in m.",
    )
    parser.add_argument(
        "--grasp-contact-position-tolerance",
        type=float,
        default=0.006,
        help="Allowed per-finger closure below the sponge's expected contact position in m.",
    )
    parser.add_argument(
        "--grasp-min-effort",
        type=float,
        default=1.0,
        help="Minimum measured per-finger effort required to accept a grasp in N.",
    )
    parser.add_argument(
        "--grasp-max-tcp-sponge-distance",
        type=float,
        default=0.025,
        help="Maximum TCP-to-sponge-root distance while grasped and carried in m.",
    )
    parser.add_argument(
        "--pan-contact-force-threshold",
        type=float,
        default=0.01,
        help="Minimum filtered sponge-to-pan contact force used for placement success in N.",
    )
    parser.add_argument(
        "--pan-contact-settle-time",
        type=float,
        default=0.25,
        help="Required continuous post-release sponge-to-pan contact time in seconds.",
    )
    parser.add_argument(
        "--placement-max-sponge-speed",
        type=float,
        default=0.05,
        help="Maximum sponge linear speed accepted at final task success in m/s.",
    )
    parser.add_argument("--close-hold", type=float, default=0.8, help="Hold time after full closure in seconds.")
    parser.add_argument("--open-hold", type=float, default=0.7, help="Hold time after fully opening.")
    parser.add_argument("--final-hold", type=float, default=1.0, help="Final home-frame recording time in seconds.")
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.020,
        help="Hard safety cap on each integrated arm-drive target step in rad.",
    )
    parser.add_argument(
        "--ik-position-gain",
        type=float,
        default=4.0,
        help="Proportional gain from Cartesian position error to desired linear velocity.",
    )
    parser.add_argument(
        "--ik-orientation-gain",
        type=float,
        default=3.0,
        help="Proportional gain from axis-angle error to desired angular velocity.",
    )
    parser.add_argument(
        "--ik-max-linear-speed",
        type=float,
        default=0.15,
        help="Maximum IK-requested hand linear speed in m/s.",
    )
    parser.add_argument(
        "--ik-max-angular-speed",
        type=float,
        default=0.8,
        help="Maximum IK-requested hand angular speed in rad/s.",
    )
    parser.add_argument(
        "--arm-command-velocity-limit",
        type=float,
        default=0.35,
        help="Maximum magnitude of each persistent arm-drive velocity target in rad/s.",
    )
    parser.add_argument(
        "--arm-command-acceleration-limit",
        type=float,
        default=1.0,
        help="Maximum change rate of each arm-drive velocity target in rad/s^2.",
    )
    parser.add_argument(
        "--arm-following-error-limit",
        type=float,
        default=0.06,
        help="Maximum persistent drive-target offset from measured arm position in rad.",
    )
    parser.add_argument(
        "--ik-damping-min",
        type=float,
        default=0.03,
        help="DLS damping used away from singular configurations.",
    )
    parser.add_argument(
        "--ik-damping-max",
        type=float,
        default=0.20,
        help="Maximum DLS damping used close to a singular configuration.",
    )
    parser.add_argument(
        "--ik-singularity-threshold",
        type=float,
        default=0.08,
        help="Smallest Jacobian singular value below which damping is increased.",
    )
    parser.add_argument(
        "--ik-posture-gain",
        type=float,
        default=0.50,
        help="Null-space velocity gain toward a validated redundant-arm posture.",
    )
    parser.add_argument(
        "--ik-joint-limit-gain",
        type=float,
        default=0.25,
        help="Null-space velocity gain that repels joints from their soft limits.",
    )
    parser.add_argument(
        "--ik-joint-limit-margin",
        type=float,
        default=0.15,
        help="Fraction of each joint range over which joint-limit repulsion activates.",
    )
    parser.add_argument(
        "--divergence-window",
        type=float,
        default=1.5,
        help="Seconds of sustained post-trajectory error growth before failing early.",
    )
    parser.add_argument(
        "--divergence-position-growth",
        type=float,
        default=0.04,
        help="Post-trajectory position-error growth in m considered divergent.",
    )
    parser.add_argument(
        "--divergence-orientation-growth",
        type=float,
        default=math.radians(20.0),
        help="Post-trajectory orientation-error growth in rad considered divergent.",
    )
    parser.add_argument("--arm-stiffness", type=float, default=160.0, help="PhysX arm-drive stiffness.")
    parser.add_argument("--arm-damping", type=float, default=30.0, help="PhysX arm-drive damping.")
    parser.add_argument(
        "--vla-inference",
        action="store_true",
        help="Run closed-loop inference from a localhost SmolVLA policy server instead of the waypoint state machine.",
    )
    parser.add_argument("--policy-host", default="127.0.0.1", help="Local SmolVLA server host.")
    parser.add_argument("--policy-port", type=int, default=5555, help="Local SmolVLA server TCP port.")
    parser.add_argument(
        "--policy-connect-timeout",
        type=float,
        default=120.0,
        help="Maximum seconds to wait for the local SmolVLA server.",
    )
    parser.add_argument(
        "--policy-max-translation-step",
        type=float,
        default=0.010,
        help="Safety cap on each predicted base-frame TCP translation delta in m.",
    )
    parser.add_argument(
        "--policy-max-rotation-step",
        type=float,
        default=0.060,
        help="Safety cap on each predicted axis-angle rotation delta in rad.",
    )
    parser.add_argument(
        "--policy-max-position-lag",
        type=float,
        default=0.050,
        help=(
            "Maximum distance in m between the actual TCP and the bounded IK servo target; "
            "the persistent policy command is retained beyond this bound."
        ),
    )
    parser.add_argument(
        "--policy-max-orientation-lag",
        type=float,
        default=math.radians(20.0),
        help=(
            "Maximum angular distance in rad between the actual TCP and the bounded IK servo "
            "target; the persistent policy command is retained beyond this bound."
        ),
    )
    parser.add_argument(
        "--policy-servo-position-gain",
        type=float,
        default=30.0,
        help="Inference Cartesian position gain; 30 tracks about 95% of a target delta per 0.1 s.",
    )
    parser.add_argument(
        "--policy-servo-orientation-gain",
        type=float,
        default=30.0,
        help="Inference Cartesian orientation gain; physical speed/acceleration limits remain active.",
    )
    parser.add_argument(
        "--policy-gripper-threshold",
        type=float,
        default=0.25,
        help="Hysteresis threshold for converting the predicted gripper scalar to -1/+1.",
    )
    parser.add_argument(
        "--policy-workspace-min",
        type=float,
        nargs=3,
        default=(0.435, -0.281, 0.080),
        metavar=("X", "Y", "Z"),
        help="Minimum allowed TCP XYZ in the robot base frame.",
    )
    parser.add_argument(
        "--policy-workspace-max",
        type=float,
        nargs=3,
        default=(0.800, 0.060, 0.413),
        metavar=("X", "Y", "Z"),
        help="Maximum allowed TCP XYZ in the robot base frame.",
    )
    parser.add_argument("--sponge-scale", type=float, default=0.8, help="Uniform manipulation-sponge scale.")
    parser.add_argument(
        "--stage-load-timeout",
        type=float,
        default=300.0,
        help="Maximum time to wait for USD payloads to load.",
    )
    parser.add_argument(
        "--save-waypoints",
        action="store_true",
        help="Persist generated waypoint Xforms into the small waypoint overlay layer.",
    )
    parser.add_argument(
        "--show-waypoints",
        action="store_true",
        help="Show colored, non-colliding waypoint markers in the stage and camera output.",
    )
    parser.add_argument(
        "--waypoints-only",
        action="store_true",
        help="Author and save visible waypoint markers, then exit before starting the rollout.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Save PNG frames and trajectory data but do not attempt MP4 encoding.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = build_parser()
args_cli = parser.parse_args()

# This machine is headless and the camera is part of the requested observation.
# Set these before constructing AppLauncher so the RTX offscreen pipeline starts.
args_cli.headless = True
args_cli.enable_cameras = True
args_cli.device = "cuda:0"
if args_cli.waypoints_only:
    args_cli.show_waypoints = True

single_gpu_kit_args = (
    f"--/renderer/activeGpu={physical_gpu_id} "
    "--/renderer/multiGpu/enabled=false "
    "--/renderer/multiGpu/autoEnable=false "
    "--/renderer/multiGpu/maxGpuCount=1 "
    "--/physics/cudaDevice=0"
)
existing_kit_args = getattr(args_cli, "kit_args", "") or ""
args_cli.kit_args = f"{existing_kit_args} {single_gpu_kit_args}".strip()

print(
    f"[data-gen] GPU isolation: physical GPU {physical_gpu_id} is the only CUDA-visible device; "
    "Isaac Lab/PhysX use logical cuda:0.",
    flush=True,
)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Sim / Isaac Lab imports must happen after AppLauncher starts Kit.
import numpy as np
import omni.usd
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)
from isaaclab_assets import FRANKA_PANDA_CFG
from pxr import Gf, Sdf, Usd, UsdGeom


ROBOT_PATH = "/World/franka"
HAND_PATH = f"{ROBOT_PATH}/panda_hand"
SPONGE_PATH = "/World/sponge_manipulation"
PAN_PATH = "/World/frying_pan"
PAN_COLLISION_PATH = f"{PAN_PATH}/pan_collision"
CAMERA_PATHS = {
    "main": "/World/main",
    "wrist": f"{HAND_PATH}/wrist",
}
WAYPOINT_ROOT = "/World/DataGenWaypoints"
WAYPOINT_LAYER_NAME = "data_gen_waypoints.usda"

WAYPOINT_COLORS = {
    "home": (0.75, 0.75, 0.75),
    "pre_grasp": (0.0, 0.8, 1.0),
    "grasp": (1.0, 0.1, 0.1),
    "lift": (1.0, 0.8, 0.0),
    "above_pan": (0.8, 0.2, 1.0),
    "place": (0.1, 1.0, 0.2),
    "retreat": (0.2, 0.4, 1.0),
}

ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
GRIPPER_OPEN_ACTION = 1.0
GRIPPER_CLOSE_ACTION = -1.0
NEUTRAL_JOINT_POSITION = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}


# Null-space-only posture references extracted from the previously successful
# episode-0 rollout. They select a validated redundant IK branch; they are
# never written directly as articulation state or drive targets.
GRASP_BRANCH_JOINT_POSITION = (
    -0.3222761,
    0.7417217,
    0.3561109,
    -1.5900592,
    -0.3076719,
    2.2630899,
    2.5036349,
)
LIFT_BRANCH_JOINT_POSITION = (
    -0.1918476,
    0.5205809,
    0.2073733,
    -1.5079854,
    -0.1027308,
    1.9299217,
    2.3991275,
)
PAN_BRANCH_JOINT_POSITION = (
    -0.4086573,
    1.1509271,
    0.3772190,
    -0.4360422,
    -0.3682108,
    1.5062273,
    2.1080210,
)


# Fixed episode table in world XYZ coordinates. Episode 0 is the mandatory,
# previously verified pose; episodes 1-4 are the supplied limits; the remaining
# poses are a deterministic low-discrepancy sample of the enclosed X-Z region.
SPONGE_EPISODE_POSITIONS_W = (
    (0.394980, 1.097700, 0.254370),  # episode 00: verified
    (0.348550, 1.097700, 0.240000),  # episode 01: reachable-region corner
    (0.417170, 1.097700, 0.240000),  # episode 02: reachable-region corner
    (0.348550, 1.097700, 0.302490),  # episode 03: corner
    (0.417170, 1.097700, 0.302490),  # episode 04: corner
    (0.382860, 1.097700, 0.260830),  # episode 05
    (0.365705, 1.097700, 0.281660),  # episode 06
    (0.400015, 1.097700, 0.246943),  # episode 07
    (0.357128, 1.097700, 0.267773),  # episode 08
    (0.391437, 1.097700, 0.288603),  # episode 09
    (0.374283, 1.097700, 0.253887),  # episode 10
    (0.408592, 1.097700, 0.274717),  # episode 11
    (0.352839, 1.097700, 0.295547),  # episode 12
    (0.387149, 1.097700, 0.242314),  # episode 13
    (0.369994, 1.097700, 0.263144),  # episode 14
    (0.404304, 1.097700, 0.283974),  # episode 15
    (0.361416, 1.097700, 0.249258),  # episode 16
    (0.395726, 1.097700, 0.270088),  # episode 17
    (0.378571, 1.097700, 0.290918),  # episode 18
    (0.412881, 1.097700, 0.256201),  # episode 19
    (0.350694, 1.097700, 0.277031),  # episode 20
    (0.385004, 1.097700, 0.297861),  # episode 21
    (0.367849, 1.097700, 0.244629),  # episode 22
    (0.402159, 1.097700, 0.265459),  # episode 23
    (0.359272, 1.097700, 0.286289),  # episode 24
    (0.393582, 1.097700, 0.251572),  # episode 25
    (0.376427, 1.097700, 0.272402),  # episode 26
    (0.410737, 1.097700, 0.293232),  # episode 27
    (0.354983, 1.097700, 0.258516),  # episode 28
    (0.389293, 1.097700, 0.279346),  # episode 29
    (0.372138, 1.097700, 0.300176),  # episode 30
    (0.406448, 1.097700, 0.240771),  # episode 31
    (0.363561, 1.097700, 0.261601),  # episode 32
    (0.397871, 1.097700, 0.282431),  # episode 33
    (0.380716, 1.097700, 0.247715),  # episode 34
    (0.415026, 1.097700, 0.268545),  # episode 35
    (0.349622, 1.097700, 0.289375),  # episode 36
    (0.383932, 1.097700, 0.254658),  # episode 37
    (0.366777, 1.097700, 0.275488),  # episode 38
    (0.401087, 1.097700, 0.296318),  # episode 39
    (0.358200, 1.097700, 0.243086),  # episode 40
    (0.392510, 1.097700, 0.263916),  # episode 41
    (0.375355, 1.097700, 0.284746),  # episode 42
    (0.409665, 1.097700, 0.250029),  # episode 43
    (0.353911, 1.097700, 0.270859),  # episode 44
    (0.388221, 1.097700, 0.291689),  # episode 45
    (0.371066, 1.097700, 0.256973),  # episode 46
    (0.405376, 1.097700, 0.277803),  # episode 47
    (0.362488, 1.097700, 0.298633),  # episode 48
    (0.396798, 1.097700, 0.245400),  # episode 49
)
SPONGE_POSITION_LIMITS_W = {
    "x": (0.348550, 0.417170),
    "y": (1.097700, 1.097700),
    "z": (0.240000, 0.302490),
}

# Held-out evaluation positions. These are at least 3 mm inside the reachable
# rectangle and 7.37-9.90 mm from the nearest one of the 50 training positions.
# They must never be added to the training episode table above.
VALIDATION_SPONGE_POSITIONS_W = (
    (0.414170, 1.097700, 0.283108),
    (0.368144, 1.097700, 0.291017),
    (0.385678, 1.097700, 0.270398),
    (0.351550, 1.097700, 0.249779),
    (0.403525, 1.097700, 0.256558),
    (0.393192, 1.097700, 0.299490),
    (0.358751, 1.097700, 0.277459),
    (0.381294, 1.097700, 0.278871),
    (0.409473, 1.097700, 0.263054),
    (0.375032, 1.097700, 0.243000),
)


def log(message: str) -> None:
    print(f"[data-gen] {message}", flush=True)


def context_is_stage_loading(context: Any) -> bool:
    """Return whether USD payloads are loading across Isaac Sim API versions."""
    get_status = getattr(context, "get_stage_loading_status", None)
    if callable(get_status):
        return int(get_status()[2]) > 0

    legacy_check = getattr(context, "is_stage_loading", None)
    if callable(legacy_check):
        return bool(legacy_check())

    raise RuntimeError("The USD context exposes no supported stage-loading status API.")


def wait_for_stage(timeout_s: float, report_interval_s: float) -> Usd.Stage:
    """Wait until the USD context and all payloads finish loading."""
    start = time.monotonic()
    last_report = start
    context = omni.usd.get_context()
    while context_is_stage_loading(context):
        now = time.monotonic()
        if now - start > timeout_s:
            raise TimeoutError(f"Stage payload loading exceeded {timeout_s:.1f} seconds.")
        if report_interval_s > 0.0 and now - last_report >= report_interval_s:
            log(f"Loading stage payloads... elapsed={now - start:.1f}s")
            last_report = now
        simulation_app.update()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("USD context did not provide a stage after loading.")
    return stage


def require_prim(stage: Usd.Stage, path: str, expected_type: type | None = None) -> Usd.Prim:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Required prim is missing: {path}")
    if expected_type is not None and not prim.IsA(expected_type):
        raise RuntimeError(f"Prim {path} is not a {expected_type.__name__}.")
    return prim
def deinstance_subtree_in_session(stage: Usd.Stage, path: str) -> list[str]:
    """Expand USD instance roots transiently so Fabric does not depend on prototypes."""
    root = require_prim(stage, path)
    deinstanced: list[str] = []
    previous_edit_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    try:
        for _ in range(32):
            instance_paths = [prim.GetPath() for prim in Usd.PrimRange(root) if prim.IsInstanceable()]
            if not instance_paths:
                break
            for instance_path in instance_paths:
                prim = stage.GetPrimAtPath(instance_path)
                if prim.IsValid() and prim.IsInstanceable():
                    prim.SetInstanceable(False)
                    deinstanced.append(str(instance_path))
        else:
            raise RuntimeError(f"Instance expansion under {path} exceeded 32 composition passes.")
    finally:
        stage.SetEditTarget(previous_edit_target)
    return deinstanced


def world_pose_from_usd(stage: Usd.Stage, path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a prim world pose as position and wxyz quaternion."""
    prim = require_prim(stage, path)
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = transform.ExtractTranslation()
    rotation = transform.ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    position = np.asarray([translation[0], translation[1], translation[2]], dtype=np.float64)
    quaternion = np.asarray(
        [rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]], dtype=np.float64
    )
    quaternion /= np.linalg.norm(quaternion)
    return position, quaternion


def aligned_world_bounds(stage: Usd.Stage, path: str) -> tuple[np.ndarray, np.ndarray]:
    prim = require_prim(stage, path)
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    bounds = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    lower = bounds.GetMin()
    upper = bounds.GetMax()
    return (
        np.asarray([lower[0], lower[1], lower[2]], dtype=np.float64),
        np.asarray([upper[0], upper[1], upper[2]], dtype=np.float64),
    )


def enforce_sponge_scale(stage: Usd.Stage, uniform_scale: float) -> tuple[float, float, float]:
    """Preserve the requested 0.8 scale and correct it in memory if needed."""
    prim = require_prim(stage, SPONGE_PATH)
    attr = prim.GetAttribute("xformOp:scale")
    if not attr.IsValid():
        raise RuntimeError(f"{SPONGE_PATH} has no xformOp:scale attribute.")
    current = attr.Get()
    current_tuple = tuple(float(current[i]) for i in range(3))
    desired = (uniform_scale, uniform_scale, uniform_scale)
    if not np.allclose(current_tuple, desired, atol=1.0e-6):
        log(f"Changing {SPONGE_PATH} scale from {current_tuple} to {desired} in memory.")
        if not attr.Set(Gf.Vec3f(*desired)):
            raise RuntimeError(f"Could not set scale on {SPONGE_PATH}.")
    else:
        log(f"Manipulation sponge scale is already {desired}.")
    return desired


def validate_sponge_episode_positions() -> None:
    if len(SPONGE_EPISODE_POSITIONS_W) != 50:
        raise RuntimeError(
            f"Expected exactly 50 sponge positions; found {len(SPONGE_EPISODE_POSITIONS_W)}."
        )
    if len(set(SPONGE_EPISODE_POSITIONS_W)) != len(SPONGE_EPISODE_POSITIONS_W):
        raise RuntimeError("The sponge episode table contains duplicate positions.")
    if SPONGE_EPISODE_POSITIONS_W[0] != (0.394980, 1.097700, 0.254370):
        raise RuntimeError("Episode 0 must remain the mandatory verified sponge position.")
    for episode_index, position in enumerate(SPONGE_EPISODE_POSITIONS_W):
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            lower, upper = SPONGE_POSITION_LIMITS_W[axis_name]
            if not lower <= position[axis_index] <= upper:
                raise RuntimeError(
                    f"Episode {episode_index} {axis_name}={position[axis_index]:.6f} is outside "
                    f"[{lower:.6f}, {upper:.6f}]."
                )
    if len(VALIDATION_SPONGE_POSITIONS_W) != 10:
        raise RuntimeError(
            f"Expected exactly 10 held-out validation positions; "
            f"found {len(VALIDATION_SPONGE_POSITIONS_W)}."
        )
    if len(set(VALIDATION_SPONGE_POSITIONS_W)) != len(VALIDATION_SPONGE_POSITIONS_W):
        raise RuntimeError("The validation sponge-position table contains duplicates.")
    overlap = set(VALIDATION_SPONGE_POSITIONS_W) & set(SPONGE_EPISODE_POSITIONS_W)
    if overlap:
        raise RuntimeError(f"Validation positions overlap training positions: {sorted(overlap)}")
    for validation_index, position in enumerate(VALIDATION_SPONGE_POSITIONS_W):
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            lower, upper = SPONGE_POSITION_LIMITS_W[axis_name]
            if not lower <= position[axis_index] <= upper:
                raise RuntimeError(
                    f"Validation {validation_index} {axis_name}={position[axis_index]:.6f} "
                    f"is outside [{lower:.6f}, {upper:.6f}]."
                )


def author_sponge_position_in_session(
    stage: Usd.Stage,
    position_w: tuple[float, float, float],
) -> None:
    """Override only the sponge translation in the anonymous USD session layer."""
    prim = require_prim(stage, SPONGE_PATH)
    translate_attr = prim.GetAttribute("xformOp:translate")
    if not translate_attr.IsValid():
        raise RuntimeError(f"{SPONGE_PATH} has no xformOp:translate attribute.")
    previous_edit_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    try:
        if not translate_attr.Set(Gf.Vec3d(*position_w)):
            raise RuntimeError(f"Could not author session-layer sponge position {position_w}.")
    finally:
        stage.SetEditTarget(previous_edit_target)
    actual_position_w, _ = world_pose_from_usd(stage, SPONGE_PATH)
    if not np.allclose(actual_position_w, position_w, atol=1.0e-6):
        raise RuntimeError(
            f"Sponge parent transform is not identity: requested world position {position_w}, "
            f"resolved {actual_position_w.tolist()}."
        )


def set_waypoint_xform(stage: Usd.Stage, name: str, pose_w: torch.Tensor, visible: bool) -> None:
    """Create/update a world-space Xform with non-colliding visual markers."""
    path = f"{WAYPOINT_ROOT}/{name}"
    xform = UsdGeom.Xform.Define(stage, path)
    xformable = UsdGeom.Xformable(xform.GetPrim())
    ordered_ops = xformable.GetOrderedXformOps()
    translate_op = next(
        (op for op in ordered_ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
        None,
    )
    orient_op = next(
        (op for op in ordered_ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient),
        None,
    )
    if translate_op is None:
        translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    if orient_op is None:
        orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    xformable.SetXformOpOrder([translate_op, orient_op])
    values = pose_w.detach().cpu().numpy().astype(np.float64)
    translate_op.Set(Gf.Vec3d(*values[0:3]))
    orient_op.Set(Gf.Quatd(values[3], Gf.Vec3d(*values[4:7])))
    xform.GetPrim().CreateAttribute("dataGen:role", Sdf.ValueTypeNames.String, custom=True).Set(name)

    color = Gf.Vec3f(*WAYPOINT_COLORS.get(name, (1.0, 1.0, 1.0)))
    marker = UsdGeom.Sphere.Define(stage, f"{path}/position_marker")
    marker.CreateRadiusAttr(0.018)
    marker.CreateDisplayColorAttr([color])
    marker.CreateDisplayOpacityAttr([0.8])

    approach_axis = UsdGeom.Cone.Define(stage, f"{path}/local_plus_z")
    approach_axis.CreateAxisAttr(UsdGeom.Tokens.z)
    approach_axis.CreateHeightAttr(0.08)
    approach_axis.CreateRadiusAttr(0.006)
    approach_axis.CreateDisplayColorAttr([color])
    approach_axis.CreateDisplayOpacityAttr([0.8])

    for visual in (marker, approach_axis):
        imageable = UsdGeom.Imageable(visual.GetPrim())
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()


def tcp_pose_from_hand(hand_pose_w: torch.Tensor, tcp_offset_b: torch.Tensor) -> torch.Tensor:
    tcp_position = hand_pose_w[:, 0:3] + quat_apply(hand_pose_w[:, 3:7], tcp_offset_b)
    return torch.cat((tcp_position, hand_pose_w[:, 3:7]), dim=-1)


def hand_target_from_tcp(tcp_pose_w: torch.Tensor, tcp_offset_b: torch.Tensor) -> torch.Tensor:
    hand_position = tcp_pose_w[:, 0:3] - quat_apply(tcp_pose_w[:, 3:7], tcp_offset_b)
    return torch.cat((hand_position, tcp_pose_w[:, 3:7]), dim=-1)


def tensor_pose(position: np.ndarray, quaternion: np.ndarray, device: str) -> torch.Tensor:
    values = np.concatenate((position, quaternion)).astype(np.float64, copy=False)
    return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)




class AdaptiveNullspaceIKController:
    """Persistent velocity-shaped IK targets for PhysX articulation drives.

    The controller converts Cartesian pose error into a bounded joint velocity,
    acceleration-limits that velocity, and integrates a persistent joint drive
    target. The target may lag the measured arm by a bounded following error so
    the PhysX drive can build corrective torque without any state teleport.
    """

    def __init__(
        self,
        *,
        physics_dt: float,
        position_gain: float,
        orientation_gain: float,
        max_linear_speed: float,
        max_angular_speed: float,
        damping_min: float,
        damping_max: float,
        singularity_threshold: float,
        posture_gain: float,
        joint_limit_gain: float,
        joint_limit_margin: float,
        joint_velocity_limit: float,
        joint_acceleration_limit: float,
        following_error_limit: float,
        max_joint_step: float,
    ):
        self.physics_dt = physics_dt
        self.position_gain = position_gain
        self.orientation_gain = orientation_gain
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.damping_min = damping_min
        self.damping_max = damping_max
        self.singularity_threshold = singularity_threshold
        self.posture_gain = posture_gain
        self.joint_limit_gain = joint_limit_gain
        self.joint_limit_margin = joint_limit_margin
        self.joint_velocity_limit = min(
            joint_velocity_limit,
            max_joint_step / physics_dt,
        )
        self.joint_acceleration_limit = joint_acceleration_limit
        self.following_error_limit = following_error_limit
        self.max_joint_step = max_joint_step
        self._joint_position_target: torch.Tensor | None = None
        self._joint_velocity_target: torch.Tensor | None = None

    @staticmethod
    def _limit_vector_norm(
        vector: torch.Tensor,
        maximum_norm: float,
    ) -> torch.Tensor:
        vector_norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
        scale = torch.clamp(
            maximum_norm / torch.clamp(vector_norm, min=1.0e-8),
            max=1.0,
        )
        return vector * scale

    def reset(self, joint_position: torch.Tensor) -> None:
        self._joint_position_target = joint_position.clone()
        self._joint_velocity_target = torch.zeros_like(joint_position)

    def compute(
        self,
        ee_position_b: torch.Tensor,
        ee_quaternion_b: torch.Tensor,
        target_position_b: torch.Tensor,
        target_quaternion_b: torch.Tensor,
        jacobian_b: torch.Tensor,
        joint_position: torch.Tensor,
        posture_reference: torch.Tensor,
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return persistent position/velocity targets and diagnostics."""
        if (
            self._joint_position_target is None
            or self._joint_velocity_target is None
        ):
            self.reset(joint_position)

        position_error, rotation_error = compute_pose_error(
            ee_position_b,
            ee_quaternion_b,
            target_position_b,
            target_quaternion_b,
            rot_error_type="axis_angle",
        )
        linear_velocity = self._limit_vector_norm(
            self.position_gain * position_error,
            self.max_linear_speed,
        )
        angular_velocity = self._limit_vector_norm(
            self.orientation_gain * rotation_error,
            self.max_angular_speed,
        )
        desired_twist = torch.cat(
            (linear_velocity, angular_velocity),
            dim=-1,
        )

        singular_values = torch.linalg.svdvals(jacobian_b)
        minimum_singular_value = torch.amin(singular_values, dim=-1)
        singularity_blend = torch.clamp(
            (self.singularity_threshold - minimum_singular_value)
            / self.singularity_threshold,
            min=0.0,
            max=1.0,
        ).square()
        damping = self.damping_min + (
            self.damping_max - self.damping_min
        ) * singularity_blend

        jacobian_transpose = jacobian_b.transpose(1, 2)
        task_identity = torch.eye(
            jacobian_b.shape[1],
            dtype=jacobian_b.dtype,
            device=jacobian_b.device,
        ).unsqueeze(0)
        task_matrix = (
            jacobian_b @ jacobian_transpose
            + damping.square()[:, None, None] * task_identity
        )
        damped_inverse = jacobian_transpose @ torch.linalg.solve(
            task_matrix,
            task_identity.expand(jacobian_b.shape[0], -1, -1),
        )
        task_joint_velocity = (
            damped_inverse @ desired_twist.unsqueeze(-1)
        ).squeeze(-1)

        joint_identity = torch.eye(
            jacobian_b.shape[2],
            dtype=jacobian_b.dtype,
            device=jacobian_b.device,
        ).unsqueeze(0)
        nullspace_projector = joint_identity - damped_inverse @ jacobian_b

        joint_range = torch.clamp(joint_upper - joint_lower, min=1.0e-6)
        limit_band = self.joint_limit_margin * joint_range
        lower_activation = torch.relu(
            (joint_lower + limit_band - joint_position) / limit_band
        )
        upper_activation = torch.relu(
            (joint_position - (joint_upper - limit_band)) / limit_band
        )
        posture_velocity = self.posture_gain * (
            posture_reference - joint_position
        )
        limit_velocity = self.joint_limit_gain * (
            lower_activation - upper_activation
        )
        nullspace_joint_velocity = (
            nullspace_projector
            @ (posture_velocity + limit_velocity).unsqueeze(-1)
        ).squeeze(-1)

        unscaled_joint_velocity = (
            task_joint_velocity + nullspace_joint_velocity
        )
        maximum_absolute_velocity = torch.amax(
            torch.abs(unscaled_joint_velocity),
            dim=-1,
            keepdim=True,
        )
        velocity_scale = torch.clamp(
            self.joint_velocity_limit
            / torch.clamp(maximum_absolute_velocity, min=1.0e-8),
            max=1.0,
        )
        desired_joint_velocity = unscaled_joint_velocity * velocity_scale

        maximum_velocity_change = (
            self.joint_acceleration_limit * self.physics_dt
        )
        velocity_change = torch.clamp(
            desired_joint_velocity - self._joint_velocity_target,
            min=-maximum_velocity_change,
            max=maximum_velocity_change,
        )
        self._joint_velocity_target = torch.clamp(
            self._joint_velocity_target + velocity_change,
            min=-self.joint_velocity_limit,
            max=self.joint_velocity_limit,
        )

        previous_joint_target = self._joint_position_target
        candidate_joint_target = (
            previous_joint_target
            + self._joint_velocity_target * self.physics_dt
        )
        target_step = torch.clamp(
            candidate_joint_target - previous_joint_target,
            min=-self.max_joint_step,
            max=self.max_joint_step,
        )
        candidate_joint_target = previous_joint_target + target_step
        candidate_joint_target = torch.maximum(
            torch.minimum(candidate_joint_target, joint_upper),
            joint_lower,
        )

        following_error = torch.clamp(
            candidate_joint_target - joint_position,
            min=-self.following_error_limit,
            max=self.following_error_limit,
        )
        self._joint_position_target = joint_position + following_error
        self._joint_position_target = torch.maximum(
            torch.minimum(self._joint_position_target, joint_upper),
            joint_lower,
        )

        at_lower_limit = (
            self._joint_position_target <= joint_lower + 1.0e-6
        ) & (self._joint_velocity_target < 0.0)
        at_upper_limit = (
            self._joint_position_target >= joint_upper - 1.0e-6
        ) & (self._joint_velocity_target > 0.0)
        self._joint_velocity_target = torch.where(
            at_lower_limit | at_upper_limit,
            torch.zeros_like(self._joint_velocity_target),
            self._joint_velocity_target,
        )

        normalized_limit_margin = torch.minimum(
            (joint_position - joint_lower) / joint_range,
            (joint_upper - joint_position) / joint_range,
        )
        diagnostics = {
            "minimum_singular_value": minimum_singular_value,
            "damping": damping,
            "linear_velocity_norm": torch.linalg.norm(
                linear_velocity, dim=-1
            ),
            "angular_velocity_norm": torch.linalg.norm(
                angular_velocity, dim=-1
            ),
            "task_velocity_norm": torch.linalg.norm(
                task_joint_velocity * velocity_scale, dim=-1
            ),
            "nullspace_velocity_norm": torch.linalg.norm(
                nullspace_joint_velocity * velocity_scale, dim=-1
            ),
            "velocity_scale": velocity_scale.squeeze(-1),
            "command_velocity_norm": torch.linalg.norm(
                self._joint_velocity_target, dim=-1
            ),
            "maximum_following_error": torch.amax(
                torch.abs(self._joint_position_target - joint_position),
                dim=-1,
            ),
            "minimum_joint_limit_margin": torch.amin(
                normalized_limit_margin, dim=-1
            ),
        }
        return (
            self._joint_position_target.clone(),
            self._joint_velocity_target.clone(),
            diagnostics,
        )


def cubic_bezier_position(control_points_w: torch.Tensor, progress: float) -> torch.Tensor:
    """Evaluate a four-point Cartesian Bezier curve at progress in [0, 1]."""
    u = float(np.clip(progress, 0.0, 1.0))
    one_minus_u = 1.0 - u
    return (
        one_minus_u**3 * control_points_w[0]
        + 3.0 * one_minus_u**2 * u * control_points_w[1]
        + 3.0 * one_minus_u * u**2 * control_points_w[2]
        + u**3 * control_points_w[3]
    )
def joint_indices(robot: Articulation, names: list[str]) -> list[int]:
    missing = [name for name in names if name not in robot.joint_names]
    if missing:
        raise RuntimeError(f"Franka is missing expected joints: {missing}. Found: {robot.joint_names}")
    return [robot.joint_names.index(name) for name in names]


def reset_robot_to_neutral(robot: Articulation) -> torch.Tensor:
    """Write both the state and drive targets for the official neutral pose."""
    joint_position = robot.data.joint_pos.clone()
    joint_velocity = torch.zeros_like(joint_position)
    for name, value in NEUTRAL_JOINT_POSITION.items():
        joint_position[:, robot.joint_names.index(name)] = value
    robot.write_joint_state_to_sim(joint_position, joint_velocity)
    robot.set_joint_position_target(joint_position)
    robot.write_data_to_sim()
    robot.reset()
    return joint_position


@dataclass(frozen=True)
class StateSpec:
    name: str
    waypoint: str
    gripper_action: float
    mode: str = "motion"
    dwell_s: float = 0.0


class PickPlaceStateMachine:
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.states = [
            StateSpec("HOME", "home", GRIPPER_OPEN_ACTION),
            StateSpec("PRE_GRASP", "pre_grasp", GRIPPER_OPEN_ACTION),
            StateSpec("GRASP", "grasp", GRIPPER_OPEN_ACTION),
            StateSpec(
                "CLOSE",
                "grasp",
                GRIPPER_CLOSE_ACTION,
                mode="dwell",
                dwell_s=config.close_hold,
            ),
            StateSpec("LIFT", "lift", GRIPPER_CLOSE_ACTION),
            StateSpec("ABOVE_PAN", "above_pan", GRIPPER_CLOSE_ACTION),
            StateSpec("PLACE", "place", GRIPPER_CLOSE_ACTION),
            StateSpec("OPEN", "place", GRIPPER_OPEN_ACTION, mode="dwell", dwell_s=config.open_hold),
            StateSpec("RETREAT", "retreat", GRIPPER_OPEN_ACTION),
            StateSpec("DONE", "retreat", GRIPPER_OPEN_ACTION, mode="done", dwell_s=config.final_hold),
        ]
        self.index = 0
        self.elapsed = 0.0
        self.settled = 0.0
        log(f"State -> {self.current.name}")

    @property
    def current(self) -> StateSpec:
        return self.states[self.index]

    def _advance(self) -> None:
        self.index += 1
        self.elapsed = 0.0
        self.settled = 0.0
        log(f"State -> {self.current.name}")

    def update(
        self,
        dt: float,
        position_error: float,
        orientation_error: float,
        trajectory_complete: bool = True,
    ) -> bool:
        self.elapsed += dt
        if self.current.name == "PRE_GRASP":
            position_tolerance = self.config.pre_grasp_position_tolerance
            orientation_tolerance = self.config.grasp_orientation_tolerance
        elif self.current.name in ("GRASP", "CLOSE"):
            position_tolerance = self.config.grasp_position_tolerance
            orientation_tolerance = self.config.grasp_orientation_tolerance
        else:
            position_tolerance = self.config.position_tolerance
            orientation_tolerance = self.config.orientation_tolerance
        pose_reached = (
            trajectory_complete
            and position_error <= position_tolerance
            and orientation_error <= orientation_tolerance
        )

        if self.current.mode == "done":
            return self.elapsed >= self.current.dwell_s

        if pose_reached:
            self.settled += dt
        else:
            self.settled = 0.0

        if self.current.mode == "motion" and self.settled >= self.config.settle_time:
            self._advance()
        elif self.current.mode == "dwell" and self.settled >= self.current.dwell_s:
            self._advance()

        if self.elapsed > self.config.state_timeout:
            raise TimeoutError(
                f"State {self.current.name} timed out: position error={position_error:.4f} m, "
                f"orientation error={math.degrees(orientation_error):.2f} deg"
            )
        return False


class RolloutRecorder:
    """Write synchronized RGB, joint/action, and task-state observations."""

    def __init__(self, output_parent: Path, camera_fps: float, save_video: bool, camera_names: tuple[str, ...], metadata: dict[str, Any]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = output_parent.expanduser().resolve() / f"run_{timestamp}"
        self.camera_names = tuple(camera_names)
        if not self.camera_names:
            raise ValueError("At least one camera name is required.")
        self.rgb_dirs = {
            name: self.run_dir / ("rgb" if name == "main" else f"rgb_{name}")
            for name in self.camera_names
        }
        for rgb_dir in self.rgb_dirs.values():
            rgb_dir.mkdir(parents=True, exist_ok=False)
        self.camera_fps = camera_fps
        self.save_video = save_video
        self.metadata = metadata
        self.records: list[dict[str, Any]] = []
        self.frame_index = 0
        self.video_writers: dict[str, Any | None] = {name: None for name in self.camera_names}
        self.video_errors: dict[str, str | None] = {name: None for name in self.camera_names}

        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "Camera output requires imageio. Install imageio and imageio-ffmpeg in the Isaac Lab environment."
            ) from exc
        self.imageio = imageio
        log(f"Writing rollout to {self.run_dir}")

    def _ensure_video_writer(self, camera_name: str) -> None:
        if (
            not self.save_video
            or self.video_writers[camera_name] is not None
            or self.video_errors[camera_name] is not None
        ):
            return
        video_name = "rollout.mp4" if camera_name == "main" else f"rollout_{camera_name}.mp4"
        try:
            self.video_writers[camera_name] = self.imageio.get_writer(
                str(self.run_dir / video_name),
                fps=self.camera_fps,
                codec="libx264",
                pixelformat="yuv420p",
                quality=8,
                macro_block_size=None,
            )
        except Exception as exc:  # FFmpeg may be unavailable on a fresh headless install.
            self.video_errors[camera_name] = str(exc)
            log(f"{camera_name} MP4 writer unavailable; PNG frames will still be saved: {exc}")

    def capture(self, rgb_by_camera: dict[str, np.ndarray], record: dict[str, Any]) -> None:
        missing = [name for name in self.camera_names if name not in rgb_by_camera]
        if missing:
            raise RuntimeError(f"Capture is missing camera images: {missing}")
        for camera_name in self.camera_names:
            rgb = rgb_by_camera[camera_name]
            if rgb.ndim != 3 or rgb.shape[-1] < 3:
                raise RuntimeError(f"Unexpected {camera_name} RGB tensor shape: {rgb.shape}")
            rgb = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
            image_path = self.rgb_dirs[camera_name] / f"{self.frame_index:06d}.png"
            self.imageio.imwrite(str(image_path), rgb)
            self._ensure_video_writer(camera_name)
            writer = self.video_writers[camera_name]
            if writer is not None:
                try:
                    writer.append_data(rgb)
                except Exception as exc:
                    self.video_errors[camera_name] = str(exc)
                    log(
                        f"Disabling {camera_name} MP4 output after encoder failure: {exc}"
                    )
                    writer.close()
                    self.video_writers[camera_name] = None
        record["frame_index"] = self.frame_index
        self.records.append(record)
        self.frame_index += 1

    def close(self, success: bool, error: str | None = None) -> None:
        for camera_name, writer in self.video_writers.items():
            if writer is not None:
                writer.close()
                self.video_writers[camera_name] = None

        arrays: dict[str, np.ndarray] = {}
        if self.records:
            for key in self.records[0]:
                arrays[key] = np.asarray([record[key] for record in self.records])
        np.savez_compressed(self.run_dir / "trajectory.npz", **arrays)

        self.metadata.update(
            {
                "success": success,
                "error": error,
                "num_frames": self.frame_index,
                "video_error": self.video_errors.get("main"),
                "video_errors": self.video_errors,
            }
        )
        with (self.run_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(self.metadata, file, indent=2)


def build_waypoints(
    stage: Usd.Stage,
    home_pose_w: torch.Tensor,
    grasp_quaternion_w: np.ndarray,
    device: str,
    release_offset_w: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Seed task-space targets from current object and collision bounds."""
    sponge_lower, sponge_upper = aligned_world_bounds(stage, SPONGE_PATH)
    sponge_center = 0.5 * (sponge_lower + sponge_upper)
    sponge_center[1] += args_cli.grasp_height_bias

    pan_prim_path = PAN_COLLISION_PATH if stage.GetPrimAtPath(PAN_COLLISION_PATH).IsValid() else PAN_PATH
    pan_lower, pan_upper = aligned_world_bounds(stage, pan_prim_path)
    pan_center = 0.5 * (pan_lower + pan_upper)
    pan_surface_y = pan_upper[1]

    grasp = tensor_pose(sponge_center, grasp_quaternion_w, device)
    pre_grasp = grasp.clone()
    pre_grasp[:, 1] += args_cli.pre_grasp_height
    lift = grasp.clone()
    lift[:, 1] += args_cli.lift_height

    above_pan = lift.clone()
    above_pan[:, 0] = float(pan_center[0])
    above_pan[:, 2] = float(pan_center[2])
    place = above_pan.clone()
    place[:, 1] = float(pan_surface_y + args_cli.place_clearance)
    release_offset_tensor = torch.tensor(release_offset_w, dtype=torch.float32, device=device)
    above_pan[:, :3] += release_offset_tensor
    place[:, :3] += release_offset_tensor
    retreat = above_pan.clone()

    waypoints = {
        "home": home_pose_w.clone(),
        "pre_grasp": pre_grasp,
        "grasp": grasp,
        "lift": lift,
        "above_pan": above_pan,
        "place": place,
        "retreat": retreat,
    }

    UsdGeom.Xform.Define(stage, WAYPOINT_ROOT)
    for name, pose in waypoints.items():
        set_waypoint_xform(stage, name, pose[0], visible=args_cli.show_waypoints)

    log(
        "Waypoint seeds: "
        f"sponge_center={np.round(sponge_center, 4).tolist()}, "
        f"pan_center={np.round(pan_center, 4).tolist()}, "
        f"pan_surface_y={pan_surface_y:.4f}"
    )
    return waypoints


def camera_rgb(camera: Camera, camera_name: str) -> np.ndarray:
    output = camera.data.output.get("rgb")
    if output is None or output.numel() == 0:
        raise RuntimeError(f"Camera {camera_name!r} produced no RGB data.")
    return output[0].detach().cpu().numpy()


def write_preview_contact_sheet(
    images: list[np.ndarray],
    output_path: Path,
    camera_name: str,
) -> None:
    """Write a labeled 5x10 sheet; individual full-resolution PNGs remain authoritative."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        log("Pillow is unavailable; skipping contact sheets (individual PNGs were still written).")
        return

    columns = 5
    label_height = 24
    image_height, image_width = images[0].shape[:2]
    sheet = Image.new(
        "RGB",
        (columns * image_width, math.ceil(len(images) / columns) * (image_height + label_height)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    for episode_index, image in enumerate(images):
        row, column = divmod(episode_index, columns)
        left = column * image_width
        top = row * (image_height + label_height)
        sheet.paste(Image.fromarray(image), (left, top))
        x, _, z = SPONGE_EPISODE_POSITIONS_W[episode_index]
        draw.text(
            (left + 4, top + image_height + 4),
            f"{episode_index:02d}: x={x:.4f} z={z:.4f}",
            fill=(255, 255, 255),
        )
    sheet.save(output_path)
    log(f"Wrote {camera_name} contact sheet: {output_path}")


def render_sponge_position_previews(
    sim: sim_utils.SimulationContext,
    sponge: RigidObject,
    cameras: dict[str, Camera],
) -> Path:
    """Render all hard-coded positions without stepping physics or modifying the USD files."""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("Preview rendering requires imageio in the Isaac environment.") from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preview_dir = args_cli.preview_output_dir.expanduser().resolve() / f"preview_{timestamp}"
    image_dirs = {name: preview_dir / name for name in cameras}
    for image_dir in image_dirs.values():
        image_dir.mkdir(parents=True, exist_ok=False)

    initial_quaternion_wxyz = sponge.data.root_pose_w[:, 3:7].clone()
    zero_velocity = torch.zeros((1, 6), dtype=torch.float32, device=sim.device)
    images_by_camera: dict[str, list[np.ndarray]] = {name: [] for name in cameras}
    manifest_positions: list[dict[str, Any]] = []

    for episode_index, position_w in enumerate(SPONGE_EPISODE_POSITIONS_W):
        position_tensor = torch.tensor([position_w], dtype=torch.float32, device=sim.device)
        pose_tensor = torch.cat((position_tensor, initial_quaternion_wxyz), dim=-1)
        sponge.write_root_pose_to_sim(pose_tensor)
        sponge.write_root_velocity_to_sim(zero_velocity)
        sim.forward()
        # Two render updates avoid returning the previous Fabric/RTX frame after a teleport.
        sim.render()
        sim.render()
        sponge.update(0.0)
        for camera in cameras.values():
            camera.update(args_cli.physics_dt, force_recompute=True)

        actual_position_w = sponge.data.root_pose_w[0, :3].detach().cpu().numpy()
        if not np.allclose(actual_position_w, position_w, atol=1.0e-5):
            raise RuntimeError(
                f"Preview episode {episode_index} pose mismatch: requested={position_w}, "
                f"actual={actual_position_w.tolist()}."
            )
        manifest_positions.append(
            {
                "episode_index": episode_index,
                "position_w_xyz": list(position_w),
                "verified_seed": episode_index == 0,
            }
        )
        for camera_name, camera in cameras.items():
            rgb = np.ascontiguousarray(camera_rgb(camera, camera_name)[..., :3].astype(np.uint8))
            images_by_camera[camera_name].append(rgb)
            imageio.imwrite(image_dirs[camera_name] / f"episode_{episode_index:02d}.png", rgb)
        log(
            f"Preview {episode_index + 1:02d}/50: episode={episode_index:02d} "
            f"sponge_w=({position_w[0]:.6f}, {position_w[1]:.6f}, {position_w[2]:.6f})"
        )

    manifest = {
        "schema_version": 1,
        "stage": str(args_cli.stage.expanduser().resolve()),
        "sponge_prim": SPONGE_PATH,
        "coordinate_frame": "world_xyz_meters",
        "camera_prims": CAMERA_PATHS,
        "image_size": [args_cli.width, args_cli.height],
        "positions": manifest_positions,
    }
    with (preview_dir / "positions.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    for camera_name, images in images_by_camera.items():
        write_preview_contact_sheet(
            images,
            preview_dir / f"contact_sheet_{camera_name}.png",
            camera_name,
        )
    log(f"Preview complete: 50 synchronized frames from each camera in {preview_dir}")
    return preview_dir


_VLA_PACKET_LENGTH = struct.Struct("!Q")
_VLA_MAX_PACKET_BYTES = 64 * 1024 * 1024
_VLA_PROTOCOL_VERSION = 2


def _vla_recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("SmolVLA server disconnected while returning an inference result.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _vla_send_packet(connection: socket.socket, **arrays: Any) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    if len(payload) > _VLA_MAX_PACKET_BYTES:
        raise ValueError(f"SmolVLA request is too large: {len(payload)} bytes.")
    connection.sendall(_VLA_PACKET_LENGTH.pack(len(payload)))
    connection.sendall(payload)


def _vla_recv_packet(connection: socket.socket) -> dict[str, np.ndarray]:
    (size,) = _VLA_PACKET_LENGTH.unpack(
        _vla_recv_exact(connection, _VLA_PACKET_LENGTH.size)
    )
    if size <= 0 or size > _VLA_MAX_PACKET_BYTES:
        raise ValueError(f"Invalid SmolVLA response size: {size} bytes.")
    payload = _vla_recv_exact(connection, size)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class LocalSmolVLAClient:
    """Small localhost-only client that keeps Isaac and LeRobot environments separate."""

    def __init__(self, host: str, port: int, connect_timeout_s: float) -> None:
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("The SmolVLA client is intentionally restricted to localhost.")
        if not 1 <= port <= 65535:
            raise ValueError("policy-port must be in [1, 65535].")
        if connect_timeout_s <= 0.0:
            raise ValueError("policy-connect-timeout must be positive.")

        family = socket.AF_INET6 if host == "::1" else socket.AF_INET
        deadline = time.monotonic() + connect_timeout_s
        last_error: OSError | None = None
        log(f"Waiting for SmolVLA server at {host}:{port} (timeout {connect_timeout_s:.0f}s).")
        while time.monotonic() < deadline:
            connection = socket.socket(family, socket.SOCK_STREAM)
            connection.settimeout(min(2.0, max(0.1, deadline - time.monotonic())))
            try:
                connection.connect((host, port))
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.settimeout(max(30.0, connect_timeout_s))
                self.connection = connection
                break
            except OSError as exc:
                last_error = exc
                connection.close()
                time.sleep(0.25)
        else:
            raise TimeoutError(
                f"Could not connect to SmolVLA server at {host}:{port}: {last_error}"
            )

        _vla_send_packet(self.connection, command=np.asarray("reset"))
        response = _vla_recv_packet(self.connection)
        if not bool(np.asarray(response.get("ok", False)).reshape(()).item()):
            raise RuntimeError("SmolVLA server rejected the policy reset request.")
        protocol = int(np.asarray(response.get("protocol", -1)).reshape(()).item())
        if protocol != _VLA_PROTOCOL_VERSION:
            self.connection.close()
            raise RuntimeError(
                f"SmolVLA protocol mismatch: client={_VLA_PROTOCOL_VERSION}, server={protocol}. "
                "Restart the policy server with the updated smolvla_inference.py."
            )
        log(f"Connected to SmolVLA server at {host}:{port}; policy queue reset.")

    def infer(
        self,
        *,
        main_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state: np.ndarray,
        task: str,
    ) -> tuple[np.ndarray, float, float, int]:
        request_start = time.perf_counter()
        _vla_send_packet(
            self.connection,
            command=np.asarray("infer"),
            main_rgb=np.ascontiguousarray(main_rgb[..., :3], dtype=np.uint8),
            wrist_rgb=np.ascontiguousarray(wrist_rgb[..., :3], dtype=np.uint8),
            state=np.asarray(state, dtype=np.float32),
            task=np.asarray(task),
        )
        response = _vla_recv_packet(self.connection)
        roundtrip_s = time.perf_counter() - request_start
        if not bool(np.asarray(response.get("ok", False)).reshape(()).item()):
            error = str(np.asarray(response.get("error", "unknown server error")).reshape(()).item())
            raise RuntimeError(f"SmolVLA inference failed: {error}")
        action = np.asarray(response["action"], dtype=np.float32).reshape(-1)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"SmolVLA returned an invalid action: shape={action.shape}, value={action}.")
        latency_s = float(np.asarray(response["latency_s"]).reshape(()).item())
        if "plan_step" not in response:
            raise RuntimeError(
                "SmolVLA server does not provide plan_step; restart it with the updated "
                "smolvla_inference.py."
            )
        plan_step = int(np.asarray(response["plan_step"]).reshape(()).item())
        if plan_step < 0:
            raise RuntimeError(f"SmolVLA returned an invalid plan_step={plan_step}.")
        return action, latency_s, roundtrip_s, plan_step

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.connection.shutdown(socket.SHUT_RDWR)
        self.connection.close()


def _limit_numpy_vector_norm(vector: np.ndarray, maximum_norm: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > maximum_norm:
        vector = vector * (maximum_norm / max(norm, 1.0e-12))
    return vector


def _axis_angle_to_quaternion_wxyz(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert batched rotation vectors to normalized Isaac-order quaternions."""
    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    xyz_scale = torch.where(
        angle > 1.0e-7,
        torch.sin(half_angle) / torch.clamp(angle, min=1.0e-7),
        0.5 - angle.square() / 48.0,
    )
    quaternion = torch.cat((torch.cos(half_angle), axis_angle * xyz_scale), dim=-1)
    return quaternion / torch.clamp(
        torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True), min=1.0e-8
    )


def run_vla_policy_rollout(
    *,
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    sponge: RigidObject,
    sponge_pan_contact_sensor: ContactSensor,
    cameras: dict[str, Camera],
    arm_joint_ids: list[int],
    finger_joint_ids: list[int],
    hand_body_id: int,
    hand_jacobian_id: int,
    actual_camera_fps: float,
    capture_stride: int,
    stage_path: Path,
    selected_sponge_position_w: tuple[float, float, float],
    sponge_position_split: str,
    sponge_position_index: int,
    sponge_scale: tuple[float, float, float],
    expected_grasp_contact_position: float,
    minimum_grasp_contact_position: float,
    franka_deinstanced_prims: list[str],
) -> int:
    """Run vision-conditioned SmolVLA actions through the existing physical drive controller."""
    if set(cameras) != {"main", "wrist"}:
        raise RuntimeError(f"SmolVLA inference requires main and wrist cameras; found {list(cameras)}.")
    if (args_cli.width, args_cli.height) != (256, 256):
        raise ValueError("This SmolVLA checkpoint was trained on 256x256 images; use --width 256 --height 256.")
    if not math.isclose(actual_camera_fps, 10.0, abs_tol=1.0e-6):
        raise ValueError("This SmolVLA checkpoint expects 10 Hz observations; use --camera-fps 10.")

    client = LocalSmolVLAClient(
        args_cli.policy_host,
        args_cli.policy_port,
        args_cli.policy_connect_timeout,
    )
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "mode": "closed_loop_smolvla_inference",
        "task": args_cli.task_description,
        "stage": str(stage_path),
        "sponge_position_split": sponge_position_split,
        "sponge_position_index": sponge_position_index,
        "sponge_position_w_xyz": list(selected_sponge_position_w),
        "sponge_scale": list(sponge_scale),
        "camera_prims": CAMERA_PATHS,
        "camera_fps": actual_camera_fps,
        "physics_dt": args_cli.physics_dt,
        "image_size": [args_cli.width, args_cli.height],
        "policy_server": {
            "host": args_cli.policy_host,
            "port": args_cli.policy_port,
            "observation_keys": [
                "observation.images.image",
                "observation.images.image2",
                "observation.state",
            ],
            "state": "tcp_xyz_b + tcp_quaternion_xyzw_b + gripper_open_fraction",
            "action": "delta_tcp_xyz_b + delta_rotation_vector_b + binary_gripper",
            "control_rate_hz": actual_camera_fps,
        },
        "action_safety": {
            "maximum_translation_step_m": args_cli.policy_max_translation_step,
            "maximum_rotation_step_rad": args_cli.policy_max_rotation_step,
            "maximum_servo_position_lag_m": args_cli.policy_max_position_lag,
            "maximum_servo_orientation_lag_rad": args_cli.policy_max_orientation_lag,
            "gripper_hysteresis_threshold": args_cli.policy_gripper_threshold,
            "workspace_min_xyz_b_m": list(args_cli.policy_workspace_min),
            "workspace_max_xyz_b_m": list(args_cli.policy_workspace_max),
        },
        "control": {
            "arm": "adaptive_damped_least_squares_IK_to_PhysX_position_velocity_drives",
            "direct_joint_state_writes_during_inference": False,
            "policy_delta_target": "persistent_base_frame_pose_accumulator",
            "gripper": "slew_limited_PhysX_prismatic_drive",
            "gripper_velocity_limit_m_s": args_cli.gripper_velocity_limit,
            "arm_stiffness": args_cli.arm_stiffness,
            "arm_damping": args_cli.arm_damping,
            "policy_servo_position_gain": args_cli.policy_servo_position_gain,
            "policy_servo_orientation_gain": args_cli.policy_servo_orientation_gain,
        },
        "franka_deinstanced_prims": franka_deinstanced_prims,
        "physical_validation": {
            "grasp_contact_position_m": expected_grasp_contact_position,
            "minimum_grasp_contact_position_m": minimum_grasp_contact_position,
            "grasp_min_effort_n": args_cli.grasp_min_effort,
            "grasp_max_tcp_sponge_distance_m": args_cli.grasp_max_tcp_sponge_distance,
            "pan_contact_force_threshold_n": args_cli.pan_contact_force_threshold,
            "pan_contact_settle_time_s": args_cli.pan_contact_settle_time,
            "placement_max_sponge_speed_m_s": args_cli.placement_max_sponge_speed,
        },
    }
    recorder: RolloutRecorder | None = None
    task_success = False
    task_failure: str | None = None
    system_error: str | None = None
    grasp_validated = False
    released_after_grasp = False
    pan_contact_elapsed = 0.0
    maximum_pan_contact_force = 0.0
    pan_contact_force = 0.0
    sponge_linear_speed = math.inf
    tcp_sponge_distance = math.inf
    policy_query_count = 0
    last_policy_latency_s = 0.0
    last_policy_roundtrip_s = 0.0

    try:
        recorder = RolloutRecorder(
            args_cli.output_dir,
            actual_camera_fps,
            save_video=not args_cli.no_video,
            camera_names=tuple(CAMERA_PATHS),
            metadata=metadata,
        )
        tcp_offset_b = torch.tensor(
            [[0.0, 0.0, args_cli.tcp_offset]], dtype=torch.float32, device=sim.device
        )
        ik = AdaptiveNullspaceIKController(
            physics_dt=args_cli.physics_dt,
            position_gain=args_cli.policy_servo_position_gain,
            orientation_gain=args_cli.policy_servo_orientation_gain,
            max_linear_speed=args_cli.ik_max_linear_speed,
            max_angular_speed=args_cli.ik_max_angular_speed,
            damping_min=args_cli.ik_damping_min,
            damping_max=args_cli.ik_damping_max,
            singularity_threshold=args_cli.ik_singularity_threshold,
            posture_gain=args_cli.ik_posture_gain,
            joint_limit_gain=args_cli.ik_joint_limit_gain,
            joint_limit_margin=args_cli.ik_joint_limit_margin,
            joint_velocity_limit=args_cli.arm_command_velocity_limit,
            joint_acceleration_limit=args_cli.arm_command_acceleration_limit,
            following_error_limit=args_cli.arm_following_error_limit,
            max_joint_step=args_cli.max_joint_step,
        )
        current_arm_position = robot.data.joint_pos[:, arm_joint_ids]
        ik.reset(current_arm_position)
        arm_joint_lower = robot.data.soft_joint_pos_limits[:, arm_joint_ids, 0].clone()
        arm_joint_upper = robot.data.soft_joint_pos_limits[:, arm_joint_ids, 1].clone()
        workspace_min_b = torch.tensor(
            [args_cli.policy_workspace_min], dtype=torch.float32, device=sim.device
        )
        workspace_max_b = torch.tensor(
            [args_cli.policy_workspace_max], dtype=torch.float32, device=sim.device
        )

        current_root_pose_w = robot.data.root_pose_w
        current_hand_pose_w = robot.data.body_pose_w[:, hand_body_id]
        target_tcp_pose_w = tcp_pose_from_hand(current_hand_pose_w, tcp_offset_b).clone()
        commanded_tcp_position_b, commanded_tcp_quaternion_b = subtract_frame_transforms(
            current_root_pose_w[:, 0:3],
            current_root_pose_w[:, 3:7],
            target_tcp_pose_w[:, 0:3],
            target_tcp_pose_w[:, 3:7],
        )
        commanded_tcp_position_b = commanded_tcp_position_b.clone()
        commanded_tcp_quaternion_b = commanded_tcp_quaternion_b.clone()
        command_position_lag = 0.0
        command_orientation_lag = 0.0
        joint_target = robot.data.joint_pos.clone()
        joint_velocity_target = torch.zeros_like(joint_target)
        gripper_position = float(torch.mean(robot.data.joint_pos[:, finger_joint_ids]).item())
        gripper_drive_target_position = gripper_position
        gripper_action = GRIPPER_OPEN_ACTION
        raw_policy_action = np.zeros(7, dtype=np.float32)
        limited_policy_action = np.zeros(7, dtype=np.float32)
        policy_plan_step = 0
        next_status_wall_time = time.monotonic()
        max_steps = int(math.ceil(args_cli.max_runtime / args_cli.physics_dt))

        log(
            "SmolVLA inference started: observations/actions at 10 Hz; IK and PhysX drives at "
            f"{1.0 / args_cli.physics_dt:.0f} Hz. Both cameras will be rendered to {recorder.run_dir}."
        )
        for step in range(max_steps):
            if not simulation_app.is_running():
                raise RuntimeError("Simulation application stopped during SmolVLA inference.")

            current_root_pose_w = robot.data.root_pose_w
            current_hand_pose_w = robot.data.body_pose_w[:, hand_body_id]
            current_tcp_pose_w = tcp_pose_from_hand(current_hand_pose_w, tcp_offset_b)
            current_tcp_position_b, current_tcp_quaternion_b = subtract_frame_transforms(
                current_root_pose_w[:, 0:3],
                current_root_pose_w[:, 3:7],
                current_tcp_pose_w[:, 0:3],
                current_tcp_pose_w[:, 3:7],
            )
            current_tcp_pose_b = torch.cat(
                (current_tcp_position_b, current_tcp_quaternion_b), dim=-1
            )
            gripper_position = float(
                torch.mean(robot.data.joint_pos[:, finger_joint_ids]).item()
            )
            gripper_open_fraction = float(
                np.clip(
                    gripper_position / args_cli.open_position,
                    0.0,
                    1.0,
                )
            )

            if step % capture_stride == 0:
                rgb_by_camera = {
                    name: np.ascontiguousarray(camera_rgb(camera, name)[..., :3], dtype=np.uint8)
                    for name, camera in cameras.items()
                }
                quaternion_wxyz_b = current_tcp_quaternion_b[0].detach().cpu().numpy()
                policy_state = np.concatenate(
                    (
                        current_tcp_position_b[0].detach().cpu().numpy(),
                        quaternion_wxyz_b[[1, 2, 3, 0]],
                        np.asarray([gripper_open_fraction], dtype=np.float32),
                    )
                ).astype(np.float32, copy=False)
                (
                    raw_policy_action,
                    last_policy_latency_s,
                    last_policy_roundtrip_s,
                    policy_plan_step,
                ) = client.infer(
                    main_rgb=rgb_by_camera["main"],
                    wrist_rgb=rgb_by_camera["wrist"],
                    state=policy_state,
                    task=args_cli.task_description,
                )
                if policy_plan_step == 0:
                    commanded_tcp_position_b = current_tcp_position_b.clone()
                    commanded_tcp_quaternion_b = current_tcp_quaternion_b.clone()
                    if policy_query_count > 0:
                        log(
                            "Policy generated a new action chunk; persistent TCP command "
                            "rebased to the measured pose."
                        )
                translation_delta_b = _limit_numpy_vector_norm(
                    raw_policy_action[:3], args_cli.policy_max_translation_step
                )
                rotation_delta_b = _limit_numpy_vector_norm(
                    raw_policy_action[3:6], args_cli.policy_max_rotation_step
                )
                previous_commanded_tcp_position_b = commanded_tcp_position_b.clone()
                commanded_tcp_position_b = torch.clamp(
                    commanded_tcp_position_b
                    + torch.tensor(
                        translation_delta_b[None, :], dtype=torch.float32, device=sim.device
                    ),
                    min=workspace_min_b,
                    max=workspace_max_b,
                )
                accepted_translation_delta_b = (
                    commanded_tcp_position_b - previous_commanded_tcp_position_b
                )
                delta_quaternion_b = _axis_angle_to_quaternion_wxyz(
                    torch.tensor(
                        rotation_delta_b[None, :], dtype=torch.float32, device=sim.device
                    )
                )
                # The training exporter stores q_delta = q_next * inverse(q_current),
                # so accumulate each delta by left-multiplying it. Keeping this
                # commanded pose persistent prevents unfinished PhysX/IK motion
                # from being discarded at the next 10 Hz policy action.
                commanded_tcp_quaternion_b = quat_mul(
                    delta_quaternion_b, commanded_tcp_quaternion_b
                )
                commanded_tcp_quaternion_b = commanded_tcp_quaternion_b / torch.clamp(
                    torch.linalg.vector_norm(
                        commanded_tcp_quaternion_b, dim=-1, keepdim=True
                    ),
                    min=1.0e-8,
                )

                command_position_error_b = commanded_tcp_position_b - current_tcp_position_b
                command_position_lag_tensor = torch.linalg.vector_norm(
                    command_position_error_b, dim=-1, keepdim=True
                )
                bounded_position_error_b = command_position_error_b * torch.clamp(
                    args_cli.policy_max_position_lag
                    / torch.clamp(command_position_lag_tensor, min=1.0e-8),
                    max=1.0,
                )
                target_tcp_position_b = current_tcp_position_b + bounded_position_error_b

                _, command_orientation_error_b = compute_pose_error(
                    current_tcp_position_b,
                    current_tcp_quaternion_b,
                    commanded_tcp_position_b,
                    commanded_tcp_quaternion_b,
                    rot_error_type="axis_angle",
                )
                command_orientation_lag_tensor = torch.linalg.vector_norm(
                    command_orientation_error_b, dim=-1, keepdim=True
                )
                bounded_orientation_error_b = command_orientation_error_b * torch.clamp(
                    args_cli.policy_max_orientation_lag
                    / torch.clamp(command_orientation_lag_tensor, min=1.0e-8),
                    max=1.0,
                )
                target_tcp_quaternion_b = quat_mul(
                    _axis_angle_to_quaternion_wxyz(bounded_orientation_error_b),
                    current_tcp_quaternion_b,
                )
                command_position_lag = float(command_position_lag_tensor.item())
                command_orientation_lag = float(command_orientation_lag_tensor.item())
                target_tcp_position_w, target_tcp_quaternion_w = combine_frame_transforms(
                    current_root_pose_w[:, 0:3],
                    current_root_pose_w[:, 3:7],
                    target_tcp_position_b,
                    target_tcp_quaternion_b,
                )
                target_tcp_pose_w = torch.cat(
                    (target_tcp_position_w, target_tcp_quaternion_w), dim=-1
                )

                previous_gripper_action = gripper_action
                if raw_policy_action[6] >= args_cli.policy_gripper_threshold:
                    gripper_action = GRIPPER_OPEN_ACTION
                elif raw_policy_action[6] <= -args_cli.policy_gripper_threshold:
                    gripper_action = GRIPPER_CLOSE_ACTION
                if gripper_action != previous_gripper_action:
                    gripper_drive_target_position = gripper_position
                    log(
                        f"Policy gripper edge -> {gripper_action:+.0f}; smooth drive starts at "
                        f"{gripper_drive_target_position:.4f} m."
                    )
                limited_policy_action = np.concatenate(
                    (
                        accepted_translation_delta_b[0].detach().cpu().numpy().astype(
                            np.float32, copy=False
                        ),
                        rotation_delta_b.astype(np.float32, copy=False),
                        np.asarray([gripper_action], dtype=np.float32),
                    )
                )
                policy_query_count += 1

                target_tcp_position_b_record, target_tcp_quaternion_b_record = (
                    subtract_frame_transforms(
                        current_root_pose_w[:, 0:3],
                        current_root_pose_w[:, 3:7],
                        target_tcp_pose_w[:, 0:3],
                        target_tcp_pose_w[:, 3:7],
                    )
                )
                target_tcp_pose_b_record = torch.cat(
                    (target_tcp_position_b_record, target_tcp_quaternion_b_record), dim=-1
                )
                position_error = float(
                    torch.linalg.vector_norm(
                        target_tcp_pose_w[:, :3] - current_tcp_pose_w[:, :3], dim=-1
                    ).item()
                )
                orientation_error = float(
                    quat_error_magnitude(
                        target_tcp_pose_w[:, 3:7], current_tcp_pose_w[:, 3:7]
                    ).item()
                )
                sponge_pose_w_tensor = sponge.data.root_pose_w[0]
                tcp_sponge_distance = float(
                    torch.linalg.vector_norm(
                        current_tcp_pose_w[0, :3] - sponge_pose_w_tensor[:3]
                    ).item()
                )
                force_matrix_w = sponge_pan_contact_sensor.data.force_matrix_w
                if force_matrix_w is None:
                    raise RuntimeError("Sponge-to-pan filtered contact-force matrix is unavailable.")
                pan_contact_force = float(
                    torch.linalg.vector_norm(force_matrix_w, dim=-1).max().item()
                )
                sponge_linear_speed = float(
                    torch.linalg.vector_norm(sponge.data.root_vel_w[0, :3]).item()
                )
                recorder.capture(
                    rgb_by_camera,
                    {
                        "sim_time": step * args_cli.physics_dt,
                        "policy_query_index": policy_query_count - 1,
                        "policy_plan_step": policy_plan_step,
                        "policy_state": policy_state,
                        "raw_policy_action": raw_policy_action.copy(),
                        "limited_policy_action": limited_policy_action.copy(),
                        "policy_latency_s": last_policy_latency_s,
                        "policy_roundtrip_s": last_policy_roundtrip_s,
                        "joint_position": robot.data.joint_pos[0].detach().cpu().numpy(),
                        "joint_velocity": robot.data.joint_vel[0].detach().cpu().numpy(),
                        "joint_target": joint_target[0].detach().cpu().numpy(),
                        "joint_velocity_target": joint_velocity_target[0].detach().cpu().numpy(),
                        "joint_applied_torque": robot.data.applied_torque[0].detach().cpu().numpy(),
                        "robot_root_pose_w": current_root_pose_w[0].detach().cpu().numpy(),
                        "tcp_pose_w": current_tcp_pose_w[0].detach().cpu().numpy(),
                        "target_tcp_pose_w": target_tcp_pose_w[0].detach().cpu().numpy(),
                        "tcp_pose_b": current_tcp_pose_b[0].detach().cpu().numpy(),
                        "target_tcp_pose_b": target_tcp_pose_b_record[0].detach().cpu().numpy(),
                        "commanded_tcp_pose_b": torch.cat(
                            (commanded_tcp_position_b, commanded_tcp_quaternion_b), dim=-1
                        )[0]
                        .detach()
                        .cpu()
                        .numpy(),
                        "command_position_lag": command_position_lag,
                        "command_orientation_lag": command_orientation_lag,
                        "gripper_action": np.float32(gripper_action),
                        "gripper_position": gripper_position,
                        "gripper_open_fraction": gripper_open_fraction,
                        "gripper_drive_target_position": gripper_drive_target_position,
                        "sponge_pose_w": sponge_pose_w_tensor.detach().cpu().numpy(),
                        "sponge_linear_speed": sponge_linear_speed,
                        "tcp_sponge_distance": tcp_sponge_distance,
                        "grasp_validated": grasp_validated,
                        "released_after_grasp": released_after_grasp,
                        "pan_contact_force": pan_contact_force,
                        "pan_contact_elapsed": pan_contact_elapsed,
                        "position_error": position_error,
                        "orientation_error": orientation_error,
                    },
                )

            gripper_goal_position = (
                args_cli.open_position if gripper_action > 0.0 else args_cli.closed_position
            )
            maximum_gripper_step = args_cli.gripper_velocity_limit * args_cli.physics_dt
            gripper_target_step = float(
                np.clip(
                    gripper_goal_position - gripper_drive_target_position,
                    -maximum_gripper_step,
                    maximum_gripper_step,
                )
            )
            gripper_drive_target_position += gripper_target_step

            target_hand_pose_w = hand_target_from_tcp(target_tcp_pose_w, tcp_offset_b)
            root_pose_w = robot.data.root_pose_w
            hand_pose_w = robot.data.body_pose_w[:, hand_body_id]
            hand_position_b, hand_quaternion_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                hand_pose_w[:, 0:3],
                hand_pose_w[:, 3:7],
            )
            target_hand_position_b, target_hand_quaternion_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                target_hand_pose_w[:, 0:3],
                target_hand_pose_w[:, 3:7],
            )
            jacobian = robot.root_physx_view.get_jacobians()[
                :, hand_jacobian_id, :, arm_joint_ids
            ].clone()
            world_to_base_rotation = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
            jacobian[:, :3, :] = torch.bmm(world_to_base_rotation, jacobian[:, :3, :])
            jacobian[:, 3:, :] = torch.bmm(world_to_base_rotation, jacobian[:, 3:, :])
            current_arm_position = robot.data.joint_pos[:, arm_joint_ids]
            desired_arm_position, desired_arm_velocity, ik_diagnostics = ik.compute(
                hand_position_b,
                hand_quaternion_b,
                target_hand_position_b,
                target_hand_quaternion_b,
                jacobian,
                current_arm_position,
                current_arm_position.detach().clone(),
                arm_joint_lower,
                arm_joint_upper,
            )
            if not (
                torch.isfinite(desired_arm_position).all()
                and torch.isfinite(desired_arm_velocity).all()
            ):
                raise FloatingPointError("IK produced a non-finite drive target during inference.")

            joint_target = robot.data.joint_pos.clone()
            joint_target[:, arm_joint_ids] = desired_arm_position
            joint_target[:, finger_joint_ids] = gripper_drive_target_position
            joint_velocity_target.zero_()
            joint_velocity_target[:, arm_joint_ids] = desired_arm_velocity
            robot.set_joint_position_target(joint_target)
            robot.set_joint_velocity_target(joint_velocity_target)
            robot.write_data_to_sim()

            sim.step(render=True)
            robot.update(args_cli.physics_dt)
            sponge.update(args_cli.physics_dt)
            sponge_pan_contact_sensor.update(args_cli.physics_dt, force_recompute=True)
            for camera in cameras.values():
                camera.update(args_cli.physics_dt)

            current_tcp_pose_w = tcp_pose_from_hand(
                robot.data.body_pose_w[:, hand_body_id], tcp_offset_b
            )
            gripper_position = float(
                torch.mean(robot.data.joint_pos[:, finger_joint_ids]).item()
            )
            gripper_effort = float(
                torch.max(torch.abs(robot.data.applied_torque[:, finger_joint_ids])).item()
            )
            tcp_sponge_distance = float(
                torch.linalg.vector_norm(
                    current_tcp_pose_w[0, :3] - sponge.data.root_pose_w[0, :3]
                ).item()
            )
            sponge_linear_speed = float(
                torch.linalg.vector_norm(sponge.data.root_vel_w[0, :3]).item()
            )
            force_matrix_w = sponge_pan_contact_sensor.data.force_matrix_w
            if force_matrix_w is None:
                raise RuntimeError("Sponge-to-pan filtered contact-force matrix is unavailable.")
            pan_contact_force = float(
                torch.linalg.vector_norm(force_matrix_w, dim=-1).max().item()
            )
            maximum_pan_contact_force = max(maximum_pan_contact_force, pan_contact_force)

            contact_upper_position = (
                expected_grasp_contact_position + args_cli.gripper_position_tolerance
            )
            if (
                not grasp_validated
                and gripper_action < 0.0
                and minimum_grasp_contact_position <= gripper_position <= contact_upper_position
                and gripper_effort >= args_cli.grasp_min_effort
                and tcp_sponge_distance <= args_cli.grasp_max_tcp_sponge_distance
            ):
                grasp_validated = True
                log(
                    f"Inference grasp validated at t={(step + 1) * args_cli.physics_dt:.2f}s: "
                    f"finger={gripper_position:.4f}m effort={gripper_effort:.2f}N "
                    f"tcp_sponge={tcp_sponge_distance:.4f}m."
                )
            open_threshold = args_cli.open_position - args_cli.gripper_position_tolerance
            if grasp_validated and gripper_action > 0.0 and gripper_position >= open_threshold:
                released_after_grasp = True
            pan_contact = pan_contact_force >= args_cli.pan_contact_force_threshold
            if released_after_grasp and pan_contact:
                pan_contact_elapsed += args_cli.physics_dt
            else:
                pan_contact_elapsed = 0.0

            target_position_error = float(
                torch.linalg.vector_norm(
                    target_tcp_pose_w[:, :3] - current_tcp_pose_w[:, :3], dim=-1
                ).item()
            )
            target_orientation_error = float(
                quat_error_magnitude(
                    target_tcp_pose_w[:, 3:7], current_tcp_pose_w[:, 3:7]
                ).item()
            )
            status_wall_time = time.monotonic()
            if args_cli.status_interval > 0.0 and status_wall_time >= next_status_wall_time:
                action_text = np.array2string(
                    limited_policy_action, precision=3, floatmode="fixed", suppress_small=True
                )
                log(
                    f"inference t={(step + 1) * args_cli.physics_dt:6.2f}s "
                    f"query={policy_query_count:04d} chunk_step={policy_plan_step:02d} "
                    f"action={action_text} "
                    f"model={last_policy_latency_s:.3f}s rpc={last_policy_roundtrip_s:.3f}s "
                    f"pos_err={target_position_error:.4f}m "
                    f"rot_err={math.degrees(target_orientation_error):.2f}deg "
                    f"grip={gripper_position:.3f}m grasp={int(grasp_validated)} "
                    f"released={int(released_after_grasp)} pan_force={pan_contact_force:.3f}N "
                    f"pan_t={pan_contact_elapsed:.2f}s "
                    f"cmd_lag={command_position_lag:.3f}m,"
                    f"{math.degrees(command_orientation_lag):.1f}deg "
                    f"ik_sigma={float(ik_diagnostics['minimum_singular_value'].item()):.4f} "
                    f"frames={recorder.frame_index}"
                )
                next_status_wall_time = status_wall_time + args_cli.status_interval

            if (
                pan_contact_elapsed >= args_cli.pan_contact_settle_time
                and sponge_linear_speed <= args_cli.placement_max_sponge_speed
            ):
                task_success = True
                log(
                    f"SmolVLA task success: sponge-to-pan contact held {pan_contact_elapsed:.2f}s "
                    f"at sponge speed {sponge_linear_speed:.4f}m/s."
                )
                break
        else:
            task_failure = (
                f"Policy did not satisfy physical task success within "
                f"max-runtime={args_cli.max_runtime:.1f}s."
            )

        if not task_success and task_failure is None:
            task_failure = "Policy rollout ended without satisfying physical task success."
        if task_failure is not None:
            log(f"Inference rollout finished without task success: {task_failure}")
    except Exception as exc:
        system_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["evaluation"] = {
            "task_success": task_success,
            "task_failure": task_failure,
            "system_error": system_error,
            "policy_queries": policy_query_count,
            "grasp_validated": grasp_validated,
            "released_after_grasp": released_after_grasp,
            "maximum_pan_contact_force_n": maximum_pan_contact_force,
            "final_pan_contact_force_n": pan_contact_force,
            "final_pan_contact_time_s": pan_contact_elapsed,
            "final_sponge_linear_speed_m_s": sponge_linear_speed,
            "final_tcp_sponge_distance_m": tcp_sponge_distance,
        }
        if recorder is not None:
            recorder.close(
                success=task_success,
                error=system_error if system_error is not None else task_failure,
            )
        client.close()

    log(
        f"Inference render complete: {recorder.frame_index if recorder is not None else 0} "
        f"synchronized frames from each camera; task_success={task_success}."
    )
    return 0


def main() -> int:
    validate_sponge_episode_positions()
    if not 0 <= args_cli.episode_index < len(SPONGE_EPISODE_POSITIONS_W):
        raise ValueError(
            f"episode-index must be in [0, {len(SPONGE_EPISODE_POSITIONS_W) - 1}]."
        )
    if args_cli.validation_index is not None:
        if not args_cli.vla_inference:
            raise ValueError("validation-index is inference-only; also pass --vla-inference.")
        if not 0 <= args_cli.validation_index < len(VALIDATION_SPONGE_POSITIONS_W):
            raise ValueError(
                f"validation-index must be in [0, {len(VALIDATION_SPONGE_POSITIONS_W) - 1}]."
            )
    if args_cli.width <= 0 or args_cli.height <= 0:
        raise ValueError("Camera width and height must be positive.")
    if args_cli.physics_dt <= 0.0 or args_cli.camera_fps <= 0.0:
        raise ValueError("physics-dt and camera-fps must be positive.")
    if args_cli.camera_fps > 1.0 / args_cli.physics_dt:
        raise ValueError("camera-fps cannot exceed the physics rate.")
    if args_cli.status_interval < 0.0:
        raise ValueError("status-interval cannot be negative.")
    if args_cli.neutral_settle_time < 0.0:
        raise ValueError("neutral-settle-time cannot be negative.")
    if args_cli.position_tolerance <= 0.0:
        raise ValueError("position-tolerance must be positive.")
    if (
        args_cli.pre_grasp_position_tolerance <= 0.0
        or args_cli.grasp_position_tolerance <= 0.0
    ):
        raise ValueError("Pre-grasp and grasp position tolerances must be positive.")
    if args_cli.orientation_tolerance <= 0.0:
        raise ValueError("orientation-tolerance must be positive.")
    if args_cli.grasp_orientation_tolerance <= 0.0:
        raise ValueError("grasp-orientation-tolerance must be positive.")
    if args_cli.max_joint_step <= 0.0:
        raise ValueError("max-joint-step must be positive.")
    if args_cli.ik_position_gain <= 0.0 or args_cli.ik_orientation_gain <= 0.0:
        raise ValueError("IK Cartesian gains must be positive.")
    if args_cli.ik_max_linear_speed <= 0.0 or args_cli.ik_max_angular_speed <= 0.0:
        raise ValueError("IK Cartesian speed limits must be positive.")
    if args_cli.arm_command_velocity_limit <= 0.0:
        raise ValueError("arm-command-velocity-limit must be positive.")
    if args_cli.arm_command_acceleration_limit <= 0.0:
        raise ValueError("arm-command-acceleration-limit must be positive.")
    if args_cli.arm_following_error_limit <= 0.0:
        raise ValueError("arm-following-error-limit must be positive.")
    if args_cli.ik_damping_min <= 0.0:
        raise ValueError("ik-damping-min must be positive.")
    if args_cli.ik_damping_max < args_cli.ik_damping_min:
        raise ValueError("ik-damping-max cannot be smaller than ik-damping-min.")
    if args_cli.ik_singularity_threshold <= 0.0:
        raise ValueError("ik-singularity-threshold must be positive.")
    if args_cli.ik_posture_gain < 0.0 or args_cli.ik_joint_limit_gain < 0.0:
        raise ValueError("IK null-space gains cannot be negative.")
    if not 0.0 < args_cli.ik_joint_limit_margin < 0.5:
        raise ValueError("ik-joint-limit-margin must be between 0 and 0.5.")
    if args_cli.divergence_window <= 0.0:
        raise ValueError("divergence-window must be positive.")
    if (
        args_cli.divergence_position_growth <= 0.0
        or args_cli.divergence_orientation_growth <= 0.0
    ):
        raise ValueError("Divergence growth thresholds must be positive.")
    if args_cli.transport_speed <= 0.0:
        raise ValueError("transport-speed must be positive.")
    if args_cli.trajectory_seed < 0:
        raise ValueError("trajectory-seed must be non-negative.")
    if args_cli.noise_clip_sigma <= 0.0:
        raise ValueError("noise-clip-sigma must be positive.")
    if any(value < 0.0 for value in args_cli.transport_noise_std):
        raise ValueError("transport-noise-std values must be non-negative.")
    if any(value < 0.0 for value in args_cli.release_noise_std):
        raise ValueError("release-noise-std values must be non-negative.")
    vertical_speeds = (
        args_cli.grasp_descent_speed,
        args_cli.lift_speed,
        args_cli.place_descent_speed,
        args_cli.retreat_speed,
    )
    if any(speed <= 0.0 for speed in vertical_speeds):
        raise ValueError("All vertical Cartesian speeds must be positive.")
    if args_cli.open_position <= 0.0:
        raise ValueError("open-position must be positive.")
    if args_cli.gripper_effort_limit <= 0.0:
        raise ValueError("gripper-effort-limit must be positive.")
    if args_cli.gripper_velocity_limit <= 0.0:
        raise ValueError("gripper-velocity-limit must be positive.")
    if args_cli.close_hold < 0.0:
        raise ValueError("close-hold cannot be negative.")
    if args_cli.open_hold < 0.0:
        raise ValueError("open-hold cannot be negative.")
    if not 0.0 <= args_cli.closed_position <= args_cli.open_position:
        raise ValueError("closed-position must be between zero and open-position.")
    if not 0.0 <= args_cli.gripper_position_tolerance < args_cli.open_position - args_cli.closed_position:
        raise ValueError("gripper-position-tolerance must be non-negative and smaller than finger travel.")
    if not 0.0 <= args_cli.grasp_contact_position_tolerance < (
        args_cli.open_position - args_cli.closed_position
    ):
        raise ValueError(
            "grasp-contact-position-tolerance must be non-negative and smaller than finger travel."
        )
    if args_cli.grasp_min_effort < 0.0:
        raise ValueError("grasp-min-effort cannot be negative.")
    if args_cli.grasp_max_tcp_sponge_distance <= 0.0:
        raise ValueError("grasp-max-tcp-sponge-distance must be positive.")
    if args_cli.pan_contact_force_threshold <= 0.0:
        raise ValueError("pan-contact-force-threshold must be positive.")
    if args_cli.pan_contact_settle_time <= 0.0:
        raise ValueError("pan-contact-settle-time must be positive.")
    if args_cli.placement_max_sponge_speed <= 0.0:
        raise ValueError("placement-max-sponge-speed must be positive.")
    if args_cli.arm_stiffness <= 0.0 or args_cli.arm_damping <= 0.0:
        raise ValueError("Arm drive stiffness and damping must be positive.")
    if args_cli.policy_max_translation_step <= 0.0:
        raise ValueError("policy-max-translation-step must be positive.")
    if args_cli.policy_max_rotation_step <= 0.0:
        raise ValueError("policy-max-rotation-step must be positive.")
    if args_cli.policy_max_position_lag <= 0.0:
        raise ValueError("policy-max-position-lag must be positive.")
    if not 0.0 < args_cli.policy_max_orientation_lag <= math.pi:
        raise ValueError("policy-max-orientation-lag must be in (0, pi].")
    if (
        args_cli.policy_servo_position_gain <= 0.0
        or args_cli.policy_servo_orientation_gain <= 0.0
    ):
        raise ValueError("Policy Cartesian servo gains must be positive.")
    if not 0.0 < args_cli.policy_gripper_threshold <= 1.0:
        raise ValueError("policy-gripper-threshold must be in (0, 1].")
    if any(
        lower >= upper
        for lower, upper in zip(
            args_cli.policy_workspace_min,
            args_cli.policy_workspace_max,
            strict=True,
        )
    ):
        raise ValueError("Every policy-workspace-min component must be below its maximum.")

    log(
        f"GPU pinning: CUDA_VISIBLE_DEVICES={physical_gpu_id}, "
        "logical device=cuda:0, RTX multi-GPU disabled."
    )

    stage_path = args_cli.stage.expanduser().resolve()
    if not stage_path.is_file():
        raise FileNotFoundError(f"Working stage does not exist: {stage_path}")

    log(f"Opening stage: {stage_path}")
    if not sim_utils.open_stage(str(stage_path)):
        raise RuntimeError(f"Isaac Sim failed to open {stage_path}")
    stage = wait_for_stage(args_cli.stage_load_timeout, args_cli.status_interval)

    up_axis = UsdGeom.GetStageUpAxis(stage)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if up_axis != UsdGeom.Tokens.y:
        raise RuntimeError(f"This controller expects the authored Y-up stage; found up axis {up_axis!r}.")
    if not math.isclose(meters_per_unit, 1.0, abs_tol=1.0e-9):
        raise RuntimeError(f"This controller expects meter units; found metersPerUnit={meters_per_unit}.")

    waypoint_layer_path = stage_path.with_name(WAYPOINT_LAYER_NAME)
    waypoint_layer = Sdf.Layer.FindOrOpen(str(waypoint_layer_path))
    if waypoint_layer is None:
        raise RuntimeError(f"Could not open waypoint overlay: {waypoint_layer_path}")
    if waypoint_layer not in stage.GetLayerStack():
        raise RuntimeError(f"The stage does not compose the waypoint overlay: {waypoint_layer_path}")

    require_prim(stage, ROBOT_PATH)
    require_prim(stage, HAND_PATH)
    require_prim(stage, SPONGE_PATH)
    require_prim(stage, PAN_PATH)
    require_prim(stage, PAN_COLLISION_PATH)
    for camera_name, camera_path in CAMERA_PATHS.items():
        require_prim(stage, camera_path, UsdGeom.Camera)
        log(f"Camera {camera_name}: {camera_path}")
    franka_deinstanced_prims = deinstance_subtree_in_session(stage, ROBOT_PATH)
    if franka_deinstanced_prims:
        log(
            f"Expanded {len(franka_deinstanced_prims)} Franka instance roots in the session layer "
            "to prevent Fabric prototype dropouts."
        )
        simulation_app.update()
        stage = wait_for_stage(args_cli.stage_load_timeout, args_cli.status_interval)
    else:
        log("Franka contains no USD instance roots requiring expansion.")
    sponge_scale = enforce_sponge_scale(stage, args_cli.sponge_scale)
    if args_cli.validation_index is None:
        sponge_position_split = "training"
        sponge_position_index = args_cli.episode_index
        selected_sponge_position_w = SPONGE_EPISODE_POSITIONS_W[sponge_position_index]
    else:
        sponge_position_split = "validation"
        sponge_position_index = args_cli.validation_index
        selected_sponge_position_w = VALIDATION_SPONGE_POSITIONS_W[sponge_position_index]
    author_sponge_position_in_session(stage, selected_sponge_position_w)
    log(
        f"Sponge position split={sponge_position_split} "
        f"index={sponge_position_index:02d} (world XYZ): "
        f"{selected_sponge_position_w}"
    )

    # Mimic the real arm's gravity-compensated mode while retaining full link
    # inertia, contacts, joint limits, and torque-limited PhysX drives.
    sim_utils.modify_rigid_body_properties(
        ROBOT_PATH, sim_utils.RigidBodyPropertiesCfg(disable_gravity=True), stage=stage
    )

    # In this Y-up stage, rotate +90 degrees about X so hand/TCP local +Z points
    # down (-Y) and the finger-separation axis (local +Y) spans world Z.
    grasp_quaternion_w = np.asarray(
        [
            math.sqrt(0.5),
            math.sqrt(0.5),
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    sponge_lower, sponge_upper = aligned_world_bounds(stage, SPONGE_PATH)
    sponge_width = float(sponge_upper[2] - sponge_lower[2])
    half_sponge_width = 0.5 * sponge_width
    if half_sponge_width > args_cli.open_position:
        raise ValueError(
            f"Sponge width {sponge_width:.4f} m exceeds the gripper opening."
        )

    expected_contact_position = half_sponge_width
    minimum_grasp_contact_position = max(
        args_cli.closed_position,
        expected_contact_position - args_cli.grasp_contact_position_tolerance,
    )
    log(
        f"Gripper command: fully-closed target={args_cli.closed_position:.4f} m per finger; "
        f"rigid contact expected near {expected_contact_position:.4f} m; "
        f"drive effort capped at {args_cli.gripper_effort_limit:.1f} N per finger."
    )

    capture_stride = max(1, int(round(1.0 / (args_cli.physics_dt * args_cli.camera_fps))))
    actual_camera_fps = 1.0 / (capture_stride * args_cli.physics_dt)
    if not math.isclose(actual_camera_fps, args_cli.camera_fps, rel_tol=1.0e-6):
        log(f"Adjusted camera rate from {args_cli.camera_fps:.3f} to {actual_camera_fps:.3f} Hz.")

    # GPU simulation needs Fabric for animated renderer transforms. Franka's
    # visual instances were expanded above to avoid UsdRT prototype dropouts.
    sim_cfg = sim_utils.SimulationCfg(
        dt=args_cli.physics_dt,
        render_interval=1 if args_cli.preview_sponge_positions else capture_stride,
        device=args_cli.device,
        gravity=(0.0, -9.81, 0.0),
        physics_prim_path="/World/physicsScene",
        physx=sim_utils.PhysxCfg(solve_articulation_contact_last=True),
        use_fabric=True,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    previous_edit_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    try:
        sim_utils.activate_contact_sensors(
            SPONGE_PATH,
            threshold=0.0,
            stage=stage,
        )
    finally:
        stage.SetEditTarget(previous_edit_target)

    robot_cfg = FRANKA_PANDA_CFG.replace(prim_path=ROBOT_PATH)
    for actuator_name in ("panda_shoulder", "panda_forearm"):
        robot_cfg.actuators[actuator_name].stiffness = args_cli.arm_stiffness
        robot_cfg.actuators[actuator_name].damping = args_cli.arm_damping
    robot_cfg.actuators["panda_hand"].effort_limit_sim = args_cli.gripper_effort_limit
    robot_cfg.actuators["panda_hand"].velocity_limit_sim = args_cli.gripper_velocity_limit
    robot_cfg.spawn = None
    log(
        f"Arm control: PhysX drives kp={args_cli.arm_stiffness:g}, kd={args_cli.arm_damping:g}, "
        "with smooth seven-joint position/velocity targets and torque limits."
    )
    log(
        "Gripper action contract: +1=open, -1=close. The binary command selects a drive endpoint; "
        "each command edge starts a new drive trajectory from the measured finger position. "
        f"The target slew and PhysX velocity cap are {args_cli.gripper_velocity_limit:.3f} m/s with "
        f"{args_cli.gripper_effort_limit:g} N."
    )
    robot = Articulation(robot_cfg)
    sponge_cfg = RigidObjectCfg(prim_path=SPONGE_PATH, spawn=None)
    sponge = RigidObject(sponge_cfg)
    sponge_pan_contact_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path=SPONGE_PATH,
            update_period=0.0,
            history_length=3,
            debug_vis=False,
            filter_prim_paths_expr=[PAN_COLLISION_PATH],
        )
    )

    cameras: dict[str, Camera] = {}
    for camera_name, camera_path in CAMERA_PATHS.items():
        cameras[camera_name] = Camera(
            CameraCfg(
                prim_path=camera_path,
                spawn=None,
                width=args_cli.width,
                height=args_cli.height,
                update_period=args_cli.physics_dt if args_cli.preview_sponge_positions else 1.0 / actual_camera_fps,
                data_types=["rgb"],
                update_latest_camera_pose=False,
            )
        )
    log(f"Configured synchronized RGB streams: {list(cameras)}")

    log("Initializing physics, articulation, and both offscreen cameras.")
    sim.reset()
    robot.reset()
    sponge.reset()
    sponge_pan_contact_sensor.reset()
    if sponge_pan_contact_sensor.contact_physx_view.filter_count != 1:
        raise RuntimeError("Sponge-to-pan contact filter did not resolve exactly one collision shape.")
    log(f"Physical placement check: {SPONGE_PATH} contact with {PAN_COLLISION_PATH}.")
    for camera in cameras.values():
        camera.reset()

    arm_joint_ids = joint_indices(robot, ARM_JOINT_NAMES)
    finger_joint_ids = joint_indices(robot, FINGER_JOINT_NAMES)
    if "panda_hand" not in robot.body_names:
        raise RuntimeError(f"Franka is missing panda_hand. Found bodies: {robot.body_names}")
    hand_body_id = robot.body_names.index("panda_hand")
    hand_jacobian_id = hand_body_id - 1 if robot.is_fixed_base else hand_body_id

    neutral_target = reset_robot_to_neutral(robot)
    neutral_settle_steps = max(5, int(math.ceil(args_cli.neutral_settle_time / args_cli.physics_dt)))
    log(f"Settling neutral pose for {neutral_settle_steps * args_cli.physics_dt:.2f} simulated seconds.")
    for _ in range(neutral_settle_steps):
        robot.set_joint_position_target(neutral_target)
        robot.set_joint_velocity_target(torch.zeros_like(neutral_target[:, arm_joint_ids]), joint_ids=arm_joint_ids)
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(args_cli.physics_dt)
        sponge.update(args_cli.physics_dt)
        sponge_pan_contact_sensor.update(args_cli.physics_dt, force_recompute=True)
        for camera in cameras.values():
            camera.update(args_cli.physics_dt)

    if args_cli.preview_sponge_positions:
        render_sponge_position_previews(sim, sponge, cameras)
        return 0

    if args_cli.vla_inference:
        return run_vla_policy_rollout(
            sim=sim,
            robot=robot,
            sponge=sponge,
            sponge_pan_contact_sensor=sponge_pan_contact_sensor,
            cameras=cameras,
            arm_joint_ids=arm_joint_ids,
            finger_joint_ids=finger_joint_ids,
            hand_body_id=hand_body_id,
            hand_jacobian_id=hand_jacobian_id,
            actual_camera_fps=actual_camera_fps,
            capture_stride=capture_stride,
            stage_path=stage_path,
            selected_sponge_position_w=selected_sponge_position_w,
            sponge_position_split=sponge_position_split,
            sponge_position_index=sponge_position_index,
            sponge_scale=sponge_scale,
            expected_grasp_contact_position=expected_contact_position,
            minimum_grasp_contact_position=minimum_grasp_contact_position,
            franka_deinstanced_prims=franka_deinstanced_prims,
        )


    episode_trajectory_seed = args_cli.trajectory_seed + args_cli.episode_index
    trajectory_rng = np.random.default_rng(episode_trajectory_seed)
    transport_noise_std_w = np.asarray(args_cli.transport_noise_std, dtype=np.float64)
    release_noise_std_w = np.asarray(args_cli.release_noise_std, dtype=np.float64)
    transport_control_noise_w = trajectory_rng.normal(
        loc=0.0,
        scale=transport_noise_std_w,
        size=(2, 3),
    )
    transport_control_noise_w = np.clip(
        transport_control_noise_w,
        -args_cli.noise_clip_sigma * transport_noise_std_w,
        args_cli.noise_clip_sigma * transport_noise_std_w,
    )
    release_offset_w = trajectory_rng.normal(loc=0.0, scale=release_noise_std_w)
    release_offset_w = np.clip(
        release_offset_w,
        -args_cli.noise_clip_sigma * release_noise_std_w,
        args_cli.noise_clip_sigma * release_noise_std_w,
    )
    log(
        f"Trajectory randomization seed={episode_trajectory_seed}: "
        f"transport_control_noise_w={np.round(transport_control_noise_w, 5).tolist()}, "
        f"release_offset_w={np.round(release_offset_w, 5).tolist()}"
    )


    tcp_offset_b = torch.tensor(
        [[0.0, 0.0, args_cli.tcp_offset]], dtype=torch.float32, device=sim.device
    )
    hand_pose_w = robot.data.body_pose_w[:, hand_body_id]
    home_tcp_pose_w = tcp_pose_from_hand(hand_pose_w, tcp_offset_b)

    grasp_quaternion_tensor = torch.tensor(
        grasp_quaternion_w, dtype=torch.float32, device=sim.device
    ).unsqueeze(0)
    approach_direction_w = quat_apply(
        grasp_quaternion_tensor,
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=sim.device),
    )[0]
    closing_axis_w = quat_apply(
        grasp_quaternion_tensor,
        torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32, device=sim.device),
    )[0]
    log(
        "Top-down grasp frame: "
        f"approach={approach_direction_w.detach().cpu().numpy().round(4).tolist()}, "
        f"finger_axis={closing_axis_w.detach().cpu().numpy().round(4).tolist()}."
    )

    previous_edit_target = stage.GetEditTarget()
    stage.SetEditTarget(waypoint_layer)
    try:
        waypoints = build_waypoints(
            stage,
            home_tcp_pose_w,
            grasp_quaternion_w,
            sim.device,
            release_offset_w,
        )
    finally:
        stage.SetEditTarget(previous_edit_target)

    if args_cli.save_waypoints or args_cli.waypoints_only:
        log(f"Saving generated waypoint Xforms into overlay: {waypoint_layer_path}")
        if not waypoint_layer.Save():
            raise RuntimeError(f"Failed to save waypoint overlay: {waypoint_layer_path}")
    if args_cli.waypoints_only:
        log(f"Saved visible waypoint markers under {WAYPOINT_ROOT}; waypoint-only mode complete.")
        return 0
    vertical_motion_specs = (
        ("GRASP", "pre_grasp", "grasp", args_cli.grasp_descent_speed),
        ("LIFT", "grasp", "lift", args_cli.lift_speed),
        ("PLACE", "above_pan", "place", args_cli.place_descent_speed),
        ("RETREAT", "place", "retreat", args_cli.retreat_speed),
    )
    vertical_trajectories: dict[str, dict[str, Any]] = {}
    for state_name, start_name, goal_name, speed in vertical_motion_specs:
        distance = float(
            torch.linalg.norm(waypoints[goal_name][:, :3] - waypoints[start_name][:, :3]).item()
        )
        duration = max(args_cli.physics_dt, distance / speed)
        vertical_trajectories[state_name] = {
            "start_waypoint": start_name,
            "goal_waypoint": goal_name,
            "speed": speed,
            "distance": distance,
            "duration": duration,
        }
        log(
            f"{state_name} vertical motion: {distance:.3f} m over {duration:.2f} s "
            f"at {speed:.3f} m/s average."
        )

    transport_start_w = waypoints["lift"][0, :3]
    transport_goal_w = waypoints["above_pan"][0, :3]
    transport_delta_w = transport_goal_w - transport_start_w
    transport_noise_tensor = torch.tensor(
        transport_control_noise_w,
        dtype=torch.float32,
        device=sim.device,
    )
    transport_control_points_w = torch.stack(
        (
            transport_start_w,
            transport_start_w + transport_delta_w / 3.0 + transport_noise_tensor[0],
            transport_start_w + 2.0 * transport_delta_w / 3.0 + transport_noise_tensor[1],
            transport_goal_w,
        ),
        dim=0,
    )
    arc_samples = torch.stack(
        [
            cubic_bezier_position(transport_control_points_w, sample_index / 64.0)
            for sample_index in range(65)
        ],
        dim=0,
    )
    transport_distance = float(
        torch.linalg.norm(arc_samples[1:] - arc_samples[:-1], dim=-1).sum().item()
    )
    transport_duration = max(args_cli.physics_dt, transport_distance / args_cli.transport_speed)
    transport_trajectory = {
        "state": "ABOVE_PAN",
        "curve": "cubic_bezier_with_smoothstep_time",
        "start_waypoint": "lift",
        "goal_waypoint": "above_pan",
        "speed": args_cli.transport_speed,
        "arc_length": transport_distance,
        "duration": transport_duration,
        "control_points_w_xyz": transport_control_points_w.detach().cpu().numpy().tolist(),
    }
    log(
        f"ABOVE_PAN transport curve: {transport_distance:.3f} m over "
        f"{transport_duration:.2f} s at {args_cli.transport_speed:.3f} m/s average."
    )


    ik = AdaptiveNullspaceIKController(
        physics_dt=args_cli.physics_dt,
        position_gain=args_cli.ik_position_gain,
        orientation_gain=args_cli.ik_orientation_gain,
        max_linear_speed=args_cli.ik_max_linear_speed,
        max_angular_speed=args_cli.ik_max_angular_speed,
        damping_min=args_cli.ik_damping_min,
        damping_max=args_cli.ik_damping_max,
        singularity_threshold=args_cli.ik_singularity_threshold,
        posture_gain=args_cli.ik_posture_gain,
        joint_limit_gain=args_cli.ik_joint_limit_gain,
        joint_limit_margin=args_cli.ik_joint_limit_margin,
        joint_velocity_limit=args_cli.arm_command_velocity_limit,
        joint_acceleration_limit=args_cli.arm_command_acceleration_limit,
        following_error_limit=args_cli.arm_following_error_limit,
        max_joint_step=args_cli.max_joint_step,
    )
    ik.reset(robot.data.joint_pos[:, arm_joint_ids])

    posture_references = {
        "neutral": torch.tensor(
            [NEUTRAL_JOINT_POSITION[name] for name in ARM_JOINT_NAMES],
            dtype=torch.float32,
            device=sim.device,
        ).unsqueeze(0),
        "grasp_branch": torch.tensor(
            GRASP_BRANCH_JOINT_POSITION,
            dtype=torch.float32,
            device=sim.device,
        ).unsqueeze(0),
        "lift_branch": torch.tensor(
            LIFT_BRANCH_JOINT_POSITION,
            dtype=torch.float32,
            device=sim.device,
        ).unsqueeze(0),
        "pan_branch": torch.tensor(
            PAN_BRANCH_JOINT_POSITION,
            dtype=torch.float32,
            device=sim.device,
        ).unsqueeze(0),
    }
    posture_reference_by_state = {
        "HOME": "neutral",
        "PRE_GRASP": "grasp_branch",
        "GRASP": "grasp_branch",
        "CLOSE": "grasp_branch",
        "LIFT": "lift_branch",
        "ABOVE_PAN": "pan_branch",
        "PLACE": "pan_branch",
        "OPEN": "pan_branch",
        "RETREAT": "pan_branch",
        "DONE": "pan_branch",
    }
    state_machine = PickPlaceStateMachine(args_cli)
    log(
        "IK control: persistent PhysX position/velocity drive targets with "
        f"joint speed <= {args_cli.arm_command_velocity_limit:.3f} rad/s, "
        f"joint acceleration <= {args_cli.arm_command_acceleration_limit:.3f} rad/s^2, "
        f"following error <= {args_cli.arm_following_error_limit:.3f} rad, "
        f"lambda=[{args_cli.ik_damping_min:.3f}, {args_cli.ik_damping_max:.3f}]. "
        "Validated posture references are null-space objectives only."
    )

    metadata: dict[str, Any] = {
        "dataset_schema_version": 3,
        "task_description": args_cli.task_description,
        "stage": str(stage_path),
        "camera_prim": CAMERA_PATHS["main"],
        "camera_prims": CAMERA_PATHS,
        "camera_width": args_cli.width,
        "camera_outputs": {
            "main": {"rgb_dir": "rgb", "video": "rollout.mp4"},
            "wrist": {"rgb_dir": "rgb_wrist", "video": "rollout_wrist.mp4"},
        },
        "camera_height": args_cli.height,
        "camera_fps": actual_camera_fps,
        "physics_dt": args_cli.physics_dt,
        "sample_timing": {
            "observation_time_field": "sim_time",
            "action_start_time_field": "action_start_time",
            "action_to_observation_latency_s": args_cli.physics_dt,
            "semantics": "action applied for one physics step before the recorded observation",
        },
        "robot_prim": ROBOT_PATH,
        "sponge_prim": SPONGE_PATH,
        "episode_index": args_cli.episode_index,
        "initial_sponge_position_w_xyz": list(selected_sponge_position_w),
        "sponge_episode_positions_w_xyz": [
            list(position) for position in SPONGE_EPISODE_POSITIONS_W
        ],
        "sponge_position_limits_w_xyz": {
            axis: list(bounds) for axis, bounds in SPONGE_POSITION_LIMITS_W.items()
        },
        "pan_prim": PAN_PATH,
        "sponge_scale": sponge_scale,
        "sponge_width_along_finger_axis": sponge_width,
        "gripper_open_position": args_cli.open_position,
        "gripper_closed_position": args_cli.closed_position,
        "gripper_expected_contact_position": expected_contact_position,
        "gripper_effort_limit": args_cli.gripper_effort_limit,
        "gripper_velocity_limit": args_cli.gripper_velocity_limit,
        "gripper_target_slew_rate": args_cli.gripper_velocity_limit,
        "gripper_position_tolerance": args_cli.gripper_position_tolerance,
        "gripper_close_hold": args_cli.close_hold,
        "gripper_open_hold": args_cli.open_hold,
        "physical_validation": {
            "grasp": {
                "strict_pre_grasp_position_tolerance_m": args_cli.pre_grasp_position_tolerance,
                "strict_grasp_position_tolerance_m": args_cli.grasp_position_tolerance,
                "strict_grasp_orientation_tolerance_rad": args_cli.grasp_orientation_tolerance,
                "minimum_per_finger_contact_position_m": minimum_grasp_contact_position,
                "minimum_effort_n": args_cli.grasp_min_effort,
                "maximum_tcp_sponge_distance_m": args_cli.grasp_max_tcp_sponge_distance,
            },
            "placement": {
                "contact_sensor_prim": SPONGE_PATH,
                "contact_filter_prim": PAN_COLLISION_PATH,
                "minimum_contact_force_n": args_cli.pan_contact_force_threshold,
                "required_continuous_contact_time_s": args_cli.pan_contact_settle_time,
                "maximum_sponge_linear_speed_m_s": args_cli.placement_max_sponge_speed,
            },
            "observed": {},
        },
        "action_schema": {
            "gripper_action": {
                "field": "gripper_action",
                "open": GRIPPER_OPEN_ACTION,
                "close": GRIPPER_CLOSE_ACTION,
                "semantics": "absolute_binary_endpoint_command_with_rate_limited_driver",
                "edge_behavior": "replan_drive_target_from_measured_finger_position",
            },
        },
        "grasp_quaternion_wxyz": grasp_quaternion_w.tolist(),
        "joint_names": robot.joint_names,
        "vertical_trajectories": vertical_trajectories,
        "transport_trajectory": transport_trajectory,
        "trajectory_randomization": {
            "base_seed": args_cli.trajectory_seed,
            "episode_seed": episode_trajectory_seed,
            "noise_clip_sigma": args_cli.noise_clip_sigma,
            "transport_control_noise_std_w_xyz": transport_noise_std_w.tolist(),
            "transport_control_noise_w_xyz": transport_control_noise_w.tolist(),
            "release_noise_std_w_xyz": release_noise_std_w.tolist(),
            "release_offset_w_xyz": release_offset_w.tolist(),
            "grasp_noise_applied": False,
            "transport_noise_semantics": "two_episode_fixed_cubic_bezier_control_point_offsets",
            "release_noise_semantics": "shared_above_pan_place_retreat_endpoint_offset",
        },
        "rendering": {
            "use_fabric": True,
            "franka_deinstanced_prim_count": len(franka_deinstanced_prims),
            "franka_deinstanced_prims": franka_deinstanced_prims,
        },
        "state_names": [state.name for state in state_machine.states],
        "neutral_joint_position": NEUTRAL_JOINT_POSITION,
        "control": {
            "type": "adaptive_dls_persistent_rate_limited_physx_position_velocity_drive",
            "arm_policy_command_field": "target_tcp_pose_w",
            "arm_policy_command_type": "absolute_world_tcp_pose_wxyz",
            "lerobot_action_type": "executed_delta_tcp_pose_in_robot_base_frame_plus_binary_gripper",
            "gripper_policy_command_field": "gripper_action",
            "gripper_drive_type": "slew_limited_implicit_prismatic_position_drive",
            "arm_drive_target_semantics": "persistent_integrated_position_plus_velocity_target",
            "arm_stiffness": args_cli.arm_stiffness,
            "arm_damping": args_cli.arm_damping,
            "max_joint_step": args_cli.max_joint_step,
            "arm_command_velocity_limit_rad_s": args_cli.arm_command_velocity_limit,
            "arm_command_acceleration_limit_rad_s2": args_cli.arm_command_acceleration_limit,
            "arm_following_error_limit_rad": args_cli.arm_following_error_limit,
            "ik_position_gain": args_cli.ik_position_gain,
            "ik_orientation_gain": args_cli.ik_orientation_gain,
            "ik_max_linear_speed_m_s": args_cli.ik_max_linear_speed,
            "ik_max_angular_speed_rad_s": args_cli.ik_max_angular_speed,
            "ik_damping_min": args_cli.ik_damping_min,
            "ik_damping_max": args_cli.ik_damping_max,
            "ik_singularity_threshold": args_cli.ik_singularity_threshold,
            "ik_posture_gain": args_cli.ik_posture_gain,
            "ik_joint_limit_gain": args_cli.ik_joint_limit_gain,
            "ik_joint_limit_margin_fraction": args_cli.ik_joint_limit_margin,
            "posture_reference_semantics": "null_space_objective_only_never_directly_commanded",
            "posture_references": {
                name: value[0].detach().cpu().numpy().tolist()
                for name, value in posture_references.items()
            },
            "posture_reference_by_state": posture_reference_by_state,
            "joint_limit_source": "articulation_soft_joint_position_limits",
            "divergence_watchdog": {
                "window_s": args_cli.divergence_window,
                "position_growth_m": args_cli.divergence_position_growth,
                "orientation_growth_rad": args_cli.divergence_orientation_growth,
                "active_after_commanded_trajectory_completes": True,
            },
            "position_tolerance_m": args_cli.position_tolerance,
            "orientation_tolerance_rad": args_cli.orientation_tolerance,
            "settle_time_s": args_cli.settle_time,
            "gripper_stiffness": float(robot_cfg.actuators["panda_hand"].stiffness),
            "gripper_damping": float(robot_cfg.actuators["panda_hand"].damping),
            "gravity_compensated_mode": "arm_gravity_disabled",
        },
        "waypoints_wxyz": {
            name: pose[0].detach().cpu().numpy().tolist() for name, pose in waypoints.items()
        },
        "robot_root_pose_wxyz": robot.data.root_pose_w[0].detach().cpu().numpy().tolist(),
    }
    recorder = RolloutRecorder(
        args_cli.output_dir,
        actual_camera_fps,
        save_video=not args_cli.no_video,
        camera_names=tuple(CAMERA_PATHS),
        metadata=metadata,
    )

    max_steps = int(math.ceil(args_cli.max_runtime / args_cli.physics_dt))
    success = False
    failure: str | None = None
    joint_target = neutral_target.clone()
    joint_velocity_target = torch.zeros_like(neutral_target)
    gripper_drive_target_position = args_cli.open_position
    previous_gripper_action = GRIPPER_OPEN_ACTION
    next_status_wall_time = time.monotonic()
    arm_joint_lower = robot.data.soft_joint_pos_limits[:, arm_joint_ids, 0].clone()
    arm_joint_upper = robot.data.soft_joint_pos_limits[:, arm_joint_ids, 1].clone()
    tracked_state_index = -1
    posture_reference_name = "neutral"
    posture_reference = posture_references[posture_reference_name]
    best_post_trajectory_position_error = math.inf
    best_post_trajectory_orientation_error = math.inf
    divergence_elapsed = 0.0
    grasp_validated = False
    pan_contact_elapsed = 0.0
    maximum_pan_contact_force = 0.0
    tcp_sponge_distance = math.inf
    sponge_linear_speed = math.inf
    pan_contact_force = 0.0

    try:
        for step in range(max_steps):
            if not simulation_app.is_running():
                failure = "Simulation application stopped before the rollout completed."
                break

            state = state_machine.current
            if state_machine.index != tracked_state_index:
                tracked_state_index = state_machine.index
                posture_reference_name = posture_reference_by_state[state.name]
                posture_reference = posture_references[
                    posture_reference_name
                ]
                best_post_trajectory_position_error = math.inf
                best_post_trajectory_orientation_error = math.inf
                divergence_elapsed = 0.0
                posture_values = (
                    posture_reference[0].detach().cpu().numpy().round(4).tolist()
                )
                log(
                    f"IK posture objective -> {state.name}: "
                    f"{posture_reference_name} {posture_values} "
                    "(null-space only)"
                )
            goal_tcp_pose_w = waypoints[state.waypoint]
            target_tcp_pose_w = goal_tcp_pose_w
            motion_progress = 1.0
            trajectory_complete = True
            vertical_trajectory = vertical_trajectories.get(state.name)
            transport_trajectory_active = state.name == transport_trajectory["state"]
            trajectory_kind = ""
            if vertical_trajectory is not None:
                trajectory_kind = "vertical"
                duration = float(vertical_trajectory["duration"])
                time_fraction = min(1.0, state_machine.elapsed / duration)
                trajectory_complete = time_fraction >= 1.0
                motion_progress = time_fraction * time_fraction * (3.0 - 2.0 * time_fraction)
                target_tcp_pose_w = goal_tcp_pose_w.clone()
                motion_start_w = waypoints[str(vertical_trajectory["start_waypoint"])]
                target_tcp_pose_w[:, :3] = (
                    motion_start_w[:, :3]
                    + motion_progress * (goal_tcp_pose_w[:, :3] - motion_start_w[:, :3])
                )
            elif transport_trajectory_active:
                trajectory_kind = "transport"
                duration = float(transport_trajectory["duration"])
                time_fraction = min(1.0, state_machine.elapsed / duration)
                trajectory_complete = time_fraction >= 1.0
                motion_progress = time_fraction * time_fraction * (3.0 - 2.0 * time_fraction)
                target_tcp_pose_w = goal_tcp_pose_w.clone()
                target_tcp_pose_w[:, :3] = cubic_bezier_position(
                    transport_control_points_w,
                    motion_progress,
                )
            gripper_action = state.gripper_action
            if gripper_action not in (GRIPPER_OPEN_ACTION, GRIPPER_CLOSE_ACTION):
                raise RuntimeError(f"Invalid gripper action {gripper_action}; expected exactly -1 or +1.")
            gripper_command_edge = gripper_action != previous_gripper_action
            if gripper_command_edge:
                gripper_drive_target_position = float(
                    torch.mean(robot.data.joint_pos[:, finger_joint_ids]).item()
                )
                previous_gripper_action = gripper_action
                log(
                    f"Gripper action -> {gripper_action:+.0f}; drive trajectory starts from measured "
                    f"position {gripper_drive_target_position:.4f} m."
                )
            gripper_goal_position = args_cli.open_position if gripper_action > 0.0 else args_cli.closed_position
            max_gripper_target_step = args_cli.gripper_velocity_limit * args_cli.physics_dt
            gripper_target_step = float(
                np.clip(
                    gripper_goal_position - gripper_drive_target_position,
                    -max_gripper_target_step,
                    max_gripper_target_step,
                )
            )
            gripper_drive_target_position += gripper_target_step
            gripper_target_position = gripper_drive_target_position
            gripper_drive_velocity = gripper_target_step / args_cli.physics_dt
            target_hand_pose_w = hand_target_from_tcp(target_tcp_pose_w, tcp_offset_b)

            root_pose_w = robot.data.root_pose_w
            hand_pose_w = robot.data.body_pose_w[:, hand_body_id]
            hand_position_b, hand_quaternion_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                hand_pose_w[:, 0:3],
                hand_pose_w[:, 3:7],
            )
            target_hand_position_b, target_hand_quaternion_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                target_hand_pose_w[:, 0:3],
                target_hand_pose_w[:, 3:7],
            )
            jacobian = robot.root_physx_view.get_jacobians()[
                :, hand_jacobian_id, :, arm_joint_ids
            ].clone()
            world_to_base_rotation = matrix_from_quat(
                quat_inv(root_pose_w[:, 3:7])
            )
            jacobian[:, :3, :] = torch.bmm(
                world_to_base_rotation, jacobian[:, :3, :]
            )
            jacobian[:, 3:, :] = torch.bmm(
                world_to_base_rotation, jacobian[:, 3:, :]
            )
            current_arm_position = robot.data.joint_pos[:, arm_joint_ids]
            (
                desired_arm_position,
                desired_arm_velocity,
                ik_diagnostics,
            ) = ik.compute(
                hand_position_b,
                hand_quaternion_b,
                target_hand_position_b,
                target_hand_quaternion_b,
                jacobian,
                current_arm_position,
                posture_reference,
                arm_joint_lower,
                arm_joint_upper,
            )
            if not (
                torch.isfinite(desired_arm_position).all()
                and torch.isfinite(desired_arm_velocity).all()
            ):
                raise FloatingPointError(
                    f"IK produced a non-finite arm drive target in {state.name}."
                )

            joint_target = robot.data.joint_pos.clone()
            joint_target[:, arm_joint_ids] = desired_arm_position
            joint_target[:, finger_joint_ids] = gripper_target_position
            joint_velocity_target.zero_()
            joint_velocity_target[:, arm_joint_ids] = desired_arm_velocity
            robot.set_joint_position_target(joint_target)
            robot.set_joint_velocity_target(joint_velocity_target)
            robot.write_data_to_sim()

            sim.step(render=True)
            robot.update(args_cli.physics_dt)
            sponge.update(args_cli.physics_dt)
            sponge_pan_contact_sensor.update(args_cli.physics_dt, force_recompute=True)
            for camera in cameras.values():
                camera.update(args_cli.physics_dt)

            current_hand_pose_w = robot.data.body_pose_w[:, hand_body_id]
            current_tcp_pose_w = tcp_pose_from_hand(current_hand_pose_w, tcp_offset_b)
            current_root_pose_w = robot.data.root_pose_w
            current_tcp_position_b, current_tcp_quaternion_b = subtract_frame_transforms(
                current_root_pose_w[:, 0:3],
                current_root_pose_w[:, 3:7],
                current_tcp_pose_w[:, 0:3],
                current_tcp_pose_w[:, 3:7],
            )
            target_tcp_position_b, target_tcp_quaternion_b = subtract_frame_transforms(
                current_root_pose_w[:, 0:3],
                current_root_pose_w[:, 3:7],
                target_tcp_pose_w[:, 0:3],
                target_tcp_pose_w[:, 3:7],
            )
            goal_tcp_position_b, goal_tcp_quaternion_b = subtract_frame_transforms(
                current_root_pose_w[:, 0:3],
                current_root_pose_w[:, 3:7],
                goal_tcp_pose_w[:, 0:3],
                goal_tcp_pose_w[:, 3:7],
            )
            current_tcp_pose_b = torch.cat((current_tcp_position_b, current_tcp_quaternion_b), dim=-1)
            target_tcp_pose_b = torch.cat((target_tcp_position_b, target_tcp_quaternion_b), dim=-1)
            goal_tcp_pose_b = torch.cat((goal_tcp_position_b, goal_tcp_quaternion_b), dim=-1)
            position_error = float(
                torch.linalg.norm(goal_tcp_pose_w[:, 0:3] - current_tcp_pose_w[:, 0:3], dim=-1).item()
            )
            orientation_error = float(
                quat_error_magnitude(goal_tcp_pose_w[:, 3:7], current_tcp_pose_w[:, 3:7]).item()
            )

            arm_speed = float(torch.max(torch.abs(robot.data.joint_vel[:, arm_joint_ids])).item())
            arm_torque = float(torch.max(torch.abs(robot.data.applied_torque[:, arm_joint_ids])).item())
            gripper_position = float(torch.mean(robot.data.joint_pos[:, finger_joint_ids]).item())
            gripper_speed = float(torch.max(torch.abs(robot.data.joint_vel[:, finger_joint_ids])).item())
            sponge_pose_w_tensor = sponge.data.root_pose_w[0]
            sponge_pose_w = sponge_pose_w_tensor.detach().cpu().numpy()
            sponge_linear_speed = float(torch.linalg.vector_norm(sponge.data.root_vel_w[0, :3]).item())
            tcp_sponge_distance = float(
                torch.linalg.vector_norm(current_tcp_pose_w[0, :3] - sponge_pose_w_tensor[:3]).item()
            )
            force_matrix_w = sponge_pan_contact_sensor.data.force_matrix_w
            if force_matrix_w is None:
                raise RuntimeError("Sponge-to-pan filtered contact-force matrix is unavailable.")
            pan_contact_force = float(
                torch.linalg.vector_norm(force_matrix_w, dim=-1).max().item()
            )
            maximum_pan_contact_force = max(maximum_pan_contact_force, pan_contact_force)
            open_threshold = args_cli.open_position - args_cli.gripper_position_tolerance
            post_release = (
                state.name in ("OPEN", "RETREAT", "DONE")
                and gripper_position >= open_threshold
            )
            pan_contact = pan_contact_force >= args_cli.pan_contact_force_threshold
            if post_release and pan_contact:
                pan_contact_elapsed += args_cli.physics_dt
            else:
                pan_contact_elapsed = 0.0
            gripper_effort = float(torch.max(torch.abs(robot.data.applied_torque[:, finger_joint_ids])).item())
            gripper_goal_reached = True
            if state.name == "CLOSE":
                contact_threshold = expected_contact_position + args_cli.gripper_position_tolerance
                gripper_goal_reached = gripper_position <= contact_threshold
            elif state.name == "OPEN":
                gripper_goal_reached = gripper_position >= open_threshold
            trajectory_complete = trajectory_complete and gripper_goal_reached
            if state.name in ("LIFT", "ABOVE_PAN", "PLACE"):
                if not grasp_validated:
                    raise RuntimeError(f"Entered {state.name} without a validated sponge grasp.")
                if tcp_sponge_distance > args_cli.grasp_max_tcp_sponge_distance:
                    raise RuntimeError(
                        f"Grasp lost in {state.name}: TCP-to-sponge distance "
                        f"{tcp_sponge_distance:.4f} m exceeds "
                        f"{args_cli.grasp_max_tcp_sponge_distance:.4f} m."
                    )
            if state.mode == "motion" and trajectory_complete:
                best_post_trajectory_position_error = min(
                    best_post_trajectory_position_error,
                    position_error,
                )
                best_post_trajectory_orientation_error = min(
                    best_post_trajectory_orientation_error,
                    orientation_error,
                )
                position_error_growth = (
                    position_error - best_post_trajectory_position_error
                )
                orientation_error_growth = (
                    orientation_error - best_post_trajectory_orientation_error
                )
                if (
                    position_error_growth
                    > args_cli.divergence_position_growth
                    or orientation_error_growth
                    > args_cli.divergence_orientation_growth
                ):
                    divergence_elapsed += args_cli.physics_dt
                else:
                    divergence_elapsed = max(
                        0.0,
                        divergence_elapsed - args_cli.physics_dt,
                    )
            else:
                position_error_growth = 0.0
                orientation_error_growth = 0.0
                divergence_elapsed = 0.0

            ik_minimum_singular_value = float(
                ik_diagnostics["minimum_singular_value"].item()
            )
            ik_damping = float(ik_diagnostics["damping"].item())
            ik_linear_velocity = float(
                ik_diagnostics["linear_velocity_norm"].item()
            )
            ik_angular_velocity = float(
                ik_diagnostics["angular_velocity_norm"].item()
            )
            ik_task_velocity_norm = float(
                ik_diagnostics["task_velocity_norm"].item()
            )
            ik_nullspace_velocity_norm = float(
                ik_diagnostics["nullspace_velocity_norm"].item()
            )
            ik_velocity_scale = float(
                ik_diagnostics["velocity_scale"].item()
            )
            ik_command_velocity_norm = float(
                ik_diagnostics["command_velocity_norm"].item()
            )
            arm_following_error = float(
                ik_diagnostics["maximum_following_error"].item()
            )
            minimum_joint_limit_margin = float(
                ik_diagnostics["minimum_joint_limit_margin"].item()
            )
            sim_time = (step + 1) * args_cli.physics_dt
            status_wall_time = time.monotonic()
            motion_status = (
                f" motion={100.0 * motion_progress:5.1f}%"
                if trajectory_kind
                else ""
            )
            if state.name in ("CLOSE", "OPEN"):
                motion_status += f" drive_ready={int(gripper_goal_reached)}"
            motion_status += (
                f" tcp_sponge={tcp_sponge_distance:.4f}m grasp_ok={int(grasp_validated)}"
                f" pan_force={pan_contact_force:.3f}N pan_contact_t={pan_contact_elapsed:.2f}s"
            )
            if args_cli.status_interval > 0.0 and status_wall_time >= next_status_wall_time:
                log(
                    f"status t={sim_time:6.2f}s "
                    f"state={state_machine.index + 1}/{len(state_machine.states)}:{state.name} "
                    f"state_t={state_machine.elapsed:5.2f}s "
                    f"pos_err={position_error:7.4f}m "
                    f"rot_err={math.degrees(orientation_error):6.2f}deg "
                    f"grip_action={gripper_action:+.0f} "
                    f"grip_goal={gripper_goal_position:.3f}m "
                    f"grip_drive_target={gripper_target_position:.3f}m "
                    f"grip_actual={gripper_position:.3f}m "
                    f"grip_speed={gripper_speed:.3f}m/s "
                    f"arm_speed={arm_speed:.3f}rad/s "
                    f"arm_tau={arm_torque:.2f}Nm grip_effort={gripper_effort:.2f}N "
                    f"ik_sigma_min={ik_minimum_singular_value:.4f} "
                    f"ik_lambda={ik_damping:.4f} "
                    f"ik_twist={ik_linear_velocity:.3f}m/s,{ik_angular_velocity:.3f}rad/s "
                    f"ik_qdot={ik_task_velocity_norm:.3f}+{ik_nullspace_velocity_norm:.3f}rad/s "
                    f"cmd_qdot={ik_command_velocity_norm:.3f}rad/s "
                    f"follow_err={arm_following_error:.4f}rad "
                    f"joint_margin={minimum_joint_limit_margin:.3f} "
                    f"diverge_t={divergence_elapsed:.2f}s "
                    f"frames={recorder.frame_index}{motion_status}"
                )
                next_status_wall_time = status_wall_time + args_cli.status_interval

            if step % capture_stride == 0:
                sponge_pose_w = sponge.data.root_pose_w[0].detach().cpu().numpy()
                recorder.capture(
                    {name: camera_rgb(camera, name) for name, camera in cameras.items()},
                    {
                        "sim_time": sim_time,
                        "action_start_time": step * args_cli.physics_dt,
                        "state_index": state_machine.index,
                        "joint_position": robot.data.joint_pos[0].detach().cpu().numpy(),
                        "joint_velocity": robot.data.joint_vel[0].detach().cpu().numpy(),
                        "joint_target": joint_target[0].detach().cpu().numpy(),
                        "joint_velocity_target": joint_velocity_target[0].detach().cpu().numpy(),
                        "joint_applied_torque": robot.data.applied_torque[0].detach().cpu().numpy(),
                        "joint_computed_torque": robot.data.computed_torque[0].detach().cpu().numpy(),
                        "robot_root_pose_w": current_root_pose_w[0].detach().cpu().numpy(),
                        "tcp_pose_w": current_tcp_pose_w[0].detach().cpu().numpy(),
                        "target_tcp_pose_w": target_tcp_pose_w[0].detach().cpu().numpy(),
                        "goal_tcp_pose_w": goal_tcp_pose_w[0].detach().cpu().numpy(),
                        "tcp_pose_b": current_tcp_pose_b[0].detach().cpu().numpy(),
                        "target_tcp_pose_b": target_tcp_pose_b[0].detach().cpu().numpy(),
                        "goal_tcp_pose_b": goal_tcp_pose_b[0].detach().cpu().numpy(),
                        "vertical_motion_progress": motion_progress,
                        "motion_progress": motion_progress,
                        "is_transport_motion": transport_trajectory_active,
                        "gripper_action": np.float32(gripper_action),
                        "gripper_command_edge": gripper_command_edge,
                        "gripper_goal_position": gripper_goal_position,
                        "gripper_target_position": gripper_target_position,
                        "gripper_drive_velocity": gripper_drive_velocity,
                        "gripper_position": gripper_position,
                        "gripper_velocity": gripper_speed,
                        "gripper_effort": gripper_effort,
                        "gripper_goal_reached": gripper_goal_reached,
                        "sponge_pose_w": sponge_pose_w,
                        "sponge_linear_speed": sponge_linear_speed,
                        "tcp_sponge_distance": tcp_sponge_distance,
                        "grasp_validated": grasp_validated,
                        "pan_contact_force": pan_contact_force,
                        "pan_contact": pan_contact,
                        "pan_contact_elapsed": pan_contact_elapsed,
                        "position_error": position_error,
                        "orientation_error": orientation_error,
                        "ik_posture_reference": posture_reference[
                            0
                        ].detach().cpu().numpy(),
                        "ik_minimum_singular_value": ik_minimum_singular_value,
                        "ik_damping": ik_damping,
                        "ik_linear_velocity_norm": ik_linear_velocity,
                        "ik_angular_velocity_norm": ik_angular_velocity,
                        "ik_task_velocity_norm": ik_task_velocity_norm,
                        "ik_nullspace_velocity_norm": ik_nullspace_velocity_norm,
                        "ik_velocity_scale": ik_velocity_scale,
                        "ik_command_velocity_norm": ik_command_velocity_norm,
                        "arm_following_error": arm_following_error,
                        "minimum_joint_limit_margin": minimum_joint_limit_margin,
                        "divergence_elapsed": divergence_elapsed,
                        "position_error_growth": position_error_growth,
                        "orientation_error_growth": orientation_error_growth,
                    },
                )

            if divergence_elapsed >= args_cli.divergence_window:
                raise RuntimeError(
                    f"IK divergence detected in {state.name}: "
                    f"position error grew by {position_error_growth:.4f} m, "
                    f"orientation error grew by "
                    f"{math.degrees(orientation_error_growth):.2f} deg, "
                    f"sigma_min={ik_minimum_singular_value:.4f}, "
                    f"lambda={ik_damping:.4f}, "
                    f"following error={arm_following_error:.4f} rad, "
                    f"joint-limit margin={minimum_joint_limit_margin:.3f}."
                )

            rollout_done = state_machine.update(
                args_cli.physics_dt,
                position_error,
                orientation_error,
                trajectory_complete=trajectory_complete,
            )
            if (
                state.name == "CLOSE"
                and state_machine.current.name == "LIFT"
            ):
                grasp_failures: list[str] = []
                if gripper_position < minimum_grasp_contact_position:
                    grasp_failures.append(
                        f"finger position {gripper_position:.4f} m is below the contact minimum "
                        f"{minimum_grasp_contact_position:.4f} m"
                    )
                if gripper_effort < args_cli.grasp_min_effort:
                    grasp_failures.append(
                        f"finger effort {gripper_effort:.2f} N is below "
                        f"{args_cli.grasp_min_effort:.2f} N"
                    )
                if tcp_sponge_distance > args_cli.grasp_max_tcp_sponge_distance:
                    grasp_failures.append(
                        f"TCP-to-sponge distance {tcp_sponge_distance:.4f} m exceeds "
                        f"{args_cli.grasp_max_tcp_sponge_distance:.4f} m"
                    )
                if grasp_failures:
                    raise RuntimeError("Grasp validation failed: " + "; ".join(grasp_failures))
                grasp_validated = True
                log(
                    f"Grasp physically validated: finger={gripper_position:.4f} m, "
                    f"effort={gripper_effort:.2f} N, tcp_sponge={tcp_sponge_distance:.4f} m."
                )
            if rollout_done:
                if not grasp_validated:
                    raise RuntimeError("Rollout reached DONE without a physically validated grasp.")
                if pan_contact_elapsed < args_cli.pan_contact_settle_time:
                    raise RuntimeError(
                        f"Placement validation failed: sponge-to-pan contact held for "
                        f"{pan_contact_elapsed:.2f} s; required {args_cli.pan_contact_settle_time:.2f} s."
                    )
                if sponge_linear_speed > args_cli.placement_max_sponge_speed:
                    raise RuntimeError(
                        f"Placement validation failed: sponge speed {sponge_linear_speed:.4f} m/s "
                        f"exceeds {args_cli.placement_max_sponge_speed:.4f} m/s."
                    )
                log(
                    f"Physical placement validated: pan_force={pan_contact_force:.3f} N, "
                    f"continuous_contact={pan_contact_elapsed:.2f} s, "
                    f"sponge_speed={sponge_linear_speed:.4f} m/s."
                )
                success = True
                break
        else:
            failure = f"Rollout exceeded max-runtime={args_cli.max_runtime:.1f} seconds."

        if not success and failure is None:
            failure = "Rollout stopped before reaching DONE."
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["physical_validation"]["observed"] = {
            "grasp_validated": grasp_validated,
            "final_tcp_sponge_distance_m": tcp_sponge_distance,
            "maximum_pan_contact_force_n": maximum_pan_contact_force,
            "final_pan_contact_force_n": pan_contact_force,
            "final_continuous_pan_contact_time_s": pan_contact_elapsed,
            "final_sponge_linear_speed_m_s": sponge_linear_speed,
            "passed": success,
        }
        recorder.close(success=success, error=failure)

    if not success:
        raise RuntimeError(failure)
    log(
        f"Rollout completed successfully with {recorder.frame_index} synchronized RGB frames "
        f"from each of {len(cameras)} cameras."
    )
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception as exc:
        print(f"[data-gen] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        # SimulationApp.close() can wait indefinitely for Replicator after
        # multi-camera Camera sensors have already finished. This is a one-shot
        # headless process; main() closes videos and trajectory files first, so
        # an immediate process exit is deterministic and the OS releases Kit,
        # CUDA, and renderer resources.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
