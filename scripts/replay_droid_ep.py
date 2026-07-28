#!/usr/bin/env python3
"""Replay a DROID episode's joint-position trajectory in the PolaRiS sim.

Usage:
    cd polaris        # so PolaRiS-Hub resolves
    unset DISPLAY
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \\
    __GLX_VENDOR_LIBRARY_NAME=nvidia \\
    CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EILA=YES \\
    uv run python ../scripts/replay_droid_ep.py \\
        --env DROID-PanClean \\
        --traj ../droid_sample/notes/ep36048_pan_sponge.json \\
        --start 600 --end 1000 \\
        --out ../runs/replay/ep36048_scrub.mp4
"""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--env", default="DROID-PanClean")
ap.add_argument("--traj", required=True, type=Path)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=None)
ap.add_argument("--out", type=Path, required=True)
args, _ = ap.parse_known_args()

# Launch Sim
_p = argparse.ArgumentParser()
_a, _ = _p.parse_known_args()
_a.enable_cameras = True
_a.headless = True
_app = AppLauncher(_a)
_sim = _app.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tqdm  # noqa: E402
import mediapy  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import polaris.environments  # noqa: F401,E402
from polaris.utils import load_eval_initial_conditions  # noqa: E402

traj = json.loads(args.traj.read_text())
T = traj["trajectory"]
start = max(0, args.start)
end = min(len(T), args.end if args.end is not None else len(T))
print(f"replay {args.env}  ep {traj.get('episode_index')}  steps [{start}..{end}) "
      f"({(end-start)/15:.1f}s)", flush=True)

env_cfg = parse_env_cfg(args.env, device="cuda", num_envs=1, use_fabric=True)
env = gym.make(args.env, cfg=env_cfg)

instruction, initial_conditions = load_eval_initial_conditions(
    usd=env.usd_file, initial_conditions_file=None, rollouts=1
)
ic = dict(initial_conditions[0])
ic["sponge"] = [0.635, -0.069, 0.150, 1.0, 0.0, 0.0, 0.0]
try:
    obs, info = env.reset(object_positions=ic)
except (KeyError, RuntimeError) as e:
    print(f"[warn] scene has no {e}; retrying reset without ICs", flush=True)
    obs, info = env.reset(object_positions={})
try:
    pan_pose = torch.tensor([[0.653, -0.085, 0.060,
                              0.196, 0.0, 0.0, -0.981]])
    env.scene["pan"].write_root_pose_to_sim(pan_pose)
    env.sim.render(); env.scene.update(0)
    print("  aligned pan via scene.write_root_pose_to_sim", flush=True)
except Exception as e:
    print(f"[warn] could not move pan: {e}", flush=True)
print(f"env reset OK, instruction={instruction!r}", flush=True)

# Seed robot to start-of-window joint pose so the first commanded step is reachable.
try:
    robot = env.scene["robot"]
    init_joint = np.zeros(robot.num_joints, dtype=np.float32)
    init_joint[:7] = T[start]["joint_pos"]      # 7 arm joints
    init_joint[7:] = float(T[start]["gripper_pos"])  # broadcast gripper to 6 finger joints
    init_t = torch.tensor(init_joint).unsqueeze(0).to(env.device)
    robot.write_joint_state_to_sim(
        init_t,
        torch.zeros_like(init_t),
        env_ids=torch.zeros(1, dtype=torch.long, device=env.device),
    )
    print(f"seeded robot joints: arm={init_joint[:7].round(3).tolist()} grip={init_joint[7]:.3f}", flush=True)
except Exception as e:
    print(f"[warn] could not seed robot pose: {e}", flush=True)

video = []
bar = tqdm.tqdm(range(start, end), desc="replay")
for i in bar:
    s = T[i]
    arm = list(map(float, s["action_joint"]))
    grip = 1.0 if float(s["action_gripper"]) > 0.5 else 0.0
    action = torch.tensor(arm + [grip]).reshape(1, -1)
    obs, rew, term, trunc, info = env.step(action, expensive=True)
    # capture wrist+exterior frame (concatenated like eval.py)
    try:
        ext = obs["splat"]["external_cam"]
        wri = obs["splat"]["wrist_cam"]
        for v in (ext, wri):
            pass
        def _to_uint8(x):
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            x = np.asarray(x)
            if x.ndim == 4:
                x = x[0]
            if x.dtype != np.uint8:
                x = (x * 255).clip(0, 255).astype(np.uint8) if x.max() <= 1.0 else x.astype(np.uint8)
            return x
        video.append(np.concatenate([_to_uint8(ext), _to_uint8(wri)], axis=1))
    except Exception as e:
        if i == start:
            print(f"[warn] no splat obs: {e}", flush=True)

args.out.parent.mkdir(parents=True, exist_ok=True)
mediapy.write_video(str(args.out), video, fps=15)
print(f"wrote {args.out}  ({len(video)} frames)", flush=True)

env.close()
_sim.close()
