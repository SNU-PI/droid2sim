#!/usr/bin/env python3
"""Replay a DROID episode's joint trajectory inside arhanjain/sim-evals DROID env.

Usage (from sim-evals dir so DATA_PATH resolves):
    cd repos/sim-evals
    unset DISPLAY
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \\
    __GLX_VENDOR_LIBRARY_NAME=nvidia \\
    CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \\
    uv run python ../../scripts/replay_droid_simeval.py \\
        --traj /home/parkprogrammer/jehyun/droid2sim/droid_sample/notes/ep36048_pan_sponge.json \\
        --start 600 --end 1000 --scene 1 \\
        --out /home/parkprogrammer/jehyun/droid2sim/runs/replay/ep36048_simeval_scene1.mp4
"""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--traj", type=Path, required=True)
ap.add_argument("--scene", type=int, default=1)
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

import sim_evals.environments  # noqa: F401,E402  registers gym envs
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

traj = json.loads(args.traj.read_text())
T = traj["trajectory"]
start = max(0, args.start)
end = min(len(T), args.end if args.end is not None else len(T))
print(f"replay scene {args.scene}  ep {traj.get('episode_index')}  "
      f"steps [{start}..{end})  ({(end-start)/15:.1f}s)", flush=True)

env_cfg = parse_env_cfg("DROID", device="cuda", num_envs=1, use_fabric=True)
env_cfg.set_scene(args.scene)
env = gym.make("DROID", cfg=env_cfg)

obs, _ = env.reset()
obs, _ = env.reset()  # second cycle so materials load
print("env reset OK", flush=True)

# Seed robot to start joint pose so initial command isn't a giant jump.
try:
    robot = env.scene["robot"]
    n = robot.num_joints
    init_joint = np.zeros(n, dtype=np.float32)
    init_joint[:7] = T[start]["joint_pos"]
    init_joint[7:] = float(T[start]["gripper_pos"])
    init_t = torch.tensor(init_joint).unsqueeze(0).to(env.unwrapped.device)
    robot.write_joint_state_to_sim(
        init_t,
        torch.zeros_like(init_t),
        env_ids=torch.zeros(1, dtype=torch.long, device=env.unwrapped.device),
    )
    print(f"seeded joints: arm={init_joint[:7].round(3).tolist()} grip={init_joint[7]:.3f}", flush=True)
except Exception as e:
    print(f"[warn] could not seed robot: {e}", flush=True)

video = []
bar = tqdm.tqdm(range(start, end), desc="replay")
for i in bar:
    s = T[i]
    arm = list(map(float, s["action_joint"]))
    grip = 1.0 if float(s["action_gripper"]) > 0.5 else 0.0
    action = torch.tensor(arm + [grip]).reshape(1, -1).to(env.unwrapped.device)
    obs, _, term, trunc, _ = env.step(action)

    try:
        ext = obs["policy"]["external_cam"][0].detach().cpu().numpy()
        wri = obs["policy"]["wrist_cam"][0].detach().cpu().numpy()

        def _u8(x):
            x = np.asarray(x)
            if x.ndim == 4:
                x = x[0]
            if x.dtype != np.uint8:
                x = (x * 255).clip(0, 255).astype(np.uint8) if x.max() <= 1.0 else x.astype(np.uint8)
            return x

        video.append(np.concatenate([_u8(ext), _u8(wri)], axis=1))
    except Exception as e:
        if i == start:
            print(f"[warn] no camera obs: {e}", flush=True)

args.out.parent.mkdir(parents=True, exist_ok=True)
mediapy.write_video(str(args.out), video, fps=15)
print(f"wrote {args.out}  ({len(video)} frames)", flush=True)

env.close()
_sim.close()
