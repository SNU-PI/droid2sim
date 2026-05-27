#!/usr/bin/env python3
"""Standalone metadata + full trajectory dump for all six PolaRiS environments.

Run via uv from the polaris/ working tree (so PolaRiS-Hub/ resolves):

    cd polaris
    unset DISPLAY
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \\
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \\
    __GLX_VENDOR_LIBRARY_NAME=nvidia \\
    CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES \\
    uv run python ../scripts/dump_metadata.py

Requires a policy server already listening on port 8001 (see README).
Loops through the six DROID-* envs sequentially; for each env runs one
rollout of pi05_droid_jointpos_polaris and writes:
    metadata/<env>/scene.json
    metadata/<env>/initial_conditions.json
    metadata/<env>/eval_summary.json   (if results/<env>/eval_results.csv exists)
    metadata/<env>/trajectory_ep0.json
"""

import argparse
import csv
import json
import re
from pathlib import Path

# Boot Isaac Sim BEFORE importing isaaclab modules
from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser()
_args, _ = _parser.parse_known_args()
_args.enable_cameras = True
_args.headless = True
_app_launcher = AppLauncher(_args)
_simulation_app = _app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tqdm  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import polaris.environments  # noqa: E402,F401  registers gym envs
from polaris.config import PolicyArgs  # noqa: E402
from polaris.policy import InferenceClient  # noqa: E402
from polaris.utils import load_eval_initial_conditions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HUB_CANDIDATES = [ROOT / "polaris" / "PolaRiS-Hub", ROOT / "data" / "PolaRiS-Hub"]
HUB = next((p for p in HUB_CANDIDATES if p.exists()), HUB_CANDIDATES[0])
OUT_DIR = ROOT / "metadata"
RESULTS = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

ENVS = [
    ("DROID-FoodBussing", "food_bussing"),
    ("DROID-BlockStackKitchen", "block_stack_kitchen"),
    ("DROID-PanClean", "pan_clean"),
    ("DROID-MoveLatteCup", "move_latte_cup"),
    ("DROID-OrganizeTools", "organize_tools"),
    ("DROID-TapeIntoContainer", "tape_into_container"),
]

CAM_INTRINSICS = {
    "wrist_cam": {
        "height": 1440, "width": 2560,
        "focal_length": 2.8, "focus_distance": 28.0,
        "horizontal_aperture": 5.376, "vertical_aperture": 3.024,
        "offset_pos": [0.011, -0.031, -0.074],
        "offset_rot_quat_wxyz": [-0.420, 0.570, 0.576, -0.409],
        "convention": "opengl",
        "attached_to": "robot/Gripper/Robotiq_2F_85/base_link",
    },
    "external_cam_default": {
        "height": 1440, "width": 2560,
        "focal_length": 1.0476,
        "horizontal_aperture": 2.5452, "vertical_aperture": 1.4721,
        "note": "Used when scene.usda does not define an external camera",
    },
}

ARTICULATION_FIELDS = (
    "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w",
    "joint_pos", "joint_vel", "joint_acc",
    "applied_torque", "computed_torque",
    "joint_pos_target", "joint_vel_target", "joint_effort_target",
    "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
)
RIGID_FIELDS = (
    "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w",
)
SENSOR_FIELDS = (
    "target_pos_source", "target_quat_source",
    "target_pos_w", "target_quat_w",
    "source_pos_w", "source_quat_w",
)


def to_list(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x).tolist()


def safe_field(d, attr, env_idx=0):
    if not hasattr(d, attr):
        return None
    try:
        v = getattr(d, attr)
        if v is None:
            return None
        try:
            v = v[env_idx]
        except (IndexError, TypeError):
            pass
        return to_list(v)
    except Exception:
        return None


def parse_scene_usda(p):
    text = p.read_text(errors="ignore")
    prims = []
    for m in re.finditer(r'^    def (?:(\w+) )?"([^"]+)"', text, re.MULTILINE):
        kind = m.group(1) or "Xform"
        name = m.group(2)
        if name in ("OmniverseKit", "Vars"):
            continue
        prims.append({"name": name, "type": kind})
    assets_re = re.compile(r"prepend payload = @\./assets/([^/@]+)/([^@]+)@")
    assets = [{"folder": am.group(1), "file": am.group(2)} for am in assets_re.finditer(text)]
    return prims, assets


def parse_ply_header(p):
    with p.open("rb") as f:
        if f.readline().strip() != b"ply":
            return None
        fmt = f.readline().strip().decode()
        vc = None
        pc = 0
        for _ in range(500):
            line = f.readline().strip().decode()
            if line == "end_header":
                break
            if line.startswith("element vertex"):
                vc = int(line.split()[2])
            elif line.startswith("property"):
                pc += 1
        return {"format": fmt, "vertex_count": vc, "property_count": pc}


def static_scene(env_short):
    env_hub = HUB / env_short
    scene = {
        "env": env_short,
        "scene_usda": str((env_hub / "scene.usda").relative_to(ROOT)) if (env_hub / "scene.usda").exists() else None,
        "world_prims": [],
        "asset_payloads": [],
        "splats": [],
        "cameras": CAM_INTRINSICS,
    }
    ic = None
    ic_path = env_hub / "initial_conditions.json"
    if ic_path.exists():
        ic = json.loads(ic_path.read_text())
        scene["instruction"] = ic.get("instruction")
        scene["num_initial_conditions"] = len(ic.get("poses", []))
    usda = env_hub / "scene.usda"
    if usda.exists():
        prims, assets = parse_scene_usda(usda)
        scene["world_prims"] = prims
        scene["asset_payloads"] = assets
        for a in assets:
            ply = env_hub / "assets" / a["folder"] / "splat.ply"
            if ply.exists():
                info = parse_ply_header(ply)
                if info:
                    scene["splats"].append({"folder": a["folder"], **info})
    return scene, ic


def aggregate_csv(env_short):
    p = RESULTS / env_short / "eval_results.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open()))
    if not rows:
        return {"rollouts": 0}
    progress = [float(r["progress"]) for r in rows]
    succ = sum(1 for r in rows if r["success"] == "True")
    return {
        "rollouts": len(rows),
        "success_count": succ,
        "success_rate": succ / len(rows),
        "avg_progress": sum(progress) / len(progress),
        "min_progress": min(progress),
        "max_progress": max(progress),
        "per_episode": [
            {"episode": int(r["episode"]),
             "episode_length": int(r["episode_length"]),
             "success": r["success"] == "True",
             "progress": float(r["progress"])}
            for r in rows
        ],
    }


def record_step(env, obs, action, rew, info, step, rerender):
    rec = {"step": int(step), "rerender": bool(rerender)}
    try:
        rec["action"] = to_list(action)
    except Exception:
        pass
    try:
        rec["reward"] = float(np.asarray(rew).reshape(-1)[0])
    except Exception:
        pass
    try:
        for k, v in obs.get("policy", {}).items():
            val = to_list(v[0] if hasattr(v, "shape") and len(v.shape) >= 1 else v)
            if val is not None:
                rec[f"obs.{k}"] = val
    except Exception:
        pass
    rigid = {}
    try:
        for name, asset in getattr(env.scene, "rigid_objects", {}).items():
            entry = {}
            for attr in RIGID_FIELDS:
                val = safe_field(asset.data, attr)
                if val is not None:
                    entry[attr] = val
            rigid[name] = entry
    except Exception:
        pass
    art = {}
    try:
        for name, asset in getattr(env.scene, "articulations", {}).items():
            entry = {}
            for attr in ARTICULATION_FIELDS:
                val = safe_field(asset.data, attr)
                if val is not None:
                    entry[attr] = val
            try:
                entry["joint_names"] = list(asset.joint_names)
            except Exception:
                pass
            try:
                entry["body_names"] = list(asset.body_names)
            except Exception:
                pass
            art[name] = entry
    except Exception:
        pass
    sensors = {}
    try:
        for name, asset in getattr(env.scene, "sensors", {}).items():
            entry = {}
            for attr in SENSOR_FIELDS:
                val = safe_field(asset.data, attr) if hasattr(asset, "data") else None
                if val is not None:
                    entry[attr] = val
            if entry:
                sensors[name] = entry
    except Exception:
        pass
    rec["rigid_objects"] = rigid
    rec["articulations"] = art
    rec["sensors"] = sensors
    if isinstance(info, dict) and "rubric" in info:
        try:
            r = info["rubric"]
            rec["rubric"] = {
                "success": bool(r.get("success", False)),
                "progress": float(r.get("progress", 0)),
            }
        except Exception:
            pass
    return rec


def run_one(env_name, env_short, policy_args):
    print(f"\n=== {env_name} ({env_short}) ===", flush=True)
    env_cfg = parse_env_cfg(env_name, device="cuda", num_envs=1, use_fabric=True)
    env = gym.make(env_name, cfg=env_cfg)

    instruction, initial_conditions = load_eval_initial_conditions(
        usd=env.usd_file, initial_conditions_file=None, rollouts=1
    )
    obs, info = env.reset(object_positions=initial_conditions[0])
    policy_client = InferenceClient.get_client(policy_args)
    policy_client.reset()

    horizon = env.max_episode_length
    trajectory = []
    bar = tqdm.tqdm(range(horizon), desc=env_short)
    for step in range(horizon):
        action, _ = policy_client.infer(obs, instruction)
        obs, rew, term, trunc, info = env.step(
            torch.tensor(action).reshape(1, -1), expensive=policy_client.rerender
        )
        try:
            trajectory.append(record_step(env, obs, action, rew, info, step, policy_client.rerender))
        except Exception as e:
            print(f"[traj] step {step} error: {e}")
        bar.update(1)
        if term[0] or trunc[0]:
            break
    bar.close()

    rubric = info.get("rubric", {}) if isinstance(info, dict) else {}
    out = {
        "env": env_short,
        "episode": 0,
        "episode_length": len(trajectory),
        "success": bool(rubric.get("success", False)) if rubric else None,
        "progress": float(rubric.get("progress", 0)) if rubric else None,
        "instruction": instruction,
        "trajectory": trajectory,
    }
    env_dir = OUT_DIR / env_short
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "trajectory_ep0.json").write_text(json.dumps(out))
    print(f"  wrote trajectory_ep0.json ({len(trajectory)} steps)")

    scene, ic = static_scene(env_short)
    (env_dir / "scene.json").write_text(json.dumps(scene, indent=2))
    if ic is not None:
        (env_dir / "initial_conditions.json").write_text(json.dumps(ic, indent=2))
    summary = aggregate_csv(env_short)
    if summary is not None:
        (env_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    env.close()


def main():
    import sys
    if len(sys.argv) < 2:
        print("usage: dump_metadata.py <env_short>   (e.g. food_bussing)")
        print("       dump_metadata.py <DROID-EnvName>")
        sys.exit(2)
    target = sys.argv[1]
    match = next(((n, s) for n, s in ENVS if s == target or n == target), None)
    if match is None:
        print(f"unknown env: {target}; valid: {[s for _, s in ENVS]}")
        sys.exit(2)
    env_name, env_short = match
    policy_args = PolicyArgs(
        client="DroidJointPos", host="0.0.0.0", port=8001, open_loop_horizon=8
    )
    try:
        run_one(env_name, env_short, policy_args)
    except Exception as e:
        print(f"!!! {env_short} failed: {e}")
        import traceback
        traceback.print_exc()
        _simulation_app.close()
        sys.exit(1)
    _simulation_app.close()


if __name__ == "__main__":
    main()
