#!/usr/bin/env python3
"""Render presentation-quality figures from metadata/<env>/trajectory_ep0.json.

Panels (per environment):
  1 Panda arm joint angles vs time
  2 End-effector 3D path with time gradient
  3 Applied joint torque (arm + gripper)
  5 Action vs joint position tracking (arm)

Plus two cross-env summary bar charts.
Output: runs/figures/<env>/panel_*.png and runs/figures/summary/S*_.png with
an HTML index at runs/figures/index.html.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "metadata"
OUT = ROOT / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

ENVS = [
    ("food_bussing",        "FoodBussing"),
    ("block_stack_kitchen", "BlockStackKitchen"),
    ("pan_clean",           "PanClean"),
    ("move_latte_cup",      "MoveLatteCup"),
    ("organize_tools",      "OrganizeTools"),
    ("tape_into_container", "TapeIntoContainer"),
]
CONTROL_HZ = 15
DPI = 160

BG = "#1b1f27"
FG = "#e6e9ef"
MUTED = "#9aa3b2"
GRID = "#3a4150"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": FG,
    "axes.titlecolor": FG,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.labelsize": 11,
    "text.color": FG,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "legend.labelcolor": FG,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "savefig.bbox": "tight",
    "savefig.dpi": DPI,
})

ARM_COLORS = plt.cm.viridis(np.linspace(0.05, 0.85, 7))
GRIP_COLORS = plt.cm.plasma(np.linspace(0.15, 0.85, 6))


def load_traj(env):
    return json.loads((META / env / "trajectory_ep0.json").read_text())


def ee_pos_array(t):
    pts = []
    for s in t["trajectory"]:
        sens = s.get("sensors", {}).get("ee_frame") or {}
        p = sens.get("target_pos_w")
        if p is None:
            b = np.array(s["articulations"]["robot"]["body_pos_w"])
            arr = b[-3]
        else:
            arr = np.array(p)
            if arr.ndim == 2:
                arr = arr[0]
        pts.append(arr)
    return np.array(pts)


def time_axis(t):
    steps = np.array([s["step"] for s in t["trajectory"]])
    secs = steps / CONTROL_HZ
    return steps, secs


def add_time_axis(ax, steps, secs):
    ax_top = ax.secondary_xaxis(
        "top",
        functions=(lambda x: x / CONTROL_HZ, lambda x: x * CONTROL_HZ),
    )
    ax_top.set_xlabel("time (s)", color=MUTED)
    ax_top.tick_params(colors=MUTED)


def panel1_joint_angles(t, env_label, out_path):
    steps, secs = time_axis(t)
    art = [s["articulations"]["robot"] for s in t["trajectory"]]
    jp = np.array([a["joint_pos"] for a in art])
    names = art[0]["joint_names"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i in range(7):
        ax.plot(steps, jp[:, i], color=ARM_COLORS[i], lw=1.6,
                label=f"j{i+1}  {names[i]}")
    ax.set_xlabel("step")
    ax.set_ylabel("joint position (rad)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    add_time_axis(ax, steps, secs)
    fig.suptitle(f"{env_label}  —  Panda arm joint angles", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def panel2_ee_path(t, env_label, out_path):
    pts = ee_pos_array(t)
    steps, _ = time_axis(t)
    fig = plt.figure(figsize=(7.2, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    for pane_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane_axis.set_pane_color((0.11, 0.13, 0.16, 1.0))
        pane_axis._axinfo["grid"]["color"] = GRID

    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(steps.min(), steps.max())
    lc = Line3DCollection(segs, cmap="viridis", norm=norm, lw=2.0)
    lc.set_array(steps[:-1])
    ax.add_collection(lc)

    pad = 0.05
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(FG)
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_zlim(pts[:, 2].min() - pad, pts[:, 2].max() + pad)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    ax.scatter(*pts[0],  c="#2ca02c", s=80, label="start", edgecolors="white")
    ax.scatter(*pts[-1], c="#d62728", s=80, label="end",   edgecolors="white")

    cb = fig.colorbar(lc, ax=ax, shrink=0.65, pad=0.05)
    cb.set_label("step", color=FG)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=MUTED)

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.view_init(elev=22, azim=-58)
    fig.suptitle(f"{env_label}  —  End-effector 3D path", fontsize=15, fontweight="bold", y=0.96)
    fig.savefig(out_path)
    plt.close(fig)


def panel3_joint_torque(t, env_label, out_path):
    steps, secs = time_axis(t)
    art = [s["articulations"]["robot"] for s in t["trajectory"]]
    tq = np.array([a["applied_torque"] for a in art])
    names = art[0]["joint_names"]

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    for i in range(7):
        axes[0].plot(steps, tq[:, i], color=ARM_COLORS[i], lw=1.4,
                     label=f"j{i+1}  {names[i]}")
    axes[0].set_ylabel("applied torque (Nm)")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0,
                   title="arm")

    n_grip = tq.shape[1] - 7
    for k in range(n_grip):
        i = 7 + k
        axes[1].plot(steps, tq[:, i], color=GRIP_COLORS[k % len(GRIP_COLORS)],
                     lw=1.2, label=names[i])
    axes[1].set_ylabel("applied torque (Nm)")
    axes[1].set_xlabel("step")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0,
                   title="gripper", fontsize=8)
    add_time_axis(axes[0], steps, secs)
    fig.suptitle(f"{env_label}  —  Applied joint torque", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def panel5_action_vs_joint(t, env_label, out_path):
    steps, secs = time_axis(t)
    actions = np.array([s["action"] for s in t["trajectory"]])
    joints = np.array([s["articulations"]["robot"]["joint_pos"][:7] for s in t["trajectory"]])
    names = t["trajectory"][0]["articulations"]["robot"]["joint_names"]

    fig, axes = plt.subplots(7, 1, figsize=(9.5, 11), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    for i in range(7):
        ax = axes[i]
        ax.fill_between(steps, actions[:, i], joints[:, i],
                        color=ARM_COLORS[i], alpha=0.18, linewidth=0,
                        label="tracking gap" if i == 0 else None)
        ax.plot(steps, actions[:, i], color=ARM_COLORS[i], lw=1.0,
                ls="--", label="action target" if i == 0 else None)
        ax.plot(steps, joints[:, i], color=ARM_COLORS[i], lw=1.6,
                label="joint position" if i == 0 else None)
        ax.set_ylabel(f"j{i+1}", rotation=0, labelpad=18, va="center")
        ax.tick_params(left=True, labelleft=True)
        ax.text(0.99, 0.92, names[i], transform=ax.transAxes,
                ha="right", va="top", color=MUTED, fontsize=9)
    axes[-1].set_xlabel("step")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.05), ncol=3,
                   frameon=False)
    add_time_axis(axes[0], steps, secs)
    fig.suptitle(f"{env_label}  —  Action target vs measured joint position",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def summary_panels():
    summary_dir = OUT / "summary"
    summary_dir.mkdir(exist_ok=True)
    data = []
    for env, label in ENVS:
        s = json.loads((META / env / "eval_summary.json").read_text())
        data.append((label, s["success_rate"], s["avg_progress"]))
    labels = [d[0] for d in data]
    sr = [d[1] for d in data]
    ap = [d[2] for d in data]

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    bars = ax.bar(labels, sr, color="#3b82f6", edgecolor=BG)
    for bar, val in zip(bars, sr):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.0%}",
                ha="center", va="bottom", color=FG, fontsize=10)
    ax.set_ylabel("success rate")
    fig.suptitle("Success rate per environment (5 rollouts)", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks(rotation=15, ha="right")
    fig.savefig(summary_dir / "S1_success_rate.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    bars = ax.bar(labels, ap, color="#f59e0b", edgecolor=BG)
    for bar, val in zip(bars, ap):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}",
                ha="center", va="bottom", color=FG, fontsize=10)
    ax.set_ylabel("avg rubric progress")
    fig.suptitle("Average rubric progress per environment (5 rollouts)", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15, ha="right")
    fig.savefig(summary_dir / "S2_avg_progress.png")
    plt.close(fig)


PANEL_FUNCS = [
    ("panel_1_joint_angles.png",    panel1_joint_angles),
    ("panel_2_ee_path.png",         panel2_ee_path),
    ("panel_3_joint_torque.png",    panel3_joint_torque),
    ("panel_5_action_vs_joint.png", panel5_action_vs_joint),
]


def composite_grid():
    import matplotlib.image as mpimg
    panel_titles = ["Joint angles", "EE 3D path", "Joint torque", "Action vs joint"]
    n_rows, n_cols = len(ENVS), len(PANEL_FUNCS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows))
    for r, (env, label) in enumerate(ENVS):
        for c, (fname, _) in enumerate(PANEL_FUNCS):
            ax = axes[r, c]
            p = OUT / env / fname
            if p.exists():
                ax.imshow(mpimg.imread(p))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(panel_titles[c], fontsize=14, fontweight="bold", pad=8)
        axes[r, 0].set_ylabel(label, fontsize=13, fontweight="bold",
                              rotation=0, ha="right", va="center", labelpad=12)
    fig.suptitle("pi0.5 rollouts — per-environment trajectory panels",
                 fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.02, 0, 1, 0.985])
    fig.savefig(OUT / "composite_6x4.png", dpi=140)
    plt.close(fig)


def build_index_html():
    html = ["<!doctype html><html><head><meta charset='utf-8'>",
            "<title>Trajectory figures</title>",
            "<style>",
            "body{background:#0f1115;color:#ddd;font-family:-apple-system,Segoe UI,sans-serif;padding:12px 8px;margin:0}",
            "h1{font-weight:500;margin:4px 0 18px;padding-left:6px}",
            "h2{margin-top:36px;padding-left:10px;border-left:4px solid #4af}",
            ".grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}",
            ".env2x2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-bottom:28px}",
            ".env2x2 .cell{background:#1b1f27;padding:0;border-radius:6px;overflow:hidden}",
            ".env2x2 .cell img{width:100%;height:520px;object-fit:contain;display:block;background:#1b1f27}",
            ".cell{background:#1b1f27;padding:10px;border-radius:8px}",
            ".cell img{width:100%;display:block;border-radius:4px;background:#fff}",
            ".cap{font-size:12px;color:#aaa;margin-top:6px}",
            "</style></head><body>",
            "<h1>Trajectory figures (episode 0 of each environment)</h1>"]
    for env, label in ENVS:
        html.append(f"<h2>{label} <span style='font-size:13px;color:#888'>({env})</span></h2>")
        html.append("<div class='env2x2'>")
        for fname, _ in PANEL_FUNCS:
            html.append(f"<div class='cell'><img src='{env}/{fname}'></div>")
        html.append("</div>")
    html.append("<h2>Summary</h2><div class='grid'>")
    html.append("<div class='cell'><img src='summary/S1_success_rate.png'><div class='cap'>S1 success rate</div></div>")
    html.append("<div class='cell'><img src='summary/S2_avg_progress.png'><div class='cap'>S2 avg progress</div></div>")
    html.append("</div>")
    html.append("</body></html>")
    (OUT / "index.html").write_text("\n".join(html))


def main():
    for env, label in ENVS:
        env_out = OUT / env
        env_out.mkdir(parents=True, exist_ok=True)
        try:
            t = load_traj(env)
        except FileNotFoundError:
            print(f"[skip] no trajectory for {env}")
            continue
        for fname, fn in PANEL_FUNCS:
            try:
                fn(t, label, env_out / fname)
            except Exception as e:
                print(f"  {env}/{fname} failed: {e}")
        # remove dropped panels (4, 6) if a previous run left them behind
        for stale in ("panel_4_object_z.png", "panel_6_progress_gripper.png"):
            p = env_out / stale
            if p.exists():
                p.unlink()
        print(f"  {env}: {len(PANEL_FUNCS)} panels")
    summary_panels()
    print("\nwrote", OUT / "index.html")


if __name__ == "__main__":
    main()
