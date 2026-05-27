#!/usr/bin/env python3
"""Build the static comparison + figures page (index.html)."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"
FIGURES = ASSETS / "figures"

ENVS = [
    ("food_bussing",        "fruitbus.mp4",  "FoodBussing",        "Put all the foods in the bowl"),
    ("block_stack_kitchen", "stack.mp4",     "BlockStackKitchen",  "Place and stack the blocks on top of the green tray"),
    ("pan_clean",           "panclean.mp4",  "PanClean",           "Use the yellow sponge to scrub the blue-handle frying pan"),
    ("move_latte_cup",      "lattecup.mp4",  "MoveLatteCup",       "Put the latte art cup on top of the cutting board"),
    ("organize_tools",      "scissor.mp4",   "OrganizeTools",      "Put the scissor into the large container"),
    ("tape_into_container", "tape.mp4",      "TapeIntoContainer",  "Put the tape into the container"),
]
EP = 0

FIGURE_PANELS = [
    "panel_1_joint_angles.png",
    "panel_2_ee_path.png",
    "panel_3_joint_torque.png",
    "panel_5_action_vs_joint.png",
]


def read_progress(env, ep):
    path = RESULTS / env / "eval_results.csv"
    for r in csv.DictReader(path.open()):
        if int(r["episode"]) == ep:
            return r["success"] == "True", float(r["progress"])
    return False, 0.0


def badge_for(success, progress):
    if success:
        return f'<span class="badge ok">&#10003; {progress:.2f}</span>'
    if progress >= 0.5:
        return f'<span class="badge meh">{progress:.2f}</span>'
    return f'<span class="badge bad">{progress:.2f}</span>'


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PolaRiS Grid - real vs sim</title>
<style>
  body {{ background:#111; color:#ddd; font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; padding:24px; }}
  h1 {{ font-weight:500; margin:0 0 16px 0; }}
  h2 {{ margin-top:36px; padding-left:10px; border-left:4px solid #4af; }}
  .pair {{ display:grid; grid-template-columns: 1fr 1fr; gap:8px; }}
  .cell {{ display:flex; flex-direction:column; align-items:center; }}
  .label {{ font-size:14px; color:#bbb; margin-bottom:6px; text-align:center; }}
  .inst {{ font-weight:400; color:#888; font-size:11px; }}
  video {{ width:400px; height:200px; object-fit:contain; background:#000; display:block; }}
  .cap {{ color:#888; font-size:12px; margin-top:2px; text-align:center; }}
  .badge {{ display:inline-block; font-size:11px; padding:1px 6px; border-radius:6px; margin-left:4px; vertical-align:middle; }}
  .ok {{ background:#1a5; color:#fff; }}
  .meh {{ background:#a73; color:#fff; }}
  .bad {{ background:#444; color:#aaa; }}
  .footer {{ margin-top:24px; color:#666; font-size:12px; }}
  .grid2 {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:14px; }}
  .env2x2 {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:8px; margin-bottom:28px; }}
  .env2x2 .fcell {{ background:#1b1f27; padding:0; border-radius:6px; overflow:hidden; }}
  .env2x2 .fcell img {{ width:100%; height:520px; object-fit:contain; display:block; background:#1b1f27; }}
  .fcell {{ background:#1b1f27; padding:10px; border-radius:8px; }}
  .fcell img {{ width:100%; display:block; border-radius:4px; }}
</style>
</head>
<body>
<h1>PolaRiS - 6 environments, real vs sim</h1>

<table style="border-collapse:separate; border-spacing:20px 16px;">
{rows}
</table>

<div class="footer">
  Sim: native 5120x1440 (exterior | wrist), 2x slow (7.5s). Real: 448x224, time-warped to 7.6s.
</div>

<h1 style="margin-top:48px">Trajectory figures</h1>
{figures}

<h2>Summary</h2>
<div class="grid2">
  <div class="fcell"><img src="assets/figures/summary/S1_success_rate.png"><div class="cap">S1 success rate</div></div>
  <div class="fcell"><img src="assets/figures/summary/S2_avg_progress.png"><div class="cap">S2 avg progress</div></div>
</div>

</body>
</html>
"""

ROW = "  <tr>\n{cells}\n  </tr>"

ENV_BLOCK = """    <td valign="top">
      <div class="label"><strong>{label}</strong> <span class="inst">- {instruction}</span></div>
      <div class="pair">
        <div class="cell"><video src="assets/real/{real_fn}" controls loop muted preload="metadata"></video><div class="cap">real</div></div>
        <div class="cell"><video src="assets/sim/{env}/episode_{ep}_mid.mp4" controls loop muted preload="metadata"></video><div class="cap">sim {badge}</div></div>
      </div>
    </td>"""


def build_figure_blocks():
    out = []
    for env, _real, label, _inst in ENVS:
        out.append(f'<h2>{label} <span style="font-size:13px;color:#888">({env})</span></h2>')
        out.append('<div class="env2x2">')
        for fname in FIGURE_PANELS:
            out.append(f'  <div class="fcell"><img src="assets/figures/{env}/{fname}"></div>')
        out.append("</div>")
    return "\n".join(out)


def main():
    rows = []
    for r in range(3):
        cells = []
        for c in range(2):
            env, real_fn, label, instruction = ENVS[r * 2 + c]
            success, prog = read_progress(env, EP)
            cells.append(ENV_BLOCK.format(
                env=env, real_fn=real_fn, label=label,
                instruction=instruction, ep=EP,
                badge=badge_for(success, prog),
            ))
        rows.append(ROW.format(cells="\n".join(cells)))
    html = HTML.format(rows="\n".join(rows), figures=build_figure_blocks())
    (ROOT / "index.html").write_text(html)
    print("wrote", ROOT / "index.html")


if __name__ == "__main__":
    main()
