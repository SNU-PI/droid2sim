"""Confound 3: does the ACTION-CONDITIONED PREDICTOR carry the physics?

Everything so far used only the frozen encoder, comparing images that were
already rendered. That never asked the world model to predict anything. This
does, and it is the procedure the whole pseudo-labelling idea depends on:

    take the context frames and the actions that were actually executed,
    let V-JEPA 2-AC predict the future in latent space,
    then find which physical parameters make the simulator reproduce
    that predicted future.

If the predictor has absorbed any physics from DROID, its prediction should
look most like the rollout of the parameters that actually generated the
context, and the energy over the grid should dip there.

The pusher rides a rail and is the only thing a robot would be holding, so it
plays the end effector: pose = [x, y, z, 0, 0, 0, gripper] and the action is
the frame-to-frame delta, matching the convention V-JEPA 2-AC was trained with.
"""

import sys, os, json, argparse
import numpy as np

VJEPA = "/data/pgc/simdroid/stage0/vjepa2"
CKPT = "/data/pgc/simdroid/stage0/vjepa2-ac-vitg.pt"
sys.path.insert(0, VJEPA)
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

sys.path.insert(0, f"{VJEPA}/notebooks")
from scene import (PushScene, MASS_GRID, MU_GRID, NOMINAL_CELL, TABLE_TOP_Z,
                   OBJ_HALF, CTRL_DT)

CLIPS = os.environ.get("CLIP_DIR", "out/stage0_zoom/clips")
OUT = os.environ.get("RES_DIR", "out/stage0_zoom")
FRAME_IDX = [8, 12, 16, 20, 24, 28, 32, 36]
NCTX = 4           # context frames given to the predictor
NPRED = 4          # frames it must predict
NT = 256


def pusher_poses(mass, mu):
    """End-effector pose per saved frame, from the (unrendered) simulation."""
    sc = PushScene()                       # physics only; no renderer touched
    st, _ = sc.rollout(mass, mu, render=False)
    y = st[FRAME_IDX, 6]                   # rail position at each saved frame
    z = TABLE_TOP_Z + OBJ_HALF[2]
    poses = np.zeros((len(FRAME_IDX), 7), np.float32)
    poses[:, 0] = 0.0
    poses[:, 1] = y
    poses[:, 2] = z
    poses[:, 6] = 0.0                      # gripper closedness, constant
    actions = np.zeros_like(poses)
    actions[:-1] = poses[1:] - poses[:-1]
    actions[-1] = actions[-2]
    return poses, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true-cell", default="2,2")
    a = ap.parse_args()
    ti, tj = [int(v) for v in a.true_cell.split(",")]
    if os.path.exists(f"{OUT}/ac_predictor_{ti}{tj}.json"):
        print(f"already done ({ti},{tj})"); return

    from src.hub.backbones import _make_vjepa2_ac_model
    from app.vjepa_droid.transforms import make_transforms

    print("building encoder + AC predictor ...", flush=True)
    enc, pred = _make_vjepa2_ac_model(model_name="vit_ac_giant", img_size=256,
                                      pretrained=False)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    e_sd = {k.replace("module.", ""): v for k, v in ck["encoder"].items()}
    p_sd = {k.replace("module.", ""): v for k, v in ck["predictor"].items()}
    me, ue = enc.load_state_dict(e_sd, strict=False)
    mp, up = pred.load_state_dict(p_sd, strict=False)
    print(f"  encoder   {len(e_sd)} tensors, {len(me)} missing, {len(ue)} unexpected")
    print(f"  predictor {len(p_sd)} tensors, {len(mp)} missing, {len(up)} unexpected",
          flush=True)
    enc = enc.to("cuda").eval(); pred = pred.to("cuda").eval()
    for p in list(enc.parameters()) + list(pred.parameters()):
        p.requires_grad_(False)
    del ck, e_sd, p_sd

    tf = make_transforms(random_horizontal_flip=False,
                         random_resize_aspect_ratio=(1., 1.),
                         random_resize_scale=(1., 1.), reprob=0.,
                         auto_augment=False, motion_shift=False, crop_size=256)

    @torch.no_grad()
    def encode(frames):
        clip = tf(frames).unsqueeze(0).cuda()
        B, C, T, H, W = clip.size()
        c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = enc(c)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2).float()
        return F.layer_norm(h, (h.size(-1),))

    def load(n):
        d = np.load(f"{CLIPS}/{n}.npz")
        return d["clip"], float(d["travel"])

    # ---- roll the predictor forward from the true cell's context ------------
    true_name = f"grid_{ti}_{tj}"
    clip, _ = load(true_name)
    poses, actions = pusher_poses(MASS_GRID[ti], MU_GRID[tj])
    z_ctx = encode(clip[:NCTX])                                  # [1, NCTX*NT, D]
    s = torch.tensor(poses[:NCTX], device="cuda").unsqueeze(0)
    act = torch.tensor(actions[:NCTX], device="cuda").unsqueeze(0)

    print(f"\nrolling the predictor {NPRED} steps from cell ({ti},{tj}) "
          f"mass={MASS_GRID[ti]:.3f} mu={MU_GRID[tj]:.3f}", flush=True)
    from utils.mpc_utils import compute_new_pose
    z, preds = z_ctx, []
    with torch.no_grad():
        for step in range(NPRED):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nxt = pred(z, act, s)[:, -NT:].float()
            nxt = F.layer_norm(nxt, (nxt.size(-1),))
            preds.append(nxt)
            new_pose = compute_new_pose(s[:, -1:], act[:, -1:]).to("cuda").float()
            z = torch.cat([z, nxt], dim=1)
            s = torch.cat([s, new_pose], dim=1)
            k = min(NCTX + step, len(actions) - 1)
            act = torch.cat([act, torch.tensor(actions[k], device="cuda")
                             .view(1, 1, 7)], dim=1)
    z_hat = torch.cat(preds, dim=1)                              # [1, NPRED*NT, D]
    print(f"  predicted latent {tuple(z_hat.shape)}", flush=True)

    # ---- which physics reproduces that prediction best? --------------------
    E = np.zeros((5, 5)); T = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            c, tr = load(f"grid_{i}_{j}")
            z_true = encode(c[NCTX:NCTX + NPRED])
            E[i, j] = float((z_hat - z_true).abs().mean())
            T[i, j] = tr
        print(f"  mass {MASS_GRID[i]:.3f}: " + " ".join(f"{v:.4f}" for v in E[i]),
              flush=True)

    nuis = {}
    for f in sorted(os.listdir(CLIPS)):
        if f.startswith("nuis_"):
            c, _ = load(f[:-4])
            nuis[f[5:-4]] = float((z_hat - encode(c[NCTX:NCTX + NPRED])).abs().mean())

    amin = np.unravel_index(np.argmin(E), E.shape)
    off = [E[i, j] for i in range(5) for j in range(5) if (i, j) != (ti, tj)]
    wn = max(nuis.values())
    print(f"\ntrue cell        ({ti},{tj})  energy {E[ti,tj]:.5f}")
    print(f"argmin cell      {amin}  energy {E[amin]:.5f}")
    print(f"  -> {'HIT: predictor points at the right physics' if amin == (ti,tj) else 'MISS: it points elsewhere'}")
    print(f"nearest other    {min(off):.5f}")
    print(f"worst nuisance   {wn:.5f}")
    print(f"SIGNAL/NUISANCE  {min(off)/max(wn,1e-9):.2f}")
    dtr = np.abs(T - T[ti, tj]).ravel()
    print(f"corr with travel {np.corrcoef(dtr, E.ravel())[0,1]:.3f}")

    json.dump({"E": E.tolist(), "travel": T.tolist(), "nuisance": nuis,
               "true_cell": [ti, tj], "argmin": [int(v) for v in amin],
               "hit": bool(amin == (ti, tj))},
              open(f"{OUT}/ac_predictor_{ti}{tj}.json", "w"), indent=2)
    print(f"\nwrote {OUT}/ac_predictor_{ti}{tj}.json")


if __name__ == "__main__":
    main()
