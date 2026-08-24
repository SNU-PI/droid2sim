"""Measure whether each world model actually encodes the physics.

Two measurements, and the difference between them is the whole argument.

ABSOLUTE ERROR -- how far the predicted object ends up from the truth. This is
the obvious metric and it is badly confounded: a model can score well by
reproducing the average behaviour of the training distribution, without ever
responding to the physics of the particular episode in front of it.

DIFFERENCE-OF-DIFFERENCES -- take the same seed under two different physics
settings. The simulator's answer changes by some amount; ask whether the world
model's answer changes by the same amount, in the same direction. Anything
systematic about the model (its blur, its bias, its rendering style) is present
in both of its predictions and cancels in the subtraction. What survives is
only its sensitivity to the physical parameter.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import WorldModel, N_PRED
from train import load_split, norm_params, get_batch, E_EVAL
from readout import centroid_px, fit_pixel_to_world, apply_map
from sim import MASSES, FRICTIONS, NOMINAL

NCELL = len(MASSES) * len(FRICTIONS)
NOM_C = NOMINAL[0] * len(FRICTIONS) + NOMINAL[1]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def calibrate(d, dev, n=4000):
    """Fit the pixel -> world map of the readout on real rendered frames,
    and report the readout's own error floor."""
    idx = torch.arange(0, min(n, d["frames"].shape[0]), device=dev)
    t = 20
    imgs = d["frames"][idx, t]                       # [N,H,W,3] uint8
    px, _ = centroid_px(imgs.float())
    px = px.cpu().numpy()
    world = d["states"][idx, t, :2].cpu().numpy()
    M, err = fit_pixel_to_world(px, world)
    return M, float(err.mean()), float(np.percentile(err, 95))


@torch.no_grad()
def predict_all(net, d, k, oracle, M, dev, bs=250):
    """Predicted and true object displacement over the prediction window,
    for every eval episode. Returns [N,2] arrays in metres."""
    N = d["frames"].shape[0]
    pred_disp = np.zeros((N, 2), np.float32)
    true_disp = np.zeros((N, 2), np.float32)
    start = d["states"][:, E_EVAL, :2].cpu().numpy()
    true_disp[:] = d["states"][:, E_EVAL + N_PRED, :2].cpu().numpy() - start

    for lo in range(0, N, bs):
        idx = torch.arange(lo, min(lo + bs, N), device=dev)
        ctx, _ = get_batch(d, idx, E_EVAL, k, dev)
        p = norm_params(d["params"][idx]) if oracle else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(ctx, d["actions"][idx], p).float()
        last = out[:, -1]                                    # [B,3,H,W] in [-1,1]
        px, _ = centroid_px(last)
        w = apply_map(px.float(), torch.as_tensor(M, device=dev, dtype=torch.float32))
        pred_disp[lo:lo + len(idx)] = (w.cpu().numpy() - start[lo:lo + len(idx)])
    return pred_disp, true_disp


def analyse(pred, true, pidx, seed):
    """Per-model metrics. Episodes are grouped by seed; within a seed the 25
    cells are exactly paired, differing only in physics."""
    seeds = np.unique(seed)
    cell = pidx[:, 0] * len(FRICTIONS) + pidx[:, 1]

    # absolute error
    abs_err = float(np.linalg.norm(pred - true, axis=1).mean())

    # build [n_seed, 25, 2] tables
    P = np.zeros((len(seeds), NCELL, 2), np.float32)
    T = np.zeros((len(seeds), NCELL, 2), np.float32)
    for si, s in enumerate(seeds):
        m = seed == s
        P[si, cell[m]] = pred[m]
        T[si, cell[m]] = true[m]

    # difference against the nominal cell, same seed
    dP = P - P[:, NOM_C:NOM_C + 1]
    dT = T - T[:, NOM_C:NOM_C + 1]
    keep = np.arange(NCELL) != NOM_C
    fp, ft = dP[:, keep].reshape(-1), dT[:, keep].reshape(-1)

    # signed change in displacement magnitude: "does it know heavier/rougher
    # means it moves less?"
    mP = np.linalg.norm(P, axis=2) - np.linalg.norm(P[:, NOM_C:NOM_C + 1], axis=2)
    mT = np.linalg.norm(T, axis=2) - np.linalg.norm(T[:, NOM_C:NOM_C + 1], axis=2)
    mp, mt = mP[:, keep].reshape(-1), mT[:, keep].reshape(-1)

    # slope of the model's response against the simulator's ("sensitivity gain":
    # 1.0 = responds exactly as much as the simulator, 0 = ignores the parameter)
    gain = float(np.polyfit(ft, fp, 1)[0])

    out = {
        "abs_err_m": abs_err,
        "delta_pearson": pearson(ft, fp),
        "delta_spearman": spearman(ft, fp),
        "delta_gain": gain,
        "mag_pearson": pearson(mt, mp),
        "mag_spearman": spearman(mt, mp),
    }

    # per-parameter: vary mass with friction nominal, and vice versa
    for name, cells in (("mass", [i * len(FRICTIONS) + NOMINAL[1] for i in range(len(MASSES))]),
                        ("friction", [NOMINAL[0] * len(FRICTIONS) + j for j in range(len(FRICTIONS))])):
        cs = [c for c in cells if c != NOM_C]
        a, b = dT[:, cs].reshape(-1), dP[:, cs].reshape(-1)
        out[f"{name}_pearson"] = pearson(a, b)
        out[f"{name}_gain"] = float(np.polyfit(a, b, 1)[0])

    return out, {"P": P, "T": T, "seeds": seeds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)

    tr = load_split(f"{a.data}/train", dev)
    ev = load_split(f"{a.data}/eval", dev)
    M, floor_mean, floor_p95 = calibrate(tr, dev)
    print(f"readout calibrated: error floor mean {floor_mean*1000:.2f} mm, "
          f"p95 {floor_p95*1000:.2f} mm", flush=True)

    pidx = ev["pidx"].cpu().numpy()
    seed = ev["seed"].cpu().numpy()

    results, tables = {}, {}
    for f in sorted(os.listdir(a.ckpt)):
        if not f.endswith(".pt"):
            continue
        ck = torch.load(f"{a.ckpt}/{f}", map_location=dev, weights_only=False)
        net = WorldModel(ck["k"], cond_params=ck["oracle"], width=ck["width"]).to(dev)
        net.load_state_dict(ck["sd"]); net.eval()
        pred, true = predict_all(net, ev, ck["k"], ck["oracle"], M, dev)
        r, tab = analyse(pred, true, pidx, seed)
        r["k"] = ck["k"]; r["oracle"] = ck["oracle"]
        results[ck["tag"]] = r
        tables[ck["tag"]] = tab
        print(f"{ck['tag']:>12}  abs {r['abs_err_m']*1000:6.1f} mm | "
              f"delta r {r['delta_pearson']:+.3f} rho {r['delta_spearman']:+.3f} "
              f"gain {r['delta_gain']:+.3f} | mass r {r['mass_pearson']:+.3f} "
              f"fric r {r['friction_pearson']:+.3f}", flush=True)
        del net
        torch.cuda.empty_cache()

    results["_readout_floor_m"] = {"mean": floor_mean, "p95": floor_p95}
    with open(f"{a.out}/results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    np.savez(f"{a.out}/tables.npz", M=M,
             **{f"{t}__{key}": v for t, tab in tables.items() for key, v in tab.items()})
    print(f"\nwrote {a.out}/results.json")


if __name__ == "__main__":
    main()
