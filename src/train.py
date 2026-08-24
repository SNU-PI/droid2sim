"""Train one world model variant.

The ablation is over how much *history* the model may look at before it has to
predict. Every variant predicts the SAME 8 future frames; only the amount of
context differs. That is the whole point: mass shows up in how fast the object
is already moving, friction in how fast that speed is decaying, so a longer
context makes more of the physics inferable from observation alone.

    --k 1        a single frame: neither parameter is observable
    --k 2,4,8    increasing slices of recent motion
    --k 16       the full push and decay
    --oracle     k=1, but handed the true (mass, friction). Upper bound: what a
                 model that *knows* the physics could achieve.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import WorldModel, N_PRED

# Prediction window. Context ends at frame e, targets are e+1 .. e+8.
# e ranges over [15, 23] so that even k=16 has full context and e+8 <= 31.
E_MIN, E_MAX = 15, 23
E_EVAL = 15          # fixed evaluation window, identical for every k


def load_split(prefix, device):
    fr = np.load(f"{prefix}_frames.npy", mmap_mode="r")
    md = np.load(f"{prefix}_meta.npz")
    frames = torch.from_numpy(np.ascontiguousarray(fr)).to(device)  # uint8 NHWC
    return {
        "frames": frames,
        "states": torch.from_numpy(md["states"]).to(device),
        "actions": torch.from_numpy(md["actions"]).to(device),
        "params": torch.from_numpy(md["params"]).to(device),
        "pidx": torch.from_numpy(md["pidx"].astype(np.int64)).to(device),
        "seed": torch.from_numpy(md["seed"].astype(np.int64)).to(device),
    }


def norm_params(p):
    """log-standardise (mass, mu) so the oracle conditioning is well scaled."""
    lp = torch.log(p)
    mu = torch.tensor([np.log(0.56), np.log(0.112)], device=p.device, dtype=p.dtype)
    sd = torch.tensor([0.42, 0.60], device=p.device, dtype=p.dtype)
    return (lp - mu) / sd


def get_batch(d, idx, e, k, device):
    """ctx [B,k,3,H,W] in [-1,1]; tgt [B,8,3,H,W]."""
    fr = d["frames"]
    ctx_i = torch.arange(e - k + 1, e + 1, device=device)
    tgt_i = torch.arange(e + 1, e + 1 + N_PRED, device=device)
    ctx = fr[idx[:, None], ctx_i[None, :]].float().div_(127.5).sub_(1.0)
    tgt = fr[idx[:, None], tgt_i[None, :]].float().div_(127.5).sub_(1.0)
    return ctx.permute(0, 1, 4, 2, 3).contiguous(), tgt.permute(0, 1, 4, 2, 3).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    tag = a.tag or (f"oracle_k{a.k}" if a.oracle else f"k{a.k}")
    dev = "cuda"
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[{tag}] loading data -> GPU", flush=True)
    tr = load_split(f"{a.data}/train", dev)
    ev = load_split(f"{a.data}/eval", dev)
    N = tr["frames"].shape[0]
    print(f"[{tag}] train {N} eps ({tr['frames'].nbytes/1e9:.1f} GB on GPU), "
          f"eval {ev['frames'].shape[0]}", flush=True)

    net = WorldModel(a.k, cond_params=a.oracle, width=a.width).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    print(f"[{tag}] {nparam/1e6:.2f} M params, {a.steps} steps", flush=True)

    g = torch.Generator(device=dev).manual_seed(0)
    t0 = time.time()
    for step in range(a.steps):
        idx = torch.randint(0, N, (a.bs,), device=dev, generator=g)
        e = int(torch.randint(E_MIN, E_MAX + 1, (1,), generator=g, device=dev))
        ctx, tgt = get_batch(tr, idx, e, a.k, dev)
        p = norm_params(tr["params"][idx]) if a.oracle else None

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(ctx, tr["actions"][idx], p)
            loss = F.l1_loss(pred, tgt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()

        if step % 1000 == 0 or step == a.steps - 1:
            net.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ei = torch.arange(0, ev["frames"].shape[0], 7, device=dev)
                c, t = get_batch(ev, ei, E_EVAL, a.k, dev)
                pp = norm_params(ev["params"][ei]) if a.oracle else None
                vl = F.l1_loss(net(c, ev["actions"][ei], pp), t).item()
            net.train()
            print(f"[{tag}] step {step:5d}  train {loss.item():.4f}  "
                  f"eval {vl:.4f}  {time.time()-t0:.0f}s", flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save({"sd": net.state_dict(), "k": a.k, "oracle": a.oracle,
                "width": a.width, "tag": tag}, f"{a.out}/{tag}.pt")
    print(f"[{tag}] done in {time.time()-t0:.0f}s -> {a.out}/{tag}.pt", flush=True)


if __name__ == "__main__":
    main()
