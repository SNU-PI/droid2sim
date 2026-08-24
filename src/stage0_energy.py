"""Stage 0: is V-JEPA's latent space sensitive to mass and friction?

The whole pseudo-labelling idea rests on one assumption -- that a pretrained
world model's representation moves when an object's physical parameters move.
If it does not, no amount of optimisation machinery downstream will recover
anything. This measures that directly, and nothing else.

Method. Render the scene at a reference (mass, mu), encode it with the frozen
V-JEPA 2 encoder, and treat that as the target. Then sweep the 5x5 parameter
grid, encode each, and plot the latent distance to the target. A usable
landscape has a single clear minimum at the reference.

The control matters as much as the sweep. A minimum on its own proves little:
the images at the reference cell are literally identical to the target, so
*some* dip is guaranteed. The real question is whether the physics signal is
large compared to things we do not care about. So the same distance is measured
under nuisance perturbations -- a nudged camera, dimmer lights, a different
cube colour -- at the unchanged reference physics. If nuisance distances swamp
the physics distances, the approach cannot work on real footage.
"""

import sys, os, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "2")

VJEPA = "/data/pgc/simdroid/stage0/vjepa2"
CKPT = "/data/pgc/simdroid/stage0/vjepa2-ac-vitg.pt"
sys.path.insert(0, VJEPA)

import torch
import torch.nn.functional as F
import mujoco

from scene import (PushScene, MASS_GRID, MU_GRID, NOMINAL_CELL,
                   NOMINAL_MASS, NOMINAL_MU, N_FRAMES)

OUT = "out/stage0"
FRAME_IDX = [8, 12, 16, 20, 24, 28, 32, 36]     # 8 frames: strike + slide
CROP = 256


def load_encoder(device="cuda"):
    from src.hub.backbones import _make_vjepa2_ac_model
    print("building ViT-g encoder ...", flush=True)
    encoder, predictor = _make_vjepa2_ac_model(
        model_name="vit_ac_giant", img_size=CROP, pretrained=False)
    print(f"loading weights from {CKPT} ...", flush=True)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    enc_sd = {k.replace("module.", ""): v for k, v in ck["encoder"].items()}
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    print(f"  encoder: {len(enc_sd)} tensors, {len(missing)} missing, "
          f"{len(unexpected)} unexpected", flush=True)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in encoder.parameters())
    print(f"  {n/1e9:.2f} B parameters on {device}", flush=True)
    return encoder


def make_transform():
    from app.vjepa_droid.transforms import make_transforms
    return make_transforms(random_horizontal_flip=False,
                           random_resize_aspect_ratio=(1., 1.),
                           random_resize_scale=(1., 1.),
                           reprob=0., auto_augment=False, motion_shift=False,
                           crop_size=CROP)


@torch.no_grad()
def encode(encoder, transform, frames, device="cuda"):
    """frames uint8 [T,H,W,3] -> layer-normed token representation [T*N, D]."""
    clip = transform(frames).unsqueeze(0).to(device)      # [1,C,T,H,W]
    B, C, T, H, W = clip.size()
    # each frame is duplicated to fill the 2-frame tubelet, as in the reference
    # implementation, so the encoder sees T independent frames
    c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = encoder(c)
    h = h.view(B, T, -1, h.size(-1)).flatten(1, 2).float()
    return F.layer_norm(h, (h.size(-1),))[0]


def energy(a, b):
    """Latent distance used as the matching objective (mean L1, as in V-JEPA)."""
    return float((a - b).abs().mean())


# ---- nuisance perturbations: things we do NOT want the metric to respond to --
def apply_nuisance(sc, kind, amount):
    m = sc.model
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "droidcam")
    if kind == "camera":
        m.cam_pos[cam, 0] += amount                       # metres, sideways
    elif kind == "light":
        m.light_diffuse[0] *= (1.0 + amount)
    elif kind == "cube_hue":
        mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "objmat")
        m.mat_rgba[mid, 0] = np.clip(m.mat_rgba[mid, 0] - amount, 0, 1)
        m.mat_rgba[mid, 1] = np.clip(m.mat_rgba[mid, 1] + amount, 0, 1)
    elif kind == "table_hue":
        mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "wood")
        m.mat_rgba[mid, 2] = np.clip(m.mat_rgba[mid, 2] + amount, 0, 1)
    else:
        raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    # ---- PHASE 1: render everything, touching no CUDA ------------------------
    # Rendering and encoding are kept strictly apart. MuJoCo's EGL context and
    # PyTorch's CUDA context share the GPU, and interleaving them silently
    # corrupted the encoder output -- latent distances collapsed to 0.0000 for
    # frames that were visibly different. Rendering first, encoding second,
    # makes the two never overlap.
    print("\nphase 1: rendering", flush=True)
    sc = PushScene()
    clips, keys, travel_l = [], [], []

    st, fr = sc.rollout(NOMINAL_MASS, NOMINAL_MU, render=True)
    clips.append(fr[FRAME_IDX].copy()); keys.append(("ref", None))
    print(f"  reference: mass={NOMINAL_MASS} kg, mu={NOMINAL_MU}", flush=True)

    travel = np.zeros((len(MASS_GRID), len(MU_GRID)))
    for i, mass in enumerate(MASS_GRID):
        for j, mu in enumerate(MU_GRID):
            st, fr = sc.rollout(mass, mu, render=True)
            clips.append(fr[FRAME_IDX].copy()); keys.append(("grid", (i, j)))
            travel[i, j] = np.linalg.norm(st[-1, :2] - st[0, :2]) * 100
            print(f"    cell[{i},{j}] m={mass:.3f} mu={mu:.3f} "
                  f"travel={travel[i,j]:5.1f}cm", flush=True)
    print(f"  {len(MASS_GRID)*len(MU_GRID)} physics cells", flush=True)

    for kind, amounts in (("camera", [0.005, 0.010, 0.020]),
                          ("light", [0.10, 0.25]),
                          ("cube_hue", [0.10, 0.25]),
                          ("table_hue", [0.10, 0.25])):
        for amt in amounts:
            s2 = PushScene()
            apply_nuisance(s2, kind, amt)
            _, fr = s2.rollout(NOMINAL_MASS, NOMINAL_MU, render=True)
            clips.append(fr[FRAME_IDX].copy()); keys.append(("nuis", f"{kind}={amt}"))
            del s2
    del sc
    print(f"  {len(clips)} clips total  ({time.time()-t0:.0f}s)", flush=True)

    # ---- PHASE 2: encode ------------------------------------------------------
    print("\nphase 2: encoding", flush=True)
    encoder = load_encoder(a.device)
    transform = make_transform()
    zs = []
    for n, c in enumerate(clips):
        zs.append(encode(encoder, transform, c, a.device))
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(clips)}  ({time.time()-t0:.0f}s)", flush=True)
    z_ref = zs[0]
    print(f"  latent {tuple(z_ref.shape)}", flush=True)

    # sanity: the encoder must be deterministic and must see the differences
    z_ref2 = encode(encoder, transform, clips[0], a.device)
    noise = energy(z_ref, z_ref2)
    print(f"  encoder noise floor (same input twice): {noise:.6f}", flush=True)

    E = np.zeros((len(MASS_GRID), len(MU_GRID)))
    nuis = {}
    for z, (kind, key) in zip(zs, keys):
        if kind == "grid":
            E[key] = energy(z, z_ref)
        elif kind == "nuis":
            nuis[key] = energy(z, z_ref)
    for i, mass in enumerate(MASS_GRID):
        print(f"  mass {mass:.3f}: " + " ".join(f"{v:.4f}" for v in E[i]), flush=True)
    print("\nnuisance controls at the reference physics", flush=True)
    for k, v in nuis.items():
        print(f"  {k:18s} -> {v:.4f}", flush=True)

    # ---- summary -------------------------------------------------------------
    ni, nj = NOMINAL_CELL
    phys_max = float(E.max())
    phys_min_off = float(np.min([E[i, j] for i in range(5) for j in range(5)
                                 if (i, j) != (ni, nj)]))
    worst_nuis = max(nuis.values())
    print(f"\nreference cell energy      : {E[ni, nj]:.5f}  (should be ~0)")
    print(f"nearest other cell         : {phys_min_off:.5f}")
    print(f"furthest cell              : {phys_max:.5f}")
    print(f"worst nuisance             : {worst_nuis:.5f}")
    print(f"signal-to-nuisance ratio   : {phys_min_off/max(worst_nuis,1e-9):.2f}"
          f"  (nearest physics change vs worst irrelevant change)")

    np.savez(f"{OUT}/energy.npz", E=E, travel=travel,
             mass=MASS_GRID, mu=MU_GRID, nominal=NOMINAL_CELL)
    with open(f"{OUT}/energy_summary.json", "w") as f:
        json.dump({"E": E.tolist(), "travel_cm": travel.tolist(),
                   "nuisance": nuis, "ref_energy": float(E[ni, nj]),
                   "nearest_other_cell": phys_min_off,
                   "furthest_cell": phys_max, "worst_nuisance": worst_nuis,
                   "frames": FRAME_IDX}, f, indent=2)
    print(f"\nwrote {OUT}/energy.npz  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
