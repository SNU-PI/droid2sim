"""Phase 2: encode the rendered clips with frozen V-JEPA 2 and score them.

Runs in its own process, after all rendering is finished. Interleaving MuJoCo's
EGL context with PyTorch CUDA on the same GPU silently corrupted the encoder
output -- latent distances collapsed to zero for frames that were visibly
different -- so the two never share a process here.
"""

import sys, os, json, time
import numpy as np

VJEPA = "/data/pgc/simdroid/stage0/vjepa2"
CKPT = "/data/pgc/simdroid/stage0/vjepa2-ac-vitg.pt"
sys.path.insert(0, VJEPA)
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

from scene import MASS_GRID, MU_GRID, NOMINAL_CELL

CLIPS = os.environ.get("CLIP_DIR", "out/stage0/clips")
OUT = os.environ.get("RES_DIR", "out/stage0")
CROP = 256


def main():
    t0 = time.time()
    from src.hub.backbones import _make_vjepa2_ac_model
    from app.vjepa_droid.transforms import make_transforms

    print("building ViT-g encoder ...", flush=True)
    enc, _ = _make_vjepa2_ac_model(model_name="vit_ac_giant", img_size=CROP,
                                   pretrained=False)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ck["encoder"].items()}
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    print(f"  {len(sd)} tensors, {len(missing)} missing, {len(unexpected)} unexpected",
          flush=True)
    enc = enc.to("cuda").eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    del ck, sd

    tf = make_transforms(random_horizontal_flip=False,
                         random_resize_aspect_ratio=(1., 1.),
                         random_resize_scale=(1., 1.), reprob=0.,
                         auto_augment=False, motion_shift=False, crop_size=CROP)

    @torch.no_grad()
    def encode(frames):
        clip = tf(frames).unsqueeze(0).cuda()
        B, C, T, H, W = clip.size()
        c = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = enc(c)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2).float()
        return F.layer_norm(h, (h.size(-1),))[0]

    def load(name):
        d = np.load(f"{CLIPS}/{name}.npz")
        return d["clip"], float(d["travel"])

    # reference
    ref_clip, ref_travel = load("ref")
    z_ref = encode(ref_clip)
    z_ref2 = encode(ref_clip)
    noise = float((z_ref - z_ref2).abs().mean())
    print(f"latent {tuple(z_ref.shape)}   encoder noise floor {noise:.8f}", flush=True)

    def E(z):
        return float((z - z_ref).abs().mean())

    # physics grid
    Emat = np.zeros((len(MASS_GRID), len(MU_GRID)))
    Tmat = np.zeros_like(Emat)
    pix = np.zeros_like(Emat)
    for i in range(len(MASS_GRID)):
        for j in range(len(MU_GRID)):
            clip, tr = load(f"grid_{i}_{j}")
            Emat[i, j] = E(encode(clip))
            Tmat[i, j] = tr
            pix[i, j] = np.abs(clip.astype(np.int16)
                               - ref_clip.astype(np.int16)).mean()
        print(f"  mass {MASS_GRID[i]:.3f}: "
              + " ".join(f"{v:.4f}" for v in Emat[i]), flush=True)

    # nuisance controls
    nuis, nuis_pix = {}, {}
    print("\nnuisance controls (reference physics, irrelevant changes)", flush=True)
    for f in sorted(os.listdir(CLIPS)):
        if not f.startswith("nuis_"):
            continue
        name = f[:-4]
        clip, _ = load(name)
        key = name[5:]
        nuis[key] = E(encode(clip))
        nuis_pix[key] = float(np.abs(clip.astype(np.int16)
                                     - ref_clip.astype(np.int16)).mean())
        print(f"  {key:22s} energy {nuis[key]:.4f}   "
              f"(pixel diff {nuis_pix[key]:.3f})", flush=True)

    ni, nj = NOMINAL_CELL
    off = [Emat[i, j] for i in range(5) for j in range(5) if (i, j) != (ni, nj)]
    worst_n = max(nuis.values())
    print(f"\nreference cell energy    {Emat[ni,nj]:.5f}   (identical clip -> ~0)")
    print(f"nearest other cell       {min(off):.5f}")
    print(f"furthest cell            {max(off):.5f}")
    print(f"worst nuisance           {worst_n:.5f}")
    print(f"SIGNAL / NUISANCE        {min(off)/max(worst_n,1e-9):.2f}")

    np.savez(f"{OUT}/energy.npz", E=Emat, travel=Tmat, pixdiff=pix,
             mass=MASS_GRID, mu=MU_GRID, nominal=NOMINAL_CELL)
    json.dump({"E": Emat.tolist(), "travel_cm": Tmat.tolist(),
               "pixdiff": pix.tolist(), "nuisance": nuis,
               "nuisance_pixdiff": nuis_pix, "noise_floor": noise,
               "ref_energy": float(Emat[ni, nj]),
               "nearest_other_cell": float(min(off)),
               "furthest_cell": float(max(off)),
               "worst_nuisance": float(worst_n)},
              open(f"{OUT}/energy_summary.json", "w"), indent=2)
    print(f"\nwrote {OUT}/energy.npz   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
