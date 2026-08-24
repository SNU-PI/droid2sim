"""How does V-JEPA's latent distance grow with the size of an image change?

Needed to read the stage 0 numbers at all. The reference cell -- whose clip
matches the target to 0.006 grey levels -- already sits at a latent distance of
0.15, while the largest physics change in the whole grid only reaches 0.28. If
the distance saturates almost immediately then it cannot resolve physics, and
the flat landscape says nothing about whether the physics is represented.

This measures the response curve directly: perturb the reference clip by a
known amount and watch the distance.
"""

import sys, os, json
import numpy as np

VJEPA = "/data/pgc/simdroid/stage0/vjepa2"
CKPT = "/data/pgc/simdroid/stage0/vjepa2-ac-vitg.pt"
sys.path.insert(0, VJEPA)
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

CLIPS = "out/stage0/clips"
OUT = "out/stage0"


def main():
    from src.hub.backbones import _make_vjepa2_ac_model
    from app.vjepa_droid.transforms import make_transforms

    enc, _ = _make_vjepa2_ac_model(model_name="vit_ac_giant", img_size=256,
                                   pretrained=False)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    enc.load_state_dict({k.replace("module.", ""): v
                         for k, v in ck["encoder"].items()}, strict=False)
    enc = enc.to("cuda").eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    del ck
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
        return F.layer_norm(h, (h.size(-1),))[0]

    ref = np.load(f"{CLIPS}/ref.npz")["clip"]
    z_ref = encode(ref)
    rng = np.random.default_rng(0)

    print(f"{'perturbation':28s} {'pixel diff':>11} {'latent dist':>12}")
    print(f"{'identical (noise floor)':28s} {0.0:11.4f} "
          f"{float((z_ref-encode(ref)).abs().mean()):12.5f}")

    rows = []
    for sigma in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        pert = np.clip(ref.astype(np.float32)
                       + rng.normal(0, sigma, ref.shape), 0, 255).astype(np.uint8)
        pd = float(np.abs(pert.astype(np.int16) - ref.astype(np.int16)).mean())
        e = float((encode(pert) - z_ref).abs().mean())
        rows.append(("noise sigma=%.2f" % sigma, pd, e))
        print(f"{rows[-1][0]:28s} {pd:11.4f} {e:12.5f}", flush=True)

    # a change of the same kind the physics produces: shift the whole clip
    for shift in (1, 2, 4):
        pert = np.roll(ref, shift, axis=2)
        pd = float(np.abs(pert.astype(np.int16) - ref.astype(np.int16)).mean())
        e = float((encode(pert) - z_ref).abs().mean())
        rows.append((f"image shift {shift}px", pd, e))
        print(f"{rows[-1][0]:28s} {pd:11.4f} {e:12.5f}", flush=True)

    json.dump([{"name": n, "pixdiff": p, "energy": e} for n, p, e in rows],
              open(f"{OUT}/calibration.json", "w"), indent=2)
    print(f"\nwrote {OUT}/calibration.json")


if __name__ == "__main__":
    main()
