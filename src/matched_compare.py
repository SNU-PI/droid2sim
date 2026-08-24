"""Compare physics and nuisance changes at MATCHED image-change magnitude.

The raw signal-to-nuisance ratio is not a fair test on its own: a 2 cm camera
nudge moves far more pixels than any cell of the physics grid, so of course it
moves the latent more. The question that matters is whether, for the same
amount of image change, a change in the physics moves the representation more
than an irrelevant change does. If the two fall on one curve, the encoder is
just responding to pixels and carries no specific sensitivity to the physics.
"""
import json, sys
import numpy as np

def load(tag):
    es = json.load(open(f"out/{tag}/energy_summary.json"))
    ta = json.load(open(f"out/{tag}/token_analysis.json"))
    pix = np.array(es["pixdiff"]).ravel()
    E = np.array(ta["summary"]["object_tokens"]["grid"]).ravel()
    npix = es["nuisance_pixdiff"]
    nE = {k: v["object_tokens"] for k, v in ta["nuisance"].items()}
    return pix, E, npix, nE

for tag, label in (("stage0", "no arm, wide"), ("stage0_franka", "arm, wide"), ("stage0_zoom", "arm, ZOOM")):
    pix, E, npix, nE = load(tag)
    keep = pix > 1e-6                       # drop the self cell
    px, ey = pix[keep], E[keep]
    nx = np.array([npix[k] for k in nE]); ny = np.array([nE[k] for k in nE])

    lo, hi = max(px.min(), nx.min()), min(px.max(), nx.max())
    mp, mn = (px >= lo) & (px <= hi), (nx >= lo) & (nx <= hi)
    print(f"\n=== {label} ===")
    print(f"overlapping pixel-change range: {lo:.3f} .. {hi:.3f} grey levels")
    print(f"  physics  n={mp.sum():2d}  energy {ey[mp].mean():.4f} "
          f"+- {ey[mp].std():.4f}   (per pixel {(ey[mp]/px[mp]).mean():.4f})")
    if mn.sum():
        print(f"  nuisance n={mn.sum():2d}  energy {ny[mn].mean():.4f} "
              f"+- {ny[mn].std():.4f}   (per pixel {(ny[mn]/nx[mn]).mean():.4f})")
        ratio = (ey[mp]/px[mp]).mean() / (ny[mn]/nx[mn]).mean()
        print(f"  physics moves the latent {ratio:.2f}x as much per unit pixel change")
        print(f"  -> {'physics-specific sensitivity' if ratio > 1.5 else 'no physics-specific sensitivity; it tracks pixels'}")
    # slope of log-log fit for each family
    for nm, x, y in (("physics", px, ey), ("nuisance", nx, ny)):
        b = np.polyfit(np.log(x), np.log(y), 1)
        print(f"  {nm:8s} log-log slope {b[0]:+.3f} (0 = fully saturated)")
