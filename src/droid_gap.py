"""Where does the sim-to-real gap live in feature space?

The real DROID clip sits at Mahalanobis 86 from the sim cloud (sim p95 = 11).
Before deciding what to do about it, find out what drives that distance:

  static content    encode a single frame repeated 8x (no motion at all) from
                    real vs sim -> is the gap already there before anything
                    moves? Then it is appearance, not dynamics.
  per-frame vs      compare the real clip against sim using only the mean
  temporal          feature over time (appearance) versus only frame-to-frame
                    differences (motion). Which channel is out of distribution?
  cross-scene       is the real clip closer to any of the other four sim
                    families? Or to sim under nuisance perturbations?
  a sim clip made   drop the sim cube's frames into the real background via
  to look real      naive compositing -- does the distance collapse? That
                    would say background/appearance is most of the gap.
"""
import sys, os, json
import numpy as np
VJEPA = "/data/pgc/simdroid/stage0/vjepa2"; CKPT = "/data/pgc/simdroid/stage0/vjepa2-ac-vitg.pt"
sys.path.insert(0, VJEPA); sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn.functional as F
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from droid_probe import build, encode, to256

FEAT = "/data/pgc/simdroid/features"
FAMS = ["slide", "roll", "bounce", "collide", "incline"]
OUT = "out/droid"


def cloud(X, k=96):
    sc = StandardScaler().fit(X); pca = PCA(n_components=k).fit(sc.transform(X))
    Z = pca.transform(sc.transform(X)); mu = Z.mean(0)
    icov = np.linalg.inv(np.cov(Z.T) + 1e-6 * np.eye(k))
    ref = np.sqrt(np.einsum("ij,jk,ik->i", Z - mu, icov, Z - mu))
    def dist(x): z = pca.transform(sc.transform(x.reshape(1, -1)))[0]; return float(np.sqrt((z-mu)@icov@(z-mu)))
    return dist, float(np.percentile(ref, 95))


def main():
    os.makedirs(OUT, exist_ok=True)
    enc, tf = build()
    d = np.load("/data/pgc/simdroid/droid/eps/ep88.npz", allow_pickle=True)
    fr = d["frames"]; s0 = 20; idx = list(range(s0, s0 + 15, 2))
    real = to256(fr[idx])                                     # the most-moving window
    f_real = encode(enc, tf, real)                            # [8, D]

    sim = np.load(f"{FEAT}/slide__train_clean.npz")["mean"].astype(np.float64)   # [N,8,D]
    R = {}

    # ---- 1. full clip, per family + nuisance sets ------------------------------
    print("=== 1. distance of the real clip to each sim cloud (full 8-frame features) ===")
    R["by_family"] = {}
    for fam in FAMS:
        X = np.load(f"{FEAT}/{fam}__train_clean.npz")["mean"].astype(np.float64)
        dist, p95 = cloud(X.reshape(len(X), -1))
        m = dist(f_real.reshape(-1)); R["by_family"][fam] = (m, p95)
        print(f"  {fam:8s} maha {m:6.1f}   (sim p95 {p95:5.1f})   ratio {m/p95:4.1f}x")
    Xa = np.concatenate([np.load(f"{FEAT}/slide__{s}.npz")["mean"].astype(np.float64)
                         for s in ("train_clean", "train_camera")])
    dist, p95 = cloud(Xa.reshape(len(Xa), -1)); m = dist(f_real.reshape(-1))
    print(f"  slide + camera-aug         maha {m:6.1f}   (p95 {p95:5.1f})   ratio {m/p95:4.1f}x")
    R["slide_aug"] = (m, p95)

    # ---- 2. appearance vs motion channels ---------------------------------------
    print("\n=== 2. which channel is out of distribution? ===")
    Xs = sim
    app_sim = Xs.mean(1); mot_sim = np.diff(Xs, axis=1).reshape(len(Xs), -1)
    app_real = f_real.mean(0); mot_real = np.diff(f_real, axis=0).reshape(-1)
    d_app, p_app = cloud(app_sim, k=64); d_mot, p_mot = cloud(mot_sim, k=64)
    ma, mm = d_app(app_real), d_mot(mot_real)
    print(f"  appearance (time-mean feature)   maha {ma:6.1f}  (p95 {p_app:5.1f})  ratio {ma/p_app:4.1f}x")
    print(f"  motion (frame differences)       maha {mm:6.1f}  (p95 {p_mot:5.1f})  ratio {mm/p_mot:4.1f}x")
    R["channels"] = {"appearance": (ma, p_app), "motion": (mm, p_mot)}

    # ---- 3. static frame: is the gap there before anything moves? ---------------
    print("\n=== 3. a single frame repeated 8x (zero motion) ===")
    still_real = np.repeat(real[:1], 8, axis=0)
    f_still = encode(enc, tf, still_real)
    dist, p95 = cloud(sim.reshape(len(sim), -1))
    m_still = dist(f_still.reshape(-1))
    # and the same for a sim frame, as a control
    simclip = np.load(sorted(__import__("glob").glob("/data/pgc/simdroid/episodes/slide/train_clean/*.npz"))[3])["clip"]
    f_simstill = encode(enc, tf, np.repeat(simclip[:1], 8, axis=0))
    m_simstill = dist(f_simstill.reshape(-1))
    print(f"  real frame x8   maha {m_still:6.1f}   |  sim frame x8  maha {m_simstill:6.1f}   (p95 {p95:5.1f})")
    R["static"] = {"real": m_still, "sim": m_simstill, "p95": p95}

    # ---- 4. composite: sim cube pasted onto real background --------------------
    print("\n=== 4. sim object composited onto the real background ===")
    # object mask from the sim clip: red cube pixels
    sc_ = simclip.astype(np.int16)
    mask = ((sc_[..., 0] - sc_[..., 1] > 60) & (sc_[..., 0] - sc_[..., 2] > 60))   # [8,H,W]
    bg = np.repeat(real[:1], 8, axis=0).astype(np.int16)
    comp = bg.copy(); comp[mask] = sc_[mask]
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    f_comp = encode(enc, tf, comp)
    m_comp = dist(f_comp.reshape(-1))
    print(f"  sim cube on real background   maha {m_comp:6.1f}   (pure sim clip {dist(encode(enc,tf,simclip).reshape(-1)):5.1f}, "
          f"real {dist(f_real.reshape(-1)):5.1f})")
    R["composite"] = m_comp
    import imageio.v2 as iio
    iio.imwrite(f"{OUT}/composite_check.png", np.concatenate([real[3], simclip[3], comp[3]], 1))

    # ---- 5. the reverse: real block region pasted onto sim background ---------
    # crude: take the real clip, grey out everything except a central region
    print("\n=== 5. how much of the real gap is 'stuff outside the workspace'? ===")
    for name, box in (("centre 60%", 0.6), ("centre 40%", 0.4)):
        h = int(256 * box); o = (256 - h) // 2
        masked = np.full_like(real, 110); masked[:, o:o+h, o:o+h] = real[:, o:o+h, o:o+h]
        m = dist(encode(enc, tf, masked).reshape(-1))
        print(f"  real clip, only {name} kept   maha {m:6.1f}")
        R[f"mask_{box}"] = m

    json.dump(R, open(f"{OUT}/gap_analysis.json", "w"), indent=2)
    print(f"\nwrote {OUT}/gap_analysis.json")


if __name__ == "__main__":
    main()
