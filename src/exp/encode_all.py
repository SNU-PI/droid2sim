"""Encode every episode with frozen V-JEPA 2 and store pooled features.

Full token grids would be 11.5 MB per clip; each clip is reduced to per-frame
pooled vectors. Pooling is per frame, not per clip: friction reveals itself in
how the motion decays, so time must survive the pooling.
"""

import sys, os, glob, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.encoder import build_encoder, encode_frames
from core.paths import EP_DIR, FEAT_DIR

ROOT, FEAT = str(EP_DIR), str(FEAT_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    a = ap.parse_args()
    os.makedirs(FEAT, exist_ok=True)
    enc, tf = build_encoder()
    for d in a.dirs:
        fam, sub = d.split("/")
        outp = f"{FEAT}/{fam}__{sub}.npz"
        if os.path.exists(outp):
            print(f"skip {d}"); continue
        files = sorted(glob.glob(f"{ROOT}/{d}/*.npz"))
        if not files:
            print(f"empty {d}"); continue
        t0 = time.time()
        M, S, P, X = [], [], [], []
        for f in files:
            z = np.load(f)
            m, s = encode_frames(enc, tf, z["clip"], pool="mean+std")
            M.append(m.astype(np.float16)); S.append(s.astype(np.float16))
            P.append(z["params"])
            X.append([float(z[k]) for k in ("travel", "vmax", "vend", "zmax",
                                            "zend", "aux")])
        np.savez_compressed(outp, mean=np.stack(M), std=np.stack(S),
                            params=np.stack(P), phys=np.array(X),
                            files=[os.path.basename(f) for f in files])
        print(f"{d}: {len(files)} clips -> {outp}  ({time.time()-t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
