"""Check the scene against theory before trusting a single downstream number.

Every check must be able to FAIL. An early version of this file reported PASS
on a scene where the object never moved: nan errors averaged to zero and
"monotone decreasing" was vacuously true over a row of zeros.

Measurements are taken from the 1 ms physics trace, not the 20 ms rendered
frames. At 20 ms a rough, heavy cube stops within a single frame and there is
nothing left to fit a deceleration to -- an artefact of the sampling rate that
looks exactly like a physics failure.

  0. Sanity            the object must actually move
  1. Coulomb law       free-slide deceleration must equal mu * g
  2. Mass invariance   that deceleration must NOT depend on mass
  3. Momentum split    release speed must follow M*v0 / (M + m)
  4. Identifiability   mass and friction must leave distinguishable signatures
  5. Timestep          release speed and deceleration must converge in dt
  6. Determinism       identical inputs -> identical outputs
  7. Framing           the object must stay on the table
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "2")

from scene import (PushScene, NOMINAL_MASS, NOMINAL_MU, TABLE_TOP_Z, OBJ_HALF,
                   CTRL_DT, N_FRAMES, PUSHER_MASS, PUSHER_V0, MASS_GRID, MU_GRID)

G = 9.81
RESULTS = []


def report(name, passed, detail=""):
    RESULTS.append((name, bool(passed)))
    print(f"   -> {name}: {'PASS' if passed else 'FAIL'} {detail}\n")


def measure(sc, mass, mu):
    """Release speed and free-slide deceleration, from the 1 ms trace.

    The free-slide phase begins once the pusher has separated (object faster
    than the rail) and the object is past its peak speed.
    """
    st, _, tr = sc.rollout(mass, mu, fine=True)
    dt = sc.model.opt.timestep
    v_obj, v_rail = tr[:, 0], tr[:, 1]
    peak = int(np.argmax(v_obj))
    v_release = float(v_obj[peak])

    # Fit the mid-slide band only: between 85% and 25% of the release speed.
    # A fixed lower cutoff instead pulls in the stiction tail, where the solver
    # brings the object to rest nonlinearly. That tail is a negligible part of
    # a long slide but dominates a short one, which is why the measured
    # "friction" used to degrade exactly as mu and mass went up.
    lo = peak + int(0.010 / dt)                    # 10 ms past the collision
    sel = np.arange(lo, len(v_obj))
    sel = sel[(v_obj[sel] < 0.85 * v_release) & (v_obj[sel] > 0.25 * v_release)
              & (v_obj[sel] > v_rail[sel])]
    if len(sel) < 20:
        return v_release, np.nan, len(sel), st
    a = -np.polyfit(sel * dt, v_obj[sel], 1)[0]
    return v_release, float(a), len(sel), st


def main():
    sc = PushScene()
    print(f"scene: {tuple(np.round(2*OBJ_HALF,3))} m cube on table z={TABLE_TOP_Z}, "
          f"dt={sc.model.opt.timestep*1000:.2f} ms, {N_FRAMES} frames @ "
          f"{CTRL_DT*1000:.0f} ms")
    print(f"action: pusher {PUSHER_MASS} kg on a rail at {PUSHER_V0} m/s\n")

    print("0) sanity")
    vr, a, n, st = measure(sc, NOMINAL_MASS, NOMINAL_MU)
    travel = float(np.linalg.norm(st[-1, :2] - st[0, :2]))
    print(f"   nominal m={NOMINAL_MASS} mu={NOMINAL_MU}: release {vr:.3f} m/s, "
          f"decel {a:.3f} m/s^2, travel {travel*100:.1f} cm, {n} trace pts")
    report("object moves", vr > 0.05 and travel > 0.01)
    if vr <= 0.05:
        return 1

    # ---- 1 & 2 -------------------------------------------------------------
    print("1+2) free-slide deceleration vs theory  (a = mu*g, mass-independent)")
    print(f"   {'mu':>5} {'mass':>6} {'a_meas':>8} {'mu*g':>7} {'err':>7} {'pts':>6}")
    errs, by_mu = [], {}
    for mu in MU_GRID:
        for mass in MASS_GRID[[0, 2, 4]]:
            _, a, n, _ = measure(sc, mass, mu)
            if not np.isfinite(a):
                print(f"   {mu:5.2f} {mass:6.2f} {'nan':>8} {mu*G:7.3f} "
                      f"{'--':>7} {n:6d}")
                errs.append(np.inf); continue
            rel = abs(a - mu * G) / (mu * G)
            errs.append(rel)
            by_mu.setdefault(mu, []).append(a)
            print(f"   {mu:5.2f} {mass:6.2f} {a:8.3f} {mu*G:7.3f} {rel*100:6.2f}% "
                  f"{n:6d}")
    worst = max(errs)
    report("Coulomb law holds", np.isfinite(worst) and worst < 0.05,
           f"(worst error {worst*100:.2f}%)" if np.isfinite(worst) else "(nan)")
    spreads = [np.ptp(v) / np.mean(v) for v in by_mu.values() if len(v) > 1]
    ms = max(spreads) if spreads else np.inf
    report("deceleration independent of mass", np.isfinite(ms) and ms < 0.05,
           f"(worst spread across masses {ms*100:.2f}%)")

    # ---- 3 -----------------------------------------------------------------
    print("3) release speed vs momentum-exchange theory  M*v0/(M+m)")
    print(f"   {'mass':>6} {'v_meas':>8} {'v_theory':>9} {'ratio':>7}")
    ratios, vr_by_mass = [], []
    for mass in MASS_GRID:
        v, _, _, _ = measure(sc, mass, NOMINAL_MU)
        vth = sc.predicted_release_speed(mass)
        ratios.append(v / vth); vr_by_mass.append(v)
        print(f"   {mass:6.2f} {v:8.3f} {vth:9.3f} {v/vth:7.3f}")
    # The reference M*v0/(M+m) assumes a perfectly INELASTIC collision. The
    # contact is deliberately compliant, so it returns a little energy and the
    # ratio sits slightly above 1. What matters is that the ratio is CONSTANT
    # across mass -- that is the evidence the M/(M+m) law is being reproduced,
    # with the offset absorbed into an effective restitution.
    e_eff = np.mean(ratios) - 1.0
    print(f"   ratio mean {np.mean(ratios):.3f}, spread {np.ptp(ratios):.3f}  "
          f"-> effective restitution e = {e_eff:+.3f}")
    report("momentum split follows theory (constant ratio across mass)",
           np.ptp(ratios) < 0.08 and 0.85 < np.mean(ratios) < 1.20,
           f"(spread {np.ptp(ratios):.3f}; a constant ratio is the law holding)")

    # ---- 4: is the parameter -> observable map invertible? -----------------
    # The requirement for stage 0 is not that the two parameters act on
    # disjoint quantities, but that every cell of the grid produces a
    # DISTINGUISHABLE signature. Mass sets the release speed, friction sets the
    # decay; together they should separate all 25 cells.
    print("4) identifiability  (are all 25 grid cells distinguishable?)")
    sig = np.zeros((len(MASS_GRID), len(MU_GRID), 2))
    for i, m in enumerate(MASS_GRID):
        for j, mu in enumerate(MU_GRID):
            v, a, _, _ = measure(sc, m, mu)
            sig[i, j] = (v, a)
    print(f"   release speed spans {sig[...,0].min():.3f}..{sig[...,0].max():.3f} m/s "
          f"({sig[...,0].max()/sig[...,0].min():.2f}x)   "
          f"decel spans {sig[...,1].min():.2f}..{sig[...,1].max():.2f} m/s^2 "
          f"({sig[...,1].max()/sig[...,1].min():.2f}x)")
    # correlation of each observable with each parameter, across the grid
    mm, uu = np.meshgrid(MASS_GRID, MU_GRID, indexing="ij")
    for nm, k in (("release speed", 0), ("deceleration", 1)):
        cm = np.corrcoef(mm.ravel(), sig[..., k].ravel())[0, 1]
        cu = np.corrcoef(uu.ravel(), sig[..., k].ravel())[0, 1]
        print(f"   {nm:14s} corr with mass {cm:+.3f}   corr with mu {cu:+.3f}")
    # nearest-neighbour separation in the normalised signature space
    flat = sig.reshape(-1, 2)
    z = (flat - flat.mean(0)) / flat.std(0)
    d = np.linalg.norm(z[:, None] - z[None, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    print(f"   closest pair of cells in normalised signature space: {d.min():.3f}")
    report("all 25 cells distinguishable", d.min() > 0.15,
           f"(min separation {d.min():.3f}; 0 would mean two cells look identical)")

    # ---- 5: convergence of the two physical quantities ---------------------
    print("5) timestep convergence of the measured physics")
    print(f"   {'dt (ms)':>8} {'release':>9} {'decel':>8}")
    prev = None
    conv = []
    for dt in (0.002, 0.001, 0.0005, 0.00025):
        s2 = PushScene(timestep=dt)
        v, a, _, _ = measure(s2, NOMINAL_MASS, NOMINAL_MU)
        print(f"   {dt*1000:8.2f} {v:9.4f} {a:8.4f}")
        if prev is not None:
            conv.append((abs(v - prev[0]) / prev[0], abs(a - prev[1]) / prev[1]))
        prev = (v, a)
    last = max(conv[-1])
    report("converged in dt", last < 0.02,
           f"(last refinement changed release/decel by {last*100:.2f}%)")

    # ---- 6 -----------------------------------------------------------------
    a1, _ = sc.rollout(NOMINAL_MASS, NOMINAL_MU)
    a2, _ = sc.rollout(NOMINAL_MASS, NOMINAL_MU)
    print("6) determinism")
    report("bitwise reproducible", np.array_equal(a1, a2),
           f"(max diff {np.abs(a1-a2).max():.2e})")

    # ---- 7 -----------------------------------------------------------------
    print("7) object stays on the table across the sweep corners")
    lim_x, lim_y = 0.60 - OBJ_HALF[0], 0.45 - OBJ_HALF[1]
    okf = True
    for mass in MASS_GRID[[0, 4]]:
        for mu in MU_GRID[[0, 4]]:
            s, _ = sc.rollout(mass, mu)
            my = np.abs(s[:, 1]).max()
            on = (np.abs(s[:, 0]).max() < lim_x) and (my < lim_y)
            okf &= bool(on)
            print(f"   m={mass:.2f} mu={mu:.2f}  travel "
                  f"{np.linalg.norm(s[-1,:2]-s[0,:2])*100:5.1f} cm  "
                  f"|y|max={my:.3f}  {'ok' if on else 'OFF TABLE'}")
    report("stays on table", okf)

    n_pass = sum(p for _, p in RESULTS)
    print("=" * 60)
    print(f"{n_pass}/{len(RESULTS)} checks passed")
    for name, p in RESULTS:
        print(f"   [{'x' if p else ' '}] {name}")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
