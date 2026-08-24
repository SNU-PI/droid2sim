"""Analytic checks for every scene family. A family that fails is not used.

Each family is swept over its own parameter and compared against the closed
form for its mechanism:

    slide     free-slide deceleration          a = mu * g
    roll      travel falls with rolling resistance, monotonically
    bounce    rebound height falls with contact damping, monotonically
    collide   the struck cube's speed follows the momentum split
    incline   down-ramp acceleration           a = g (sin t - mu cos t)
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "2")
from scenes import Scene, CTRL_DT, TABLE_TOP_Z, G

RAMP_T = 0.45          # must match the ramp euler in scenes.py
RES = []


def rep(n, ok, d=""):
    RES.append((n, bool(ok))); print(f"   -> {n}: {'PASS' if ok else 'FAIL'} {d}\n")


def fine_speed(sc, p, col=6):
    st, _ = sc.rollout(p)
    return st


def check_slide():
    print("slide: free-slide deceleration must equal mu*g  (1 ms trace)")
    sc = Scene("slide"); errs = []
    dt = sc.model.opt.timestep
    for mu in (0.15, 0.25, 0.40):
        for mass in (0.12, 0.35):
            _, _, tr = sc.rollout({"mass": mass, "mu": mu}, fine=True)
            v = tr[:, 0]; pk = int(np.argmax(v))
            sel = np.arange(pk + int(0.010 / dt), len(v))
            sel = sel[(v[sel] < 0.85 * v[pk]) & (v[sel] > 0.25 * v[pk])]
            if len(sel) < 20:
                errs.append(np.inf); continue
            a = -np.polyfit(sel * dt, v[sel], 1)[0]
            e = abs(a - mu * G) / (mu * G); errs.append(e)
            print(f"   mu={mu:.2f} m={mass:.2f}  a={a:6.3f}  mu*g={mu*G:6.3f}  err {e*100:5.1f}%")
    rep("slide obeys Coulomb", max(errs) < 0.12, f"(worst {max(errs)*100:.1f}%)")


def check_roll():
    print("roll: travel must fall monotonically with rolling resistance")
    sc = Scene("roll"); ds = []
    for rf in (0.0005, 0.0012, 0.003, 0.006):
        st, _ = sc.rollout({"mass": 0.2, "roll_fric": rf})
        d = np.linalg.norm(st[-1, :2] - st[0, :2]) * 100; ds.append(d)
        print(f"   roll_fric={rf:.4f}  travel={d:6.2f} cm")
    rep("rolling resistance monotone", all(np.diff(ds) < 0) and ds[0] - ds[-1] > 2.0,
        f"(range {ds[0]-ds[-1]:.1f} cm)")


def check_bounce():
    print("bounce: rebound height must fall with contact damping")
    sc = Scene("bounce"); hs = []
    for dmp in (10.0, 40.0, 120.0, 220.0):
        st, _ = sc.rollout({"mass": 0.2, "damping": dmp})
        z = st[:, 2] - (TABLE_TOP_Z + 0.035)
        land = int(np.argmin(z[:20]))
        h = float(z[land:].max()) * 100
        hs.append(h)
        print(f"   damping={dmp:6.1f}  rebound={h:6.2f} cm")
    rep("restitution monotone", all(np.diff(hs) < 0.02) and hs[0] - hs[-1] > 0.5,
        f"(range {hs[0]-hs[-1]:.2f} cm)")


def check_collide():
    print("collide: struck cube's speed must fall as its mass rises")
    sc = Scene("collide"); vs = []
    for m2 in (0.08, 0.18, 0.35, 0.60):
        st, _ = sc.rollout({"mass2": m2, "mu": 0.25})
        d2 = abs(st[-1, 7] - st[0, 7]) * 100
        vs.append(d2)
        print(f"   mass2={m2:.2f}  second cube moved {d2:6.2f} cm")
    rep("momentum split monotone", all(np.diff(vs) < 0) and vs[0] - vs[-1] > 1.0,
        f"(range {vs[0]-vs[-1]:.1f} cm)")


def check_incline():
    print(f"incline: along-ramp acceleration must equal g(sin t - mu cos t), t={RAMP_T} rad")
    sc = Scene("incline"); errs = []
    for mu in (0.12, 0.22, 0.35):
        st, _ = sc.rollout({"mass": 0.2, "mu": mu})
        v = st[:, 6]
        _, _, tr = sc.rollout({"mass": 0.2, "mu": mu}, fine=True)
        # speed ALONG the ramp: the horizontal component alone is short by
        # cos(theta), which showed up as a constant 10% deficit against theory
        v = np.hypot(tr[:, 0], tr[:, 1]); dt = sc.model.opt.timestep
        sel = np.arange(int(0.02 / dt), len(v))
        sel = sel[(v[sel] > 0.08) & (v[sel] < 0.85 * v.max())]
        if len(sel) < 20:
            print(f"   mu={mu:.2f}  did not slide"); errs.append(np.inf); continue
        a = np.polyfit(sel * dt, v[sel], 1)[0]
        th = G * (np.sin(RAMP_T) - mu * np.cos(RAMP_T))
        e = abs(a - th) / th; errs.append(e)
        print(f"   mu={mu:.2f}  a={a:6.3f}  theory={th:6.3f}  err {e*100:5.1f}%")
    rep("incline obeys the inclined-plane law", max(errs) < 0.20,
        f"(worst {max(errs)*100:.1f}%)")


if __name__ == "__main__":
    for fn in (check_slide, check_roll, check_bounce, check_collide, check_incline):
        try:
            fn()
        except Exception as e:
            print(f"   EXCEPTION {type(e).__name__}: {e}\n"); RES.append((fn.__name__, False))
    print("=" * 56)
    print(f"{sum(o for _, o in RES)}/{len(RES)} families verified")
    for n, o in RES:
        print(f"   [{'x' if o else ' '}] {n}")
