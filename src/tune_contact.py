"""Isolate sliding friction and find contact settings that obey Coulomb's law.

No pusher, no collision: the cube is simply given an initial velocity and left
to slide. Deceleration must come out at mu*g for every mass. Anything else
means the contact model, not the physics we intend, is setting the friction.

The suspect is solimp's width parameter. MuJoCo ramps contact impedance over
the first `width` metres of penetration, and a heavier object settles deeper,
so with a wide ramp the effective contact stiffness -- and with it the friction
impulse -- becomes a function of mass. Holding impedance constant (d0 == dmax)
should remove that coupling.
"""

import sys, os
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "2")

G = 9.81
HALF = 0.035
TOP = 0.40

TEMPLATE = """
<mujoco>
  <option timestep="{dt}" integrator="implicitfast" gravity="0 0 -9.81"
          cone="{cone}" impratio="{impratio}" iterations="200" ls_iterations="50"
          noslip_iterations="{noslip}" tolerance="1e-12"/>
  <worldbody>
    <geom name="table" type="box" size="1.2 1.2 0.02" pos="0 0 {tz}"/>
    <body name="obj" pos="0 0 {oz}">
      <freejoint name="f"/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 1e-4 1e-4"/>
      <geom name="obj" type="box" size="{h} {h} {h}" condim="3"/>
    </body>
  </worldbody>
  <contact>
    <pair name="p" geom1="obj" geom2="table" condim="3"
          friction="0.3 0.3 0.004 0.0001 0.0001"
          solref="{solref}" solimp="{solimp}"/>
  </contact>
</mujoco>
"""


def build(dt=0.001, cone="elliptic", impratio=10, noslip=0,
          solref="0.010 1", solimp="0.90 0.96 0.002"):
    xml = TEMPLATE.format(dt=dt, cone=cone, impratio=impratio, noslip=noslip,
                          solref=solref, solimp=solimp,
                          tz=TOP - 0.02, oz=TOP + HALF, h=HALF)
    return mujoco.MjModel.from_xml_string(xml)


def slide_decel(model, mass, mu, v0=0.5, T=0.5):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj")
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, "p")
    model.body_mass[bid] = mass
    i = mass * (2 * (2 * HALF) ** 2) / 12.0
    model.body_inertia[bid] = [i, i, i]
    model.pair_friction[pid, 0] = mu
    model.pair_friction[pid, 1] = mu

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[2] = TOP + HALF
    mujoco.mj_forward(model, data)
    # let it settle so we measure steady sliding, not a settling transient
    for _ in range(int(0.10 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    data.qvel[1] = v0
    pen0 = TOP + HALF - data.qpos[2]

    ts, vs = [], []
    n = int(T / model.opt.timestep)
    for k in range(n):
        mujoco.mj_step(model, data)
        v = data.qvel[1]
        if v < 0.05:
            break
        ts.append(k * model.opt.timestep)
        vs.append(v)
    if len(vs) < 20:
        return np.nan, pen0
    a = -np.polyfit(ts, vs, 1)[0]
    return a, pen0


CONFIGS = [
    ("baseline (current)",        dict()),
    ("constant impedance",        dict(solimp="0.95 0.95 0.001")),
    ("const imp + stiff",         dict(solref="0.004 1", solimp="0.95 0.95 0.001")),
    ("const imp + very stiff",    dict(solref="0.002 1", solimp="0.98 0.98 0.001")),
    ("const imp, pyramidal",      dict(cone="pyramidal", solimp="0.95 0.95 0.001")),
    ("const imp, impratio=1",     dict(impratio=1, solimp="0.95 0.95 0.001")),
    ("const imp + noslip 10",     dict(noslip=10, solimp="0.95 0.95 0.001")),
]

MASSES = (0.08, 0.20, 0.50)
MUS = (0.10, 0.30, 0.60)

print(f"target: a = mu*g, identical for every mass "
      f"(mu=0.1 -> {0.1*G:.3f}, 0.3 -> {0.3*G:.3f}, 0.6 -> {0.6*G:.3f})\n")
print(f"{'config':26s} {'mu':>5} " + "".join(f"{f'm={m}':>9}" for m in MASSES)
      + f"{'max err':>9}{'pen(mm)':>9}")

best, best_err = None, 1e9
for name, kw in CONFIGS:
    model = build(**kw)
    worst = 0.0
    lines = []
    pen_report = 0.0
    for mu in MUS:
        row, errs = [], []
        for m in MASSES:
            a, pen = slide_decel(model, m, mu)
            row.append(a)
            errs.append(abs(a - mu * G) / (mu * G) if np.isfinite(a) else np.inf)
            pen_report = max(pen_report, pen)
        worst = max(worst, max(errs))
        lines.append((mu, row, max(errs)))
    for mu, row, e in lines:
        print(f"{name:26s} {mu:5.2f} " + "".join(f"{v:9.3f}" for v in row)
              + f"{e*100:8.1f}%{pen_report*1000:9.3f}")
    print(f"{'':26s} {'WORST':>5} {'':27s}{worst*100:8.1f}%")
    if worst < best_err:
        best, best_err = name, worst
    print()

print(f"best config: {best}  (worst error {best_err*100:.1f}%)")
