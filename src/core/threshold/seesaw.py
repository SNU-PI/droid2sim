"""Seesaw: two same-material cubes on a hinged plank.

The only degree of freedom is the hinge, so the outcome is purely the sign of
the net torque -- no sliding dynamics mixed in. Both cubes share a density and
a colour, so size and lever arm are the whole story and both are visible."""

import numpy as np
import mujoco

from core.threshold.base import (Base, wrap, TABLE, TABLE_Z, DENSITY, G)


def _cube(name, half, x, mat):
    m = DENSITY * (2 * half) ** 3
    return (f'<geom name="{name}" type="box" size="{half} {half} {half}" '
            f'pos="{x} 0 {half + 0.008}" material="{mat}" mass="{m:.5f}"/>')


class Seesaw(Base):
    """Plank on a hinge; two same-material cubes at different arms and sizes.

    The cubes are welded to the plank so the only degree of freedom is the
    hinge: outcome is purely the sign of the net torque, with no sliding
    dynamics mixed in. margin is the normalised net torque.
    """

    n_frames = 40
    settle_steps = 0

    def __init__(self, p0=None):
        self.p = p0 or dict(s1=0.028, d1=0.15, s2=0.022, d2=0.22)
        super().__init__()

    def xml(self):
        p = self.p
        body = TABLE + f"""
    <body name="stand" pos="0 0 {TABLE_Z}">
      <geom name="standg" type="box" size="0.012 0.05 0.035" pos="0 0 0.035" material="dark" contype="0" conaffinity="0"/>
    </body>
    <body name="plank" pos="0 0 {TABLE_Z + 0.078}">
      <joint name="tilt" type="hinge" axis="0 1 0" damping="0.005"/>
      <geom name="plankg" type="box" size="0.30 0.05 0.008" material="grey" mass="0.25"/>
      {_cube("c1", p["s1"], -p["d1"], "red")}
      {_cube("c2", p["s2"], p["d2"], "red")}
    </body>
"""
        return wrap("seesaw", body)

    def set_params(self, p):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.data = mujoco.MjData(self.model)
        self._r = None

    def observe(self):
        return [self.data.qpos[0]]

    @staticmethod
    def margin_of(p):
        m1 = DENSITY * (2 * p["s1"]) ** 3
        m2 = DENSITY * (2 * p["s2"]) ** 3
        t1, t2 = m1 * p["d1"], m2 * p["d2"]
        return (t2 - t1) / (t2 + t1)

    def labels(self, p, trace):
        ang = trace[:, 0]
        outcome = int(ang[-1] > 0)                    # 1 = tips toward cube 2 (+x)
        moved = np.abs(ang) > 0.05
        event = int(np.argmax(moved)) if moved.any() else self.n_frames - 1
        return dict(margin=self.margin_of(p), outcome=outcome,
                    event_frame=event, aux=float(ang[-1]))

    @staticmethod
    def sample(rng):
        return dict(s1=rng.uniform(0.018, 0.032), d1=rng.uniform(0.10, 0.26),
    s2=rng.uniform(0.018, 0.032), d2=rng.uniform(0.10, 0.26))
