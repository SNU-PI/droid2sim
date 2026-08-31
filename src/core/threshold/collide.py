"""Two-ball collision on a near-frictionless floor.

Equal density, different radii, so the mass ratio is visible as size. The
light ball bounces back iff the restitution exceeds mA/mB."""

import numpy as np
import mujoco

from core.threshold.base import (Base, wrap, TABLE, TABLE_Z, DENSITY, G)


class Collide(Base):
    """Light ball slides into a heavier resting ball on a near-frictionless
    floor. Equal-density spheres, different radii, so the mass ratio is
    visible as size. Bounces back iff e > mA/mB."""

    cam = "side"
    n_frames = 50
    settle_steps = 0
    RA, RB = 0.030, 0.038
    V0 = 0.8

    def __init__(self, p0=None):
        self.p = p0 or dict(damp=40.0)
        super().__init__()

    def xml(self):
        rho = 800.0
        mA = rho * 4 / 3 * np.pi * self.RA ** 3
        mB = rho * 4 / 3 * np.pi * self.RB ** 3
        d = self.p["damp"]
        body = f"""
    <body name="A" pos="-0.25 0 {self.RA + 0.001}">
      <freejoint/><geom name="gA" type="sphere" size="{self.RA}" material="red" mass="{mA:.5f}"/>
    </body>
    <body name="B" pos="0 0 {self.RB + 0.001}">
      <freejoint/><geom name="gB" type="sphere" size="{self.RB}" material="blue" mass="{mB:.5f}"/>
    </body>
"""
        contact = f"""
  <contact>
    <pair geom1="gA" geom2="floor" condim="3" friction="0.0002 0.0002 0.0001 0.0001 0.0001"
          solref="0.006 1" solimp="0.95 0.95 0.001"/>
    <pair geom1="gB" geom2="floor" condim="3" friction="0.0002 0.0002 0.0001 0.0001 0.0001"
          solref="0.006 1" solimp="0.95 0.95 0.001"/>
    <pair geom1="gA" geom2="gB" condim="1" solref="-12000 -{d:.2f}"
          solimp="0.95 0.95 0.001"/>
  </contact>
"""
        return wrap("collide", body, contact)

    def set_params(self, p):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.data = mujoco.MjData(self.model)
        self._r = None

    def init_state(self, p):
        self.data.qvel[0] = self.V0

    def observe(self):
        return [self.data.qpos[0], self.data.qvel[0],
                self.data.qpos[7], self.data.qvel[6]]

    def mass_ratio(self):
        return (self.RA / self.RB) ** 3

    def labels(self, p, trace):
        vA, vB = trace[-1, 1], trace[-1, 3]
        v_before = self.V0
        e = (vB - vA) / v_before
        outcome = int(vA < -0.01)                     # 1 = A bounced back
        hit = trace[:, 3] > 0.02
        event = int(np.argmax(hit)) if hit.any() else self.n_frames - 1
        return dict(margin=float(e - self.mass_ratio()), outcome=outcome,
                    event_frame=event, aux=float(e))

    @staticmethod
    def sample(rng):
        return dict(damp=float(np.exp(rng.uniform(np.log(5), np.log(300)))))
