"""Leaning rod: the classic ladder problem.

Rod between a frictional floor and a frictionless wall. It holds iff
tan(theta) > 1/(2*mu). Floor friction is fixed, so the threshold is a pure
angle -- readable from geometry alone."""

import numpy as np
import mujoco

from core.threshold.base import (Base, wrap, TABLE, TABLE_Z, DENSITY, G)


class Lean(Base):
    """Rod between floor and a frictionless wall. Floor friction fixed at 0.5,
    so the threshold angle is atan(1/(2*0.5)) = 45 deg, visible as geometry."""

    cam = "side"
    n_frames = 40
    settle_steps = 0
    MU = 0.5
    L = 0.14                     # capsule half-length
    R = 0.012
    WALL_X = 0.20

    def __init__(self, p0=None):
        self.p = p0 or dict(theta=0.9)
        super().__init__()

    def xml(self):
        th = self.p["theta"]
        cx = self.WALL_X - self.L * np.cos(th) - self.R
        cz = TABLE_Z + self.L * np.sin(th) + self.R
        body = TABLE + f"""
    <body name="wall" pos="{self.WALL_X + 0.02} 0 {TABLE_Z + 0.20}">
      <geom name="wallg" type="box" size="0.02 0.16 0.20" material="dark"/>
    </body>
    <body name="rod" pos="{cx:.5f} 0 {cz:.5f}" euler="0 {np.pi/2 - th:.5f} 0">
      <freejoint/>
      <geom name="rodg" type="capsule" size="{self.R} {self.L}" material="red" mass="0.12"/>
    </body>
"""
        contact = f"""
  <contact>
    <pair geom1="rodg" geom2="table_top" condim="3"
          friction="{self.MU} {self.MU} 0.002 0.0001 0.0001"
          solref="0.008 1" solimp="0.95 0.95 0.001"/>
    <pair geom1="rodg" geom2="wallg" condim="3"
          friction="0.0001 0.0001 0.0001 0.0001 0.0001"
          solref="0.008 1" solimp="0.95 0.95 0.001"/>
  </contact>
"""
        return wrap("lean", body, contact)

    def set_params(self, p):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.data = mujoco.MjData(self.model)
        self._r = None

    def observe(self):
        return list(self.data.qpos[0:3])

    @staticmethod
    def margin_of(p, mu=0.5):
        return p["theta"] - np.arctan(1.0 / (2 * mu))

    def labels(self, p, trace):
        slide = np.linalg.norm(trace[:, :2] - trace[0, :2], axis=1)
        outcome = int(slide[-1] < 0.02)               # 1 = holds
        moved = slide > 0.01
        event = int(np.argmax(moved)) if moved.any() else self.n_frames - 1
        return dict(margin=self.margin_of(p), outcome=outcome,
                    event_frame=event, aux=float(slide[-1]))

    @staticmethod
    def sample(rng):
        return dict(theta=rng.uniform(0.55, 1.05))
