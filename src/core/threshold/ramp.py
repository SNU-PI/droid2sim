"""A frictionless slider climbing a visible ramp toward a goal gate."""

import numpy as np
import mujoco

from core.threshold.base import Base, G, wrap


class Ramp(Base):
    """One-DoF conservative ramp: succeeds iff 1/2 v^2 exceeds g h."""

    cam = "side"
    n_frames = 12
    settle_steps = 0
    capture_dt = 1 / 16
    LENGTH = 0.62
    HEIGHT = 0.14
    BALL_R = 0.03
    START_X = -0.46
    START_Z = 0.055

    def __init__(self, p0=None):
        self.p = p0 or self.params_for_margin(0.20)
        super().__init__()

    @classmethod
    def params_for_margin(cls, margin, h=None):
        height = cls.HEIGHT if h is None else float(h)
        return dict(v0=float(np.sqrt(2 * G * height * (1 + margin))), h=height)

    @staticmethod
    def normalized_margin_of(p):
        return 0.5 * p["v0"] ** 2 / (G * p["h"]) - 1

    margin_of = normalized_margin_of

    def _geometry(self):
        h = self.p["h"]
        horizontal = np.sqrt(self.LENGTH ** 2 - h ** 2)
        ux, uz = horizontal / self.LENGTH, h / self.LENGTH
        nx, nz = -uz, ux
        offset = self.BALL_R + 0.009
        start = np.array([
            self.START_X + nx * offset,
            0.0,
            self.START_Z + nz * offset,
        ])
        end = np.array([self.START_X + horizontal, 0.0, self.START_Z + h])
        return ux, uz, start, end

    def xml(self):
        ux, uz, start, end = self._geometry()
        gate = end + np.array([0.0, 0.0, 0.055])
        body = f"""
    <geom name="rampg" type="capsule"
          fromto="{self.START_X} 0 {self.START_Z} {end[0]:.6f} 0 {end[2]:.6f}"
          size="0.009" material="grey" contype="0" conaffinity="0"/>
    <geom name="gate" type="box" size="0.008 0.055 0.055"
          pos="{gate[0]:.6f} 0 {gate[2]:.6f}" material="blue"
          contype="0" conaffinity="0"/>
    <body name="slider" pos="{start[0]:.6f} 0 {start[2]:.6f}">
      <joint name="ramp_slide" type="slide" axis="{ux:.8f} 0 {uz:.8f}" damping="0"/>
      <geom name="sliderg" type="sphere" size="{self.BALL_R}"
            material="red" mass="0.1" contype="0" conaffinity="0"/>
    </body>
"""
        return wrap("energy_ramp", body)

    def set_params(self, p):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.data = mujoco.MjData(self.model)
        self._r = None

    def init_state(self, p):
        self.data.qvel[0] = p["v0"]

    def observe(self):
        return [self.data.qpos[0], self.data.qvel[0]]

    def labels(self, p, trace):
        position = trace[:, 0]
        reached = position >= self.LENGTH
        outcome = int(reached.any())
        event = int(np.argmax(reached)) if reached.any() else self.n_frames - 1
        return dict(
            margin=float(self.normalized_margin_of(p)),
            outcome=outcome,
            event_frame=event,
            aux=float(position.max()),
        )
