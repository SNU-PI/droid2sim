"""A rigid pendulum given an initial push at its lowest point."""

import numpy as np
import mujoco

from core.threshold.base import Base, G, wrap


class Pendulum(Base):
    """Conservative swing-over threshold with a visible arm length and speed."""

    cam = "pendulum_side"
    n_frames = 12
    settle_steps = 0
    capture_dt = 1 / 16
    LENGTH = 0.20
    BALL_R = 0.03
    PIVOT_Z = 0.64

    def __init__(self, p0=None):
        self.p = p0 or self.params_for_margin(0.20)
        super().__init__()

    @classmethod
    def params_for_margin(cls, margin, length=None):
        arm = cls.LENGTH if length is None else float(length)
        inertia_per_mass = arm ** 2 + 0.4 * cls.BALL_R ** 2
        barrier_per_mass = 2 * G * arm
        v0 = arm * np.sqrt(
            2 * (1 + margin) * barrier_per_mass / inertia_per_mass
        )
        return dict(v0=float(v0), length=arm)

    @classmethod
    def normalized_margin_of(cls, p):
        arm = p.get("length", cls.LENGTH)
        inertia_per_mass = arm ** 2 + 0.4 * cls.BALL_R ** 2
        kinetic_per_mass = 0.5 * inertia_per_mass * (p["v0"] / arm) ** 2
        barrier_per_mass = 2 * G * arm
        return kinetic_per_mass / barrier_per_mass - 1

    margin_of = normalized_margin_of

    def xml(self):
        arm = self.p.get("length", self.LENGTH)
        body = f"""
    <camera name="pendulum_side" pos="0 -1.05 0.61"
            xyaxes="1 0 0 0 0.10 0.995"/>
    <geom name="stand_left" type="box" size="0.018 0.035 0.29"
          pos="-0.25 0 0.35" material="dark" contype="0" conaffinity="0"/>
    <geom name="stand_right" type="box" size="0.018 0.035 0.29"
          pos="0.25 0 0.35" material="dark" contype="0" conaffinity="0"/>
    <geom name="stand_top" type="box" size="0.27 0.035 0.018"
          pos="0 0 {self.PIVOT_Z + 0.025}" material="dark" contype="0" conaffinity="0"/>
    <body name="pendulum" pos="0 0 {self.PIVOT_Z}">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0"/>
      <geom name="arm" type="capsule" fromto="0 0 0 0 0 {-arm}"
            size="0.006" material="grey" mass="0.000001"
            contype="0" conaffinity="0"/>
      <geom name="bob" type="sphere" size="{self.BALL_R}" pos="0 0 {-arm}"
            material="red" mass="0.1" contype="0" conaffinity="0"/>
    </body>
"""
        return wrap("energy_pendulum", body)

    def set_params(self, p):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.data = mujoco.MjData(self.model)
        self._r = None

    def init_state(self, p):
        arm = p.get("length", self.LENGTH)
        self.data.qvel[0] = p["v0"] / arm

    def observe(self):
        return [self.data.qpos[0], self.data.qvel[0]]

    def labels(self, p, trace):
        angle = trace[:, 0]
        completed = angle >= np.pi + 0.02
        outcome = int(completed.any())
        event = int(np.argmax(completed)) if completed.any() else self.n_frames - 1
        return dict(
            margin=float(self.normalized_margin_of(p)),
            outcome=outcome,
            event_frame=event,
            aux=float(angle.max()),
        )
