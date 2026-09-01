"""A bead constrained to a visible circular hill.

The bead follows the grey arc without contact loss.  It crosses the crest iff
its initial kinetic energy exceeds the gravitational barrier, making this a
clean conservative counterpart to the contact-rich rolling-ball Hill scene.
"""

import numpy as np
import mujoco

from core.threshold.base import Base, G, wrap


class EnergyHill(Base):
    """Conservative bead-on-arc threshold with every causal variable visible."""

    cam = "side"
    n_frames = 12
    settle_steps = 0
    capture_dt = 1 / 16
    R = 0.34                  # bead-centre path radius
    BALL_R = 0.03
    TRACK_R = R - BALL_R - 0.007
    PIVOT_Z = 0.055
    THETA0 = 0.90

    def __init__(self, p0=None):
        self.p = p0 or self.params_for_margin(0.20)
        super().__init__()

    @classmethod
    def params_for_margin(cls, margin, theta0=None):
        theta = cls.THETA0 if theta0 is None else float(theta0)
        inertia_per_mass = cls.R ** 2 + 0.4 * cls.BALL_R ** 2
        barrier_per_mass = G * cls.R * (1 - np.cos(theta))
        v0 = cls.R * np.sqrt(
            2 * (1 + margin) * barrier_per_mass / inertia_per_mass
        )
        return dict(v0=float(v0), theta0=theta)

    @classmethod
    def normalized_margin_of(cls, p):
        theta = p.get("theta0", cls.THETA0)
        inertia_per_mass = cls.R ** 2 + 0.4 * cls.BALL_R ** 2
        kinetic_per_mass = 0.5 * inertia_per_mass * (p["v0"] / cls.R) ** 2
        barrier_per_mass = G * cls.R * (1 - np.cos(theta))
        return kinetic_per_mass / barrier_per_mass - 1

    margin_of = normalized_margin_of

    @classmethod
    def _track_geoms(cls):
        angles = np.linspace(-1.45, 1.45, 31)
        points = [
            (cls.TRACK_R * np.sin(a), cls.PIVOT_Z + cls.TRACK_R * np.cos(a))
            for a in angles
        ]
        geoms = []
        for i, ((x1, z1), (x2, z2)) in enumerate(zip(points[:-1], points[1:])):
            geoms.append(
                f'<geom name="track{i}" type="capsule" '
                f'fromto="{x1:.6f} 0 {z1:.6f} {x2:.6f} 0 {z2:.6f}" '
                f'size="0.007" material="grey" contype="0" conaffinity="0"/>'
            )
        return "\n".join(geoms)

    def xml(self):
        body = f"""
    {self._track_geoms()}
    <body name="bead" pos="0 0 {self.PIVOT_Z}">
      <joint name="arc" type="hinge" axis="0 1 0" damping="0"/>
      <geom name="beadg" type="sphere" size="{self.BALL_R}" pos="0 0 {self.R}"
            material="red" mass="0.1" contype="0" conaffinity="0"/>
    </body>
"""
        return wrap("energy_hill", body)

    def set_params(self, p):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.data = mujoco.MjData(self.model)
        self._r = None

    def init_state(self, p):
        self.data.qpos[0] = -p.get("theta0", self.THETA0)
        self.data.qvel[0] = p["v0"] / self.R

    def observe(self):
        return [self.data.qpos[0], self.data.qvel[0]]

    def labels(self, p, trace):
        angle = trace[:, 0]
        crossed = angle > 0.02
        outcome = int(crossed.any())
        event = int(np.argmax(crossed)) if crossed.any() else self.n_frames - 1
        return dict(
            margin=float(self.normalized_margin_of(p)),
            outcome=outcome,
            event_frame=event,
            aux=float(angle.max()),
        )
