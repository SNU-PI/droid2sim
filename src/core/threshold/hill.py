"""Ball rolling at a smooth ridge.

Rolling without slip, it crosses iff (7/10) v0^2 > g h. Mass cancels
completely, so the threshold depends only on two visible quantities: how fast
the ball moves and how tall the ridge is."""

import numpy as np
import mujoco

from core.threshold.base import (Base, wrap, TABLE, TABLE_Z, DENSITY, G)


class Hill(Base):
    """Ball rolls at v0 toward a smooth gaussian ridge of height h.
    Rolling without slip: crosses iff (7/10) v0^2 > g h. Mass cancels."""

    cam = "side"
    n_frames = 60
    settle_steps = 0
    NR = 80

    def __init__(self, p0=None):
        self.p = p0 or dict(v0=0.9, h=0.05)
        super().__init__()

    def _hfield(self):
        x = np.linspace(-1, 1, self.NR)
        prof = np.exp(-x ** 2 / (2 * 0.18 ** 2))
        return np.tile(prof, (2, 1))

    def xml(self):
        h = self.p["h"]
        x0 = self.p.get("x0", -0.45)
        extra = (f'<hfield name="hill" nrow="2" ncol="{self.NR}" '
                 f'size="0.60 0.30 {h} 0.001"/>')
        body = f"""
    <geom name="hillg" type="hfield" hfield="hill" pos="0 0 0.001" material="grey"/>
    <body name="ball" pos="{x0} 0 0.0315">
      <freejoint/>
      <geom name="ballg" type="sphere" size="0.03" material="red" mass="0.1"/>
    </body>
"""
        contact = """
  <contact>
    <pair geom1="ballg" geom2="floor" condim="3" friction="0.8 0.8 0.004 0.0001 0.0001"
          solref="0.006 1" solimp="0.95 0.95 0.001"/>
    <pair geom1="ballg" geom2="hillg" condim="3" friction="0.8 0.8 0.004 0.0001 0.0001"
          solref="0.006 1" solimp="0.95 0.95 0.001"/>
  </contact>
"""
        return wrap("hill", body, contact, extra_asset=extra)

    def set_params(self, p):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.model.hfield_data[:] = self._hfield().ravel()
        self.data = mujoco.MjData(self.model)
        self._r = None

    def init_state(self, p):
        self.data.qvel[0] = p["v0"]
        self.data.qvel[4] = p["v0"] / 0.03            # rolling spin about y

    def observe(self):
        return [self.data.qpos[0], self.data.qvel[0]]

    @staticmethod
    def margin_of(p):
        return 0.7 * p["v0"] ** 2 - G * p["h"]

    def labels(self, p, trace):
        x = trace[:, 0]
        outcome = int(x[-1] > 0.25)                   # 1 = crossed
        near = np.abs(x) < 0.19
        event = int(np.argmax(near)) if near.any() else self.n_frames - 1
        return dict(margin=self.margin_of(p), outcome=outcome,
                    event_frame=event, aux=float(x[-1]))

    @staticmethod
    def sample(rng):
        return dict(v0=rng.uniform(0.45, 1.30), h=rng.uniform(0.02, 0.09))
