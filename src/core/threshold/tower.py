"""Three-cube tower with lateral offsets.

Stands iff every partial centre of mass stays over its support. margin is the
tighter of the two interface margins."""

import numpy as np
import mujoco

from core.threshold.base import (Base, wrap, TABLE, TABLE_Z, DENSITY, G)


class Tower(Base):
    """Three cubes with lateral offsets; stands iff every partial CoM stays
    over its support. margin = the smallest of the two interface margins."""

    n_frames = 44
    settle_steps = 0
    W = 0.03

    def __init__(self, p0=None):
        self.p = p0 or dict(o1=0.01, o2=0.01)
        super().__init__()

    def xml(self):
        w, (o1, o2) = self.W, (self.p["o1"], self.p["o2"])
        z0 = TABLE_Z + w
        body = TABLE + f"""
    <body name="b1" pos="0 0 {z0}">
      <freejoint/><geom name="g1" type="box" size="{w} {w} {w}" material="red" mass="0.15"/>
    </body>
    <body name="b2" pos="{o1} 0 {z0 + 2*w + 0.0004}">
      <freejoint/><geom name="g2" type="box" size="{w} {w} {w}" material="blue" mass="0.15"/>
    </body>
    <body name="b3" pos="{o1 + o2} 0 {z0 + 4*w + 0.0008}">
      <freejoint/><geom name="g3" type="box" size="{w} {w} {w}" material="grey" mass="0.15"/>
    </body>
"""
        contact = """
  <contact>
    <pair geom1="g1" geom2="table_top" condim="3" friction="0.8 0.8 0.004 0.0001 0.0001"
          solref="0.008 1" solimp="0.95 0.95 0.001"/>
    <pair geom1="g2" geom2="g1" condim="3" friction="0.8 0.8 0.004 0.0001 0.0001"
          solref="0.008 1" solimp="0.95 0.95 0.001"/>
    <pair geom1="g3" geom2="g2" condim="3" friction="0.8 0.8 0.004 0.0001 0.0001"
          solref="0.008 1" solimp="0.95 0.95 0.001"/>
  </contact>
"""
        return wrap("tower", body, contact)

    def set_params(self, p):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.data = mujoco.MjData(self.model)
        self._r = None

    def observe(self):
        return [self.data.qpos[14], self.data.qpos[16]]      # top cube x, z

    @staticmethod
    def margin_of(p, w=0.03):
        m_top = w - abs(p["o2"])
        m_bot = w - abs(p["o1"] + p["o2"] / 2)
        return min(m_top, m_bot)

    def labels(self, p, trace):
        z = trace[:, 1]
        outcome = int(z[-1] > z[0] - 0.02)            # 1 = stands
        moved = np.abs(z - z[0]) > 0.005
        event = int(np.argmax(moved)) if moved.any() else self.n_frames - 1
        return dict(margin=self.margin_of(p), outcome=outcome,
                    event_frame=event, aux=float(z[-1] - z[0]))

    @staticmethod
    def sample(rng):
        return dict(o1=rng.uniform(-0.028, 0.028), o2=rng.uniform(-0.036, 0.036))
