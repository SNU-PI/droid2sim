"""Domino chain, spacing swept.

The chain either propagates to the last domino or dies. There is no clean
closed form, so the threshold is measured by bisection and the gate checks
monotonicity instead."""

import numpy as np
import mujoco

from core.threshold.base import (Base, wrap, TABLE, TABLE_Z, DENSITY, G)


class Domino(Base):
    """Five dominoes; the first is tipped. Sweep the spacing: the chain either
    propagates to the last domino or dies. No closed form; the empirical
    threshold comes from bisection and the gate checks monotonicity."""

    n_frames = 60
    settle_steps = 0
    HX, HY, HZ = 0.004, 0.018, 0.036
    N = 5

    def __init__(self, p0=None):
        self.p = p0 or dict(s=0.07)
        super().__init__()

    def xml(self):
        s = self.p["s"]
        z = TABLE_Z + self.HZ
        blocks, pairs = [], []
        for i in range(self.N):
            mat = "red" if i == 0 else ("blue" if i == self.N - 1 else "grey")
            blocks.append(f"""
    <body name="d{i}" pos="{i * s - 0.15:.5f} 0 {z}">
      <freejoint/>
      <geom name="dg{i}" type="box" size="{self.HX} {self.HY} {self.HZ}"
            material="{mat}" mass="0.02"/>
    </body>""")
            pairs.append(f'    <pair geom1="dg{i}" geom2="table_top" condim="3" '
                         f'friction="0.6 0.6 0.004 0.0001 0.0001" '
                         f'solref="0.006 1" solimp="0.95 0.95 0.001"/>')
            if i:
                pairs.append(f'    <pair geom1="dg{i-1}" geom2="dg{i}" condim="3" '
                             f'friction="0.3 0.3 0.002 0.0001 0.0001" '
                             f'solref="0.006 1" solimp="0.95 0.95 0.001"/>')
        return wrap("domino", TABLE + "\n".join(blocks),
                     "  <contact>\n" + "\n".join(pairs) + "\n  </contact>\n")

    def set_params(self, p):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.data = mujoco.MjData(self.model)
        self._r = None

    def init_state(self, p):
        # start the first domino already past its own tipping angle, rotating
        # about +y; qpos layout per free joint is [x y z qw qx qy qz]
        th = 0.42
        self.data.qpos[3] = np.cos(th / 2)
        self.data.qpos[5] = np.sin(th / 2)
        self.data.qpos[2] = TABLE_Z + self.HZ * np.cos(th) + self.HX * np.sin(th)
        self.data.qvel[4] = 2.5

    def observe(self):
        out = []
        for i in range(self.N):
            q = self.data.qpos[i * 7 + 3:i * 7 + 7]
            pitch = 2 * np.arcsin(np.clip(abs(q[2]), 0, 1))
            out.append(pitch)
        return out

    def labels(self, p, trace):
        last = trace[-1, self.N - 1]
        outcome = int(last > 1.0)                      # 1 = chain completed
        moved = trace[:, self.N - 1] > 0.15
        event = int(np.argmax(moved)) if moved.any() else self.n_frames - 1
        return dict(margin=float(0.064 - p["s"]), outcome=outcome,
                    event_frame=event, aux=float(last))

    @staticmethod
    def sample(rng):
        return dict(s=rng.uniform(0.040, 0.088))
