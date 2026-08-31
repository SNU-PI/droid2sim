"""Shared scaffolding for the threshold scenes.

Each scene lives in its own module and subclasses Base; this file holds only
what all of them need: the common XML preamble, the table and camera blocks,
and the rollout loop that settles the scene, captures frames, and hands the
trace to the scene's own labelling code.
"""

import numpy as np
import mujoco

G = 9.81
IMG = 256
CTRL_DT = 0.02

HEAD = f"""
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" integrator="implicitfast"
          cone="elliptic" impratio="10" gravity="0 0 -9.81"
          iterations="150" ls_iterations="50" tolerance="1e-10"/>
  <visual>
    <global offwidth="{IMG}" offheight="{IMG}" azimuth="140" elevation="-25"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="grad" type="skybox" builtin="gradient"
             rgb1="0.42 0.45 0.50" rgb2="0.16 0.18 0.21" width="256" height="256"/>
    <texture name="woodtex" type="2d" builtin="flat" rgb1="0.62 0.47 0.32"
             width="512" height="512" mark="random" markrgb="0.55 0.41 0.27" random="0.25"/>
    <material name="wood" texture="woodtex" texrepeat="2 2" specular="0.15" shininess="0.25"/>
    <material name="floor" rgba="0.28 0.29 0.32 1" specular="0.05"/>
    <material name="red"   rgba="0.80 0.28 0.22 1" specular="0.35" shininess="0.4"/>
    <material name="blue"  rgba="0.22 0.42 0.72 1" specular="0.35" shininess="0.4"/>
    <material name="grey"  rgba="0.55 0.57 0.60 1" specular="0.3"/>
    <material name="dark"  rgba="0.24 0.25 0.28 1" specular="0.1"/>
  </asset>
"""

LIGHTS = """
    <light name="key"  pos="0.45 -0.55 1.65" dir="-0.3 0.35 -1" directional="false"
           diffuse="0.75 0.73 0.70" specular="0.25 0.25 0.25" castshadow="true"/>
    <light name="fill" pos="-0.6 0.5 1.4" dir="0.4 -0.3 -1" directional="false"
           diffuse="0.28 0.30 0.34" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor"/>
"""

TABLE_Z = 0.40
TABLE = f"""
    <body name="table" pos="0 0 {TABLE_Z - 0.02}">
      <geom name="table_top" type="box" size="0.60 0.45 0.02" material="wood"/>
      <geom name="leg1" type="box" size="0.03 0.03 0.18" pos=" 0.55  0.40 -0.20" material="dark"/>
      <geom name="leg2" type="box" size="0.03 0.03 0.18" pos="-0.55  0.40 -0.20" material="dark"/>
      <geom name="leg3" type="box" size="0.03 0.03 0.18" pos=" 0.55 -0.40 -0.20" material="dark"/>
      <geom name="leg4" type="box" size="0.03 0.03 0.18" pos="-0.55 -0.40 -0.20" material="dark"/>
    </body>
"""

CAM = """
    <camera name="close" pos="0.396 -0.375 0.702" xyaxes="0.74 0.68 0 -0.26 0.28 0.92"/>
    <camera name="side"  pos="0.0 -0.95 0.62" xyaxes="1 0 0 0 0.25 0.97"/>
"""

DENSITY = 600.0


def wrap(name, body, contact="", extra_asset=""):
    return (f'<mujoco model="{name}">' + HEAD.replace("</asset>", extra_asset + "  </asset>")
            + "  <worldbody>" + LIGHTS + body + CAM + "  </worldbody>\n"
            + contact + "</mujoco>")


class Base:
    """Shared rollout machinery: settle, frame capture, warm-started renderer."""

    cam = "close"
    n_frames = 40
    settle_steps = 300           # 0.3 s at dt=1ms before frame 0
    capture_dt = CTRL_DT
    render_height = IMG
    render_width = IMG

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(self.xml())
        self.data = mujoco.MjData(self.model)
        self._r = None

    def render(self):
        if self._r is None:
            self.model.vis.global_.offwidth = max(
                self.model.vis.global_.offwidth,
                self.render_width,
            )
            self.model.vis.global_.offheight = max(
                self.model.vis.global_.offheight,
                self.render_height,
            )
            self._r = mujoco.Renderer(
                self.model,
                height=self.render_height,
                width=self.render_width,
            )
            for _ in range(3):
                self._r.update_scene(self.data, camera=self.cam)
                self._r.render()
        self._r.update_scene(self.data, camera=self.cam)
        return self._r.render()

    def run(self, p, render=False):
        """Returns dict with margin, outcome, event_frame, frames, trace."""
        self.set_params(p)
        mujoco.mj_resetData(self.model, self.data)
        self.init_state(p)
        mujoco.mj_forward(self.model, self.data)
        if self.settle_steps:
            self.freeze(True)
            for _ in range(self.settle_steps):
                mujoco.mj_step(self.model, self.data)
            self.freeze(False)
        frames = []
        trace = []
        sub = int(round(self.capture_dt / self.model.opt.timestep))
        if render:
            frames.append(self.render())
        trace.append(self.observe())
        for k in range(self.n_frames - 1):
            for _ in range(sub):
                mujoco.mj_step(self.model, self.data)
            trace.append(self.observe())
            if render:
                frames.append(self.render())
        trace = np.asarray(trace)
        out = self.labels(p, trace)
        out["frames"] = np.stack(frames) if render else None
        out["trace"] = trace
        return out

    def freeze(self, on):
        """Hold everything still during settling if the scene needs it."""
        pass

    def set_params(self, p):
        raise NotImplementedError

    def init_state(self, p):
        pass

    def observe(self):
        raise NotImplementedError

    def labels(self, p, trace):
        raise NotImplementedError
