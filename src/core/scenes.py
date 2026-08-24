"""A family of tabletop scenes, each with a different physical signature.

Stage 0 tested one interaction: a cube struck and left to slide, where mass
shows up in the release speed and friction in the decay. A single mechanism
cannot tell us whether V-JEPA is blind to physics in general or blind to that
one cue, so this adds four more, each of which makes a different property the
thing that governs the outcome:

    slide     cube struck, slides to a stop     mass, sliding friction
    roll      sphere struck, rolls              mass, rolling friction
    bounce    cube dropped onto the table       restitution, mass
    collide   cube struck into a second cube    mass ratio, friction
    incline   cube released on a ramp           friction, ramp angle

Every family keeps the conventions that Stage 0 had to fight for: friction set
through an explicit contact pair rather than combined from two geoms, mass and
inertia written together followed by mj_setConst, an elliptic friction cone,
and a renderer warm-up. Each also carries an analytic check, because a scene
that looks plausible on screen can still have the wrong physics.
"""

import os
import numpy as np
import mujoco

TABLE_TOP_Z = 0.40
IMG = 256
CTRL_DT = 0.02
G = 9.81

_HERE = os.path.dirname(os.path.abspath(__file__))


def _franka_blocks():
    try:
        with open(os.path.join(_HERE, "franka_asset.xml")) as f:
            a = f.read()
        with open(os.path.join(_HERE, "franka_body.xml")) as f:
            b = f.read()
        return a, '    <body name="franka_static" pos="0 0 0">\n' + b + "    </body>"
    except FileNotFoundError:
        return "", ""


COMMON_HEAD = """
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" integrator="implicitfast"
          cone="elliptic" impratio="10" gravity="0 0 -9.81"
          iterations="150" ls_iterations="50" noslip_iterations="8"
          tolerance="1e-10"/>
  <visual>
    <global offwidth="256" offheight="256" azimuth="140" elevation="-25"/>
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
    <material name="objmat" rgba="0.80 0.28 0.22 1" specular="0.35" shininess="0.4"/>
    <material name="obj2mat" rgba="0.22 0.42 0.72 1" specular="0.35" shininess="0.4"/>
    <material name="steel" rgba="0.62 0.64 0.68 1" specular="0.7" shininess="0.7"/>
    <material name="riser" rgba="0.24 0.25 0.28 1" specular="0.1"/>
{FRANKA_ASSET}
  </asset>
"""

TABLE = f"""
    <light name="key"  pos="0.45 -0.55 1.65" dir="-0.3 0.35 -1" directional="false"
           diffuse="0.75 0.73 0.70" specular="0.25 0.25 0.25" castshadow="true"/>
    <light name="fill" pos="-0.6 0.5 1.4" dir="0.4 -0.3 -1" directional="false"
           diffuse="0.28 0.30 0.34" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor"/>
    <body name="table" pos="0 0 {TABLE_TOP_Z - 0.02}">
      <geom name="table_top" type="box" size="0.60 0.45 0.02" material="wood"/>
      <geom name="leg1" type="box" size="0.03 0.03 0.18" pos=" 0.55  0.40 -0.20" material="riser"/>
      <geom name="leg2" type="box" size="0.03 0.03 0.18" pos="-0.55  0.40 -0.20" material="riser"/>
      <geom name="leg3" type="box" size="0.03 0.03 0.18" pos=" 0.55 -0.40 -0.20" material="riser"/>
      <geom name="leg4" type="box" size="0.03 0.03 0.18" pos="-0.55 -0.40 -0.20" material="riser"/>
    </body>
"""

CAMS = """
    <camera name="wide"  pos="0.72 -0.78 0.92"   xyaxes="0.74 0.68 0 -0.26 0.28 0.92"/>
    <camera name="close" pos="0.396 -0.375 0.702" xyaxes="0.74 0.68 0 -0.26 0.28 0.92"/>
"""


def _wrap(name, body, contact):
    fa, fb = _franka_blocks()
    head = COMMON_HEAD.replace("{FRANKA_ASSET}", fa)
    return (f'<mujoco model="{name}">' + head + "  <worldbody>" + TABLE + body
            + fb + CAMS + "  </worldbody>\n" + contact + "</mujoco>")


def box_inertia(mass, half):
    a, b, c = 2 * np.asarray(half, dtype=float)
    return mass / 12.0 * np.array([b*b + c*c, a*a + c*c, a*a + b*b])


def sphere_inertia(mass, r):
    v = 0.4 * mass * r * r
    return np.array([v, v, v])



PUSHER = """
    <body name="pusher" pos="0 0 {pz}">
      <joint name="rail" type="slide" axis="0 1 0" damping="0" frictionloss="0"
             limited="true" range="-0.26 {stop}"
             solreflimit="0.002 1" solimplimit="0.99 0.999 0.001"/>
      <inertial pos="0 0 0" mass="0.20" diaginertia="0.001 0.001 0.001"/>
      <geom name="finger" type="capsule" size="0.011 0.038" euler="0 1.5708 0"
            material="steel" contype="2" conaffinity="2"/>
      <geom name="shaft" type="capsule" size="0.013 0.085" pos="0 -0.02 0.115"
            euler="0.25 0 0" material="steel" contype="0" conaffinity="0"/>
    </body>
"""

BODIES = {
"slide": (f"""
    <body name="obj" pos="0 0 {TABLE_TOP_Z + 0.035}">
      <freejoint name="objfree"/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="1 1 1"/>
      <geom name="obj" type="box" size="0.035 0.035 0.035" material="objmat" condim="3"/>
    </body>
""" + PUSHER.format(pz=TABLE_TOP_Z + 0.035, stop=-0.020), """
  <contact>
    <pair name="obj_table" geom1="obj" geom2="table_top" condim="3"
          friction="0.25 0.25 0.004 0.0001 0.0001"
          solref="0.010 1" solimp="0.90 0.96 0.002"/>
    <pair name="obj_finger" geom1="obj" geom2="finger" condim="3"
          friction="0.9 0.9 0.005 0.0001 0.0001"
          solref="0.015 1" solimp="0.95 0.95 0.001"/>
  </contact>
"""),

"roll": (f"""
    <body name="obj" pos="0 0 {TABLE_TOP_Z + 0.040}">
      <freejoint name="objfree"/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="1 1 1"/>
      <geom name="obj" type="sphere" size="0.040" material="objmat" condim="6"/>
    </body>
""" + PUSHER.format(pz=TABLE_TOP_Z + 0.040, stop=-0.020), """
  <contact>
    <!-- high sliding friction so the ball rolls rather than skids; the swept
         parameter is the ROLLING resistance, a different mechanism entirely -->
    <pair name="obj_table" geom1="obj" geom2="table_top" condim="6"
          friction="1.0 1.0 0.005 0.0002 0.0002"
          solref="0.010 1" solimp="0.90 0.96 0.002"/>
    <pair name="obj_finger" geom1="obj" geom2="finger" condim="3"
          friction="0.9 0.9 0.005 0.0001 0.0001"
          solref="0.015 1" solimp="0.95 0.95 0.001"/>
  </contact>
"""),

"bounce": (f"""
    <body name="obj" pos="0 0 {TABLE_TOP_Z + 0.28}">
      <freejoint name="objfree"/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="1 1 1"/>
      <geom name="obj" type="box" size="0.035 0.035 0.035" material="objmat" condim="3"/>
    </body>
""", """
  <contact>
    <!-- solref in the negative convention is (-stiffness, -damping): the swept
         parameter is the damping, which is what sets how much of the drop is
         given back as bounce -->
    <pair name="obj_table" geom1="obj" geom2="table_top" condim="3"
          friction="0.4 0.4 0.004 0.0001 0.0001"
          solref="-8000 -40" solimp="0.95 0.95 0.001"/>
  </contact>
"""),

"collide": (f"""
    <body name="obj" pos="0 0 {TABLE_TOP_Z + 0.035}">
      <freejoint name="objfree"/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="1 1 1"/>
      <geom name="obj" type="box" size="0.035 0.035 0.035" material="objmat" condim="3"/>
    </body>
    <body name="obj2" pos="0 0.11 {TABLE_TOP_Z + 0.035}">
      <freejoint name="obj2free"/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="1 1 1"/>
      <geom name="obj2" type="box" size="0.035 0.035 0.035" material="obj2mat" condim="3"/>
    </body>
""" + PUSHER.format(pz=TABLE_TOP_Z + 0.035, stop=-0.020), """
  <contact>
    <pair name="obj_table" geom1="obj" geom2="table_top" condim="3"
          friction="0.25 0.25 0.004 0.0001 0.0001"
          solref="0.010 1" solimp="0.90 0.96 0.002"/>
    <pair name="obj2_table" geom1="obj2" geom2="table_top" condim="3"
          friction="0.25 0.25 0.004 0.0001 0.0001"
          solref="0.010 1" solimp="0.90 0.96 0.002"/>
    <pair name="obj_obj2" geom1="obj" geom2="obj2" condim="3"
          friction="0.4 0.4 0.004 0.0001 0.0001"
          solref="0.008 1" solimp="0.95 0.95 0.001"/>
    <pair name="obj_finger" geom1="obj" geom2="finger" condim="3"
          friction="0.9 0.9 0.005 0.0001 0.0001"
          solref="0.015 1" solimp="0.95 0.95 0.001"/>
  </contact>
"""),

"incline": (f"""
    <body name="ramp" pos="0 0 0.530" euler="-0.45 0 0">
      <geom name="ramp" type="box" size="0.16 0.22 0.012" material="riser"/>
    </body>
    <body name="obj" pos="0 -0.1334 0.6445" euler="-0.45 0 0">
      <freejoint name="objfree"/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="1 1 1"/>
      <geom name="obj" type="box" size="0.032 0.032 0.032" material="objmat" condim="3"/>
    </body>
""", """
  <contact>
    <pair name="obj_table" geom1="obj" geom2="ramp" condim="3"
          friction="0.25 0.25 0.004 0.0001 0.0001"
          solref="0.010 1" solimp="0.90 0.96 0.002"/>
  </contact>
"""),
}

# swept parameters, log-uniform in these ranges
PARAMS = {
    "slide":   [("mass", 0.10, 0.40), ("mu", 0.12, 0.45)],
    "roll":    [("mass", 0.10, 0.40), ("roll_fric", 0.0004, 0.006)],
    "bounce":  [("mass", 0.10, 0.40), ("damping", 8.0, 220.0)],
    "collide": [("mass2", 0.08, 0.60), ("mu", 0.12, 0.45)],
    "incline": [("mass", 0.10, 0.40), ("mu", 0.12, 0.42)],
}
FAMILIES = list(PARAMS)
# incline is shorter: at the lowest friction the cube clears the 0.39 m ramp
# in about 25 frames, and everything after that is a fall onto the table
N_FRAMES = {"slide": 40, "roll": 40, "bounce": 40, "collide": 40, "incline": 26}
FRAME_IDX = {f: list(np.linspace(6, N_FRAMES[f] - 2, 8).astype(int)) for f in FAMILIES}
FRAME_IDX["bounce"] = [2, 5, 8, 11, 14, 18, 24, 32]


class Scene:
    def __init__(self, family, cam="close", timestep=0.001):
        assert family in BODIES, family
        self.family = family
        body, contact = BODIES[family]
        self.model = mujoco.MjModel.from_xml_string(_wrap(family, body, contact))
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.cam = cam
        self._r = None
        self.bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self.qadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "objfree")]
        self.vadr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "objfree")]
        self.pair = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_PAIR, "obj_table")
        self.has_rail = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "rail") >= 0
        if self.has_rail:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rail")
            self.rq, self.rv = self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]
        if family == "collide":
            self.bid2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj2")
            self.q2 = self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj2free")]
            self.v2 = self.model.jnt_dofadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj2free")]

    # parameters
    def set_params(self, p):
        m = self.model
        f = self.family
        if f in ("slide", "incline"):
            mass = p["mass"]
            m.body_mass[self.bid] = mass
            h = 0.032 if f == "incline" else 0.035
            m.body_inertia[self.bid] = box_inertia(mass, [h, h, h])
            m.pair_friction[self.pair, 0:2] = p["mu"]
        elif f == "roll":
            mass = p["mass"]
            m.body_mass[self.bid] = mass
            m.body_inertia[self.bid] = sphere_inertia(mass, 0.040)
            m.pair_friction[self.pair, 3:5] = p["roll_fric"]
        elif f == "bounce":
            mass = p["mass"]
            m.body_mass[self.bid] = mass
            m.body_inertia[self.bid] = box_inertia(mass, [0.035]*3)
            m.pair_solref[self.pair, 1] = -abs(p["damping"])
        elif f == "collide":
            m.body_mass[self.bid2] = p["mass2"]
            m.body_inertia[self.bid2] = box_inertia(p["mass2"], [0.035]*3)
            for nm in ("obj_table", "obj2_table"):
                pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_PAIR, nm)
                m.pair_friction[pid, 0:2] = p["mu"]
        # REQUIRED after touching mass/inertia: the mass matrix is otherwise
        # left at its compile-time value while gravity uses the new mass
        mujoco.mj_setConst(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        if self.has_rail:
            self.data.qpos[self.rq] = -0.26
            self.data.qvel[self.rv] = 1.20
        mujoco.mj_forward(self.model, self.data)

    def rollout(self, p, render=False, frames_at=None, fine=False):
        self.set_params(p)
        self.reset()
        n = N_FRAMES[self.family]
        want = set(frames_at if frames_at is not None else range(n))
        sub = int(round(CTRL_DT / self.model.opt.timestep))
        st = np.zeros((n, 8))
        out = []
        tr = [] if fine else None
        for k in range(n):
            for _ in range(sub):
                mujoco.mj_step(self.model, self.data)
                if fine:
                    vv = self.data.qvel[self.vadr:self.vadr+3]
                    tr.append((float(np.hypot(vv[0], vv[1])), float(vv[2])))
            q = self.data.qpos[self.qadr:self.qadr+7]
            v = self.data.qvel[self.vadr:self.vadr+3]
            st[k, :3] = q[:3]
            st[k, 3:6] = v
            st[k, 6] = np.hypot(v[0], v[1])
            if self.family == "collide":
                st[k, 7] = self.data.qpos[self.q2 + 1]
            elif self.has_rail:
                st[k, 7] = self.data.qpos[self.rq]
            if render and k in want:
                out.append(self.render())
        frames = np.stack(out) if render and out else None
        if fine:
            return st, frames, np.asarray(tr)
        return st, frames

    def render(self):
        if self._r is None:
            self._r = mujoco.Renderer(self.model, height=IMG, width=IMG)
            for _ in range(3):                     # warm-up; see render_clips.py
                self._r.update_scene(self.data, camera=self.cam)
                self._r.render()
        self._r.update_scene(self.data, camera=self.cam)
        return self._r.render()


def sample_params(family, rng):
    """Log-uniform draw over the family's parameter ranges."""
    return {n: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            for n, lo, hi in PARAMS[family]}
