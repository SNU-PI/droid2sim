"""DROID-like tabletop strike-and-slide scene.

The interaction is deliberately a *strike and release*, not a sustained push.
While a gripper stays in contact the object's motion is dictated by the
gripper's position, and mass and friction barely register. Once the pusher lets
go, the object's own physics is the only thing driving it:

    peak speed after the strike   ~ 1 / mass       (momentum transfer)
    deceleration while sliding    = mu * g         (Coulomb, mass-independent)

Those are the two quantities the whole pilot rests on, so the scene is built so
that each one is controlled exactly and can be checked against theory.

Rigour choices, and why:
  * an explicit <pair> between object and table sets the friction directly.
    Without it MuJoCo derives contact friction by combining the two geoms'
    values, and "the friction" would not be a single number we set.
  * cone="elliptic" with impratio=10 -- the default pyramidal cone biases
    sliding direction and would corrupt the mu -> deceleration relationship.
  * condim=3, so friction means sliding friction only. condim 4/6 would add
    torsional and rolling friction, which we are not sweeping.
  * solref/solimp pinned on the pair, so contact softness does not drift as a
    hidden second variable when mu changes.
  * mass and inertia are always set together from the box formula. Setting
    body_mass alone leaves inertia inconsistent with the geometry.
"""

import os
import numpy as np
import mujoco

# ---- object geometry (fixed; only mass and friction are swept) --------------
OBJ_HALF = np.array([0.035, 0.035, 0.035])     # 7 cm cube. A cube rather than a
# taller box: struck at mid-height, a tall box picks up a tipping moment from
# base friction and rocks, which breaks table contact and corrupts the sliding
# friction we are trying to measure.
TABLE_TOP_Z = 0.40

# ---- the two swept parameters ----------------------------------------------
# Ranges are chosen so that EVERY cell produces visible motion. Outside them the
# physics is fine but useless: at mu = 0.6 with a 0.5 kg cube the object travels
# 7 mm, which is under two pixels at 256 px and leaves nothing for either the
# deceleration fit or the encoder to see.
MASS_GRID = np.array([0.120, 0.155, 0.200, 0.260, 0.340])   # kg
MU_GRID = np.array([0.150, 0.190, 0.245, 0.310, 0.400])     # object <-> table
NOMINAL_MASS = 0.200
NOMINAL_MU = 0.245
NOMINAL_CELL = (2, 2)

# ---- the fixed action -------------------------------------------------------
# The pusher is a DYNAMIC body on a frictionless rail, not a kinematic one.
# That distinction decides whether mass is identifiable at all: a
# position-driven pusher drags the object up to its own speed regardless of
# how heavy it is, so every mass would leave contact at the same velocity. A
# free body carrying momentum splits that momentum by the mass ratio,
#
#     v_object after the strike  =  M * v0 / (M + m)
#
# which is both mass-dependent and checkable against theory.
PUSHER_MASS = 0.20         # kg
PUSHER_V0 = 1.20           # m/s, initial rail velocity
PUSHER_START_Y = -0.26
# First contact is at y = -OBJ_HALF[1] - capsule_radius = -0.046. The rail limit
# must sit well past that: if it is close, the joint limit -- not the object --
# absorbs the pusher's momentum, and every mass leaves contact at the same
# speed. With the limit here the momentum exchange completes first, the two
# travel together briefly, then the pusher is stopped and the object runs on.
PUSHER_STOP_Y = -0.020

CTRL_DT = 0.02             # one rendered frame per 20 ms (50 Hz)
N_FRAMES = 40              # 0.8 s episode
IMG_W, IMG_H = 256, 256    # V-JEPA 2-AC was post-trained at 256 px

XML = f"""
<mujoco model="droidlike_push">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" integrator="implicitfast"
          cone="elliptic" impratio="10" gravity="0 0 -9.81"
          iterations="150" ls_iterations="50" noslip_iterations="8"
          tolerance="1e-10"/>
  <visual>
    <global offwidth="{IMG_W}" offheight="{IMG_H}" azimuth="140" elevation="-25"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
  </visual>

  <asset>
    <texture name="grad" type="skybox" builtin="gradient"
             rgb1="0.42 0.45 0.50" rgb2="0.16 0.18 0.21" width="256" height="256"/>
    <texture name="woodtex" type="2d" builtin="flat" rgb1="0.62 0.47 0.32"
             width="512" height="512" mark="random" markrgb="0.55 0.41 0.27" random="0.25"/>
    <material name="wood" texture="woodtex" texrepeat="2 2" specular="0.15" shininess="0.25"/>
    <material name="floor" rgba="0.28 0.29 0.32 1" specular="0.05" shininess="0.05"/>
    <material name="objmat" rgba="0.80 0.28 0.22 1" specular="0.35" shininess="0.4"/>
    <material name="steel" rgba="0.62 0.64 0.68 1" specular="0.7" shininess="0.7"/>
    <material name="riser" rgba="0.24 0.25 0.28 1" specular="0.1"/>
<!--FRANKA_ASSET-->
  </asset>

  <worldbody>
    <light name="key"  pos="0.45 -0.55 1.65" dir="-0.3 0.35 -1" directional="false"
           diffuse="0.75 0.73 0.70" specular="0.25 0.25 0.25" castshadow="true"/>
    <light name="fill" pos="-0.6 0.5 1.4" dir="0.4 -0.3 -1" directional="false"
           diffuse="0.28 0.30 0.34" specular="0.0 0.0 0.0" castshadow="false"/>

    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 0" material="floor"/>

    <!-- table: top surface exactly at TABLE_TOP_Z -->
    <body name="table" pos="0 0 {TABLE_TOP_Z - 0.02}">
      <geom name="table_top" type="box" size="0.60 0.45 0.02" material="wood"/>
      <geom name="leg1" type="box" size="0.03 0.03 {(TABLE_TOP_Z-0.04)/2}"
            pos=" 0.55  0.40 {-(TABLE_TOP_Z-0.04)/2 - 0.02}" material="riser"/>
      <geom name="leg2" type="box" size="0.03 0.03 {(TABLE_TOP_Z-0.04)/2}"
            pos="-0.55  0.40 {-(TABLE_TOP_Z-0.04)/2 - 0.02}" material="riser"/>
      <geom name="leg3" type="box" size="0.03 0.03 {(TABLE_TOP_Z-0.04)/2}"
            pos=" 0.55 -0.40 {-(TABLE_TOP_Z-0.04)/2 - 0.02}" material="riser"/>
      <geom name="leg4" type="box" size="0.03 0.03 {(TABLE_TOP_Z-0.04)/2}"
            pos="-0.55 -0.40 {-(TABLE_TOP_Z-0.04)/2 - 0.02}" material="riser"/>
    </body>

    <!-- the object whose mass and friction we sweep -->
    <body name="obj" pos="0 0 {TABLE_TOP_Z + OBJ_HALF[2]}">
      <freejoint name="objfree"/>
      <inertial pos="0 0 0" mass="{NOMINAL_MASS}"
                diaginertia="1 1 1"/>
      <geom name="obj" type="box" size="{OBJ_HALF[0]} {OBJ_HALF[1]} {OBJ_HALF[2]}"
            material="objmat" condim="3"/>
    </body>

    <!-- dynamic pusher on a frictionless rail along +y, stopped by a joint limit -->
    <body name="pusher" pos="0 0 {TABLE_TOP_Z + OBJ_HALF[2]}">
      <joint name="rail" type="slide" axis="0 1 0" damping="0" frictionloss="0"
             limited="true" range="{PUSHER_START_Y} {PUSHER_STOP_Y}"
             solreflimit="0.002 1" solimplimit="0.99 0.999 0.001"/>
      <inertial pos="0 0 0" mass="{PUSHER_MASS}" diaginertia="0.001 0.001 0.001"/>
      <geom name="finger" type="capsule" size="0.011 0.038" euler="0 1.5708 0"
            material="steel" condim="3" contype="2" conaffinity="2"/>
      <geom name="shaft" type="capsule" size="0.013 0.085"
            pos="0 -0.02 0.115" euler="0.25 0 0" material="steel"
            contype="0" conaffinity="0"/>
    </body>

<!--FRANKA_BODY-->
    <camera name="droidcam" pos="0.72 -0.78 0.92" xyaxes="0.74 0.68 0 -0.26 0.28 0.92"/>
    <!-- Same viewing direction, moved in along it. Confound 2: the whole physics
         sweep only shifts the wide view by 0.1-1.0 grey levels, while a 5 mm
         camera nudge shifts it by 1.5, so the comparison against nuisances was
         never fair. Closing in raises the physics signal without changing what
         the scene is. -->
    <camera name="closecam" pos="0.396 -0.375 0.702" xyaxes="0.74 0.68 0 -0.26 0.28 0.92"/>
  </worldbody>

  <contact>
    <!-- object <-> table: the ONLY place sliding friction is defined -->
    <!-- solref is deliberately soft-and-damped (10 ms, critically damped).
         A stiffer contact overshoots on penetration recovery, the cube leaves
         the table for a frame or two at a time, and the friction impulse
         disappears exactly when we are trying to measure it. -->
    <pair name="obj_table" geom1="obj" geom2="table_top" condim="3"
          friction="{NOMINAL_MU} {NOMINAL_MU} 0.004 0.0001 0.0001"
          solref="0.010 1" solimp="0.90 0.96 0.002"/>
    <!-- pusher <-> object: fixed, high friction, so the strike itself does not
         change when we sweep the table friction -->
    <!-- A slightly compliant strike (15 ms) rather than a near-rigid one.
         At solref=0.004 the whole collision resolves in about four 1 ms steps,
         so the transferred momentum still moved by ~3% every time the timestep
         was halved. Spreading the contact over ~15 ms makes the exchange
         timestep-independent, and a compliant tip is the more realistic
         model of a gripper finger anyway. -->
    <pair name="obj_finger" geom1="obj" geom2="finger" condim="3"
          friction="0.9 0.9 0.005 0.0001 0.0001"
          solref="0.015 1" solimp="0.95 0.95 0.001"/>
  </contact>
</mujoco>
"""


def box_inertia(mass, half):
    """Diagonal inertia of a uniform box about its centre."""
    a, b, c = 2 * np.asarray(half)
    return mass / 12.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])


def build_xml(with_franka=False):
    """Scene XML, optionally with a static Franka spliced in.

    The arm is appearance only -- baked geoms, collisions off, no joints and no
    mass -- so that turning it on cannot move a single number in the physics.
    That is the point: it isolates the DROID domain gap as the one variable.
    """
    xml = XML
    if with_franka:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "franka_asset.xml")) as f:
            asset = f.read()
        with open(os.path.join(here, "franka_body.xml")) as f:
            body = f.read()
        xml = xml.replace("<!--FRANKA_ASSET-->", asset)
        xml = xml.replace("<!--FRANKA_BODY-->",
                          '    <body name="franka_static" pos="0 0 0">\n'
                          + body + "    </body>")
    return xml


class PushScene:
    """Strike-and-slide scene with exactly two swept physical parameters."""

    PAIR = "obj_table"

    def __init__(self, width=IMG_W, height=IMG_H, timestep=0.001, franka=False,
                 cam="droidcam"):
        self.franka = franka
        self.cam = cam
        self.model = mujoco.MjModel.from_xml_string(build_xml(franka))
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        self._w, self._h = width, height

        self.bid_obj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj")
        jrail = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rail")
        self.rail_q = self.model.jnt_qposadr[jrail]
        self.rail_v = self.model.jnt_dofadr[jrail]
        self.pair_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_PAIR, self.PAIR)
        self.qadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "objfree")]
        self.vadr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "objfree")]

    # ---- the two swept parameters ------------------------------------------
    def set_physics(self, mass, mu):
        """mass in kg, mu = object/table sliding friction. Nothing else changes."""
        self.model.body_mass[self.bid_obj] = mass
        self.model.body_inertia[self.bid_obj] = box_inertia(mass, OBJ_HALF)
        self.model.pair_friction[self.pair_id, 0] = mu     # slide along tangent 1
        self.model.pair_friction[self.pair_id, 1] = mu     # slide along tangent 2
        # REQUIRED. Writing body_mass alone updates the gravitational force but
        # NOT the inertia the solver uses: MuJoCo derives dof_M0/invweight at
        # compile time and keeps using them. Gravity then scales with the new
        # mass while acceleration still divides by the old one, so a = F/m
        # silently becomes a = F/m_xml. It is invisible in any single run --
        # everything looks physical -- and it corrupts exactly the mass sweep
        # this pilot depends on.
        mujoco.mj_setConst(self.model, self.data)

    def reset(self, obj_xy=(0.0, 0.0), obj_yaw=0.0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr + 0] = obj_xy[0]
        self.data.qpos[self.qadr + 1] = obj_xy[1]
        self.data.qpos[self.qadr + 2] = TABLE_TOP_Z + OBJ_HALF[2]
        self.data.qpos[self.qadr + 3:self.qadr + 7] = [
            np.cos(obj_yaw / 2), 0, 0, np.sin(obj_yaw / 2)]
        # the action: pusher starts back on the rail, already moving at v0
        self.data.qpos[self.rail_q] = PUSHER_START_Y
        self.data.qvel[self.rail_v] = PUSHER_V0
        mujoco.mj_forward(self.model, self.data)

    def predicted_release_speed(self, mass):
        """Momentum-exchange prediction, ignoring table friction during contact."""
        return PUSHER_MASS * PUSHER_V0 / (PUSHER_MASS + mass)

    def rollout(self, mass, mu, obj_xy=(0.0, 0.0), obj_yaw=0.0,
                n_frames=N_FRAMES, render=False, fine=False):
        """Returns states [T,7] = (x, y, yaw, vx, vy, speed, pusher_y) and frames.

        fine=True additionally returns the per-physics-step speed trace. Frames
        are 20 ms apart, which is far too coarse to fit a deceleration when a
        rough, heavy object stops in under two frames; the verification needs
        the 1 ms trace to measure friction at all.
        """
        self.set_physics(mass, mu)
        self.reset(obj_xy, obj_yaw)

        sub = max(1, int(round(CTRL_DT / self.model.opt.timestep)))
        states = np.zeros((n_frames, 7), np.float64)
        frames = np.zeros((n_frames, self._h, self._w, 3), np.uint8) if render else None
        trace = [] if fine else None

        for f in range(n_frames):
            for s in range(sub):
                mujoco.mj_step(self.model, self.data)
                if fine:
                    vv = self.data.qvel[self.vadr:self.vadr + 2]
                    trace.append((float(np.hypot(vv[0], vv[1])),
                                  float(self.data.qvel[self.rail_v])))
            q = self.data.qpos[self.qadr:self.qadr + 7]
            v = self.data.qvel[self.vadr:self.vadr + 3]
            w, x, y, z = q[3], q[4], q[5], q[6]
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            states[f] = [q[0], q[1], yaw, v[0], v[1],
                         float(np.hypot(v[0], v[1])),
                         self.data.qpos[self.rail_q]]
            if render:
                frames[f] = self.render()
        if fine:
            return states, frames, np.asarray(trace)
        return states, frames

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self._h, width=self._w)
            # Warm-up. The first frames out of a fresh EGL context come back
            # wrong -- the shadow map is not populated yet -- and the error is
            # large enough (about 15 grey levels averaged over the image) to
            # dwarf the physics signal we are trying to measure. They look
            # perfectly plausible, so nothing catches them downstream.
            for _ in range(3):
                self._renderer.update_scene(self.data, camera=self.cam)
                self._renderer.render()
        self._renderer.update_scene(self.data, camera=self.cam)
        return self._renderer.render()

    def in_contact_with_finger(self):
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "finger")
        gob = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obj")
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if {c.geom1, c.geom2} == {gid, gob}:
                return True
        return False
