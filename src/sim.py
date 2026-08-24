"""MuJoCo push environment with controllable mass and friction.

A box is pushed by a constant external force for a short window, then slides
freely to a stop. The two physical parameters leave distinct temporal
signatures:

    mass      -> peak speed        (v = J / m),   visible in the first frames
    friction  -> deceleration      (a = mu * g),  visible only over a longer window

That separation is the point of the whole pilot: it lets us ask, for a given
context length, *which* physical parameter is even inferable from observation.
"""

import numpy as np
import mujoco

XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global offwidth="256" offheight="256"/>
    <quality shadowsize="1024"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="0.06 0.07 0.09"
             width="64" height="64"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.30 0.33 0.38"
             rgb2="0.22 0.25 0.29" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.0"/>
    <material name="boxmat" rgba="0.92 0.26 0.21 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.9 0.9 0.9" specular="0.1 0.1 0.1"/>
    <light pos="1.5 -1.5 2" dir="-0.5 0.5 -1" diffuse="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="2.5 2.5 0.05" material="grid"
          friction="0.3 0.005 0.0001" condim="3"/>
    <body name="box" pos="0 0 0.05">
      <freejoint name="boxjnt"/>
      <geom name="boxgeom" type="box" size="0.05 0.05 0.05" material="boxmat"
            friction="0.3 0.005 0.0001" condim="3" mass="0.5"/>
    </body>
    <camera name="topcam" pos="0 0 2.6" xyaxes="1 0 0 0 1 0" fovy="37"/>
  </worldbody>
</mujoco>
"""

# --- physical parameter grid -------------------------------------------------
# log-spaced around the nominal so that sensitivity is comparable across cells
MASSES = np.array([0.35, 0.44, 0.56, 0.71, 0.90])
FRICTIONS = np.array([0.050, 0.075, 0.112, 0.167, 0.250])
NOMINAL = (2, 2)  # (mass_idx, friction_idx) -> the "normal" condition

FRAME_SKIP = 10          # physics steps per rendered frame -> dt = 0.02 s
N_FRAMES = 32            # 0.64 s episode. Long enough that every context length
                         # k <= 16 can be evaluated on the *same* prediction
                         # window, which is what makes the ablation fair.
IMG = 96

# camera half-extent in world units at the object plane (z=0.05)
CAM_HALF = (2.6 - 0.05) * np.tan(np.deg2rad(37) / 2)


class PushEnv:
    def __init__(self, img=IMG):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self._img = img
        self._renderer = None      # lazily created; frames come from raster.py
        self.bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        self.gid_box = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "boxgeom")
        self.gid_floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def set_physics(self, mass, mu):
        self.model.body_mass[self.bid] = mass
        # keep inertia consistent with the box geometry so rotation stays sane
        s = 0.05
        i = mass * (2 * (2 * s) ** 2) / 12.0
        self.model.body_inertia[self.bid] = np.array([i, i, i])
        # MuJoCo combines pairwise friction as the max of the two geoms,
        # so set both to the target value
        self.model.geom_friction[self.gid_box, 0] = mu
        self.model.geom_friction[self.gid_floor, 0] = mu

    def rollout(self, mass, mu, init_xy, push_dir, impulse, render=True):
        """Run one episode.

        The action is an *impulse* J delivered at t=0, not a sustained force.
        Two reasons: an impulse has no static-friction threshold (so no grid
        cell is degenerate), and it makes the mass signature clean, since the
        resulting speed is exactly v0 = J / m.

        init_xy / push_dir / impulse are held identical across physics
        parameters, so episodes are exactly paired cell-to-cell.
        """
        self.set_physics(mass, mu)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:2] = init_xy
        self.data.qpos[2] = 0.05
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        # impulse -> velocity. This is the only place mass enters the action.
        v0 = impulse / mass
        self.data.qvel[0:2] = np.array([np.cos(push_dir), np.sin(push_dir)]) * v0
        mujoco.mj_forward(self.model, self.data)

        frames = np.zeros((N_FRAMES, IMG, IMG, 3), np.uint8) if render else None
        states = np.zeros((N_FRAMES, 5), np.float32)   # x, y, yaw, vx, vy

        for t in range(N_FRAMES):
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(self.model, self.data)
            q = self.data.qpos
            # yaw from the free-joint quaternion
            w, x, y, z = q[3], q[4], q[5], q[6]
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            states[t] = [q[0], q[1], yaw, self.data.qvel[0], self.data.qvel[1]]
            if render:
                if self._renderer is None:
                    self._renderer = mujoco.Renderer(
                        self.model, height=self._img, width=self._img)
                self._renderer.update_scene(self.data, camera="topcam")
                frames[t] = self._renderer.render()

        # the action seen by the world model: impulse vector, applied at t=0
        action = np.array([np.cos(push_dir) * impulse,
                           np.sin(push_dir) * impulse], np.float32)
        return frames, states, action


def sample_episode_params(rng):
    """Initial condition + action. Deliberately independent of (mass, mu) so the
    same draw can be replayed under every physics setting."""
    r = rng.uniform(0.0, 0.10)
    a = rng.uniform(0, 2 * np.pi)
    init_xy = np.array([r * np.cos(a), r * np.sin(a)])
    push_dir = rng.uniform(0, 2 * np.pi)
    impulse = rng.uniform(0.35, 0.45)
    return init_xy, push_dir, impulse


if __name__ == "__main__":
    # probe: check the box stays inside the camera frame across the whole grid
    env = PushEnv()
    rng = np.random.default_rng(0)
    worst, disp, cell = 0.0, [], np.zeros((5, 5))
    for _ in range(40):
        ic, pd, pm = sample_episode_params(rng)
        for i, m in enumerate(MASSES):
            for j, mu in enumerate(FRICTIONS):
                _, s, _ = env.rollout(m, mu, ic, pd, pm, render=False)
                worst = max(worst, np.abs(s[:, :2]).max() + 0.05)
                d = np.linalg.norm(s[-1, :2] - ic)
                disp.append(d)
                cell[i, j] += d / 40
    px = 2 * CAM_HALF / IMG
    print(f"camera half-extent : {CAM_HALF:.3f} m   ({px*1000:.1f} mm / pixel at {IMG}px)")
    print(f"max |xy| + box     : {worst:.3f} m   "
          f"({'OK' if worst < CAM_HALF * 0.95 else 'OUT OF FRAME'})")
    print(f"displacement  min/med/max : {np.min(disp):.3f} / {np.median(disp):.3f} "
          f"/ {np.max(disp):.3f} m   ({np.min(disp)/px:.1f} / {np.median(disp)/px:.1f} "
          f"/ {np.max(disp)/px:.1f} px)")
    print("\nmean displacement per grid cell (m), rows=mass, cols=friction:")
    print("        " + "".join(f"mu={v:<7.3f}" for v in FRICTIONS))
    for i, m in enumerate(MASSES):
        print(f"m={m:<5.2f} " + "".join(f"{cell[i, j]:<10.3f}" for j in range(5)))
