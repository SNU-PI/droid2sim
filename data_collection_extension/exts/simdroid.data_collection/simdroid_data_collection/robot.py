"""Panda adapter using NVIDIA Lula IK and NVIDIA trajectory generation."""
from dataclasses import dataclass
import numpy as np
import omni.kit.app
from pxr import Usd, UsdGeom, UsdPhysics
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import (
    LulaKinematicsSolver, LulaCSpaceTrajectoryGenerator, ArticulationTrajectory,
)
from isaacsim.robot_motion.motion_generation.interface_config_loader import (
    load_supported_lula_kinematics_solver_config,
)
from isaacsim.core.utils.rotations import rot_matrix_to_quat
from .domain import Pose, scaled_joint_limits
from .stage_store import world_pose


@dataclass
class Segment:
    waypoint: object
    trajectory: object
    start: np.ndarray
    end: np.ndarray
    duration: float


class FrankaBinding:
    def __init__(self, stage, path, tcp):
        self.stage, self.path, self.tcp = stage, path, tcp
        root = stage.GetPrimAtPath(path)
        if not root:
            raise ValueError("Robot prim no longer exists.")
        self.links = {p.GetName(): p for p in Usd.PrimRange(root)}
        required = ["panda_link0", "panda_hand", "panda_leftfinger", "panda_rightfinger",
                    "panda_finger_joint1", "panda_finger_joint2"]
        if any(n not in self.links for n in required):
            raise ValueError("Select a standard Franka Panda with its parallel gripper.")
        self.units = UsdGeom.GetStageMetersPerUnit(stage)
        world_pose(self.links["panda_hand"])
        world_pose(self.links["panda_link0"])
        self.art = None
        self.physics_view = None
        self.hand = self.base = self.ik = self.generator = None
        self.gripper_target = None
        self.gripper_elapsed = 0.
        self._initialized = False

    def initialize(self):
        self._initialized = False
        # In Isaac Sim 5.1 the core prim wrappers use SimulationManager's shared
        # view (the legacy initialize(view) argument is ignored internally).
        # Do not reset that global view behind other controllers' backs.
        self.physics_view = SimulationManager.get_physics_sim_view()
        if self.physics_view is None:
            raise ValueError("Physics is not ready. Press Play, wait for the scene, then Bind.")
        if not self.physics_view.is_valid:
            raise ValueError("Physics was rebuilt after a scene edit. Press Stop, then Play and Bind again.")
        self.art = SingleArticulation(self.path, name="franka_waypoint_robot", reset_xform_properties=False)
        self.art.initialize()
        names = list(self.art.dof_names)
        config = load_supported_lula_kinematics_solver_config("Franka")
        self.ik = LulaKinematicsSolver(**config)
        self.ik.bfgs_max_iterations = 100
        self.ik.ccd_max_iterations = 50
        self.generator = LulaCSpaceTrajectoryGenerator(**config)
        # Cache the bundled model's unscaled, per-joint limits. Speed=1 means
        # these native limits, not the former uniform .7/1.5/10 slow profile.
        self.motion_limits = scaled_joint_limits((
            self.generator.get_c_space_velocity_limits(),
            self.generator.get_c_space_acceleration_limits(),
            self.generator.get_c_space_jerk_limits()), 1.)
        if self.generator.get_active_joints() != self.ik.get_joint_names():
            raise ValueError("Panda IK and trajectory generator joint orders disagree.")
        self.arm_indices = np.array([names.index(n) for n in self.ik.get_joint_names()])
        self.finger_indices = np.array([names.index(f"panda_finger_joint{i}") for i in (1, 2)])
        if len(self.arm_indices) != 7:
            raise ValueError("The bundled Panda model must control seven arm joints.")
        self.hand = SingleRigidPrim(str(self.links["panda_hand"].GetPath()), name="waypoint_hand",
                                    reset_xform_properties=False)
        self.base = SingleRigidPrim(str(self.links["panda_link0"].GetPath()), name="waypoint_base",
                                    reset_xform_properties=False)
        self.hand.initialize()
        self.base.initialize()
        self.sync_base()
        # Detect USD / kinematic-model disagreement before sending a command.
        fk_p, fk_r = self.ik.compute_forward_kinematics("panda_hand", self.arm_positions())
        fk = Pose(np.asarray(fk_p)*self.units, rot_matrix_to_quat(fk_r))
        error = fk.error(self.hand_pose())
        if error[0] > .015 or error[1] > np.radians(5):
            raise ValueError(f"USD robot does not match bundled Panda kinematics: "
                             f"{error[0]*1000:.1f} mm / {np.degrees(error[1]):.1f} deg.")
        self.open_positions = np.array([
            min(.04 / self.units, float(UsdPhysics.PrismaticJoint(
                self.links[f"panda_finger_joint{i}"]).GetUpperLimitAttr().Get()))
            for i in (1, 2)])
        self.gripper_target = self.art.get_joint_positions(self.finger_indices).copy()
        self._initialized = True

    def ready(self):
        return (self._initialized and self.physics_view is not None and self.physics_view.is_valid
                and self.physics_view is SimulationManager.get_physics_sim_view()
                and self.art is not None and bool(self.art.handles_initialized))

    def sync_base(self):
        p, q = self.base.get_world_pose()
        self.ik.set_robot_base_pose(np.asarray(p), np.asarray(q))

    def base_pose(self):
        p, q = self.base.get_world_pose()
        return Pose(np.asarray(p)*self.units, q)

    def arm_positions(self):
        return np.asarray(self.art.get_joint_positions(self.arm_indices), dtype=float)

    def hand_pose(self):
        if self.hand is None or (self._initialized and not self.ready()):
            return world_pose(self.links["panda_hand"])
        p, q = self.hand.get_world_pose()
        return Pose(np.asarray(p)*self.units, q)

    def tcp_pose(self):
        return self.hand_pose().compose(self.tcp)

    def command_arm(self, positions, velocities=None):
        self.art.apply_action(ArticulationAction(
            joint_positions=np.asarray(positions),
            joint_velocities=np.zeros(7) if velocities is None else np.asarray(velocities),
            joint_indices=self.arm_indices))

    def command_gripper(self, action):
        if action == "open":
            self.gripper_target = self.open_positions.copy()
        elif action == "close":
            self.gripper_target = np.zeros(2)
        elif action != "keep":
            raise ValueError("Unknown gripper action.")
        self.gripper_elapsed = 0.

    def tick_gripper(self, dt):
        self.gripper_elapsed += dt
        if self.gripper_target is not None:
            self.art.apply_action(ArticulationAction(joint_positions=self.gripper_target,
                                                    joint_velocities=np.zeros(2),
                                                    joint_indices=self.finger_indices))

    def gripper_ready(self):
        if self.gripper_target is None:
            return True
        q = self.art.get_joint_positions(self.finger_indices)
        v = self.art.get_joint_velocities(self.finger_indices)
        stopped = bool(np.max(np.abs(v))*self.units < .002)
        at_target = bool(np.max(np.abs(q-self.gripper_target))*self.units < .002)
        closing = bool(np.allclose(self.gripper_target, 0))
        return self.gripper_elapsed >= .25 and stopped and (at_target or closing)

    async def plan(self, waypoints, progress=None):
        """Sample in TCP space, solve with Lula, then time-parameterize with Lula.

        Consecutive IK solves are seeded with the preceding joint solution. This
        handles an arbitrary fixed TCP offset without altering NVIDIA's URDF.
        """
        self.sync_base()
        seed = self.arm_positions()
        pose = self.tcp_pose()
        segments = []
        for wp in waypoints:
            if progress:
                progress(f"Planning {wp.path.rsplit('/', 1)[-1]}…")
            distance, angle = pose.error(wp.pose)
            count = max(2, int(np.ceil(distance/.008)), int(np.ceil(angle/np.radians(3))))
            if count > 1000:
                raise ValueError("Waypoint is too far away for a Panda segment.")
            start = seed.copy()
            samples = [seed.copy()]
            for i in range(1, count+1):
                desired = pose.interpolate(wp.pose, i/count).compose(self.tcp.inverse())
                solved, success = self.ik.compute_inverse_kinematics(
                    "panda_hand", np.asarray(desired.position)/self.units,
                    np.asarray(desired.orientation), warm_start=seed,
                    position_tolerance=min(.001, wp.position_tolerance/3)/self.units,
                    orientation_tolerance=min(.01, wp.orientation_tolerance/3))
                if not success or not np.all(np.isfinite(solved)):
                    raise ValueError(f"{wp.path}: Lula IK failed at sample {i}/{count}.")
                solved = np.asarray(solved)
                if np.max(np.abs(solved-seed)) > .35:
                    raise ValueError(f"{wp.path}: discontinuous IK branch; add an intermediate waypoint.")
                seed = solved
                if np.max(np.abs(seed-samples[-1])) > 1e-5:
                    samples.append(seed.copy())
                if i % 4 == 0:
                    await omni.kit.app.get_app().next_update_async()
            if len(samples) == 1:
                trajectory, duration = None, 0.
            else:
                velocity, acceleration, jerk = scaled_joint_limits(self.motion_limits, wp.speed)
                self.generator.set_c_space_velocity_limits(velocity)
                self.generator.set_c_space_acceleration_limits(acceleration)
                self.generator.set_c_space_jerk_limits(jerk)
                trajectory = self.generator.compute_c_space_trajectory(np.asarray(samples))
                if trajectory is None:
                    raise ValueError(f"{wp.path}: Lula could not generate a bounded trajectory.")
                duration = float(trajectory.end_time-trajectory.start_time)
            if duration + wp.dwell + .5 >= wp.timeout:
                raise ValueError(f"{wp.path}: motion {duration:.1f}s + wait {wp.dwell:.1f}s + "
                                 f"settling allowance 0.5s = {duration+wp.dwell+.5:.1f}s; "
                                 f"Max s ({wp.timeout:g}) must be larger. Increase Max s or Speed (up to 1).")
            segments.append(Segment(wp, trajectory, start, seed.copy(), duration))
            pose = wp.pose
            await omni.kit.app.get_app().next_update_async()
        return segments

    def playback(self, segment, dt):
        return ArticulationTrajectory(self.art, segment.trajectory, dt) if segment.trajectory else None
