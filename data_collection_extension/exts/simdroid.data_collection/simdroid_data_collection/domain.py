"""Pose conventions and sequence state, independent of Kit and the simulator.

Positions are metres, orientations are unit quaternions (w, x, y, z).
The TCP uses hand axes and a configurable fixed transform from panda_hand.
"""
from dataclasses import dataclass
import math
import numpy as np


def unit_quat(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or np.max(np.abs(q)) < 1e-10:
        raise ValueError("Orientation must be a finite, nonzero quaternion.")
    # Scale first so finite but large user-authored quaternion values cannot
    # overflow the norm and silently become the invalid all-zero quaternion.
    q = q / np.max(np.abs(q))
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def conjugate(q):
    return np.asarray(q) * [1, -1, -1, -1]


def rotate(q, v):
    q = unit_quat(q)
    return quat_mul(quat_mul(q, np.r_[0., v]), conjugate(q))[1:]


def angular_error(a, b):
    return 2 * math.acos(float(np.clip(abs(np.dot(unit_quat(a), unit_quat(b))), 0, 1)))


def slerp(a, b, t):
    a, b = unit_quat(a), unit_quat(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        return unit_quat(a + t * (b-a))
    theta = math.acos(np.clip(dot, -1, 1))
    return (math.sin((1-t)*theta)*a + math.sin(t*theta)*b) / math.sin(theta)


def euler_quat(degrees):
    """Extrinsic XYZ / roll-pitch-yaw: Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = np.radians(degrees) / 2
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    return unit_quat([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                      cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


def quat_euler(q):
    w, x, y, z = unit_quat(q)
    return np.degrees([math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                       math.asin(np.clip(2*(w*y-z*x), -1, 1)),
                       math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))])


def scaled_joint_limits(limits, speed):
    """Time-scale Panda model limits, never replace them with arbitrary caps.

    For a speed fraction s, velocity/acceleration/jerk scale as s/s²/s³.
    Always start from the unscaled model values so repeated plans do not
    compound the scaling or depend on the previous waypoint's speed.
    """
    if not math.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError("Speed must be greater than 0 and at most 1 (model speed fraction).")
    values = np.asarray(limits, dtype=float)
    if values.shape != (3, 7) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Panda model must provide seven positive finite velocity, acceleration and jerk limits.")
    return tuple(values[i].copy() * speed**(i+1) for i in range(3))


@dataclass(frozen=True)
class Pose:
    position: tuple
    orientation: tuple

    def __post_init__(self):
        p = np.asarray(self.position, dtype=float)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError("Position must contain three finite values.")
        object.__setattr__(self, "position", tuple(p))
        object.__setattr__(self, "orientation", tuple(unit_quat(self.orientation)))

    def compose(self, other):
        return Pose(np.asarray(self.position) + rotate(self.orientation, other.position),
                    quat_mul(self.orientation, other.orientation))

    def inverse(self):
        q = conjugate(self.orientation)
        return Pose(-rotate(q, self.position), q)

    def interpolate(self, other, t):
        return Pose((1-t)*np.asarray(self.position) + t*np.asarray(other.position),
                    slerp(self.orientation, other.orientation, t))

    def error(self, other):
        return (float(np.linalg.norm(np.asarray(self.position)-other.position)),
                angular_error(self.orientation, other.orientation))


@dataclass(frozen=True)
class Waypoint:
    path: str
    pose: Pose
    enabled: bool = True
    gripper: str = "keep"
    dwell: float = .5
    speed: float = .3
    position_tolerance: float = .005
    orientation_tolerance: float = math.radians(3)
    timeout: float = 30.

    def __post_init__(self):
        if self.gripper not in ("keep", "open", "close"):
            raise ValueError("Gripper action must be keep, open, or close.")
        if not all(math.isfinite(v) for v in (self.dwell, self.speed, self.position_tolerance,
                                              self.orientation_tolerance, self.timeout)):
            raise ValueError("Waypoint settings must be finite.")
        if not 0 < self.speed <= 1:
            raise ValueError("Speed must be greater than 0 and at most 1 (model speed fraction).")
        if not (0 <= self.dwell and self.timeout > self.dwell
                and 0 < self.position_tolerance <= .1 and 0 < self.orientation_tolerance <= math.pi):
            raise ValueError("Invalid speed, tolerance, dwell, or timeout.")


class ArrivalGate:
    """An action is emitted once after sustained arrival; dwell follows it.

    Closing on an object need not reach zero finger separation. Gripper readiness
    is provided by the robot adapter and is not a grasp-success predicate.
    """
    def __init__(self, waypoint, settle=.15):
        self.waypoint = waypoint
        self.settle = settle
        self.elapsed = self.settled = self.dwelled = 0.
        self.acted = False
        self.complete = False

    def step(self, dt, pose_error, motion_complete, gripper_ready):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Physics timestep must be positive and finite.")
        self.elapsed += dt
        if self.complete:
            return None
        if self.elapsed > self.waypoint.timeout:
            raise TimeoutError(f"{self.waypoint.path}: timed out (pose error {pose_error}).")
        reached = (motion_complete and pose_error[0] <= self.waypoint.position_tolerance
                   and pose_error[1] <= self.waypoint.orientation_tolerance)
        if not self.acted:
            self.settled = self.settled + dt if reached else 0.
            if self.settled >= self.settle:
                self.acted = True
                return self.waypoint.gripper
        else:
            self.dwelled = self.dwelled + dt if reached and gripper_ready else 0.
            self.complete = self.dwelled >= max(self.waypoint.dwell, dt)
        return None
