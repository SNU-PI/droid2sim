"""Physics-driven sequence execution. No task progress is coupled to UI frames."""
import json
import math
import time
from pathlib import Path
import numpy as np
from .domain import ArrivalGate


class SequenceRunner:
    def __init__(self, status, clock=time.monotonic):
        self.status = status
        self.clock = clock
        self.robot = None
        self.segments = []
        self.state = "idle"
        self.events = []
        self.samples = []
        self.elapsed = self.sample_clock = 0.
        self.hold = None
        self.wall_elapsed = 0.
        self.wall_tick = None

    def measure_wall_time(self):
        now = self.clock()
        if self.wall_tick is not None:
            self.wall_elapsed += max(0., now-self.wall_tick)
        self.wall_tick = now

    @property
    def simulation_rate(self):
        return self.elapsed/self.wall_elapsed if self.wall_elapsed > 1e-6 else 0.

    @property
    def active(self):
        return self.state in ("planning", "running", "paused")

    def prepare(self, robot):
        if self.active:
            raise ValueError("Abort the current run before starting another.")
        self.robot = robot
        if not robot.ready():
            raise ValueError("Robot physics is not ready.")
        self.hold = self._arm()
        self.base_at_start = robot.base_pose()
        self.commanded_arm = self.hold.copy()
        self.state = "planning"

    def _arm(self):
        positions = np.asarray(self.robot.arm_positions(), dtype=float)
        if positions.shape != (7,) or not np.all(np.isfinite(positions)):
            raise ValueError("Robot returned invalid arm joint positions.")
        return positions.copy()

    def check_base(self):
        distance, angle = self.robot.base_pose().error(self.base_at_start)
        if distance > .002 or angle > np.radians(.5):
            raise ValueError("Robot base moved; abort and re-plan from its new pose.")

    def start(self, segments):
        if self.state != "planning":
            raise ValueError("Planning is no longer active; cancelled plans cannot start.")
        if not self.robot.ready():
            raise ValueError("Robot physics changed while planning.")
        self.check_base()
        if not segments:
            raise ValueError("No planned segments.")
        if np.max(np.abs(self._arm()-segments[0].start)) > .08:
            raise ValueError("Robot moved while planning. Re-plan from its current position.")
        self.segments, self.index = segments, 0
        self.events, self.samples = [], []
        self.elapsed = self.sample_clock = 0.
        self.wall_elapsed, self.wall_tick = 0., self.clock()
        self.metadata = {"version": 1, "robot": self.robot.path, "units": "metres/radians",
                         "quaternion_order": "wxyz", "pose_frame": "world",
                         "tcp_in_hand": vars(self.robot.tcp),
                         "waypoints": [dict(vars(s.waypoint), pose=vars(s.waypoint.pose)) for s in segments],
                         "planned_segments": [{"waypoint": s.waypoint.path,
                                               "motion_seconds": s.duration} for s in segments]}
        self.state = "running"
        self._enter()

    def _enter(self):
        self.segment = self.segments[self.index]
        self.gate = ArrivalGate(self.segment.waypoint)
        self.segment_time = 0.
        self.playback = None
        self.events.append({"time": self.elapsed, "type": "waypoint_started",
                            "waypoint": self.segment.waypoint.path})

    def pause(self):
        if self.state == "running":
            self.measure_wall_time()
            self.wall_tick = None
            self.hold = self._arm()
            self.robot.command_arm(self.hold)
            self.state = "paused"
            self.events.append({"time": self.elapsed, "type": "paused"})
            self.status("Paused: arm held; gripper command preserved.")

    def resume(self):
        if self.state == "paused":
            # Resume only near the paused trajectory target, avoiding a jump.
            self.check_base()
            current = self._arm()
            if (np.max(np.abs(current-self.hold)) > .08 or
                    np.max(np.abs(current-self.commanded_arm)) > .08):
                raise ValueError("Robot moved while paused. Abort and re-plan.")
            self.state = "running"
            self.wall_tick = self.clock()
            self.events.append({"time": self.elapsed, "type": "resumed"})

    def abort(self, reason="Aborted", hold=True):
        if self.state == "running":
            self.measure_wall_time()
        self.wall_tick = None
        if self.active and self.robot and hold:
            try:
                if self.robot.ready():
                    self.robot.command_arm(self._arm())
            except Exception:
                pass  # Physics handles may already be invalid after a stage/timeline change.
        # Cancelled validation/planning must not corrupt the previous recording.
        if self.state in ("running", "paused"):
            self.events.append({"time": self.elapsed, "type": "aborted", "reason": reason})
        self.state = "idle"
        self.segments = []
        self.status(reason)

    def step(self, dt):
        if not self.active:
            return
        try:
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("Physics timestep must be positive and finite.")
            if not self.robot or not self.robot.ready():
                self.abort("Physics handles changed; stop/play and bind the robot again.", hold=False)
                return
            if self.active and not self.robot.stage.GetPrimAtPath(self.robot.path):
                raise RuntimeError("Selected robot was deleted.")
            self.check_base()
            if self.state in ("planning", "paused"):
                self.robot.command_arm(self.hold)
                self.robot.tick_gripper(dt)
                return
            if self.state != "running":
                return
            self.measure_wall_time()
            segment = self.segment
            prim = self.robot.stage.GetPrimAtPath(segment.waypoint.path)
            if not prim or not prim.IsActive():
                raise RuntimeError("Active waypoint was deleted.")
            self.elapsed += dt
            self.segment_time = min(self.segment_time+dt, segment.duration)
            if segment.trajectory:
                if self.playback is None:
                    self.playback = self.robot.playback(segment, dt)
                action = self.playback.get_action_at_time(
                    segment.trajectory.start_time+self.segment_time)
                self.commanded_arm = np.asarray(action.joint_positions).copy()
                self.robot.art.apply_action(action)
            else:
                self.commanded_arm = segment.end.copy()
                self.robot.command_arm(segment.end)
            self.robot.tick_gripper(dt)
            measured = self.robot.tcp_pose()
            error = measured.error(segment.waypoint.pose)
            action = self.gate.step(dt, error, self.segment_time >= segment.duration,
                                    self.robot.gripper_ready())
            if action is not None:
                self.robot.command_gripper(action)
                self.events.append({"time": self.elapsed, "type": "gripper_action",
                                    "action": action, "waypoint": segment.waypoint.path})
            self.sample_clock += dt
            if self.sample_clock >= .05:
                self.sample_clock %= .05
                # Bound memory during accidental very long runs.
                if len(self.samples) >= 100000:
                    raise RuntimeError("Telemetry buffer full; export and start a new run.")
                self.samples.append({"time": self.elapsed, "wall_time": self.wall_elapsed,
                                     "simulation_rate": self.simulation_rate,
                                     "segment_time": self.segment_time,
                                     "planned_motion_seconds": segment.duration,
                                     "waypoint": segment.waypoint.path,
                                     "joint_positions": self.robot.art.get_joint_positions().tolist(),
                                     "joint_velocities": self.robot.art.get_joint_velocities().tolist(),
                                     "commanded_arm": self.commanded_arm.tolist(),
                                     "measured_tcp": vars(measured), "goal_tcp": vars(segment.waypoint.pose),
                                     "gripper_target": self.robot.gripper_target.tolist()})
                self.status(f"{self.index+1}/{len(self.segments)} {segment.waypoint.path.rsplit('/', 1)[-1]}: "
                            f"{error[0]*1000:.1f} mm / {np.degrees(error[1]):.1f}° "
                            f"({'action/dwell' if self.gate.acted else 'moving/settling'}) | "
                            f"motion {self.segment_time:.1f}/{segment.duration:.1f}s | "
                            f"sim {self.simulation_rate:.2f}x real time")
            if self.gate.complete:
                self.events.append({"time": self.elapsed, "type": "waypoint_finished",
                                    "waypoint": segment.waypoint.path})
                self.index += 1
                if self.index == len(self.segments):
                    self.state = "complete"
                    self.events.append({"time": self.elapsed, "type": "sequence_complete"})
                    self.wall_tick = None
                    self.status(f"Sequence complete: {self.elapsed:.1f}s simulation / "
                                f"{self.wall_elapsed:.1f}s wall time (pauses excluded). "
                                "This does not certify grasp/task success.")
                else:
                    self._enter()
        except Exception as exc:
            self.abort(f"Execution failed: {exc}")

    def export(self, filename):
        if self.active:
            raise ValueError("Finish or abort the sequence before exporting.")
        if not self.samples:
            raise ValueError("No recorded trajectory samples yet.")
        path = Path(filename).expanduser().resolve()
        if path.suffix != ".json" or not path.parent.is_dir():
            raise ValueError("Choose a .json file in an existing directory.")
        payload = json.dumps({"metadata": self.metadata, "events": self.events, "samples": self.samples},
                             indent=2, allow_nan=False)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
