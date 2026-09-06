"""Runner behavior with a deterministic fake articulation; no Kit required."""
from pathlib import Path
import importlib.util
import sys
import types
import unittest
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "exts/simdroid.data_collection/simdroid_data_collection"
package = types.ModuleType("runner_test_package")
package.__path__ = [str(SOURCE)]
sys.modules[package.__name__] = package
for name in ("domain", "runner"):
    spec = importlib.util.spec_from_file_location(f"runner_test_package.{name}", SOURCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from runner_test_package.domain import Pose, Waypoint
from runner_test_package.runner import SequenceRunner


class FakeRobot:
    def __init__(self):
        self.path = "/robot"
        self.tcp = Pose((0, 0, .107), (1, 0, 0, 0))
        self.pose = Pose((0, 0, 0), (1, 0, 0, 0))
        self.base = Pose((0, 0, 0), (1, 0, 0, 0))
        self.q = np.zeros(7)
        self.gripper_target = np.zeros(2)
        self.commands = []
        self.valid = True
        self.exists = True
        self.stage = types.SimpleNamespace(GetPrimAtPath=lambda path:
            types.SimpleNamespace(IsActive=lambda: True) if self.exists else None)
        self.art = types.SimpleNamespace(get_joint_positions=lambda: self.q.copy(),
                                         get_joint_velocities=lambda: np.zeros(7))

    def ready(self):
        return self.valid

    def arm_positions(self):
        return self.q.copy()

    def tcp_pose(self):
        return self.pose

    def base_pose(self):
        return self.base

    def command_arm(self, positions):
        self.last_arm = np.asarray(positions).copy()

    def tick_gripper(self, dt):
        pass

    def command_gripper(self, action):
        self.commands.append(action)

    def gripper_ready(self):
        return True


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot()
        self.messages = []
        self.wall_clock = 0.
        self.runner = SequenceRunner(self.messages.append, clock=lambda: self.wall_clock)
        self.runner.prepare(self.robot)

    def start(self, action="close", timeout=30.):
        waypoint = Waypoint("/goal", self.robot.pose, gripper=action, dwell=.1, timeout=timeout)
        segment = types.SimpleNamespace(waypoint=waypoint, trajectory=None,
                                        start=np.zeros(7), end=np.zeros(7), duration=0.)
        self.runner.start([segment])

    def test_action_and_completion(self):
        self.start()
        for _ in range(20):
            self.runner.step(.05)
        self.assertEqual(self.runner.state, "complete")
        self.assertEqual(self.robot.commands, ["close"])
        self.assertEqual(self.runner.events[-1]["type"], "sequence_complete")
        self.assertGreater(len(self.runner.samples), 0)

    def test_pause_freezes_sequence_clock(self):
        self.start()
        self.runner.step(.05)
        self.runner.pause()
        elapsed, gate_elapsed = self.runner.elapsed, self.runner.gate.elapsed
        for _ in range(50):
            self.runner.step(.1)
        self.assertEqual(self.runner.elapsed, elapsed)
        self.assertEqual(self.runner.gate.elapsed, gate_elapsed)
        self.runner.resume()
        self.assertEqual(self.runner.state, "running")

    def test_wall_clock_reports_slow_simulation(self):
        self.start()
        self.wall_clock = .2
        self.runner.step(.05)
        self.assertAlmostEqual(self.runner.wall_elapsed, .2)
        self.assertAlmostEqual(self.runner.simulation_rate, .25)
        self.assertAlmostEqual(self.runner.samples[-1]["wall_time"], .2)
        self.assertIn("sim 0.25x real time", self.messages[-1])
        self.assertIn("planned_segments", self.runner.metadata)

    def test_wall_clock_excludes_planning_and_pauses(self):
        self.wall_clock = 50.
        self.start()
        self.wall_clock = 50.2
        self.runner.step(.05)
        self.runner.pause()
        self.wall_clock = 150.2
        self.runner.step(.05)
        self.assertAlmostEqual(self.runner.wall_elapsed, .2)
        self.runner.resume()
        self.wall_clock = 150.4
        self.runner.step(.05)
        self.assertAlmostEqual(self.runner.wall_elapsed, .4)
        self.assertAlmostEqual(self.runner.elapsed, .1)

    def test_cancelled_validation_preserves_previous_timing(self):
        self.start()
        for i in range(10):
            self.wall_clock += .1
            self.runner.step(.1)
        self.assertEqual(self.runner.state, "complete")
        elapsed = self.runner.wall_elapsed
        self.runner.prepare(self.robot)
        self.wall_clock += 100.
        self.runner.abort("Validation cancelled")
        self.assertEqual(self.runner.wall_elapsed, elapsed)

    def test_resume_rejects_displaced_robot(self):
        self.start()
        self.runner.pause()
        self.robot.q[0] = .2
        with self.assertRaises(ValueError):
            self.runner.resume()
        self.assertEqual(self.runner.state, "paused")

    def test_abort_preserves_gripper(self):
        self.start()
        self.runner.abort()
        self.runner.step(.1)
        self.assertEqual(self.robot.commands, [])
        self.assertFalse(self.runner.active)
        np.testing.assert_allclose(self.robot.last_arm, self.robot.q)

    def test_missing_prim_aborts(self):
        self.start()
        self.robot.exists = False
        self.runner.step(.1)
        self.assertFalse(self.runner.active)
        self.assertIn("deleted", self.messages[-1])

    def test_invalid_handles_abort(self):
        self.start()
        self.robot.valid = False
        self.runner.step(.1)
        self.assertFalse(self.runner.active)
        self.assertIn("Physics handles", self.messages[-1])

    def test_timeout_aborts_without_action(self):
        self.start(timeout=.5)
        self.robot.pose = Pose((1, 0, 0), (1, 0, 0, 0))
        for _ in range(10):
            self.runner.step(.1)
        self.assertFalse(self.runner.active)
        self.assertEqual(self.robot.commands, [])
        self.assertIn("timed out", self.messages[-1])

    def test_rejects_moved_start(self):
        self.robot.q[0] = .3
        with self.assertRaises(ValueError):
            self.start()

    def test_aborted_plan_cannot_start(self):
        self.runner.abort()
        with self.assertRaises(ValueError):
            self.start()

    def test_invalid_timestep_never_commands_motion(self):
        self.start()
        self.runner.step(float("nan"))
        self.assertFalse(self.runner.active)
        self.assertEqual(self.runner.elapsed, 0.)

    def test_handle_exception_is_contained(self):
        self.start()
        def broken_ready():
            raise RuntimeError("stale handle")
        self.robot.ready = broken_ready
        self.runner.step(.1)
        self.assertFalse(self.runner.active)

    def test_cancelled_validation_preserves_previous_recording(self):
        self.start()
        for _ in range(10):
            self.runner.step(.1)
        events = list(self.runner.events)
        self.runner.prepare(self.robot)
        self.runner.abort("Validation cancelled")
        self.assertEqual(self.runner.events, events)

    def test_base_motion_aborts(self):
        self.start()
        self.robot.base = Pose((.1, 0, 0), (1, 0, 0, 0))
        self.runner.step(.1)
        self.assertFalse(self.runner.active)

    def test_base_motion_during_planning_cannot_start(self):
        self.robot.base = Pose((.1, 0, 0), (1, 0, 0, 0))
        with self.assertRaises(ValueError):
            self.start()

    def test_pause_rejects_large_tracking_error_on_resume(self):
        self.start()
        self.runner.step(.05)
        self.robot.q[0] = .3
        self.runner.pause()
        with self.assertRaises(ValueError):
            self.runner.resume()


if __name__ == "__main__":
    unittest.main()
