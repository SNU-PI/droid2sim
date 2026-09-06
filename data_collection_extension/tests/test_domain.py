"""Run with any Python environment containing NumPy; no Kit required."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "exts/simdroid.data_collection/simdroid_data_collection/domain.py"
spec = importlib.util.spec_from_file_location("waypoint_domain", SOURCE)
d = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = d
spec.loader.exec_module(d)


class DomainTests(unittest.TestCase):
    def test_tcp_inverse(self):
        goal = d.Pose((.2, -.7, .3), d.euler_quat((20, 80, -60)))
        offset = d.Pose((0, 0, .107), d.euler_quat((0, 0, 45)))
        hand = goal.compose(offset.inverse())
        np.testing.assert_allclose(hand.compose(offset).position, goal.position, atol=1e-12)
        self.assertLess(hand.compose(offset).error(goal)[1], 1e-7)

    def test_rotated_tcp_offset(self):
        hand = d.Pose((1, 2, 3), d.euler_quat((0, 90, 0)))
        result = hand.compose(d.Pose((0, 0, .107), (1, 0, 0, 0)))
        np.testing.assert_allclose(result.position, (1.107, 2, 3), atol=1e-12)

    def test_quaternion_double_cover(self):
        q = d.euler_quat((40, -20, 140))
        self.assertLess(d.angular_error(q, -q), 1e-7)
        self.assertLess(d.angular_error(q, d.slerp(q, -q, .5)), 1e-7)

    def test_large_finite_quaternion_is_normalized(self):
        q = d.unit_quat((1e308, 1e308, 0, 0))
        np.testing.assert_allclose(q, (2**-.5, 2**-.5, 0, 0))

    def test_slerp_shortest_path(self):
        middle = d.slerp(d.euler_quat((0, 0, 170)), d.euler_quat((0, 0, -170)), .5)
        self.assertLess(d.angular_error(middle, d.euler_quat((0, 0, 180))), 1e-7)

    def test_euler_roundtrip(self):
        q = d.euler_quat((45, -35, 130))
        self.assertLess(d.angular_error(q, d.euler_quat(d.quat_euler(q))), 1e-7)

    def test_validation(self):
        with self.assertRaises(ValueError):
            d.Pose((0, 0, 0), (0, 0, 0, 0))
        pose = d.Pose((0, 0, 0), (1, 0, 0, 0))
        for kwargs in ({"speed": 0}, {"dwell": -1}, {"timeout": .1}, {"gripper": "toggle"},
                       {"position_tolerance": float("nan")}):
            with self.assertRaises(ValueError):
                d.Waypoint("/goal", pose, **kwargs)

    def test_native_joint_limits_at_full_speed(self):
        limits = np.arange(1, 22, dtype=float).reshape(3, 7)
        actual = d.scaled_joint_limits(limits, 1.)
        np.testing.assert_array_equal(actual, limits)
        actual[0][0] = 999
        self.assertEqual(limits[0, 0], 1., "Returned limits must not alias the native model")

    def test_joint_limits_use_derivative_time_scaling(self):
        limits = np.arange(1, 22, dtype=float).reshape(3, 7)
        for speed in (.3, .7, 1., .3):
            actual = d.scaled_joint_limits(limits, speed)
            for i in range(3):
                np.testing.assert_allclose(actual[i], limits[i]*speed**(i+1))
        np.testing.assert_array_equal(limits, np.arange(1, 22).reshape(3, 7))

    def test_invalid_joint_limits_and_speed_are_rejected(self):
        for limits in (np.ones((3, 6)), np.zeros((3, 7)), -np.ones((3, 7)),
                       np.full((3, 7), np.nan), np.full((3, 7), np.inf)):
            with self.assertRaises(ValueError):
                d.scaled_joint_limits(limits, 1.)
        for speed in (0, -1, 1.01, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "Speed"):
                d.scaled_joint_limits(np.ones((3, 7)), speed)

    def test_arrival_action_once_and_dwell(self):
        wp = d.Waypoint("/goal", d.Pose((0, 0, 0), (1, 0, 0, 0)), gripper="close", dwell=.2)
        gate = d.ArrivalGate(wp)
        self.assertIsNone(gate.step(.1, (0, 0), True, True))
        self.assertIsNone(gate.step(.1, (.1, 0), True, True))
        self.assertIsNone(gate.step(.1, (0, 0), True, True))
        self.assertEqual(gate.step(.1, (0, 0), True, True), "close")
        self.assertIsNone(gate.step(.1, (0, 0), True, False))
        self.assertFalse(gate.complete)
        self.assertIsNone(gate.step(.1, (0, 0), True, True))
        self.assertIsNone(gate.step(.1, (0, 0), True, True))
        self.assertTrue(gate.complete)
        self.assertIsNone(gate.step(.1, (0, 0), True, True))

    def test_orientation_motion_and_timeout_gates(self):
        wp = d.Waypoint("/goal", d.Pose((0, 0, 0), (1, 0, 0, 0)), timeout=1.)
        gate = d.ArrivalGate(wp)
        self.assertIsNone(gate.step(.3, (0, 1), True, True))
        self.assertIsNone(gate.step(.3, (0, 0), False, True))
        with self.assertRaises(TimeoutError):
            gate.step(.5, (0, 1), True, True)
        with self.assertRaises(ValueError):
            d.ArrivalGate(wp).step(0, (0, 0), True, True)


if __name__ == "__main__":
    unittest.main()
