"""Read-only-on-disk saved-scene motion regression; run inside kit_smoke.py."""
from pathlib import Path
import time
import uuid
import numpy as np
from audit_cases import enabled_owner, frames
from ui_interaction_cases import settle


async def saved_motion_audit(context, manager, runtime, scene_filename, waypoint_filename, probe_gravity=False):
    from simdroid_data_collection.domain import Pose, scaled_joint_limits
    from isaacsim.core.utils.rotations import rot_matrix_to_quat
    from pxr import PhysxSchema

    sources = [Path(scene_filename).resolve(), Path(waypoint_filename).resolve()]
    original_stats = [(p.stat().st_size, p.stat().st_mtime_ns) for p in sources]
    opened, error = await context.open_stage_async(str(sources[0]))
    assert opened, error
    await settle(1.)
    owner = enabled_owner(manager)
    owner.layer_file = str(sources[1])
    owner.load()  # Load before Bind so no anonymous sequence is created.
    owner.bind()
    snapshot = owner.store.snapshot()
    assert len(snapshot) >= 2, "The saved-motion comparison needs at least two waypoints"
    await settle(.6)
    print(f"Saved goals: {[(w.path, w.speed) for w in snapshot]}", flush=True)
    original_gravity = None
    try:
        owner.timeline.play()
        await frames(40)
        owner.bind()
        await settle(.5)
        if not owner.robot.ready():
            owner.timeline.stop()
            await frames(12)
            owner.timeline.play()
            await frames(40)
            owner.bind()
        robot = owner.robot
        assert robot.ready()
        print(f"Scene joint gains: {robot.art.get_articulation_controller().get_gains()}", flush=True)
        print(f"Scene gravity disabled: {[(n, PhysxSchema.PhysxRigidBodyAPI(p).GetDisableGravityAttr().Get()) for n, p in robot.links.items() if n.startswith('panda_link') or n == 'panda_hand']}", flush=True)
        owner.runner.prepare(robot)  # Hold the arm during all timing comparisons.
        owner.preview.set_hidden(True)
        native_limits = robot.motion_limits
        original_q = robot.arm_positions()
        # Capture one set of IK samples and time-parameterize it twice. Comparing
        # separate IK plans is not exact: the real arm settles slightly under
        # gravity even while holding. Never freeze physics or teleport it here.
        generator = robot.generator
        sampled_paths = []

        class RecordingGenerator:
            def __getattr__(self, name):
                return getattr(generator, name)

            def compute_c_space_trajectory(self, samples):
                sampled_paths.append(np.array(samples, copy=True))
                return generator.compute_c_space_trajectory(samples)

        try:
            robot.generator = RecordingGenerator()
            new_segments = await robot.plan(snapshot)
        finally:
            robot.generator = generator
        assert np.max(np.abs(robot.arm_positions()-original_q)) < .02
        old_durations = []
        path_iterator = iter(sampled_paths)
        for new in new_segments:
            old_duration = 0.
            if new.trajectory:
                speed = new.waypoint.speed
                generator.set_c_space_velocity_limits(np.full(7, .7*speed))
                generator.set_c_space_acceleration_limits(np.full(7, 1.5*speed**2))
                generator.set_c_space_jerk_limits(np.full(7, 10.*speed**3))
                old = generator.compute_c_space_trajectory(next(path_iterator))
                assert old is not None
                old_duration = old.end_time-old.start_time
            old_durations.append(old_duration)
            print(f"MOTION {new.waypoint.path}: speed={new.waypoint.speed:g}; "
                  f"old={old_duration:.3f}s -> native={new.duration:.3f}s simulation", flush=True)
            assert new.duration <= old_duration+.05, "Native limits unexpectedly slowed the path"
            if new.trajectory:
                velocity, acceleration, jerk = scaled_joint_limits(native_limits, new.waypoint.speed)
                times = np.linspace(new.trajectory.start_time, new.trajectory.end_time, 1001)
                for derivative, limit in ((1, velocity), (2, acceleration), (3, jerk)):
                    actual = np.array([new.trajectory.trajectory.eval(t, derivative) for t in times])
                    assert np.all(np.abs(actual) <= limit*(1+1e-3)+1e-5), (new.waypoint.path, derivative)
        velocity, acceleration, jerk = scaled_joint_limits(native_limits, snapshot[-1].speed)
        generator.set_c_space_velocity_limits(velocity)
        generator.set_c_space_acceleration_limits(acceleration)
        generator.set_c_space_jerk_limits(jerk)
        assert sum(s.duration for s in new_segments) < sum(old_durations)*.7
        print("PASS motion: same saved poses, faster timing, native velocity/acceleration/jerk bounds", flush=True)

        owner.runner.start(new_segments)
        deadline = time.monotonic()+90.
        next_report = time.monotonic()+2.
        while owner.runner.active and time.monotonic() < deadline:
            await frames(1)
            if time.monotonic() >= next_report:
                next_report = time.monotonic()+2.
                print(f"RUN: {owner.message}", flush=True)
            if (original_gravity is None and owner.runner.state == "running" and not owner.runner.gate.acted
                    and owner.runner.gate.elapsed > owner.runner.segment.duration+5.):
                for label, joints in (("commanded", owner.runner.commanded_arm), ("measured", robot.arm_positions())):
                    p, r = robot.ik.compute_forward_kinematics("panda_hand", joints)
                    fk_tcp = Pose(np.asarray(p)*robot.units, rot_matrix_to_quat(r)).compose(robot.tcp)
                    print(f"SETTLING {label}: model TCP error={fk_tcp.error(owner.runner.segment.waypoint.pose)}; "
                          f"model-vs-physics TCP={fk_tcp.error(robot.tcp_pose())}; joints={joints}", flush=True)
                if not probe_gravity:
                    break
                # Diagnostic only, not production behavior. Restore these live
                # tensor flags before stopping; never author USD gravity flags.
                view = robot.art._articulation_view
                original_gravity = view.get_body_disable_gravity().copy()
                robot.art.disable_gravity()
                owner.runner.metadata["diagnostic_robot_gravity_disabled"] = True
                print("DIAGNOSTIC: disabled gravity on this robot only; no gain or tolerance changes", flush=True)
        assert owner.runner.state == "complete", owner.message
        error = robot.tcp_pose().error(snapshot[-1].pose)
        assert error[0] <= snapshot[-1].position_tolerance, error
        assert error[1] <= snapshot[-1].orientation_tolerance, error
        samples = owner.runner.samples
        measured = np.array([s["joint_positions"] for s in samples])[:, robot.arm_indices]
        commanded = np.array([s["commanded_arm"] for s in samples])
        peak_tracking = float(np.max(np.abs(measured-commanded)))
        qualifier = " (after gravity diagnostic, NOT unmodified-scene execution)" if original_gravity is not None else ""
        print(f"PASS motion execution{qualifier}: {owner.runner.elapsed:.3f}s sim / "
              f"{owner.runner.wall_elapsed:.3f}s wall; "
              f"rate={owner.runner.simulation_rate:.3f}x; "
              f"final={error[0]*1000:.3f}mm/{np.degrees(error[1]):.3f}deg; "
              f"peak sampled joint tracking error={peak_tracking:.5f}rad", flush=True)
        output = runtime / f"saved_motion_{uuid.uuid4().hex[:8]}.json"
        owner.runner.export(str(output))
        print(f"Motion telemetry: {output}", flush=True)
    finally:
        if original_gravity is not None and robot.ready():
            robot.art._articulation_view.set_body_disable_gravity(original_gravity)
        owner.abort()
        if owner.runner.samples:
            output = runtime / f"saved_motion_diagnostic_{uuid.uuid4().hex[:8]}.json"
            owner.runner.export(str(output))
            print(f"Diagnostic telemetry: {output}", flush=True)
        owner.timeline.stop()
        await frames(12)
        for source, before in zip(sources, original_stats):
            assert (source.stat().st_size, source.stat().st_mtime_ns) == before, f"Modified source: {source}"
        print("PASS motion: input scene and saved waypoint files unchanged on disk", flush=True)
