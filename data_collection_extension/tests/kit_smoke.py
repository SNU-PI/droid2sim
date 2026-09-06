"""Headless integration: extension lifecycle, USD edits, and optional real Panda.

Use Isaac Sim's Python: python -B tests/kit_smoke.py --gpu 7 [--robot-usd URL]
All test outputs/caches are directed into this project's ignored .runtime folder.
"""
import argparse
import asyncio
import os
from pathlib import Path
import sys
import traceback
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=7, help="Physical GPU index; only this CUDA device is visible")
parser.add_argument("--robot-usd")
parser.add_argument("--up-axis", choices=("Y", "Z"), default="Z")
parser.add_argument("--audit", action="store_true")
parser.add_argument("--ui-test", action="store_true", help="Exercise native mouse/keyboard interactions")
parser.add_argument("--scene", help="Open a real scene for an in-memory UI/motion smoke test; never save it")
parser.add_argument("--waypoints", help="With --scene, compare old/native timing and execute this saved waypoint layer")
parser.add_argument("--probe-gravity", action="store_true", help="Diagnostic only: test robot-only gravity disabling if arrival stalls")
args = parser.parse_args()
if args.waypoints and not args.scene:
    parser.error("--waypoints requires --scene")
if args.probe_gravity and not args.waypoints:
    parser.error("--probe-gravity requires --waypoints")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
sys.argv = [sys.argv[0]]
sys.dont_write_bytecode = True
project = Path(__file__).resolve().parents[1]
runtime = project / ".runtime"
for key, folder in (("XDG_CACHE_HOME", "cache"), ("XDG_CONFIG_HOME", "config"),
                    ("XDG_DATA_HOME", "data"), ("XDG_STATE_HOME", "state"),
                    ("CUDA_CACHE_PATH", "cuda"), ("TMPDIR", "tmp"),
                    ("__GL_SHADER_DISK_CACHE_PATH", "gl")):
    destination = runtime / folder
    destination.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(destination)

from isaacsim import SimulationApp
app = SimulationApp({
    # Vulkan uses the physical index; CUDA remaps the sole visible GPU to zero.
    "headless": True, "hide_ui": False, "active_gpu": args.gpu, "physics_gpu": 0,
    "multi_gpu": False, "max_gpu_count": 1, "fast_shutdown": True,
    "extra_args": ["--portable-root", str(runtime), "--/app/settings/persistent=false",
                   f"--/log/file={runtime / 'kit.log'}", "--/crashreporter/enabled=false",
                   "--/renderer/multiGpu/autoEnable=false",
                   "--ext-folder", str(project / "exts")],
})


async def check():
    import numpy as np
    import omni.kit.app
    import omni.kit.undo
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("simdroid.data_collection", True)
    assert manager.is_extension_enabled("simdroid.data_collection")
    from simdroid_data_collection.domain import Pose, Waypoint, euler_quat
    from simdroid_data_collection.stage_store import StageStore, world_pose, set_world_pose
    from simdroid_data_collection.preview import PreviewManager
    import omni.ui
    for _ in range(12):
        await omni.kit.app.get_app().next_update_async()
    assert omni.ui.Workspace.get_window("Franka Waypoint Editor") is not None
    context = omni.usd.get_context()
    await context.new_stage_async()
    stage = context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Y")
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/TestRobot")
    store = StageStore(stage)
    sequence = store.create_sequence("/World/TestRobot")
    pose = Pose((.2, .6, -.1), euler_quat((20, 70, -15)))
    a = store.add(pose)
    b = store.add(Pose((.3, .5, 0), (1, 0, 0, 0)))
    UsdGeom.Xform.Define(stage, "/World/UnrelatedXform")
    assert len(store.snapshot()) == 2
    store.reorder(b, -1)
    assert store.paths() == [b, a]
    store.remove(b)
    assert store.paths() == [a]
    omni.kit.undo.undo()
    assert store.paths() == [b, a]
    omni.kit.undo.redo()
    assert store.paths() == [a]
    preview = PreviewManager(stage)
    preview.update(store)
    assert stage.GetPrimAtPath(preview.root)
    assert preview.root not in stage.GetRootLayer().ExportToString()
    preview.destroy()
    assert not stage.GetPrimAtPath(preview.root)
    filename = runtime / f"waypoints_{uuid.uuid4().hex[:8]}.usda"
    previous_target = stage.GetEditTarget()
    store.save(str(filename))
    assert stage.GetEditTarget() == previous_target
    omni.kit.undo.undo()
    assert store.paths() == [b, a], "Undo must still work after saving the authoring layer"
    omni.kit.undo.redo()
    assert store.paths() == [a]
    store.save(str(filename))
    loaded_stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(loaded_stage, 1.)
    UsdGeom.SetStageUpAxis(loaded_stage, "Y")
    loaded = StageStore(loaded_stage)
    loaded.load(str(filename))
    assert loaded.sequence == sequence and loaded.paths() == [a]
    np.testing.assert_allclose(loaded.read(a).pose.position, pose.position, atol=1e-9)
    assert loaded.read(a).pose.error(pose)[1] < 1e-7
    print("PASS: Kit enable, tagged goals, order, undo/redo, session previews, USD save/load", flush=True)

    if args.ui_test:
        from ui_interaction_cases import interaction_audit
        await interaction_audit(context, manager, runtime)

    if args.scene:
        if args.waypoints:
            from motion_cases import saved_motion_audit
            await saved_motion_audit(context, manager, runtime, args.scene, args.waypoints, args.probe_gravity)
        else:
            from ui_interaction_cases import scene_smoke
            await scene_smoke(context, manager, runtime, args.scene)

    if args.audit:
        from audit_cases import authoring_audit
        await authoring_audit(context, manager, runtime)

    if args.robot_usd:
        import builtins
        # Run the World API in its extension/async mode inside this coroutine.
        builtins.ISAAC_LAUNCHED_FROM_TERMINAL = True
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage
        from simdroid_data_collection.robot import FrankaBinding
        from simdroid_data_collection.runner import SequenceRunner
        await context.new_stage_async()
        stage = context.get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, "Z")
        add_reference_to_stage(args.robot_usd, "/World/Franka")
        world = World(stage_units_in_meters=1., physics_dt=1/60, rendering_dt=1/60)
        await world.initialize_simulation_context_async()
        UsdGeom.SetStageUpAxis(stage, args.up_axis)
        world.get_physics_context().set_gravity(-9.81)
        if args.up_axis == "Y":
            set_world_pose(stage.GetPrimAtPath("/World/Franka"),
                           Pose((.3, .1, -.2), euler_quat((-90, 0, 0))))
        store = StageStore(stage)
        store.create_sequence("/World/Franka")
        preview = PreviewManager(stage)
        await world.reset_async()
        binding = FrankaBinding(stage, "/World/Franka", store.tcp())
        binding.initialize()
        # Test fixture only: start bent away from the stock asset's straight-arm
        # workspace boundary. The extension's execution path never teleports.
        nominal = np.array([0., -.5, 0., -2., 0., 1.5, .7854])
        binding.art.set_joint_positions(nominal, binding.arm_indices)
        binding.art.set_joint_velocities(np.zeros(7), binding.arm_indices)
        binding.command_arm(nominal)
        for _ in range(20):
            await omni.kit.app.get_app().next_update_async()
        initial = binding.tcp_pose()
        delta = (0, .025, 0) if args.up_axis == "Y" else (0, 0, .025)
        target = Pose(np.asarray(initial.position)+delta, initial.orientation)
        first = store.add(target)
        second = store.add(initial)
        count = preview.show_gripper(binding, store.read(first), "close")
        assert count > 0
        actual = world_pose(stage.GetPrimAtPath(preview.root+"/Gripper")).compose(binding.tcp)
        assert actual.error(target)[0] < 1e-8 and actual.error(target)[1] < 1e-7
        for prim in Usd.PrimRange(stage.GetPrimAtPath(preview.root)):
            assert not prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.HasAPI(UsdPhysics.CollisionAPI)
        preview.hide_gripper()
        preview.set_hidden(True)
        runner = SequenceRunner(lambda msg: None)
        runner.prepare(binding)
        world.add_physics_callback("waypoint_test", runner.step)
        segments = await binding.plan((Waypoint(first, target, gripper="close"),
                                       Waypoint(second, initial, gripper="open")), print)
        runner.start(segments)
        for _ in range(3600):
            await omni.kit.app.get_app().next_update_async()
            if not runner.active:
                break
        assert runner.state == "complete", (runner.state, runner.events[-3:])
        assert [e["action"] for e in runner.events if e["type"] == "gripper_action"] == ["close", "open"]
        assert binding.tcp_pose().error(initial)[0] < .005
        runner.export(str(runtime / f"run_{uuid.uuid4().hex[:8]}.json"))
        world.remove_physics_callback("waypoint_test")
        if args.audit:
            from audit_cases import enabled_owner, controller_audit
            owner = enabled_owner(manager)
            owner.reconcile_configuration()
            owner.choose_sequence(store.sequence)
            owner.refresh_robots()
            owner.choose_robot("/World/Franka")
            owner.bind()
            await controller_audit(owner)
            owner.launch("validate")
            pending = owner.task
            router = owner.edit_router
            import weakref
            owner_ref = weakref.ref(owner)
            owner = None
            manager.set_extension_enabled_immediate("simdroid.data_collection", False)
            assert not router.callbacks
            assert omni.kit.commands.get_command_class("FrankaWaypointLayerEdit") is None
            for _ in range(5):
                await omni.kit.app.get_app().next_update_async()
            assert pending.done()
            assert pending.cancelled() or pending.exception() is None
            assert owner_ref() is None, "Planning retained the unloaded extension instance"
            assert binding.ready(), "Removing editor previews invalidated the robot's physics handles"
            print("PASS audit: unload during async planning", flush=True)
        await world.stop_async()
        assert not store.layer.GetPrimAtPath("/World/Franka"), "Physics Stop wrote robot transforms into the waypoint layer"
        preview.destroy()
        World.clear_instance()
        print(f"PASS: {args.up_axis}-up Panda FK, {count} physics-free preview meshes, Lula trajectory, close/open, telemetry", flush=True)

    if args.audit:
        owner = None
    was_enabled = manager.is_extension_enabled("simdroid.data_collection")
    manager.set_extension_enabled_immediate("simdroid.data_collection", False)
    assert not manager.is_extension_enabled("simdroid.data_collection")
    # Kit can re-register Command subclasses retained in its undo history on a
    # later script-change event. Check explicit unregister immediately, not
    # after unrelated event-loop iterations that invoke Kit's class scanner.
    if was_enabled:
        assert omni.kit.commands.get_command_class("FrankaWaypointLayerEdit") is None
    assert not any(str(p.GetPath()).startswith("/__FrankaWaypointPreview_") for p in context.get_stage().Traverse())
    print("PASS: extension shutdown", flush=True)


exit_code = 0
try:
    task = asyncio.ensure_future(check())
    while not task.done():
        app.update()
    task.result()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    # Kit fast shutdown can terminate the process itself; pass the failure code
    # to Kit before cleanup instead of relying only on the final sys.exit().
    app.app.post_uncancellable_quit(exit_code)
    app.close()
sys.exit(exit_code)
