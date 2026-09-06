"""Kit-only regressions exercising the actual enabled extension instance."""
import asyncio
from dataclasses import replace
import uuid
import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.kit.undo
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from simdroid_data_collection.domain import Pose, euler_quat
from simdroid_data_collection.stage_store import DEFAULT_TCP, StageStore, author, pose_matrix, set_world_pose, world_pose


def enabled_owner(manager):
    # Test-only introspection of Kit 107.3's actual IExt instance. Production
    # code never depends on Kit's private extension bookkeeping.
    from omni.ext._impl import _internal
    ext_id = manager.get_enabled_extension_id("simdroid.data_collection")
    return _internal._extensions[ext_id]._started_extensions[0][0]


async def frames(count=4):
    for _ in range(count):
        await omni.kit.app.get_app().next_update_async()


def expect_error(fn, text=None):
    try:
        fn()
    except (ValueError, RuntimeError) as exc:
        if text:
            assert text in str(exc), str(exc)
        return
    raise AssertionError("Operation unexpectedly succeeded")


async def authoring_audit(context, manager, runtime):
    await context.new_stage_async()
    stage = context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Y")
    # Minimal stopped-stage Panda fixtures: no simulation is started here.
    for root in ("/World/PandaA", "/World/PandaB"):
        prim = UsdGeom.Xform.Define(stage, root).GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        for name in ("panda_link0", "panda_hand", "panda_leftfinger", "panda_rightfinger"):
            UsdGeom.Xform.Define(stage, root+"/"+name)
        for i in range(1, 8):
            UsdPhysics.RevoluteJoint.Define(stage, root+f"/panda_joint{i}")
        for i in (1, 2):
            UsdPhysics.PrismaticJoint.Define(stage, root+f"/panda_finger_joint{i}")
    owner = enabled_owner(manager)
    import omni.kit.window.property
    omni.kit.window.property.get_window().set_visible(True)
    await frames(20)
    owner.refresh_robots()
    owner.choose_robot("/World/PandaA")
    owner.bind(Pose((0, 0, .2), (1, 0, 0, 0)))
    seq_a = owner.store.sequence
    owner.add_at_tcp()
    goal_a = owner.selected
    owner.new_sequence()
    seq_b = owner.store.sequence
    assert owner.robot is None, "New sequence must invalidate the old TCP binding"
    assert owner.ensure_robot().tcp == DEFAULT_TCP
    owner.add_at_tcp()
    goal_b = owner.selected
    owner.bind(Pose((0, 0, .3), (1, 0, 0, 0)))
    omni.kit.undo.undo()
    assert owner.ensure_robot().tcp == DEFAULT_TCP, "Undo TCP calibration must refresh the binding"
    owner.choose_robot("/World/PandaB")
    owner.bind()
    omni.kit.undo.undo()
    assert owner.ensure_robot().path == "/World/PandaA", "Undo robot binding must not be overwritten"
    owner.select(goal_a)
    await frames()
    assert owner.store.sequence == seq_a and owner.selected == goal_a
    owner.select(goal_b)
    await frames()
    assert owner.store.sequence == seq_b and owner.selected == goal_b
    owner.choose_sequence(seq_a)
    owner.store.sequence = seq_b
    owner.observed_config = owner.configuration()
    owner.select(goal_b)
    root_layer = stage.GetRootLayer()
    assert stage.GetEditTarget().GetLayer() == root_layer
    context.get_selection().set_selected_prim_paths(["/World/PandaA"], False)
    owner.sync_selection()
    assert stage.GetEditTarget().GetLayer() == root_layer
    print("PASS audit: new-sequence TCP, undo calibration/robot, cross-sequence selection, edit target", flush=True)

    store = owner.store
    wp = store.read(goal_b)
    with Usd.EditContext(stage, store.layer):
        set_world_pose(stage.GetPrimAtPath(seq_b), Pose((.2, .3, -.1), euler_quat((20, 0, 40))))
        UsdGeom.Xformable(stage.GetPrimAtPath(goal_b)).SetResetXformStack(True)
    target = Pose((.4, .5, .6), euler_quat((35, -20, 75)))
    store.update(replace(wp, pose=target))
    distance, angle = store.read(goal_b).pose.error(target)
    assert distance < 1e-8 and angle < 1e-6
    bad = store.add(target)
    with Usd.EditContext(stage, store.layer):
        prim = stage.GetPrimAtPath(bad)
        author(prim, "enabled", Sdf.ValueTypeNames.Bool, False)
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d().SetScale(2.))
    assert len(store.snapshot()) == 1, "Disabled malformed pose must not block valid enabled goals"
    owner.select(bad)
    owner.preview.update(store)
    owner.ui.build()
    assert bad not in owner.preview.marker_paths
    owner.delete()
    assert bad not in store.paths(), "Malformed goals must remain removable"
    missing = store.add(target)
    with Usd.EditContext(stage, store.layer):
        stage.RemovePrim(missing)
    store.remove(missing)
    assert missing not in store.paths()
    marker = owner.preview.marker_paths[goal_b]
    with Usd.EditContext(stage, owner.preview.layer):
        stage.RemovePrim(marker)
    owner.preview.update(store)
    assert stage.GetPrimAtPath(owner.preview.marker_paths[goal_b])
    print("PASS audit: reset-stack poses, disabled/invalid/deleted goals, resilient UI and markers", flush=True)

    filename = runtime / f"audit_layer_{uuid.uuid4().hex[:8]}.usda"
    owner.select(goal_b)
    previous_pose = store.read(goal_b).pose
    moved_pose = Pose((.7, .8, .9), (1, 0, 0, 0))
    ok, _ = omni.kit.commands.execute("TransformPrim", path=goal_b,
                                      new_transform_matrix=pose_matrix(moved_pose))
    assert ok and store.read(goal_b).pose.error(moved_pose)[0] < 1e-8
    store.save(str(filename))
    context.get_selection().set_selected_prim_paths(["/World/PandaA"], False)
    owner.sync_selection()
    omni.kit.undo.undo()
    assert store.read(goal_b).pose.error(previous_pose)[0] < 1e-8
    assert not stage.GetRootLayer().GetPrimAtPath(goal_b), "Gizmo undo leaked opinions into the root layer"
    omni.kit.undo.redo()
    assert store.read(goal_b).pose.error(moved_pose)[0] < 1e-8
    assert stage.GetEditTarget().GetLayer() == root_layer
    store.save(str(filename))
    units = UsdGeom.GetStageMetersPerUnit(stage)
    UsdGeom.SetStageMetersPerUnit(stage, .01)
    expect_error(store.snapshot, "matching units")
    owner.ui.build()
    UsdGeom.SetStageMetersPerUnit(stage, units)
    other_stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(other_stage, 1.)
    UsdGeom.SetStageUpAxis(other_stage, "Y")
    UsdGeom.Xform.Define(other_stage, "/World/FrankaPaths")
    expect_error(lambda: StageStore(other_stage).load(str(filename)), "unrelated")
    expect_error(lambda: store.save(str(runtime/"kit.log")))
    old_layer = store.layer
    stage.GetRootLayer().subLayerPaths.remove(str(filename))
    expect_error(store._layer, "detached")
    stage.GetRootLayer().subLayerPaths.insert(0, str(filename))
    assert store._layer() == old_layer
    print("PASS audit: unit mismatch, occupied paths, protected files, detached authoring layers", flush=True)
    context.get_selection().clear_selected_prim_paths()
    await frames(12)


async def controller_audit(owner):
    """Use the UI's actual launch/cancel/physics callbacks, not a second runner."""
    paths = owner.store.paths()
    for path, action in ((paths[0], 2), (paths[-1], 1)):
        owner.select(path)
        owner.ui.build()
        owner.ui.gripper.model.get_item_value_model().set_value(action)
        owner.ui.apply_waypoint()
    owner.launch("validate")
    task = owner.task
    owner.abort()
    await frames(5)
    assert task.done() and owner.task is None and not owner.runner.active
    owner.launch("validate")
    for _ in range(600):
        await frames(1)
        if owner.task is None:
            break
    assert owner.task is None and owner.message.startswith("Validated"), owner.message
    owner.launch("sequence")
    for _ in range(600):
        await frames(1)
        if owner.runner.state != "planning":
            break
    assert owner.runner.state == "running", owner.message
    await frames(5)
    owner.runner.pause()
    elapsed = owner.runner.elapsed
    await frames(8)
    assert owner.runner.elapsed == elapsed
    owner.resume()
    for _ in range(3600):
        await frames(1)
        if not owner.runner.active:
            break
    assert owner.runner.state == "complete", owner.message
    assert [e["action"] for e in owner.runner.events if e["type"] == "gripper_action"] == ["close", "open"]
    assert np.min(owner.robot.art.get_joint_positions(owner.robot.finger_indices)) > .035
    # Check actual finger motion at the first waypoint, not only event labels.
    first_samples = [sample for sample in owner.runner.samples if sample["waypoint"] == paths[0]]
    assert max(np.asarray(first_samples[-1]["joint_positions"])[owner.robot.finger_indices]) < .003
    events = list(owner.runner.events)
    owner.launch("validate")
    owner.abort()
    await frames(4)
    assert owner.runner.events == events, "Cancelled validation corrupted the previous run"
    assert not owner.store.layer.GetPrimAtPath(owner.robot_path), "Robot opinions leaked into the waypoint layer"
    print("PASS audit: real extension validate/cancel/run/pause/resume, preserved recordings", flush=True)
