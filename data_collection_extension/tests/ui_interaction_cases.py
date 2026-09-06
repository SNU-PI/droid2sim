"""Regression tests for persistent controls and actual Kit mouse/keyboard input.

Unlike the motion tests, these intentionally keep controls open across many app
updates and deliver redundant model notifications like those from popup views.
"""
import time
import omni.kit.app
import omni.ui as ui
from pxr import UsdGeom, UsdPhysics
from audit_cases import enabled_owner, frames


async def settle(seconds=.4):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        await frames(1)


async def interaction_audit(context, manager, runtime):
    manager.set_extension_enabled_immediate("omni.kit.ui_test", True)
    import omni.kit.ui_test as ui_test
    from carb.input import KeyboardInput
    await context.new_stage_async()
    stage = context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Y")
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
    owner.refresh_robots()
    owner.ui.window.width = 600
    owner.ui.window.height = 900
    owner.ui.window.position_x = 20
    owner.ui.window.position_y = 20
    owner.ui.window.visible = True
    await settle()

    prefix = "Franka Waypoint Editor//Frame/**/"

    def button(text):
        return ui_test.find(prefix+f"Button[*].text=='{text}'")

    def robot_combo():
        return ui_test.find_all(prefix+"ComboBox[*]")[0]

    # First reproduce the no-op item-notification loop even without a mouse.
    owner.bind()
    await settle()
    bound = owner.robot
    original_combo = robot_combo().widget
    original_field = owner.ui.tcp_p[0]
    original_field.set_value(.0123)  # unsaved field edits must survive refresh
    original_combo.model._item_changed(None)
    await settle()
    assert owner.robot is bound, "Unchanged dropdown notification discarded the robot binding"
    assert robot_combo().widget is original_combo, "Unchanged dropdown notification rebuilt the controls"
    assert owner.ui.tcp_p[0] is original_field and abs(original_field.as_float-.0123) < 1e-6
    print("PASS UI: redundant dropdown notifications preserve controls, binding and field edits", flush=True)

    # Force repeated no-op refresh requests, including while a dropdown is open.
    await robot_combo().click()
    for _ in range(8):
        owner.dirty = True
        await settle(.05)
    assert robot_combo().widget is original_combo, "Background refresh destroyed an open dropdown"
    await ui_test.emulate_keyboard_press(KeyboardInput.ESCAPE)
    await settle()

    # Real mouse selection through the popup changes the chosen robot once.
    await robot_combo().click()
    await settle()
    from omni.kit.ui_test import Vec2
    combo_ref = robot_combo()
    # The native popup starts directly below the control; click row two.
    await ui_test.emulate_mouse_move_and_click(combo_ref.position+Vec2(60, combo_ref.size.y*2.5))
    await settle()
    assert owner.robot_path == "/World/PandaB", owner.robot_path

    # Hold the mouse down across several refresh intervals before releasing it.
    # Rebuilding the Button between down/up used to lose this click entirely.
    from omni.kit.ui_test.input import emulate_mouse
    from carb.input import MouseEventType
    bind_button = button("Bind")
    await bind_button.bring_to_front()
    await ui_test.emulate_mouse_move(bind_button.center)
    await emulate_mouse(MouseEventType.LEFT_BUTTON_DOWN)
    for _ in range(6):
        owner.dirty = True
        await settle(.05)
    await emulate_mouse(MouseEventType.LEFT_BUTTON_UP)
    await settle()
    assert owner.robot and owner.robot.path == "/World/PandaB", owner.message
    assert "bound" in owner.ui.status_label.text.lower(), owner.ui.status_label.text
    status_widget = owner.ui.status_label
    owner.ui.sections["Robot and TCP"].collapsed = True
    await settle()

    await button("Add at Current TCP").click()
    await settle()
    assert len(owner.store.paths()) == 1, owner.message
    assert owner.ui.status_label is status_widget, "Action feedback must not be recreated with the form"
    assert owner.ui.sections["Robot and TCP"].collapsed, "A form refresh expanded a collapsed section"
    assert "Add at Current TCP" in owner.ui.status_label.text, owner.ui.status_label.text
    print("PASS UI: real dropdown selection, Bind and Add at Current TCP mouse clicks", flush=True)

    # Selected waypoint form stays intact across unrelated no-op refreshes.
    field = owner.ui.pos[0]
    field.set_value(.0345)
    owner.dirty = True
    await settle()
    assert owner.ui.pos[0] is field and abs(field.as_float-.0345) < 1e-6
    owner.guard(lambda: (_ for _ in ()).throw(ValueError("Visible test failure")), label="Test action")
    assert "ERROR" in owner.ui.status_label.text and "Visible test failure" in owner.ui.status_label.text
    assert "ScrollingFrame" not in ui_test.find(prefix+"Label[*].identifier=='status'").realpath
    print("PASS UI: persistent top-level feedback, visible errors and unsaved field preservation", flush=True)
    context.get_selection().clear_selected_prim_paths()
    await frames(12)


async def scene_smoke(context, manager, runtime, filename):
    """Use the user's scene through normal timeline Play, without a World fixture.

    All waypoint edits are anonymous and preview opinions are session-only.
    This test never calls Save/Export on the source scene or its sublayers.
    """
    from dataclasses import replace
    from pathlib import Path
    import numpy as np
    from simdroid_data_collection.domain import Pose
    manager.set_extension_enabled_immediate("omni.kit.ui_test", True)
    import omni.kit.ui_test as ui_test
    source = Path(filename).resolve()
    before = (source.stat().st_size, source.stat().st_mtime_ns)
    opened, error = await context.open_stage_async(str(source))
    assert opened, error
    await settle(1.)
    owner = enabled_owner(manager)
    owner.refresh_robots()
    assert "/World/franka" in owner.robot_paths, owner.robot_paths
    owner.choose_robot("/World/franka")
    owner.ui.window.visible = True
    owner.ui.window.width = 600
    owner.ui.window.height = 900
    owner.ui.window.position_x = 20
    owner.ui.window.position_y = 20
    await settle()
    owner.ui.sections["Robot and TCP"].collapsed = False
    owner.ui.scroll.scroll_y = 0
    await settle()
    prefix = "Franka Waypoint Editor//Frame/**/"

    def button(text):
        return ui_test.find(prefix+f"Button[*].text=='{text}'")

    await button("Bind").click()
    await settle()
    assert owner.robot and owner.robot.path == "/World/franka", owner.message
    owner.ui.sections["Robot and TCP"].collapsed = True
    await settle()
    await button("Add at Current TCP").click()
    await settle()
    assert owner.selected and len(owner.store.paths()) == 1, owner.message
    await button("Show Gripper Preview").click()
    await settle()
    assert owner.preview.waypoint == owner.selected, owner.message
    print("PASS scene: actual data_gen.usda robot discovery, mouse Bind/Add/Preview", flush=True)

    try:
        owner.timeline.play()
        await frames(40)
        assert owner.guard(owner.bind, label="Bind for scene motion"), owner.message
        assert owner.robot.ready()
        initial = owner.robot.tcp_pose()
        owner.add_at_tcp()
        delta = np.array((0., .025, 0.)) if UsdGeom.GetStageUpAxis(owner.store.stage) == "Y" else np.array((0., 0., .025))
        wp = owner.store.read(owner.selected)
        target = Pose(np.asarray(initial.position)+delta, initial.orientation)
        owner.store.update(replace(wp, pose=target))
        # Like a user finishing a gizmo edit and then clicking Run, allow the
        # stage/metrics notifications to settle before starting the controller.
        await settle(.5)
        if not owner.robot.ready():
            # Metrics assembly in this scene can invalidate the simulation-wide
            # view. Verify a clear, safe failure, then the user-facing recovery.
            assert not owner.guard(lambda: owner.launch("selected"), label="Run after scene edit")
            assert "Stop, then Play" in owner.message, owner.message
            print("PASS scene: invalidated shared physics view yields actionable feedback", flush=True)
            owner.timeline.stop()
            await frames(12)
            owner.timeline.play()
            await frames(40)
            assert owner.guard(owner.bind, label="Bind after Stop/Play"), owner.message
        owner.launch("selected")
        for _ in range(2400):
            await frames(1)
            if not owner.runner.active:
                break
        assert owner.runner.state == "complete", owner.message
        assert owner.robot.tcp_pose().error(target)[0] < .005
        print("PASS scene: normal Play without World, native Lula 2.5 cm motion completed", flush=True)
    finally:
        owner.timeline.stop()
        await frames(12)
        assert (source.stat().st_size, source.stat().st_mtime_ns) == before, "Source scene was modified"
    # Capture the actual UI after real scene interactions, not a drawn mockup.
    owner.set_status("Scene test passed. Source scene was not saved or modified.")
    owner.ui.scroll.scroll_y = 0
    manager.set_extension_enabled_immediate("omni.kit.renderer.capture", True)
    import omni.kit.renderer_capture
    omni.kit.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(
        str(runtime / "ui_scene_after.png"))
    await settle(.6)
    context.get_selection().clear_selected_prim_paths()
    await frames(12)
