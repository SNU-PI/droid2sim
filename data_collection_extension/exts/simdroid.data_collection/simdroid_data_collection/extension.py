"""Kit lifecycle, selection synchronization, and orchestration of the editor."""
import asyncio
from pathlib import Path
import time
import weakref
import carb
import omni.ext
import omni.kit.app
import omni.kit.commands
import omni.physx
import omni.timeline
import omni.usd
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items
from pxr import Tf, Usd, UsdGeom
from .domain import Pose
from .stage_store import StageStore, FrankaWaypointLayerEdit, find_robots
from .preview import PreviewManager
from .robot import FrankaBinding
from .runner import SequenceRunner
from .ui import EditorUI
from .edit_router import WaypointEditRouter


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self.context = omni.usd.get_context()
        self.timeline = omni.timeline.get_timeline_interface()
        self.message = "Select a Franka and create a waypoint sequence."
        self.store = self.preview = self.robot = self.ui = None
        self.robot_path = self.selected = ""
        self.robot_paths = []
        self.task = self.notice = None
        self.runner = SequenceRunner(self.set_status)
        self.dirty = True
        self.last_update = 0.
        self.last_state = "idle"
        self.observed_config = None
        self.layer_file = str(Path(__file__).resolve().parents[4] / "franka_paths.usda")
        self.log_file = str(Path(__file__).resolve().parents[4] / "franka_run.json")
        omni.kit.commands.register(FrankaWaypointLayerEdit)
        self.edit_router = WaypointEditRouter(self)
        self.menu = [MenuItemDescription(name="Franka Waypoint Editor", onclick_fn=self.show)]
        add_menu_items(self.menu, "Window")
        self.stage_sub = self.context.get_stage_event_stream().create_subscription_to_pop(self.on_stage)
        self.timeline_sub = self.timeline.get_timeline_event_stream().create_subscription_to_pop(self.on_timeline)
        self.physics_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self.on_physics)
        self.update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self.on_update)
        self.attach_stage()
        self.show()

    def set_status(self, message):
        self.message = message
        if self.ui:
            self.ui.status(message)

    def guard(self, fn, label=None):
        if label:
            self.set_status(f"{label}: working...")
            carb.log_info(f"[Franka Waypoint Editor] Click: {label}")
        before = self.message
        try:
            fn()
        except Exception as exc:
            self.set_status(f"ERROR - {label or 'Action'}: {exc}")
            carb.log_warn(f"[Franka Waypoint Editor] {exc}")
            self.dirty = True
            return False
        if label:
            if self.message == before:
                self.set_status(f"OK - {label}")
            carb.log_info(f"[Franka Waypoint Editor] {self.message}")
        self.dirty = True
        return True

    def show(self):
        if not self.ui:
            self.ui = EditorUI(self)
        self.ui.window.visible = True
        self.dirty = True

    def require_idle(self):
        if self.runner.active:
            raise ValueError("Abort or finish the active run before editing its configuration.")
        if not self.store:
            raise ValueError("Open a USD scene first.")
        self.reconcile_configuration()

    def configuration(self):
        if not self.store.sequence:
            return None
        tcp = self.store.tcp()
        return (self.store.sequence, self.store.robot_path(), tcp.position, tcp.orientation)

    def reconcile_configuration(self):
        sequences = self.store.sequences()
        if self.store.sequence not in sequences:
            self.store.sequence = sequences[0] if sequences else ""
            self.selected = ""
        current = self.configuration()
        if current != self.observed_config:
            if self.runner.active:
                self.cancel_task()
                self.runner.abort("Sequence/robot/TCP configuration changed; re-plan before running.")
            self.robot = None
            self.hide_preview()
            if current is not None:
                self.robot_path = current[1]
            self.observed_config = current
            self.dirty = True
        if self.selected and (not self.store.sequence or self.selected not in self.store.paths()):
            self.selected = ""

    def detach_stage(self):
        self.edit_router.flush()
        self.cancel_task()
        self.runner.abort("Stage detached", hold=False)
        self.runner.robot = None
        if self.notice:
            self.notice.Revoke()
        self.notice = None
        if self.preview:
            self.preview.destroy()
        self.preview = self.store = self.robot = None
        self.selected = ""
        self.observed_config = None

    def attach_stage(self):
        self.detach_stage()
        stage = self.context.get_stage()
        if stage:
            self.store = StageStore(stage)
            choices = self.store.sequences()
            if choices:
                self.store.sequence = choices[0]
                self.robot_path = self.store.robot_path()
            self.preview = PreviewManager(stage)
            self.observed_config = self.configuration()
            self.notice = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self.on_usd_change, stage)
            self.refresh_robots()
        self.dirty = True

    def on_usd_change(self, notice, sender):
        if self.store:
            paths = list(notice.GetResyncedPaths()) + list(notice.GetChangedInfoOnlyPaths())
            collection = "/World/FrankaPaths"
            if any(str(p).startswith(collection) or collection.startswith(str(p).rstrip("/")+"/")
                   for p in paths):
                self.dirty = True

    def on_stage(self, event):
        if event.type == int(omni.usd.StageEventType.SELECTION_CHANGED):
            self.guard(self.sync_selection)
        elif event.type == int(omni.usd.StageEventType.OPENED):
            self.guard(self.attach_stage)
        elif event.type == int(omni.usd.StageEventType.CLOSING):
            self.guard(self.detach_stage)

    def on_timeline(self, event):
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self.cancel_task()
            self.runner.abort("Timeline stopped; bind again before running.", hold=False)
            self.runner.robot = self.robot = None
            if self.preview:
                self.preview.set_hidden(False)
        elif event.type == int(omni.timeline.TimelineEventType.PAUSE):
            if self.runner.state == "planning":
                self.cancel_task()
                self.runner.abort("Planning cancelled when the timeline paused.")
            else:
                self.runner.pause()
        self.dirty = True

    def on_physics(self, dt):
        self.runner.step(dt)
        if self.task and self.runner.state != "planning":
            self.cancel_task()

    def on_update(self, event):
        self.edit_router.flush()
        now = time.monotonic()
        if now-self.last_update < .1:
            return
        self.last_update = now
        if self.runner.state != self.last_state:
            self.last_state = self.runner.state
            self.dirty = True
            if self.preview:
                self.preview.set_hidden(self.runner.active)
        try:
            if self.store and self.dirty:
                self.reconcile_configuration()
        except Exception as exc:
            if self.runner.active:
                self.cancel_task()
                self.runner.abort(f"Invalid scene configuration: {exc}")
            self.set_status(f"Scene configuration: {exc}")
        try:
            if self.preview and self.store:
                self.preview.update(self.store)
        except Exception as exc:
            self.set_status(f"Preview: {exc}")
        # A malformed preview/goal must not prevent rebuilding the editor UI.
        try:
            if self.dirty and self.ui and self.ui.window.visible:
                self.dirty = False
                self.ui.build()
        except Exception as exc:
            self.set_status(f"Editor refresh: {exc}")

    def refresh_robots(self):
        self.robot_paths = find_robots(self.store.stage) if self.store else []
        if not self.robot_path:
            self.robot_path = self.robot_paths[0] if self.robot_paths else ""
        self.dirty = True
        self.set_status(f"Found {len(self.robot_paths)} Panda robot(s). Choose one and click Bind."
                        if self.robot_paths else "No Panda found. Load a standard Panda and click Refresh.")

    def choose_robot(self, path):
        if path != self.robot_path and path in self.robot_paths and not self.runner.active:
            self.robot_path = path
            self.robot = None
            self.dirty = True
            self.set_status(f"Selected {path}. Click Bind to use this robot.")

    def use_selected_robot(self):
        self.require_idle()
        self.refresh_robots()
        selected = self.context.get_selection().get_selected_prim_paths()
        matches = [p for p in self.robot_paths if any(s == p or s.startswith(p+"/") for s in selected)]
        if len(matches) != 1:
            raise ValueError("Select one Franka root or one of its links in the Stage.")
        self.choose_robot(matches[0])
        self.bind()

    def new_sequence(self):
        self.require_idle()
        self.store.create_sequence(self.robot_path)
        self.robot = None
        self.observed_config = self.configuration()
        self.hide_preview()
        self.selected = ""
        self.set_status("Sequence created. Add a waypoint or capture the current TCP.")

    def choose_sequence(self, path):
        self.require_idle()
        if path == self.store.sequence or path not in self.store.sequences():
            return
        self.store.sequence = path
        self.robot_path = self.store.robot_path()
        self.robot = None
        self.observed_config = self.configuration()
        self.selected = ""
        self.hide_preview()
        self.dirty = True

    def bind(self, tcp=None):
        self.require_idle()
        if not self.robot_path:
            raise ValueError("No Franka selected. Load one and click Refresh.")
        if not self.store.sequence:
            self.new_sequence()
        tcp = tcp or self.store.tcp()
        robot = FrankaBinding(self.store.stage, self.robot_path, tcp)
        if self.timeline.is_playing():
            robot.initialize()
        self.store.bind(self.robot_path, tcp)
        self.robot = robot
        self.observed_config = self.configuration()
        self.hide_preview()
        self.set_status("Robot bound. Verify TCP offset using a preview. " +
                        ("Physics ready." if robot.ready() else "Press Play to enable execution."))

    def ensure_robot(self, physics=False):
        self.reconcile_configuration()
        if physics and not self.timeline.is_playing():
            raise ValueError("Press Isaac Sim Play before validating or running a trajectory.")
        if self.robot is None:
            self.bind()
        if physics and not self.robot.ready():
            self.robot.initialize()
        return self.robot

    def sync_selection(self):
        if not self.store:
            return
        paths = self.context.get_selection().get_selected_prim_paths()
        self.selected = ""
        if len(paths) == 1:
            p = paths[0]
            preview_target = self.preview.selection_target(self.store.stage.GetPrimAtPath(p))
            p = preview_target or p
            sequence = self.store.sequence_for_waypoint(p)
            if sequence and sequence != self.store.sequence and not self.runner.active:
                self.choose_sequence(sequence)
            if self.store.sequence and p in self.store.paths():
                self.selected = p
                if preview_target:
                    self.context.get_selection().set_selected_prim_paths([p], False)
                # Native gizmo commands are scoped by edit_router, and panel
                # edits by StageStore. Never leave a different global target
                # selected across physics steps or PhysX's Stop/reset operation.
        self.dirty = True

    def select(self, path):
        self.context.get_selection().set_selected_prim_paths([path], False)
        self.sync_selection()

    def add_waypoint(self):
        self.require_idle()
        if not self.store.sequence:
            self.new_sequence()
        # Use the selected target as a convenient editing origin when available.
        if self.selected:
            pose = self.store.read(self.selected).pose
        elif self.robot:
            pose = self.robot.tcp_pose()
        else:
            pos = (.5, .4, 0) if UsdGeom.GetStageUpAxis(self.store.stage) == "Y" else (.5, 0, .4)
            pose = Pose(pos, (1, 0, 0, 0))
        self.select(self.store.add(pose))

    def add_at_tcp(self):
        self.require_idle()
        robot = self.ensure_robot()
        self.select(self.store.add(robot.tcp_pose()))

    def duplicate(self):
        self.require_idle()
        wp = self.store.read(self.selected)
        self.select(self.store.add(wp.pose, source=wp.path))

    def reorder(self, direction):
        self.require_idle()
        self.store.reorder(self.selected, direction)

    def delete(self):
        self.require_idle()
        self.store.remove(self.selected)
        self.selected = ""
        self.context.get_selection().clear_selected_prim_paths()
        self.hide_preview()

    def show_preview(self):
        self.require_idle()
        robot = self.ensure_robot()
        wp = self.store.read(self.selected)
        count = self.preview.show_gripper(robot, wp, wp.gripper)
        self.set_status(f"Preview: {count} visual meshes. Orange gripper's TCP matches the waypoint.")

    def hide_preview(self):
        if self.preview:
            self.preview.hide_gripper()

    def manual_gripper(self, action):
        self.require_idle()
        robot = self.ensure_robot(physics=True)
        robot.command_gripper(action)
        robot.tick_gripper(0.)
        self.set_status(f"Gripper drive target: {action}.")

    def launch(self, mode):
        self.require_idle()
        if mode == "selected" and not self.selected:
            raise ValueError("Select a waypoint first.")
        robot = self.ensure_robot(physics=True)
        snapshot = self.store.snapshot(self.selected if mode == "selected" else None)
        self.runner.prepare(robot)
        self.preview.set_hidden(True)
        self.task = asyncio.ensure_future(self._plan_and_start(weakref.ref(self), robot, snapshot, mode))

    @staticmethod
    async def _plan_and_start(owner_ref, robot, snapshot, mode):
        def progress(message):
            owner = owner_ref()
            if owner is not None and getattr(owner, "task", None) is asyncio.current_task():
                owner.set_status(message)
        try:
            segments = await robot.plan(snapshot, progress)
            self = owner_ref()
            if self is None:
                return
            if getattr(self, "task", None) is asyncio.current_task():
                self.reconcile_configuration()
            if (getattr(self, "task", None) is not asyncio.current_task() or self.runner.state != "planning"
                    or self.robot is not robot or not self.timeline.is_playing()):
                return
            for segment in segments:
                carb.log_info(f"[Franka Waypoint Editor] Planned {segment.waypoint.path}: "
                              f"{segment.duration:.3f}s simulation motion, Speed={segment.waypoint.speed:g} "
                              "of native Panda limits.")
            if mode == "validate":
                self.runner.abort(f"Validated {len(segments)} waypoints: "
                                  f"{sum(s.duration for s in segments):.1f}s simulation motion "
                                  "plus waits/settling. No collision checks.")
            else:
                self.runner.start(segments)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self = owner_ref()
            if self is not None and getattr(self, "task", None) is asyncio.current_task():
                self.runner.abort(f"Planning failed: {exc}")
            carb.log_warn(f"[Franka Waypoint Editor] Planning failed: {exc}")
        finally:
            # Kit clears IExt.__dict__ immediately after on_shutdown. Cancelled
            # coroutines may resume later, so never touch a torn-down instance.
            self = owner_ref()
            if self is not None and getattr(self, "task", None) is asyncio.current_task():
                self.task = None
                self.dirty = True
                if self.preview:
                    self.preview.set_hidden(self.runner.active)

    def cancel_task(self):
        if self.task and not self.task.done():
            self.task.cancel()
        self.task = None

    def abort(self):
        self.cancel_task()
        self.runner.abort()
        self.dirty = True

    def resume(self):
        if not self.timeline.is_playing():
            raise ValueError("Press Isaac Sim Play, then Resume.")
        self.runner.resume()

    def save(self):
        self.require_idle()
        self.store.save(self.layer_file)
        self.set_status("Waypoint layer saved. Save the scene to persist its layer link.")

    def load(self):
        self.require_idle()
        self.store.load(self.layer_file)
        self.hide_preview()
        self.selected = ""
        self.robot_path = self.store.robot_path() if self.store.sequence else ""
        self.robot = None
        self.observed_config = self.configuration()
        self.set_status("Waypoint layer loaded.")

    def export(self):
        self.runner.export(self.log_file)
        self.set_status(f"Saved telemetry to {self.log_file}")

    def on_shutdown(self):
        self.abort()
        self.edit_router.destroy()
        self.physics_sub = self.update_sub = self.stage_sub = self.timeline_sub = None
        self.detach_stage()
        remove_menu_items(self.menu, "Window")
        omni.kit.commands.unregister(FrankaWaypointLayerEdit)
        if self.ui:
            self.ui.destroy()
        self.ui = None
