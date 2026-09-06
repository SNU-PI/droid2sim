"""Thin omni.ui view. Buttons delegate authoring/execution to the extension."""
from functools import partial
import numpy as np
import omni.ui as ui
from .domain import Pose, Waypoint, euler_quat, quat_euler
from .stage_store import DEFAULT_TCP


class EditorUI:
    def __init__(self, owner):
        self.owner = owner
        self.window = ui.Window("Franka Waypoint Editor", width=440, height=850)
        self._signature = None
        self._status_count = 0
        self.sections = {}
        self.section_states = {}
        self.scroll = None
        # Feedback is outside the scrolling/rebuilt form, so every action is
        # visible even when the Execution section is collapsed or off screen.
        with self.window.frame:
            with ui.VStack(spacing=6):
                self.status_label = ui.Label("", identifier="status", word_wrap=True,
                                             height=66, alignment=ui.Alignment.LEFT_TOP)
                self.content_frame = ui.Frame()
        self.status(owner.message)

    def destroy(self):
        if self.window:
            self.window.destroy()
        self.sections.clear()
        self.window = self.status_label = self.content_frame = self.scroll = None
        self.owner = None

    def status(self, text):
        if self.status_label:
            self._status_count += 1
            self.status_label.text = f"[{self._status_count}] {text}"
            self.status_label.tooltip = text
            self.status_label.style = {"color": 0xff9090ff if text.startswith("ERROR") else 0xffd6ebd6}

    def button(self, text, fn, **kwargs):
        return ui.Button(text, clicked_fn=lambda: self.owner.guard(fn, label=text), height=28,
                         style={"Button": {"background_color": 0xff383838},
                                "Button:hovered": {"background_color": 0xff70583d},
                                "Button:pressed": {"background_color": 0xffa7793d},
                                "Button:disabled": {"background_color": 0xff292929}}, **kwargs)

    def section(self, title, collapsed=False):
        frame = ui.CollapsableFrame(title, height=0,
                                   collapsed=self.section_states.get(title, collapsed))
        self.sections[title] = frame
        return frame

    def choose(self, model, choices, robot=False):
        index = model.as_int
        if not 0 <= index < len(choices):
            return
        path = choices[index]
        o = self.owner
        current = o.robot_path if robot else o.store.sequence
        if path == current:
            return
        fn = o.choose_robot if robot else o.choose_sequence
        o.guard(partial(fn, path), label="Select robot" if robot else "Select sequence")

    def snapshot(self):
        """Only composed form data can invalidate widgets, never UI notices.

        Item-model notifications also occur when popup contents refresh. Treating
        those as selection changes used to create a destroy/recreate loop that
        interrupted mouse-up events and erased unsaved field values.
        """
        o = self.owner
        store = o.store
        if not store:
            return (None,)
        try:
            paths = store.paths() if store.sequence else []
            tcp = store.tcp() if store.sequence else DEFAULT_TCP
            values = []
            for path in paths:
                try:
                    values.append(store.read(path))
                except Exception as exc:
                    values.append((path, str(exc)))
            data = (tcp, tuple(values))
        except Exception as exc:
            data = (str(exc),)
        return (id(store), store.sequence, tuple(store.sequences()), tuple(o.robot_paths),
                o.robot_path, o.selected, o.runner.state, data)

    def fields(self, labels, values):
        models = []
        with ui.HStack(height=24, spacing=4):
            for label, value in zip(labels, values):
                ui.Label(label, width=35)
                model = ui.SimpleFloatModel(float(value))
                ui.FloatField(model=model, width=ui.Fraction(1))
                models.append(model)
        return models

    def build(self):
        if not self.window:
            return False
        signature = self.snapshot()
        if signature == self._signature:
            return False
        self.section_states.update({title: frame.collapsed for title, frame in self.sections.items()})
        self.sections.clear()
        scroll_y = self.scroll.scroll_y if self.scroll else 0.
        self._build()
        if self.scroll:
            self.scroll.scroll_y = scroll_y
        self._signature = signature
        return True

    def _build(self):
        o = self.owner
        store = o.store
        if not self.window:
            return
        with self.content_frame:
            with ui.ScrollingFrame() as self.scroll:
                with ui.VStack(spacing=8, height=0):
                    ui.Label("Franka Waypoint Editor", height=26)
                    ui.Label("Isaac Sim 5.1 | Lula IK + trajectory generation", height=22)
                    if not store:
                        ui.Label("Open a USD scene to begin.")
                        return
                    busy = o.runner.active
                    try:
                        paths = store.paths() if store.sequence else []
                        tcp = store.tcp() if store.sequence else DEFAULT_TCP
                    except Exception as exc:
                        paths, tcp = [], DEFAULT_TCP
                        ui.Label(f"Invalid sequence: {exc}", word_wrap=True, height=50)
                    with self.section("Robot and TCP"):
                        with ui.VStack(spacing=4, height=0):
                            with ui.HStack(height=26):
                                ui.Label("Panda", width=60)
                                robot_paths = list(o.robot_paths)
                                if o.robot_path and o.robot_path not in robot_paths:
                                    robot_paths.insert(0, o.robot_path)
                                robot_paths = robot_paths or ["No Panda found"]
                                selected = robot_paths.index(o.robot_path) if o.robot_path in robot_paths else 0
                                combo = ui.ComboBox(selected, *robot_paths, identifier="robot_choice",
                                                    enabled=not busy and bool(o.robot_paths))
                                combo.model.get_item_value_model().add_value_changed_fn(
                                    partial(self.choose, choices=tuple(robot_paths), robot=True))
                            with ui.HStack(height=28, spacing=4):
                                self.button("Refresh", o.refresh_robots, enabled=not busy)
                                self.button("Use Selected Robot", o.use_selected_robot, enabled=not busy)
                                self.button("Bind", o.bind, enabled=not busy)
                            ui.Label("TCP offset from panda_hand (metres; hand axes)", height=22)
                            self.tcp_p = self.fields(("X", "Y", "Z"), tcp.position)
                            self.tcp_r = self.fields(("Roll", "Pitch", "Yaw"), quat_euler(tcp.orientation))
                            ui.Label("RPY in degrees; verify the offset for your finger asset.", height=22)
                            self.button("Apply TCP and Bind", self.apply_tcp, enabled=not busy)
                            with ui.HStack(height=28, spacing=4):
                                self.button("Open Gripper", partial(o.manual_gripper, "open"), enabled=not busy)
                                self.button("Close Gripper", partial(o.manual_gripper, "close"), enabled=not busy)
                    with self.section("Sequence"):
                        with ui.VStack(spacing=4, height=0):
                            sequences = store.sequences()
                            with ui.HStack(height=26):
                                choices = sequences or ["Create a sequence"]
                                index = choices.index(store.sequence) if store.sequence in choices else 0
                                combo = ui.ComboBox(index, *choices, identifier="sequence_choice",
                                                    enabled=not busy and bool(sequences),
                                                    tooltip="Click New to create a sequence." if not sequences else "")
                                combo.model.get_item_value_model().add_value_changed_fn(
                                    partial(self.choose, choices=tuple(choices)))
                                self.button("New", o.new_sequence, width=60, enabled=not busy)
                            if store.sequence:
                                for index, path in enumerate(paths):
                                    try:
                                        wp = store.read(path)
                                        label = f"{index+1}. {path.rsplit('/', 1)[-1]}   [{wp.gripper}]"
                                        if not wp.enabled:
                                            label += " (disabled)"
                                    except Exception:
                                        label = f"{index+1}. INVALID: {path}"
                                    self.button(("> " if path == o.selected else "") + label,
                                                partial(o.select, path))
                            with ui.HStack(height=28, spacing=4):
                                self.button("Add Waypoint", o.add_waypoint, enabled=not busy)
                                self.button("Add at Current TCP", o.add_at_tcp, enabled=not busy)
                            with ui.HStack(height=28, spacing=4):
                                can_edit = bool(o.selected) and not busy
                                self.button("Duplicate", o.duplicate, enabled=can_edit)
                                self.button("Up", partial(o.reorder, -1), enabled=can_edit)
                                self.button("Down", partial(o.reorder, 1), enabled=can_edit)
                                self.button("Delete", o.delete, enabled=can_edit)
                    wp = None
                    if o.selected and store.sequence:
                        try:
                            wp = store.read(o.selected)
                        except Exception:
                            pass
                    with self.section("Selected waypoint"):
                        with ui.VStack(spacing=4, height=0):
                            if wp:
                                ui.Label(wp.path, word_wrap=True, height=35)
                                ui.Label("World position (m)", height=20)
                                self.pos = self.fields(("X", "Y", "Z"), wp.pose.position)
                                ui.Label("World orientation (degrees; Rz Ry Rx)", height=20)
                                self.rot = self.fields(("Roll", "Pitch", "Yaw"), quat_euler(wp.pose.orientation))
                                with ui.HStack(height=24):
                                    self.enabled = ui.SimpleBoolModel(wp.enabled)
                                    ui.CheckBox(model=self.enabled, width=24)
                                    ui.Label("Enabled", width=90)
                                    ui.Label("On arrival", width=85)
                                    self.gripper = ui.ComboBox(("keep", "open", "close").index(wp.gripper),
                                                              "Keep", "Open", "Close")
                                self.dwell, self.speed = self.fields(("Wait s", "Speed"), (wp.dwell, wp.speed))
                                ui.Label("Speed: 0 < value <= 1; 1 = native Panda model limits (not m/s).",
                                         word_wrap=True, height=32)
                                self.ptol, self.rtol = self.fields(("Tol m", "Tol °"),
                                    (wp.position_tolerance, np.degrees(wp.orientation_tolerance)))
                                self.timeout, = self.fields(("Max s",), (wp.timeout,))
                                self.button("Apply Pose and Settings", self.apply_waypoint, enabled=not busy)
                                with ui.HStack(height=28, spacing=4):
                                    self.button("Show Gripper Preview", o.show_preview, enabled=not busy)
                                    self.button("Hide Preview", o.hide_preview)
                            else:
                                ui.Label("Select a tagged waypoint in this list or in the Stage.",
                                         word_wrap=True, height=40)
                    with self.section("Execution"):
                        with ui.VStack(spacing=4, height=0):
                            ui.Label("Press Isaac Sim Play before running. No obstacle avoidance.",
                                     word_wrap=True, height=36)
                            with ui.HStack(height=28, spacing=4):
                                self.button("Validate", partial(o.launch, "validate"), enabled=not busy)
                                self.button("Run Selected", partial(o.launch, "selected"), enabled=not busy and bool(wp))
                                self.button("Run Sequence", partial(o.launch, "sequence"), enabled=not busy)
                            with ui.HStack(height=28, spacing=4):
                                self.button("Pause", o.runner.pause, enabled=o.runner.state == "running")
                                self.button("Resume", o.resume, enabled=o.runner.state == "paused")
                                self.button("Abort", o.abort, enabled=busy)
                    with self.section("Save / load / telemetry", collapsed=True):
                        with ui.VStack(spacing=4, height=0):
                            ui.Label("Waypoint layer (.usda)", height=22)
                            layer_model = ui.SimpleStringModel(o.layer_file)
                            ui.StringField(model=layer_model, height=24)
                            layer_model.add_value_changed_fn(lambda m: setattr(o, "layer_file", m.as_string))
                            with ui.HStack(height=28, spacing=4):
                                self.button("Save Waypoints", o.save, enabled=not busy)
                                self.button("Load Waypoints", o.load, enabled=not busy)
                            ui.Label("Save the scene in Isaac Sim to persist its layer link.", height=24)
                            log_model = ui.SimpleStringModel(o.log_file)
                            ui.StringField(model=log_model, height=24)
                            log_model.add_value_changed_fn(lambda m: setattr(o, "log_file", m.as_string))
                            self.button("Export Run JSON", o.export, enabled=not busy)
                            ui.Label("Robot telemetry only; camera capture is not included in v0.1.",
                                     word_wrap=True, height=36)

    def apply_tcp(self):
        tcp = Pose([m.as_float for m in self.tcp_p], euler_quat([m.as_float for m in self.tcp_r]))
        self.owner.bind(tcp)

    def apply_waypoint(self):
        self.owner.require_idle()
        wp = Waypoint(self.owner.selected,
            Pose([m.as_float for m in self.pos], euler_quat([m.as_float for m in self.rot])),
            self.enabled.as_bool, ("keep", "open", "close")[self.gripper.model.get_item_value_model().as_int],
            self.dwell.as_float, self.speed.as_float, self.ptol.as_float,
            np.radians(self.rtol.as_float), self.timeout.as_float)
        self.owner.store.update(wp)
        self.owner.hide_preview()
        self.owner.dirty = True
