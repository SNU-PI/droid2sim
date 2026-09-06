# Franka Waypoint Editor

A native Isaac Sim / Kit extension for authoring TCP waypoint Xforms and running
Franka Panda sequences. Implementation baseline: **Isaac Sim 5.1.0**, matching
the installed `env_isaaclab` environment. Other Isaac Sim versions are not yet
verified. There is no Mink, MuJoCo, custom IK solver, or old-project runtime dependency.

## Load the extension

1. Open Isaac Sim and its Extensions window (`Window > Extensions`).
2. In the extension search-path settings, add this **parent of extension folders**:

   ```text
   /home/sangjunpark/Documents/SimDROID/data_collection_extension/exts
   ```

3. Search for `Franka Waypoint Editor` / `simdroid.data_collection` and enable it.
4. Open `Window > Franka Waypoint Editor` if the panel is hidden.

Alternatively, append these arguments to your normal Isaac Sim launcher:

```bash
--ext-folder /home/sangjunpark/Documents/SimDROID/data_collection_extension/exts \
--enable simdroid.data_collection
```

Do not run `extension.py` with regular Python. Kit imports the package and calls
its startup/shutdown methods. This follows NVIDIA's [extension loading workflow](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/extensions_basic.html).

## First sequence

Author waypoints with the main timeline **stopped**, then press Play to execute.
The copied scene uses Isaac Sim's metrics assembler; adding prims while playing
can invalidate the shared physics view. If the editor reports this, use
**Stop → Play → Bind** before running again. The extension does not reset global
physics automatically or keep sending commands through invalid handles.

1. Open a scene containing a standard Franka Panda. `../data_gen.usda` is the
   copied project scene; keep `../data_gen_waypoints.usda` and `../textures/`
   beside it. Its remote NVIDIA assets still need to resolve.
2. Click **Refresh**, choose the Panda, then **Bind**. Alternatively select a
   robot/link in the Stage and click **Use Selected Robot**.
3. Verify the TCP offset. The default is `(0, 0, 0.107)` metres relative to
   `panda_hand`, with the hand's orientation. This is the finger closing region,
   not the wrist origin. It is configurable because replacement fingers or
   modified robot assets may require calibration.
4. Click **Add at Current TCP**, or **Add Waypoint**. Select the new Xform in the
   panel or Stage, then move/rotate it with Isaac Sim's viewport gizmos.
5. The inspector shows world position in metres and roll/pitch/yaw in degrees.
   For typed edits, click **Apply Pose and Settings**. Choose **Keep**, **Open**,
   or **Close** under **On arrival**, and set dwell, speed, tolerances and timeout.
6. Click **Show Gripper Preview**. This creates an independent orange copy of
   only the hand and fingers. Its TCP matches the waypoint position/orientation;
   moving the waypoint moves this preview. It has no articulation, collision,
   rigid-body, or drive APIs. It does not move the real robot.
7. Duplicate/reorder waypoints as needed. A typical sequence is approach (Keep),
   grasp (Close), lift (Keep), place (Open), retreat (Keep).
8. Press Isaac Sim **Play**, then **Validate** or **Run Sequence**. Start with a
   short unobstructed motion. **Run Selected** goes only to the selected goal.

Action feedback stays at the **top of the panel**, outside the scrolling form.
Each accepted button click updates the numbered message with a result or an
`ERROR` explanation; hovering the message shows its full text. Buttons also
have distinct hover/pressed colors. The empty sequence dropdown is intentionally
disabled: click **New**, or **Bind** to create the first sequence automatically.
Unchanged refreshes preserve dropdowns, typed-but-unapplied values and bindings.
Collapsed sections and scroll position are retained when the form changes.

The gripper action fires once, after the measured TCP is within both tolerances
for 0.15 simulation seconds. The runner then waits for the fingers to settle and
for the requested dwell before proceeding. Closing against an object may settle
before the fingers reach zero separation; this is **not** a grasp-success test.

**Pause** freezes sequence time and holds the arm; the gripper keeps its current
command. **Resume** continues only if the arm has not moved significantly and
was tracking the trajectory closely enough to avoid jumping back onto it.
**Abort** stops advancing the sequence and holds the current arm position; it
does not automatically release a held object. Stopping the main timeline clears
the binding; press Play and bind again. Scene changes/unload cancel execution.

## What counts as a waypoint?

Only Xforms explicitly tagged with `frankaPath:kind = "waypoint"`, schema version
1, and listed in the selected sequence's ordered `frankaPath:waypoints`
relationship are executable. Names are for readability, not detection.

```text
/World/FrankaPaths
  Sequence                  # robot relationship, TCP calibration, ordered goals
    Waypoints
      Waypoint              # pose + enabled/action/dwell/speed/tolerances/timeout
      Waypoint_001
```

Ordinary Xforms and the old project's `dataGen:*` targets are ignored. There is
no automatic legacy FSM conversion. Create new targets with this editor.
Each sequence binds one Panda; multiple sequences/robots can be authored, with
one active run at a time.

## Saving and telemetry

- **Save Waypoints** writes a small separate USD layer and links it into the
  scene in memory. Saving the main scene is **optional**: it persists the link,
  not the waypoint definitions themselves. To leave `data_gen.usda` unchanged,
  save only the waypoint file, then reopen the base scene and **Load Waypoints**
  from that file next time, before Bind/New/Add creates a new authoring layer.
  If you also want to preserve other scene edits or the automatic layer link,
  use Isaac Sim **Save As** to a new scene filename.
- Before saving the waypoint layer, it is anonymous/in-memory. Do not close
  without saving. Loading rejects different stage units/up-axis and refuses to
  replace unsaved work. Existing unrelated files cannot be overwritten.
- Authoring buttons support Kit undo/redo, including after the first layer save.
  Panel edits and native viewport transform commands use the waypoint layer
  only for the duration of the command, including undo/redo after deselection.
  Merely selecting a goal does not change the scene's edit target, which also
  keeps physics reset state out of the waypoint layer. Use this inspector or
  viewport gizmos; arbitrary edits in other USD layers through the native
  Properties panel are not automatically rerouted. Conflicting stronger-layer
  opinions must be removed before the editor can change those properties.
- Preview geometry lives in the stage's session layer and is hidden during execution.
  Normal scene/layer saves do not include it. Flatten/export operations can
  include session opinions; disable the extension before independent capture
  or flattened export to remove both gripper previews and waypoint markers.
- **Export Run JSON** writes 20 Hz simulation-time robot telemetry plus waypoint
  and gripper events, including pause/resume. It records commanded arm positions,
  measured joints/velocities, measured/goal TCP poses, gripper targets and run
  configuration. It also records planned segment durations, active wall time,
  and the simulation/wall-time ratio. Existing files are not replaced. Cancelling a validation does
  not change the previous run's recording.
- This first version is the waypoint editor and execution foundation. **RGB/depth
  capture, Replicator writers, randomization, dataset packaging and automated
  task-success detection are not implemented.**

## Motion and limits

NVIDIA's bundled Panda configuration feeds [Lula IK](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/manipulators/manipulators_lula_kinematics.html).
Targets refer to the TCP; each is transformed into a `panda_hand` goal before
solving. The robot base pose is synchronized with the scene. A forward-kinematics
check rejects obvious disagreement between the USD robot and the bundled model.

The planner samples a position interpolation and shortest-path orientation
interpolation, seeds each IK solve from the previous joint solution, and uses
NVIDIA's [Lula configuration-space trajectory generator](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/manipulators/manipulators_lula_trajectory_generator.html)
for velocity/acceleration/jerk-limited timing. Physics-step callbacks apply
`ArticulationAction` drive targets. It does not teleport arm joints.

### Speed and timing (v0.1.1)

**Speed is a per-waypoint fraction in `(0, 1]`, not metres/second.** `1.0` now
uses the native per-joint velocity, acceleration and jerk limits from NVIDIA's
bundled Panda model. Lower values time-scale those limits as `s`, `s²`, `s³`.
The former uniform caps of `0.7 rad/s`, `1.5 rad/s²`, and `10 rad/s³` have been
removed. Existing waypoint files need no format migration, but the same saved
speed now produces faster motion. Preview paths and test unobstructed moves
before relying on existing sequences; full model limits are not a certification
of collision safety, grasp stability, or tracking accuracy for a modified asset.

**Validate** reports planned simulation-time motion; dwell and settling add more
time. **Max s** is the whole waypoint timeout, not a requested duration or speed.
Timeout errors show the motion + dwell + settling allowance explicitly.

During execution, the status includes segment motion progress and
`sim ...x real time`. `1.00x` means approximately real-time simulation; `0.25x`
means one simulation second takes about four wall-clock seconds. This is an
average over the active run, excluding planning and pauses, not the streaming
client's display FPS. Faster trajectories cannot fix a low simulation rate.
The completion message reports both elapsed times.

**Known arrival issue in the supplied `test.usda` sequence:** the longer move
reaches a steady approximately 6.2 mm error with scene gravity enabled, outside
its saved 5 mm tolerance. It then waits for arrival until the timeout; this is
not slow trajectory playback. The native IK endpoint is accurate, but the
physical joints settle off their drive targets under gravity. A robot-only,
temporary gravity-disable diagnostic reduced the error to approximately
0.077 mm. Production code does **not** disable gravity, change drive gains, or
relax waypoint tolerances. A gravity-control policy still needs to be chosen;
the faster motion profile alone does not fix this scene's final arrival offset.

- This is a fixed-base Panda controller, not a general robot-model importer.
  Base motion beyond 2 mm or 0.5 degrees invalidates a plan/run; abort and
  re-plan after repositioning the robot.
- No obstacle avoidance, collision-free path certification, force control or
  real-hardware support. Author approach/lift/retreat waypoints explicitly.
- Joint-space spline interpolation approximates the sampled TCP path; it is not
  an exact Cartesian-path constraint between samples.
- Unreachable IK, large branch jumps, incompatible transforms, missing prims,
  invalid physics handles and execution timeouts stop the operation.
- Robot/waypoint ancestor scale, shear and reflections are rejected. Prefer the
  standard metre-scale Panda. The copied scene is metre-scale and Y-up.
- Run configuration/goal poses are snapshotted before planning. Moving a goal
  during a run does not retarget it live. Abort, edit, and re-run instead.
- Do not run another arm controller or the old `run_data_gen.py` concurrently
  against the same robot. Existing scene drive gains are preserved.

## Code and tests

`exts/simdroid.data_collection/simdroid_data_collection/` contains:

- `extension.py`: Kit lifecycle, robot/goal selection, async planning orchestration.
- `ui.py`: native `omni.ui` panel.
- `stage_store.py`: tagged USD schema, undoable authoring, persistence.
- `edit_router.py`: scoped native viewport transform and undo/redo routing.
- `domain.py`: simulator-independent pose math and arrival/action/dwell logic.
- `robot.py`: Panda model binding, native Lula IK/trajectories, drive commands.
- `runner.py`: physics-step execution and telemetry.
- `preview.py`: session-only waypoint markers and physics-free gripper meshes.

Unit tests (Python + NumPy):

```bash
python -B -m unittest discover -s data_collection_extension/tests -p 'test_*.py' -v
```

Headless integration test, using the installed Isaac Sim Python:

```bash
/home/sangjunpark/miniconda3/envs/env_isaaclab/bin/python -B \
  data_collection_extension/tests/kit_smoke.py --gpu 7 --audit \
  --robot-usd https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd
```

The test defaults to physical GPU **7**, sets `CUDA_VISIBLE_DEVICES=7`, selects
renderer GPU 7, disables multi-GPU rendering and uses CUDA's remapped device 0
for physics. It directs test outputs/caches into `.runtime/`. Omitting
`--robot-usd` runs the authoring/lifecycle checks without loading the robot.
`--audit` adds malformed-goal, undo/calibration, native transform routing, actual
extension control callbacks, and unload-during-planning regression checks.
`--ui-test` adds native mouse interaction checks: popup selection, holding a
button across refresh intervals, adding a goal, and persistent visible feedback.
It also verifies that redundant dropdown notifications do not discard the
binding or erase unsaved field values. `--scene /absolute/path/to/data_gen.usda`
adds a real-scene UI and short-motion check using normal timeline Play without
a test World; it does not save the scene or its sublayers.
Nothing in the extension imports, modifies, or executes the old `droid_sim` code.

With `--scene`, adding `--waypoints /absolute/path/to/test.usda` runs a saved-goal
motion regression instead of the short anonymous scene/UI motion. It compares
the old and native limit profiles, samples generated derivative bounds, executes
the native trajectory, and reports measured arrival error and simulation rate.
Both input files are checked for unchanged size/modification time; neither is
saved. Run telemetry is written only under `.runtime/`.
`--probe-gravity` explicitly enables a diagnostic-only robot gravity override
after a stalled arrival and restores the original live flags before stopping.
Its output labels such a completion separately: it does not count as passing
the original scene unchanged. Without that option, the supplied `test.usda`
currently reproduces the known arrival failure above.

Verification on the installed Isaac Sim 5.1.0 environment:

- 24 unit tests: TCP/quaternion math, arrival/dwell, one-shot actions,
  pause/resume, cancelled planning, base motion, invalid timesteps, preserved
  recordings, abort, timeout, deleted prims and invalid handles.
- Headless Kit checks: enable/window creation, tagged membership, ordering,
  undo/redo before and after save/deselection, TCP and robot binding undo,
  layer reload, malformed/deleted goals, preview exclusion and shutdown.
- Stock Panda: native FK check, three independent hand/finger meshes without
  physics APIs, TCP alignment, Lula motion, measured close/open finger motion,
  JSON export, real extension validate/cancel/run/pause/resume, and unload during
  async planning without invalidating the still-running robot's physics handles.
- Both Z-up and translated/rotated Y-up robot cases passed. Add `--up-axis Y`
  to the test command for the latter. GPU tests are restricted to physical GPU 7.

Native dropdown and button input is covered by `--ui-test`. An actual remote
WebRTC client connection and complete dataset-collection tasks are not automated
by these tests; start with the short unobstructed sequence above.
See [AUDIT.md](AUDIT.md) for the review findings and remaining limits.
