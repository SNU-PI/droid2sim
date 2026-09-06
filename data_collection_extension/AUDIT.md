# Extension regression audit — 2026-09-06

Scope: the extension in this folder, on the installed Isaac Sim 5.1.0 runtime.
All GPU integration tests use physical GPU **7**, with multi-GPU rendering
disabled. The old `Documents/droid_sim` project and the copied scene/assets were
not modified during this audit.

## Corrected defects

- **Stale robot/TCP configuration:** creating a sequence or undoing calibration
  and robot binding could leave a cached controller using the previous values.
  Configuration reconciliation now invalidates stale bindings and previews;
  selecting another sequence's waypoint selects its owning sequence.
- **Cancelled planning and unsafe resumption:** a cancelled plan could be
  started, and a paused arm with large tracking error could jump back toward the
  trajectory. Start/resume gates now check runner state, arm position, tracking
  error and base stability. Invalid timesteps and invalid-handle exceptions are
  contained before further drive commands are applied.
- **Recording corruption:** cancelling validation after a completed run appended
  an abort to the old run. Validation cancellation now leaves that recording
  unchanged. Telemetry also includes commanded arm positions.
- **USD transform and undo integrity:** reset-transform-stack waypoints were
  assigned incorrect world poses. Native gizmo undo after deselection could
  write into the root layer, and changing the global edit target on selection
  interfered with physics reset. Transform commands now use narrowly scoped
  authoring-layer routing; selection leaves the global target unchanged.
- **Malformed goals and layer changes:** disabled goals with invalid geometry
  could block valid goals, deleted/broken references could be difficult to
  remove, and bad goals could stop UI/preview updates. Membership validation,
  deletion, marker recovery and UI error handling are now defensive. Detached
  authoring layers, changed coordinate conventions, occupied import paths and
  stronger-layer property conflicts are rejected instead of silently altered.
- **Extension unload during planning:** Kit clears extension instance state on
  unload. Pending planning cleanup could access that cleared state or keep the
  extension alive. Cancellation now uses weak ownership and checks task identity.
  Native command callbacks and UI ownership are released on shutdown.
- **Preview cleanup invalidating physics:** removing a preview session sublayer
  could invalidate active PhysX tensor views. Previews now own only a unique
  prim subtree in the existing session layer, and cleanup removes that subtree
  without changing the layer stack.
- **Numerical edge case:** very large finite quaternion inputs could overflow
  during normalization. Normalization now rescales first.

## Verification

- **24/24 CPU unit tests passed.** New reproductions were first run against the
  earlier code and exposed failures before the corresponding fixes.
- **Expanded Isaac Sim integration tests passed for Z-up and translated/rotated
  Y-up Panda scenes**, using the stock NVIDIA asset and native Lula planning.
  Coverage includes real extension/UI callbacks, validation cancellation, run,
  pause/resume, actual finger closure/reopening, TCP preview alignment, USD
  persistence, undo/redo, malformed goals, and unload while planning.
- Native viewport transform commands were exercised with save, deselection,
  undo and redo, checking that waypoint opinions did not leak into the root
  layer. Robot/reset opinions were also checked not to leak into the waypoint
  authoring layer.
- Test shutdown verifies callback removal, release of the extension instance,
  removal of preview roots, and preservation of live robot physics handles.
- Final integration runs returned exit code 0 without error-level Kit log
  entries. Headless display warnings, masked-GPU enumeration warnings and the
  bundled Lula finger-mimic warning remain; they did not prevent these tests.

Commands are in [README.md](README.md#code-and-tests). `--audit` enables the
extended regression cases; add `--up-axis Y` for the rotated Y-up case. Generated
test scenes, telemetry and the latest Kit log are under ignored `.runtime/`.

## Not certified by this audit

This is not a guarantee of zero bugs. The initial audit did not test actual
mouse interactions or the large copied scene; the follow-up below adds that
coverage for specific actions, not complete data-collection tasks. Verify TCP
calibration before collecting data. Other Isaac Sim versions and modified
Panda assets remain unverified.

The extension does not certify collision-free motion or successful grasps.
Camera recording, dataset writers and automatic task-success detection remain
outside the implemented first-version scope. Native Properties-panel edits in
arbitrary USD layers are not automatically migrated into the waypoint layer;
use the extension inspector and viewport gizmos for managed waypoints.

## Follow-up: flickering streamed UI and missing click feedback

The user's streamed session exposed a gap in the original controller-oriented
tests: opening dropdowns could trigger generic item-model notifications even
without a changed selection. The old callback discarded the robot binding and
marked the entire form for rebuilding. Rebuilding could destroy a pressed
button before mouse-up, close a popup and erase typed-but-unapplied values.

The new regression first failed on the old code with `Unchanged dropdown
notification discarded the robot binding`. The fix:

- listens to the selected index's value changes rather than every item notice;
- ignores selecting the already-selected robot/sequence;
- rebuilds the form only when its composed data actually changes;
- retains collapsed-section state and scroll position;
- keeps numbered action/error feedback outside the scrolling/rebuilt form;
- adds explicit button hover/pressed colors and logs accepted button actions;
- replaces unsupported arrow glyphs with `Up` and `Down` labels;
- disables the empty sequence selector with a tooltip directing users to New.

`--ui-test` now exercises native Kit mouse input, not just controller methods:
opening and selecting a popup row, holding a button across multiple refresh
intervals, Bind, Add at Current TCP, persistent status, collapsed sections, and
unsaved field/model preservation. These checks passed on GPU 7.

### Actual copied-scene check

`--scene /home/sangjunpark/Documents/SimDROID/data_gen.usda` opened the real scene
in memory and passed robot discovery, mouse-driven Bind/Add/Gripper Preview,
and a 2.5 cm Lula motion using normal timeline Play without a World fixture.
The test checked that the input scene's size and modification timestamp stayed
unchanged. A rendered UI capture is in `.runtime/ui_scene_after.png`.

This also exposed a separate **scene-specific limitation**: adding prims while
playing can make the metrics assembler rebuild robot collision shapes, which
invalidates Isaac Sim's shared physics view. Merely checking whether articulation
handles exist did not catch that. The adapter now checks the public
`SimulationManager.get_physics_sim_view().is_valid` state and avoids redundant
USD binding writes. It does not replace or reset that global view automatically.
It reports **Stop, then Play and Bind again** before trying to use invalid
handles. The real-scene test exercised that explicit recovery and then completed
the short motion successfully. Prefer authoring with the timeline stopped.

The real scene still emits metrics-assembler errors for instanced geometry and
a convex-hull fallback for the dynamic Coca-Cola mesh; those pre-existing scene
issues were not suppressed or edited away. The earlier clean-log statement
applies to the small stock-Panda audit, not this real-scene run.

Actual WebRTC transport/client interaction and complete dataset-generation
episodes remain outside the automated tests. Relaunch the streaming app and
confirm the updated controls from the client as the final interactive check.

## Follow-up: native speed profile and the saved-sequence arrival stall

The user saved two waypoints to `/home/sangjunpark/Documents/SimDROID/test.usda`
and closed the streaming session. All subsequent GPU tests used physical GPU 7
only; neither this waypoint file nor `data_gen.usda` was saved or edited.

### Implemented in v0.1.1

- Cache the seven per-joint velocity, acceleration and jerk limits from NVIDIA's
  bundled Panda model, validate them, and use them at Speed=1. Lower speeds scale
  derivatives by s, s² and s³ from the original values, without compounding.
- Remove the former uniform 0.7 rad/s, 1.5 rad/s², 10 rad/s³ caps. Existing speed
  values are therefore faster; the UI and README explain the changed meaning.
- Show simulation/wall-time ratio and motion progress in running status. Keep
  wall-time accounting separate from physics-step sequence time and exclude
  planning/pauses. Export these measurements and planned durations in telemetry.
- Include dwell and settling allowance in timeout messages rather than saying
  that a 29.7-second movement alone exceeds a 30-second budget.

Thirty CPU unit tests passed. Native mouse UI regressions, authoring/undo,
Y-up stock-Panda close/open execution, validate/cancel/pause/resume, recording
preservation, and extension unload during planning also passed on GPU 7.

### Exact saved-path timing comparison

The regression records one set of IK joint samples, then time-parameterizes
those identical samples with the old and native profiles. An initial comparison
of separately solved paths was rejected because the real arm settled by about
0.0027 rad between solves; the corrected comparison does not freeze or teleport
the robot. All generated native trajectories passed sampled velocity,
acceleration and jerk-bound checks.

| Saved waypoint | Speed | Old motion | Native motion |
| --- | --- | --- | --- |
| Waypoint | 0.3 | 1.160 s | 0.239 s |
| Waypoint_001 | 1.0 | 7.399 s | 2.185 s |

These are simulation-time **motion durations**, excluding dwell and settling.

### Remaining issue: gravity-related drive tracking, not IK or playback speed

Executing the second waypoint in the unchanged scene did **not** pass arrival:
it stopped approximately 6.230 mm from its goal, outside the 5 mm tolerance,
and eventually timed out. The commanded joint solution predicts only 0.044 mm
TCP position error. FK evaluated at the measured joints agrees with the actual
physics TCP within 0.00035 mm, separating drive tracking from TCP/model mismatch.

A diagnostic-only `SingleArticulation.disable_gravity()` call, applied only to
this robot after five extra simulation seconds of unsuccessful settling,
reduced the error to **0.077 mm / 0.045 degrees** and allowed completion. Native
gravity flags were restored before stopping, with no USD gravity opinions
authored. No drive gains or waypoint tolerances changed. This matches NVIDIA's
firsthand explanation of a [Franka PD/gravity offset and robot-only workaround](https://forums.developer.nvidia.com/t/end-effector-offset-of-rmp-ik/228158/19).
That explanation is historical; the installed 5.1 scene was independently tested.

The diagnostic run reported 0.781x real-time simulation and 9.233 s simulated /
11.823 s wall time, **including the intentionally induced stall**. It is not a
normal end-to-end performance result. Telemetry is in
`.runtime/saved_motion_16c73a53.json` and explicitly tags the gravity diagnostic;
the no-override failure is `.runtime/saved_motion_diagnostic_c02bd51e.json`.

Production gravity behavior is deliberately unchanged pending a user decision.
`--scene ... --waypoints ...` reproduces the remaining failure; adding the
explicit test flag `--probe-gravity` diagnoses it and labels the modified-runtime
completion separately. Do not describe the unchanged saved sequence as passing.

One intermediate diagnostic launch encountered a Vulkan GPU pagefault while
starting the real scene's physics, before motion planning. It exited, released
GPU 7, and a repeat using the previous UI-test setup completed the diagnostics
and broader suite. No driver reset, multi-GPU use, source-scene mutation, or
renderer workaround was performed. Existing metrics-assembler and Coca-Cola
collision warnings also remain. WebRTC client performance was not measured.
