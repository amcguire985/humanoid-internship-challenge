# One controlled scripted manipulation attempt

The experiment combines a Panda-specific rim grasp and release with the existing
human-derived object transport path. It uses the same Spatial task 0, object
`akita_black_bowl_1`, and target `plate_1` as the geometry-only replay. There are no
learned components, grasp welds, object-state resets, or automatic parameter searches.

## Human timing and provenance

Edit `config/test_007_demo.json`:

```json
{
  "grasp_time_seconds": 3.77,
  "release_time_seconds": 10.5,
  "transport_start_time_seconds": 4.33,
  "transport_end_time_seconds": 10.3
}
```

These are approximate video annotations, not Panda wall-clock commands. They
resolve to source frames at 3.770000, 10.508333, 4.336667, and 10.306667 seconds,
respectively. The transport contains 180 samples. The grasp annotation initially
uses the video's reviewed pickup/onset time; it is not an automatically detected
contact event. Release is the approximate hand-opening annotation.

Non-null named annotations take precedence over legacy `phase_overrides`, then
automatic estimates. Null named values mean no named override; legacy overrides
still support explicit null/unavailable bounds. Transport must lie between grasp
and release, and the requested events must lie within the recording. Times snap
to source timestamps; annotations far inside unresolved gaps are rejected.

`automatic_phase_estimates` remains untouched. Metadata records named manual
annotations, requested/effective timestamps, their sources, and the review
rationale. The manipulation CLI reads the current demo config on every run and
resolves its source interval even if a previously generated metadata file has old
manual values. Changing human timing never changes robot phase durations.

## Geometry and grasp choice

The environment settles for 20 control steps before recording bowl and plate
positions. Observations are `akita_black_bowl_1_pos`, `plate_1_pos`,
`robot0_eef_pos`, `robot0_eef_quat`, and `robot0_gripper_qpos`.

Compiled bowl collision geometry is approximately 107 mm wide; the open Panda
finger joints span approximately 78 mm. The first controlled attempt therefore
uses a top-down rim pinch, with nominal world-frame EEF-minus-object offset
`[0, 0.045, 0.035]` m. This offset is a robot-specific choice, not human hand
imitation. `grasp_offset[2]` is the configurable grasp height offset. Pregrasp is
vertically above that rim pose, using `pregrasp_height` for its Z component.

Both finger pads must contact the bowl for the configured confirmation window
before acquisition is declared. By default the measured EEF-minus-object offset
at that point is frozen for subsequent object-to-EEF conversion, provided it is
within 30 mm of the nominal pose. Use `grasp_offset_mode: "configured"` to hold
the configured offset instead. Fixed orientation is a downward EEF quaternion
`[1,0,0,0]` (xyzw). The object is never kinematically attached to the robot.

The observed object positions are model body origins, not necessarily geometric
centres. The bowl origin is about 1.59 mm below its lowest collision geometry.
The plate collision top is about 16.44 mm above the plate origin. The explicit
`placement_object_height_offset: 0.018` m puts the bowl bottom roughly 3.15 mm
above the highest plate collision geometry for a short physical settling drop.
This is a conservative geometry-based initial placement, not a validated grasp
or placement guarantee. Asset `bottom_site` / `top_site` values are not used:
their generic values differ substantially from the compiled collision geometry.

## State machine

All phases enter only after the preceding phase completes. Aborts stop the
sequence; later phases are not attempted.

| Phase | EEF behavior / completion | Gripper |
|---|---|---|
| SCENE_SETTLE | Hold reset position for configured steps; freeze anchors | open |
| PREGRASP | Quintic move above rim; reach position tolerance | open |
| DESCEND | Vertical move to nominal rim pose; reach tolerance | open |
| GRASP | Hold pose for `grasp_close_steps` | close |
| GRASP_SETTLE | Hold for `grasp_settle_steps`; require sustained bilateral pad contact; freeze offset | close |
| LIFT | Raise EEF by `postgrasp_lift_height`; confirm actual bowl height increased | close |
| TRANSPORT | Existing human retargeting from actual post-lift bowl pose to target hover pose | close |
| PLACE | Lower to physical placement origin plus frozen grasp offset; reach tolerance | close |
| RELEASE | Hold for `release_open_steps` | open |
| RELEASE_SETTLE | Hold for `release_settle_steps` | open |
| RETRACT | Move upward by `retract_height`; hold to evaluate retention | open |
| DONE | Report final result; no additional command | open |

Transport uses the unchanged `retarget_transport()` mapping and its configurable
axis, lateral, and vertical scales. The transport goal is the physical placement
origin plus `transport_clearance_height`; PLACE performs the final vertical
lowering. The actual post-lift bowl position anchors the first transport sample,
so the scripted lift is not undone. No +0.18 m visualization offset is used.
The original smooth endpoint correction is logged and capped at 60 mm; larger
corrections abort for review rather than being silently forced through.

## Controller, gripper, and failure checks

Cartesian/orientation actions use the existing `inspect_controller()` and
`desired_pose_to_action()` affine inverse for world-frame delta OSC_POSE.
`PandaGripper.format_action()` explicitly defines -1 as open and +1 as close;
it integrates command sign into opposite finger actuator directions. The script
checks this convention without changing the live internal gripper state.
A smaller positive scalar would not request a partly closed aperture. Physical
closing progresses over the configured close steps.

`approach_step_size` limits reference position changes (metres/control step).
`max_eef_speed_m_s` additionally limits reference speed. `max_command_step_m`
limits feedback displacement and `action_clip` limits normalized pose actions.
These are reference/command bounds, not promises about actual robot velocity.
LIBERO returns `done = task_success`, even while the bowl remains held; the
script records this signal but continues release/retraction. Only a true
underlying simulator termination or an unexplained non-success done aborts.
Every phase also has its own `phase_max_steps` budget, workspace checks, an EEF
tracking-error threshold/window, and object/target validity checks.

Intentional finger/bowl contact is allowed from DESCEND onward. Other robot/scene
contacts abort. Bowl/support and bowl/plate contacts are normal physics. During
LIFT, TRANSPORT, and PLACE, missing bilateral contact has a short configured grace
period; excessive EEF-minus-object offset drift or sustained contact loss marks
`grasp_lost` and aborts. No parameters are automatically changed after a failure.

## Evaluation and logging

`metrics.json` distinguishes acquisition (bilateral pad contact), lift (measured
height above settled start), retention during transport, geometric target-region
arrival, post-release retention, and LIBERO's own `check_success()` predicate.
The geometric target region uses configured XY radius and physical-placement
height tolerance. Final success requires completing the sequence and maintaining
region membership, LIBERO success, and no bilateral grasp through the final
retract hold. Unattempted transport/release outcomes are null, not successes.

Every executed step, including scene settling and a failing post-step observation,
is logged in `replay.csv`: phase, time, EEF reference/feedback/error, actual and
applicable desired object positions, live target position, gripper command and
joint states, distance to target, all seven actions, bilateral contact, offset
drift, object quaternion, LIBERO success, and contact geometry names. Additional
CSV reference data is written if TRANSPORT is reached. Failure phase/reason and
final contact evidence are saved even for a controlled abort.

Every completed step is also appended immediately to `steps.jsonl`, and the
executed implementation is copied into the output directory. SIGTERM triggers a
graceful diagnostic abort when Python can handle it; an uncatchable termination
can still leave the durable per-step log for inspection.

Five requested diagnostic plots and per-phase camera snapshots are saved. Camera
snapshot failures are recorded separately and do not masquerade as physics failures.

## Run

```bash
conda activate libero
MUJOCO_GL=osmesa NUMBA_CACHE_DIR=/tmp/libero-numba MPLCONFIGDIR=/tmp/libero-mpl \
  python scripts/scripted_libero_pick_place.py
```

Robot settings: `config/libero_pick_place.json`. Mapping:
`config/libero_retarget.json`. Human annotations: `config/test_007_demo.json`.
Use `--config`, `--mapping-config`, `--demo-config`, `--demo`, `--seed`, or
`--output` to change them. The default output is
`results/libero_pick_place_attempt_001`; a nonempty output directory is refused
so an attempt cannot silently overwrite earlier evidence. An unsuccessful
attempt writes diagnostics then exits with status 2. Running another attempt is
an explicit action, not part of an automatic tuning loop.


## Measured outcome

[Verification attempt 003](../results/libero_pick_place_attempt_003/README.md)
completed all phases and passed the final post-release retention and LIBERO
success checks. The bowl was acquired, lifted, retained through transport,
placed on the plate, released, and left there after retraction. EEF RMSE was
1.894 mm; maximum EEF error was 15.758 mm; maximum object lift was 128.709 mm.
Final object-to-initial-target-anchor error was 11.650 mm. Total: 2,128 steps.

The first attempt stopped on LIBERO's early success signal in PLACE; diagnosis
identified the API interpretation error. A verification process was then
interrupted by SIGTERM before grasp. Attempt 003 verified the corrected sequence
with identical physical parameters. Each attempt and its evidence is preserved;
there was no automated grasp parameter tuning. Thirty relevant tests and an
independent per-step audit passed. This validates one task/seed, not general
robustness across layouts.
