# Transport-only object retargeting

This experiment maps the processed phone-video object path into LIBERO Spatial
 task 0, `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`.
The object path is virtual; only the Panda EEF tracks it, with an open gripper
and a fixed 18 cm world-Z offset. There is no grasp, object manipulation, or training.

## Inspection before implementation

`scripts/replay_libero_transport.py` contains the old human loader,
`OffScreenRenderEnv` creation, controller inspection, affine action conversion,
closed-loop replay, metrics, and plotting. `experiments/explore_libero.py` uses the
same task. The old experiment's recorded RMSE is 2.429 mm. The new entry point
imports its controller conversion, action-contract checks, resampling, contact
check, and metric helpers without changing that experiment.

New input is `results/test_007_block_only_raw/task_demo/processed_demo.csv` with
`time`, `object_x/y/z`, object offsets, speed, `phase`, `valid_measurement`,
`valid_processed`, interpolation/rejection flags, `transport_progress`, raw object
positions, and `goal_x/y/z`. Units are seconds and metres in the ID0 tag/world
frame. `metadata.json` provides `goal_position`, start position, displacement,
phase boundaries, processing parameters, and quality flags.
`cleaned_trajectory.csv` has only `time,x_world,y_world,z_world`; `transport.csv`
includes the wider pickup-to-release interval. Neither is silently treated as the
inner transport segment. The new loader selects metadata `transport_start` through
`transport_end` from the processed CSV, includes both boundary samples, and rejects
invalid values, unresolved gaps, missing boundaries, and non-increasing timestamps.
The human start is the selected first object position, not metadata's clip start.

The reviewed test_007 interval is **4.336667–10.306667 s** (180 samples),
including carrying, lowering, and the final hold. The old automatic interval
4.836667–6.405 s truncated this action. Video annotations in
`config/test_007_demo.json` are snapped to actual source timestamps by the
processor; metadata retains the automatic estimates, annotation rationale, and
requested/effective boundary times. Release is annotated at 10.508333 s. See
[the phase review](../results/test_007_block_only_raw/task_demo/phase_review.md).

Verified observation keys:

| Meaning | Key |
|---|---|
| Task object | `akita_black_bowl_1_pos` |
| Plate target anchor | `plate_1_pos` |
| EEF world position | `robot0_eef_pos` |
| EEF orientation, xyzw quaternion | `robot0_eef_quat` |

The BDDL goal is `On akita_black_bowl_1 plate_1`. For this geometric experiment the
plate origin plus configurable `target_offset` is the target, not a computed bowl
contact placement pose. The robot first holds its reset pose for one second while objects settle under
gravity. Both anchors are then captured and held fixed. EEF clearance is separate from
object coordinates and never changes endpoint geometry.

## Mathematical mapping

Let `A = diag(axis_signs) @ I[axis_permutation]`, a signed permutation that defines
which human axes are horizontal and vertical. Let

```
q(t) = A @ (p_h(t) - p_h(0))
d_h  = A @ (g_h - p_h(0))
d_s  = g_s - s_s
L_h  = norm(d_h.xy); L_s = norm(d_s.xy)
e_h  = d_h.xy/L_h;    e_s = d_s.xy/L_s
n_h  = [-e_h.y, e_h.x]; n_s = [-e_s.y, e_s.x]
u(t) = dot(q(t).xy, e_h)/L_h
l(t) = dot(q(t).xy, n_h)
v(t) = q(t).z - u(t)*d_h.z
B(t) = s_s + u(t)*d_s + horizontal_scale*l(t)*[n_s.x,n_s.y,0]
       + vertical_scale*v(t)*[0,0,1]
r    = g_s - B(T)
b(t) = 10*tau^3 - 15*tau^4 + 6*tau^5, tau=t/T
P(t) = B(t) + b(t)*r
EEF(t) = P(t) + grasp_offset
```

Longitudinal horizontal scale is `L_s/L_h`, dictated by endpoint geometry.
`horizontal_scale` configures lateral shape in metres/metre; `vertical_scale`
configures lift relative to the start-to-goal height baseline. Axis flips and
permutation are applied once. Purely vertical or zero-length tasks are rejected.

The measured endpoint does not coincide with the human goal centre, even after
the interval correction (the goal is a region). A direction-alignment transform
alone therefore cannot meet the requested simulated endpoint. The
quintic correction distributes that mismatch over the full interval with zero
first and second blend derivatives at either boundary. It has no final-sample
snap, but it DOES modify the measured path shape; its vector and norm are logged.
If the measured path already ends at its goal, the correction is zero. The
sampled curve is linearly interpolated and slowed to bound reference speed.
There is no invented pickup lift: only lift present inside transport is retained.

## Controller and execution

Panda uses the existing verified world-frame delta `OSC_POSE`, fixed impedance,
seven-dimensional action `[dx,dy,dz,rx,ry,rz,gripper]`. Actions are normalized, not
Cartesian metres. Runtime controller bounds are inspected and inverted exactly:

```
a = (delta - (output_min+output_max)/2)
    * (input_max-input_min)/(output_max-output_min)
    + (input_min+input_max)/2
```

For the verified installation, action 1 means 0.05 m translation or 0.5 rad
rotation. The existing helper computes world-frame axis-angle orientation error
and applies action clipping. A fixed reset orientation is maintained; gripper is
always -1. Defaults bound reference speed to 0.02 m/s, commanded translation error
to 0.025 m per step, normalized action to +/-0.5, and EEF workspace to
`[-0.6,-0.6,0.85]..[0.6,0.6,1.5]` metres. These are command/reference bounds, not a
guarantee on actual robot velocity. Robot/scene contact and workspace exits stop
the replay. The full planned EEF path is checked after scene settling and before approach.

A quintic approach moves from reset EEF pose to the first desired EEF pose. Its
duration accounts for the blend's peak speed. Replay proceeds only if approach
error is within 5 mm after a one-second start hold. Transport ends with a one-second fixed-target hold.
Tracking errors compare each command's desired pose against post-step feedback.
Overall, scene-settling, approach, transport, and hold metrics are separate.

## Run and outputs

```bash
conda activate libero
MUJOCO_GL=osmesa NUMBA_CACHE_DIR=/tmp/libero-numba MPLCONFIGDIR=/tmp/libero-mpl \
  python scripts/retarget_libero_object.py
python -m unittest discover -s tests -p 'test_libero*.py' -v
```

Optional `--demo`, `--config`, `--output`, and `--seed` select inputs and outputs.
Settings are in `config/libero_retarget.json`. Default output:
`results/libero_retarget/`.

- `metrics.json`: settings, source metadata/quality, controller bounds, frame
  mapping, anchor coordinates, endpoint correction, metrics, and stop reason.
- `retargeted_trajectory.csv`: full time-resampled object and EEF references,
  source-relative time, and longitudinal task progress.
- `replay.csv`: every executed step, stage, relative source time, temporal source
  progress, longitudinal task progress, original human phase, object reference,
  EEF reference/feedback/errors, seven actions, clipping/limiting and orientation
  error. `desired_object_*` and `retargeted_object_*` intentionally coincide: this
  experiment has one virtual object reference, not a manipulated-object trace.
  During scene settling the logged virtual object reference denotes the anchor
  captured after settling (no object trajectory is executed then). During
  approach that reference remains at the object start, while the EEF
  interpolates from its reset pose. Source time is relative to transport start;
  add metadata `transport_start` to recover video time.
- `retargeting_3d.png`: both anchors, object reference, desired and actual EEF.
- `top_down.png`: anchors and virtual object path.
- `tracking_timeseries.png`: EEF x/y/z and error for the complete run.
- `tracking_error.png`: standalone tracking error.
- `tracking_3d.png`: complete EEF run including approach.

The source metadata explicitly marks `usable_for_retargeting=false` because height
and video calibration are unverified and phase/placement review flags remain.
This requested geometry-only experiment carries those flags into the report and
prints them; it does not certify physical calibration or grasp feasibility.

## Dependency recovery and current installation

The installed `/home/student/robot_challenge/LIBERO` had null-filled BDDL/source
files. A fresh upstream checkout was placed in `/tmp/libero-retarget-dependency`
at commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`, with paths configured in
`/tmp/libero-retarget-config/config.yaml`. The installed checkout and user config
were not edited during that measured run. It used the existing `libero` conda environment:

```bash
PYTHONPATH=/tmp/libero-retarget-dependency \
LIBERO_CONFIG_PATH=/tmp/libero-retarget-config \
MUJOCO_GL=osmesa NUMBA_CACHE_DIR=/tmp/libero-numba MPLCONFIGDIR=/tmp/libero-mpl \
/home/student/miniconda3/envs/libero/bin/python scripts/retarget_libero_object.py
```

The original installation has since been restored from that same clean commit at
`/home/student/robot_challenge/LIBERO`, including Git metadata and regenerated
package metadata. All 1,116 tracked files were verified against the clean copy.
The damaged installation is preserved at
`/home/student/robot_challenge/LIBERO.damaged-backup-bec396d4`.
Use the standard run command above; temporary `PYTHONPATH` and
`LIBERO_CONFIG_PATH` overrides are no longer needed. The current saved results
were regenerated after the phase correction using the repaired installation;
the temporary-checkout command above documents the original recovery run.
