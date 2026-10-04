# Object-motion trajectory following

Run in the existing LIBERO Python environment from the repository root:

```bash
conda activate libero
MUJOCO_GL=osmesa NUMBA_CACHE_DIR=/tmp/libero-numba MPLCONFIGDIR=/tmp/libero-mpl \
  python scripts/replay_libero_transport.py
```

The default scene is exactly the exploration script's `libero_spatial` task 0,
`pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`,
using `OffScreenRenderEnv`, Panda and OSC_POSE. Camera observations are disabled
for this position-only test (the scene and physics are unchanged). The horizon is
extended for the trajectory test. Scene objects remain in place; the gripper is
commanded open throughout and replay stops on detected robot/scene contact.
No grasping or learning is performed. Seed defaults to 0.

## Measured input

Default source:
`results/test_006_slower_raw/object/cube_center/gap_filled_10_frames/gentle_smoothing/savgol_7.csv`.
This is the seven-sample quadratic Savitzky–Golay object centre trajectory after
bounded gap filling. Hand columns/files are never used.

The CSV has 1,264 rows over 42.13 seconds. Decoder `time_s` is in seconds;
`x_m,y_m,z_m` are metres, with median sample interval 1/30 second. Use timestamps,
not a hard-coded video rate. The fixed coordinate frame is ID0's centre:
X from decoded corner 0 to 1, Y from corner 3 to 0, Z = X cross Y out of the
printed face. Other world tags are registered to ID0, not used as changing frames.
This is a tag frame; its alignment with the robot is a chosen hypothesis.

Complete columns: `frame,time_s,status,candidate_count,candidate_tag_ids,source_tag_id,source_instance,world_source_id,error_px,ambiguity_gap_px,max_candidate_disagreement_m,x_m,y_m,z_m,distance_m,source_status,pose_source,smoothing_method`.

There are 18 unresolved missing position rows. The default experiment follows
only the first continuous segment: 1,022 samples, 0–34.058333 seconds. It stops
source loading at the first missing row or timestamp gap over 0.15 seconds,
never silently bridges the remaining gap. `completed_full_source` is therefore
false even when the selected segment completes. Existing `task_space/trajectory.csv`
(`time,x_rel,y_rel,z_rel,speed,phase`) is also accepted; gaps remain detectable
through timestamps.

## Mapping and actions

Edit `config/libero_transport.json`. For robot axis `i`:

```text
p_rel = p_object - p_object[0]
mapped[i] = axis_signs[i] * scale_factors[i] * p_rel[axis_permutation[i]]
p_desired = p_eef_immediately_after_reset + mapped
```

Default permutation `[0,1,2]`, signs `[1,1,1]`, scales `[0.5,0.5,0.5]`:
robot X/Y/Z receive half the human X/Y/Z displacement. To swap X/Y while
preserving handedness use permutation `[1,0,2]`, signs `[-1,1,1]`.
Set scales to `[1,1,1]` for full size; no automatic workspace fitting is applied.
The full desired path must lie inside the configured workspace box.

`map_to_libero_frame` is the single mapping implementation. The starting EEF
position and orientation come from the reset observation before any action.
Positions are linearly resampled at the live control frequency (20 Hz here).
Default time stretching is at least 2× and increases if necessary to keep
requested Cartesian speed at most 0.1 m/s. The actual stretch is recorded.
A one-second final hold is included in metrics and marked by `tracking_steps`.

The installed classic robosuite controller's `set_goal` adds a scaled position
delta to its current world EEF position. `set_goal_orientation` left-multiplies
current orientation by the axis-angle delta rotation. Controller configuration
is checked at runtime: OSC_POSE, delta mode, fixed impedance, world frame,
seven actions. Live input/output bounds determine the inverse affine transform:

```text
error = desired - actual_before_step
rotation_error = rotvec(R_reset @ R_actual.T)
action_pose = (concat(error, rotation_error) - output_midpoint)
              * input_range / output_range + input_midpoint
```

The inspected defaults are input ±1, output ±0.05 m for translation and ±0.5 rad
for rotation. Thus a 1 cm X error gives action X = 0.2. Commands are clipped to
the intersection of live bounds and ±0.5. Rotation correction holds the initial
orientation rather than accumulating drift; action 6 is always -1 (open).

## Outputs and interpretation

Outputs default to `results/libero_transport/`:

- `preflight.json`: source metadata, explicit mapping matrix, limits.
- `desired_trajectory.csv`: the complete requested resampled path, including reset.
- `replay.csv`: one row per executed `env.step`, with desired XYZ, post-step actual
  `robot0_eef_pos` XYZ, all seven commanded actions, signed post-step XYZ errors,
  pre-step XYZ command errors, Euclidean error, orientation error and clipping.
- `metrics.json`: mean Euclidean error, sqrt(mean squared Euclidean error), maximum
  error, final-position error, completion status, reset pose and live controller bounds.
- `tracking_3d.png`: desired versus measured 3D path, axes in metres.
- `tracking_timeseries.png`: X/Y/Z versus simulation time and error in millimetres.

The timestamp is the end of each control step; its desired value is the target
commanded during that step. Logging is at the action/control rate, not every
internal MuJoCo integration substep. Reset is not counted as an executed action
or a zero-error metric sample. Final-position error compares the last observed
position with the last executed target; the additional
`endpoint_error_to_full_requested_target_m` exposes early termination.

A small error on a completed segment supports following that scaled, slowed
measured path in this scene. It does not establish full-speed/full-scale tracking,
calibrated real-to-simulation alignment, full-recording coverage or manipulation
success. Check completion/contact status before interpreting error metrics.

Options: positional source CSV, `--config`, `--output`, `--seed`, `--bddl` for an
explicit alternate scene, and `--inspect-only` to inspect the source and mapping
without starting LIBERO. Use a separate output directory for comparisons.

Tests:

```bash
python -m unittest discover -s tests -p test_libero_transport.py -v
```
