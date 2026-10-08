# Condition C: unwanted bowl rotation diagnosis

## Finding

The dominant visible rotation was introduced by the hybrid controller's full 3D orientation objective. The phone trajectory contains an arbitrary absolute object heading, and the controller tried to match that heading rather than only the bowl's opening-axis direction. The actual bowl yaw at transport entry was 98.42 degrees; the first human target yaw was -36.79 degrees, a -135.21 degree mismatch. The bowl subsequently turned about 120.07 degrees while the human target yaw changed only 5.55 degrees.

This is an inappropriate orientation-control objective, not evidence that the human demonstrator rotated the object 120 degrees. Both a sustained hybrid yaw correction and smaller SmolVLA yaw actions contributed. Later placement instability has additional contact/release effects that cannot be fully attributed from this one recording.

## Recorded evidence

The latest corrected C trace supplied is `trajectory (3).jsonl`, associated with `summary (9).json` and the physical collision-surface `placement_target.json`. Its SHA256 is `b5e00bd58f0a93d9e32a4b2fff25be8be7ac40ac7929b25062563c7432adcf2d`. `trajectory (4).jsonl` is B and was not used as a new comparison here. The supplied `agentview.mp4` accompanied the earlier C trace (`trajectory (2).jsonl`); no correctly paired video of the corrected run was supplied. The earlier video was inspected as corroborating evidence of rotation, but not used for corrected-run step timing.

| Signal | Observation |
|---|---|
| Initial bowl yaw / human yaw | 98.42 / -36.79 degrees |
| Bowl yaw change during guidance | -120.07 degrees |
| Human yaw change during guidance | +5.55 degrees |
| Largest adjacent logged human rotation | 3.28 degrees |
| Sum of hybrid world-Z normalized corrections | -15.43 |
| Sum of policy world-Z normalized actions | -4.12 |
| Relative grasp rotation drift through step 200 | At most 1.18 degrees |
| Relative drift at RELEASE, step 261 | About 13.07 degrees |
| Relative drift at final step 280 | About 25.27 degrees |
| Bowl tilt at step 240 / RELEASE / final | 13.32 / 12.93 / 20.97 degrees |
| Standard LIBERO success | False |

Sums of normalized action components describe commanded direction and scale, not actual angle integrals: the inner OSC dynamics must not be replaced with direct kinematic integration. At step 100, for example, the policy's world-Z component was about -0.0338, the hybrid added -0.1282, and the final command was -0.1620. The large sustained hybrid term explains the dominant yaw command.

The final full rotational correction at step 261 was approximately `[-0.01263, 0.03729, -0.02206]`. On step 262 the runner passed through raw policy rotation, dropping all correction components to zero. This is a control discontinuity. Tilt subsequently rose to 20.97 degrees and relative rotation drift to 25.27 degrees. Release, reclosure/contact dynamics, and removing orientation feedback all coincide; the trace does not establish their separate causal contributions.

The published transport metric (17.71 degrees) ends before RELEASE and therefore hides the later 20.97 degree tilt. Placement and post-release tilt must be inspected separately.

## Coordinate, interpolation, and controller audit

The human object frame uses the user-confirmed top tag ID6, world ID0 face upward, and matching camera calibration. Positional retargeting rotates about gravity and endpoint-aligns; it does not align the arbitrary decoded tag XY heading with the simulated bowl body heading. That is legitimate for recording object motion, but inappropriate as an absolute yaw task objective.

Simulator bowl quaternions are logged as WXYZ. Reordering to SciPy's XYZW reproduces every logged bowl matrix to within approximately 2e-15. Human CSV quaternions are XYZW; SLERP uses SciPy rotations consistently. End-effector control uses simulator site rotation matrices directly. No quaternion-ordering fault was found.

The rotation error is `log(R_desired @ R_actual.T)` in world coordinates. This agrees with [robosuite 1.4.0 goal composition](https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v1.4.0/robosuite/utils/control_utils.py), which applies the delta on the left of the current orientation. Gripper-to-bowl composition is `inverse(T_world_gripper) @ T_world_bowl`, with desired gripper pose derived through its inverse. These multiplication orders were checked.

The source interpolator has consistent XYZW quaternion handling, SLERP continuity, and endpoint clamping. The largest adjacent recorded human orientation step is 3.28 degrees, not a 120-degree discontinuity. The control ramp interpolated full orientations from the grasp heading to the mismatched human heading over one second; bounded actions then continued attempting that unnecessary yaw change.

The final rotational correction never exactly hit its +/-0.15 component cap (maximum about 0.14915). The *inner pose tracking* can still saturate: assuming the known-good +/-0.5 rad OSC component scale, the reconstructed full-orientation error exceeds that range on 96 guided samples, with a maximum component about 2.17 rad. The exact C `controller.json` was not attached, so that saturation count is conditional on the documented standard scale. Action attribution `policy + correction = executed pose command` matched exactly on all legacy guided steps.

Gripper-to-bowl rotation drift was small while most yaw occurred, so slipping does not explain the main rotation. Drift becomes substantial near contact/release and is a plausible contributor to terminal tilt.

Only the original-to-liquid instruction switch resets the policy queue and processors. There is no extra language switch or queue reset at RELEASE. Both the queue behavior and exact prompt override are unit-tested; there is no evidence of a release-time prompt change.

## Minimal justified correction

Condition C now uses a gravity-relative opening-axis objective. If `u = R_bowl @ [0,0,1]` and `g = [0,0,1]`, the world-frame error is the shortest swing:

`delta = atan2(norm(cross(u,g)), dot(u,g)) * cross(u,g) / norm(cross(u,g))`.

For an already upright bowl, the error is zero for every yaw. Right-multiplying any bowl rotation by local-Z yaw leaves this correction unchanged. A deterministic perpendicular axis handles exact inversion; normal rim-grasp operation remains far from that singular case. Existing correction caps, slew limits, blend/gain values, and activation ramp are retained.

The target bowl orientation is the current bowl orientation rotated by that bounded/ramped gravity-aligning swing. The same world swing is applied to the actual gripper orientation, avoiding a full-yaw or frozen-transform twist-restoration objective. The measured grasp transform remains monitored for slip. SmolVLA retains its rotational contribution; this removes the human absolute-yaw target rather than locking the robot's yaw.

Human object positions, timestamps, velocities/accelerations, resampling, endpoint alignment, translation gains/blends, and source timing remain unchanged. Desired EEF translation is recomputed with the yaw-free bowl orientation because the grasp-offset compensation depends on orientation; this necessary geometric consequence is explicitly distinguished from changing the human trajectory. Raw human orientations remain logged for audit, but are no longer motor targets. The phone demonstration remains an actual control input through its timed position and acceleration profile.

At RELEASE, only the last rotational correction slews toward zero using the existing normalized correction slew rate. Position and gripper remain under the same release policy behavior as before. This removes an abrupt rotational handoff without adding a new grasp/release strategy. No changes were made to Condition B's action behavior, release safeguard, or prompt schedule.

The version is `gravity_opening_axis_v3`. Do not combine it with legacy C runs as if the intervention were identical.

## Artifacts and verification

`scripts/analyze_bowl_rotation.py` creates `orientation_signals.csv`, `orientation_diagnosis.json`, `orientation_sources.png`, and `rotation_actions.png` from an existing JSONL; it never starts a simulator. Existing-run artifacts are under ignored `results/bowl_hybrid_preflight/rotation_diagnosis/`.

The updated C runner automatically saves these diagnostics under `C/ground_truth/episode_000/orientation_analysis/`, in addition to both camera videos and complete ground-truth/action logging. New fields distinguish raw human orientation from the gravity-aligned control target, opening-axis error, policy/executed rotation, physically scaled OSC rotational delta, and release fading. The report separately exposes terminal/post-release tilt.

Local tests cover yaw invariance, world-frame composition under a changed coordinate basis, exact inversion, gravity alignment, quaternion sign and endpoint continuity, unchanged phone position/timing/velocity/acceleration data, no yaw-drift restoration objective, release correction slew, and unchanged release translation/gripper commands. All 21 hybrid tests and 11 existing bowl tests passed locally (32 total). Bash/Python syntax and whitespace checks also passed.

The local reconstructed human reference matches the recorded target to 1.11e-16 m for position and 6.66e-16 for rotation entries. A recorded-state action replay reduced mean world-Z hybrid correction from -0.07753 to +0.00679 using the same recorded policy actions and states. That is a controller calculation, not a new closed-loop simulation and not evidence of reduced actual tilt.

Missing evidence includes the correctly paired latest C agent/wrist video, exact attached C controller settings, internal OSC goals/torques, and contact forces. The trace has enough data to identify the yaw objective and handoff discontinuity, but not to prove all terminal contact dynamics.

## One permitted validation rollout

No new simulation has been run locally. CUDA/LIBERO is in the user's Colab runtime, which this workspace cannot execute directly. After publishing the code, use this exact replacement cell for **one C validation only**:

```bash
%%bash
set -euo pipefail
cd /content/humanoid-internship-challenge
git pull --ff-only origin upright-mug-orientation
unset BOWL_RUN_ROOT
export BOWL_CONDITIONS=C
bash config/smolvla/bowl_hybrid_evaluate.bash
```

Source files to publish are `scripts/bowl_transport_guidance.py`, `scripts/evaluate_bowl_hybrid.py`, `scripts/analyze_bowl_rotation.py`, `tests/test_bowl_hybrid.py`, and this report. B must not be run or modified for this diagnosis.

## Outcome status

- Cause identified: full-orientation yaw matching, with a secondary abrupt release handoff and late contact/relative-transform drift.
- Code corrected and checked locally.
- Whether placement tilt improved: **not yet measured in a new rollout**.
- Whether grasp stability was preserved after the change: **not yet measured**.
- Whether the changed controller places the bowl successfully: **not yet measured**; the existing corrected C run failed.
- Remaining limitations: the permissive 6 cm release neighborhood and policy reclosure remain untouched because this task concerns orientation. Fixing yaw does not guarantee fixing those placement problems.
