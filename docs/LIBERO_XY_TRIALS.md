# Ten controlled initial-position trials

The batch in `results/libero_xy_trials_10` uses the successful
`libero_pick_place_attempt_003` as its nominal scene. The ten requested offsets
are enumerated in `scripts/libero_xy_trials.py`; there are no retries or
controller/grasp parameter searches.

Each episode repeats the seed-0 reset and the existing 20-step scene settling
sequence. Immediately before PREGRASP, only the manipulated bowl's free-joint
X and Y coordinates are changed. Z, quaternion, robot joints, other object
poses, and velocities are preserved. The full generalized position and velocity
vectors are checked against the nominal setup with a 1e-10 numerical tolerance;
the perturbation operation itself is checked for exact XY-only modification.
Initial-state arrays and a readable audit are saved for every trial.

The plate starts at the same pose and the retargeting goal stays fixed. During
the episode, all bodies retain the original contact physics. The plate is not
artificially pinned. Human transport is retargeted anew after each grasp/lift,
using the observed bowl position and the same fixed target hover point. The
measured grasp offset and speed-based timing follow the unchanged nominal
algorithm; they are outcomes of that algorithm, not tuned parameters.

## Preflight and the reduced tenth offset

Before any manipulation trial, a separate MuJoCo data instance checks the
compiled collision geometry for table support, table XY bounds, and intersections
with other bodies. Resting table contact is allowed. Non-support intersections
greater than 1e-7 m are rejected. Joint-limited numerical Panda IK checks the
approach, grasp, lift, all 180 human transport knots, placement, and retraction
with the original fixed EEF orientation and configured workspace limits.
This establishes kinematic feasibility, not a guarantee of successful grasping
or dynamic tracking. The separate nominal setup performs only the common
20-step settling sequence and is not a manipulation attempt.

Requested offsets 1–9 pass unchanged. Trial 10's requested (+0.04, 0) m
intersects the plate by up to 5.115 mm. Its applied offset is
**(+0.0250671875, 0) m**. A descending search along the same ray followed by
bisection resolves the contact boundary to less than 1 micrometre. The requested
invalid pose, contact pairs, applied pose, and search bracket are recorded in
`preflight.json`. Every applied pose passes all four preflight criteria.

## Artifacts and execution

`preflight.json` stores input hashes, settings, nominal state, requested/applied
offsets, geometry checks, and IK residuals. Each `trial_NN` directory contains
initial-state arrays, a parameter/initialization audit, per-step JSONL and CSV,
transport references when reached, phase images, diagnostic plots, and metrics.
`summary.csv` and `summary.json` aggregate results. The batch result README gives
the outcome table and final audit.

In the LIBERO Python environment, a new output directory can be used with:

```bash
MUJOCO_GL=osmesa python scripts/libero_xy_trials.py --output results/new_xy_batch --prepare
MUJOCO_GL=osmesa python scripts/libero_xy_trials.py --output results/new_xy_batch --run-all
```

Preparation refuses nonempty output. Trials refuse any already-started index.
Batch continuation skips already-started indices, including failures, so it
cannot silently retry them. All physical control code and configuration inputs
must retain their preflight hashes. Unit tests exercise the XY-only contract,
unchanged valid offsets, direction-preserving boundary reduction, and rejection
of an invalid nominal pose without adding physical trials.
