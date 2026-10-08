# Bounded persistent ALIGN setpoint

Robosuite 1.4.0 OSC `set_goal` passes scaled delta and current `ee_pos` to
`set_goal_position`, which computes `current_position + delta`. Goals are
reconstructed on each action, not accumulated across policy steps.
Source: https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/osc.py#L233
and https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/utils/control_utils.py#L101

Only PLACE/ALIGN translation changes. At entry the persistent setpoint starts
from measured EEF position. The first ALIGN action latches the full geometrically
computed EEF translation target (plate-centered bowl at handoff height, with the
existing gravity correction and frozen grasp transform). Each subsequent action
advances the setpoint toward this fixed target by at most the original 3 mm.
OSC receives setpoint minus measured EEF through the unchanged scaling helper.

Accumulated lead is bounded to four existing increments, 12 mm, further bounded
by translation output capacity. This is a new explicit anti-windup design choice,
not a phone-derived threshold or changed placement tolerance. An obstructed arm
cannot grow an unbounded error; command lead saturates. Near the target the
setpoint reaches the endpoint without stepping past it. Entering/leaving ALIGN
clears the setpoint/latched target. LOWER/RELEASE/VERIFY retain their old logic.
Existing rotation corrections, gripper safeguards, transport constraints, gains,
12 mm tolerance, 65-step timeout, task and checkpoint remain unchanged.
Algorithm manifest version is `shared_geometry_placement_v5_persistent_align`.

## Exactly one Condition C phone-constraints validation

After Drive mount and preceding initialization cells, replace cell 8 with:

```bash
%%bash
set -euo pipefail
cd /content/humanoid-internship-challenge
git fetch origin
git switch upright-mug-orientation
git pull --ff-only origin upright-mug-orientation
git log -1 --oneline
bash config/smolvla/bowl_phone_constraints_colab_cell.bash
```

The cell prints its unique Drive output folder and runs one rollout. Do not run
other diagnostic cells. Existing results are preserved. No local simulation has
been run; runtime convergence is unvalidated.

Inspect `C/placement_alignment_validation.json` for minimum XY distance,
alignment threshold achieved, ALIGN action steps, LOWER/RELEASE entry,
LIBERO success, grasp loss and orientation diagnostics. New JSONL fields:
`placement_align_setpoint_m`, `placement_align_fixed_target_m`,
`placement_align_lead_limit_m`, `placement_align_lead_capped`, and
`placement_align_tracking_error_m`. Existing full EEF target remains logged.
Use trace/video to assess unwanted rotation; no metric alone proves its absence.

If failure persists, inspect saturation frequency and tracking error alongside
observed EEF motion. Tracking error with capped lead suggests tracking or
obstruction, but contact forces, all robot collision pairs, joint-limit margins,
and OSC internal goals/torques remain necessary to separate contact, reachability
and target geometry. Do not relax tolerance or extend timeout.

Changed files: scripts/bowl_placement.py, scripts/evaluate_bowl_hybrid.py,
tests/test_bowl_placement.py, this document. Lag tests use an analytic response
fixture and do not constitute simulated robot validation.

Validation: all 50 bowl tests passed; syntax and diff checks passed. No CUDA/LIBERO validation rollout has been run locally.
