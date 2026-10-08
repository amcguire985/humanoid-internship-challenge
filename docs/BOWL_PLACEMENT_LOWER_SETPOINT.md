# Bounded persistent LOWER setpoint

Only LOWER translation/setpoint lifecycle and its requested safeguards change.
ALIGN progression is unchanged. LOWER entry seeds a fresh Cartesian setpoint from
measured EEF position and computes the fixed support EEF target using the existing
gripper-to-bowl transform and gravity swing. The world XY support target remains
plate-centered. Each command advances at most the existing 3 mm toward that
endpoint, limits accumulated lead to the same four increments (12 mm) used in
ALIGN and OSC capacity, and clamps Z to the target floor. Target/setpoint reset
on leaving LOWER. Upright correction, action scaling, gains, height geometry,
transport, 12 mm XY / 4 mm height tolerance, 65-step LOWER timeout and release
requirements are unchanged. Manifest: shared_geometry_placement_v6_persistent_lower.

## Requested safeguards and their new detection criteria

These criteria are explicit implementation choices, not measured phone limits:
- Any observed bowl/plate support contact holds further downward setpoint advance.
  Horizontal and orientation control remain active; release still needs every
  original condition for the original three confirmations.
- Support contact more than the existing 4 mm height tolerance above goal for
  three confirmations fails as placement_lower_unexpected_support_contact.
- Over nine intervals (3 ? existing confirm_steps), net descent <=0.4 mm (one
  tenth of existing height tolerance), with significant accumulated tracking lead
  and height still above tolerance, fails as placement_lower_tracking_stall.
- Tracking error exceeding lead plus one original increment (15 mm), while lagging
  behind the target and still above support tolerance, fails as
  placement_lower_excessive_tracking_error. Fast progress toward support is not
  rejected as lag. Lead saturation prevents continuing to accumulate command error.

Only bowl/plate support contact is currently available. These safeguards cannot
identify arm/table or other contacts; a tracking-stall result needs forces/contact
pairs and joint-limit evidence to establish its cause. Actual physical overshoot
is not guaranteed absent by a commanded setpoint floor.

## Exactly one Condition C validation (cell 8)

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

Use preceding initialization and Drive mount as before. Output folder prints
before/after execution. This runs one rollout, no other conditions.
Existing outputs are preserved. Attach C/placement_alignment_validation.json,
C/placement_lower_validation.json, C/summary.json and trajectory.jsonl afterward.

New LOWER trace fields: persistent setpoint, fixed target, lead cap, tracking
error, contact hold, stall window and minimum descent. The separate LOWER report
contains entry/end height error, support-height threshold achievement, action
count, release entry, opening command/confirmation, success, grasp loss, contact
holds, failure reason and max LOWER tilt. Existing trace/video remains necessary
to assess unexpected contact or rotation. Alignment report remains available.

Changed files: scripts/bowl_placement.py, scripts/evaluate_bowl_hybrid.py,
tests/test_bowl_placement.py, this document. No simulations were run locally.
Tests use analytic lag fixtures, not robot-dynamics predictions.

Local validation: all 56 bowl tests passed, including six new LOWER tests. No CUDA/LIBERO rollout has been run locally.
