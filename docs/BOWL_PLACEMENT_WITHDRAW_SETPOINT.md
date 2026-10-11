# Final bounded WITHDRAW setpoint and release diagnostics

VERIFY/WITHDRAW now seeds a fresh vertical setpoint from the EEF pose on entry;
the fixed endpoint remains that pose plus the existing 60 mm withdrawal.
Advance by the existing 3 mm per action, with accumulated lead capped at four
increments (12 mm), limited further by OSC positive-Z output capacity.
Command zero XY translation and nonnegative Z translation. Once at/past target,
command zero translation rather than pulling down or correcting sideways.
The gripper remains explicitly open (-1) throughout VERIFY. Orientation handling
is unchanged. The existing >=50 mm rise requirement, ten-step settling minimum,
35-step withdrawal timeout and LIBERO predicate remain unchanged. Earlier
GRASP/TRANSPORT/ALIGN/LOWER/RELEASE behavior is unchanged.
Algorithm manifest: shared_geometry_placement_v7_persistent_withdraw.

Intentional release is latched when an opening command is actually sent during
RELEASE/VERIFY. Subsequent relative-pose changes are logged as released-object
relative changes; grasp drift fields become null, grasp_drift_applicable=false,
and slip_detected=false. These are diagnostics, not modifications to release.
Summary grasp_lost checks only confirmed-transport observations before deliberate
opening (including unintended loss during placement). A later withdrawal failure
cannot turn expected released contact loss into grasp loss.

Summary and printable report separate:
- libero_success: original predicate achieved at any observation, unchanged.
- placement_success: deliberate release, no current grasp, final plate contact
  and final LIBERO predicate true; independent of withdrawal completion.
- controller_completed: state machine reached DONE.
- failure_reason: original failure, including withdrawal timeout if applicable.
Original final_libero_success and placement_controller_completed stay available.
Contact-force/rotation attribution still requires trace and video inspection.

## Exactly one Condition C validation

Use the existing initialization cells and mounted Drive, then run:

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

It prints the unique Drive folder and launches one rollout. No local simulations
have been run. Attach C/summary.json and C/ground_truth/episode_000/trajectory.jsonl
plus video if rotation/contact looks unexpected. New per-action fields record
withdrawal setpoint Z, fixed target Z, maximum lead and tracking error.

Changed files: scripts/bowl_placement.py (WITHDRAW lifecycle/translation only),
scripts/evaluate_bowl_hybrid.py (release-aware diagnostics and separate outcomes),
tests/test_bowl_placement.py, and this document. No phone constraints, checkpoint,
gains, tolerances or timeouts changed. This iteration stops after implementation
and preparation of that one user-executed validation rollout.

Validation: all 60 bowl tests passed, including four new focused withdrawal/reporting tests. No local rollout was launched.
