# Short action chunks: one nominal rollout

**Result: unsuccessful.** Predicting 12 actions and executing four before
replanning did not fix the closed-loop failure on this nominal reset.

Same two successful demonstrations (4,230 transitions), same small image
encoder and 18D proprioception, 167,116 parameters. No additional data, RL,
SmolVLA, observation history, timestep/phase input, or scripted policy actions.

| Training run (40 epochs each) | Standardized MSE | Weighted objective | Raw action MSE |
|---|---:|---:|---:|
| Gripper weight 1, no rollout | 0.049776 | 0.049776 | 0.002627 |
| Gripper weight 3, evaluated | 0.050113 | 0.054653 | 0.002264 |

The weighted run has first-action MSE 0.001143 and chunk gripper sign accuracy
99.64%. These are training-set results. Weight 3 was tried because the
unweighted model missed both demonstrated closing onsets, not in response to
rollout performance. Both runs and their checkpoints are preserved; the first
is in `../libero_chunk_bc_unweighted/`.

## Demonstrated gripper and model predictions

Open=-1; close=+1. There are 2,349 open commands (55.53%) and 1,881 closed
commands (44.47%). Episode 1: open 840 / closed 944 / open 325 steps.
Episode 2: open 859 / closed 937 / open 325. Only one closing event per episode;
4 of 4,228 adjacent within-episode pairs change class (0.095%).

| Observation | Demonstrated current command | Predicted first action, episode 1 / 2 |
|---|---:|---:|
| One step before close onset (839 / 858) | -1; next action +1 | -0.722 / -0.733 |
| Close onset (840 / 859) | +1 | -0.721 / -0.732 |
| Immediately before physical grasp (849 / 868) | +1 | +1.246 / +1.206 |

All 12 predicted gripper actions remain negative at the first two rows in
both episodes. The stronger check—anticipating the close command before the
fingers move—**fails**, even with 3x gripper loss. The pre-physical-grasp check
**passes**, but those observations already contain closing fingers. The first
bilateral grasp occurs after action 850 / 869, ten steps after close onset.
Outputs above +1 are raw regression values; execution thresholds them to +1.
See `gripper_audit.json` and `gripper_around_grasp.png` for full chunks.

## One nominal rollout (2,400 actions, 120 simulated seconds)

| Requested outcome | Observed result |
|---|---|
| Reaches object | No; minimum EEF-to-object-center distance 0.37117 m |
| Closes gripper | No; 0 / 2,400 close commands |
| Lifts object | No; maximum height change approximately zero |
| Final LIBERO success | False (also never true during rollout) |
| Failure mode | Fails to approach; remains near initial pose with open fingers until horizon expires |

EEF moves from approximately (-0.2090, 0.0000, 1.1728) m after the first action
to (-0.2033, 0.0037, 1.1905) m after the last. The bowl remains stationary.
Executed raw gripper predictions range from -1.037 to -0.817; finger positions
remain near +0.04/-0.04 m. There is no bilateral acquisition or lift. The
successful demonstration's EEF-to-object-center distance at first grasp is
approximately 0.060 m, for context; the rollout never approaches that geometry.

This single test provides no evidence that chunking solves the failure. The
on-demonstration checks also show that high aggregate sign accuracy still
hides failure to initiate closing. The immediate rollout failure is earlier:
the model does not reproduce the approach. No second rollout or subsequent
tuning was performed.

Validation: two chunk tests pass (episode boundaries/end masking and output
shape/normalization); checkpoint reload predictions match; 600 replans execute
exactly four actions each; initialization matches the existing nominal reset.
Details and reproduction: `docs/LIBERO_CHUNK_BC.md` in the repository.
