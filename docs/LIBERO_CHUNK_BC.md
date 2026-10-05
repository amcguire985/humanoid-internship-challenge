# Short action-chunk BC experiment

Uses only the original successful `episode_001.h5` and `episode_002.h5`
(4,230 transitions). No new demonstrations, RL, or pretrained models.

The image encoder and proprioception input are identical to SmallBC: three
small convolutions (16/24/32 channels), a 64-dimensional image embedding,
and 18-dimensional normalized robot proprioception. Only the last linear
layer changes, from 7 to 84 outputs, reshaped to 12 seven-dimensional LIBERO
actions. The model has no timestep, phase, EEF pose, or observation history.
At 20 Hz the predicted horizon is 0.6 seconds. The evaluator executes four
successive actions (0.2 seconds), discards the remaining eight, then replans
from the new image and proprioception. No action averaging or scripted grasp.

Each observation at t targets actions t through t+11 in the same episode.
Targets beyond the episode end are masked out of the loss, never borrowed
from the next episode. Action/state normalization, image preprocessing,
Adam learning rate .001, batch size 128, seed 0, and the 40-epoch budget match
the original baseline. Gripper output is thresholded at zero at runtime;
pose actions retain the original [-.5,.5] clipping.

The first run uses unchanged standardized MSE. Its complete results are in
`results/libero_chunk_bc_unweighted`. Because it predicts no positive gripper
action at either demonstrated closing onset, a second run uses a modest 3x
multiplier on the gripper dimension. The architecture and data are unchanged.
Both runs are training-set measurements; neither establishes generalization.
Only the weighted model receives a simulator evaluation: one nominal rollout,
with the original initialization, safety checks, and 2,400-step horizon.

Gripper labels are -1=open and +1=close. There are 2,349 open and 1,881 closed
commands (55.53% / 44.47%). Each episode has just one closing and one opening
transition. Episode 1 runs open for 840 steps, closed for 944, then open for
325; episode 2 runs open for 859, closed for 937, then open for 325. Only 4 of
4,228 within-episode adjacent pairs switch command (0.095%). Thus the important
imbalance is temporal, despite relatively balanced command classes.
Closing begins at t=840 / 859, and bilateral physical grasp first registers
after executing t=850 / 869. Physical contact follows command onset by
0.5 seconds. The audit explicitly distinguishes pre-closing observations from
observations immediately before physical grasp, which already show closing
fingers. A positive prediction in the latter is a weaker check.

Reproduce with `/home/student/miniconda3/envs/libero/bin/python`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/train_libero_chunk_bc.py \
  --output results/new_chunk_bc --gripper-weight 3
MPLCONFIGDIR=/tmp/libero_matplotlib python scripts/inspect_libero_chunk_bc.py results/new_chunk_bc
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=2 \
  python scripts/evaluate_libero_bc.py --checkpoint results/new_chunk_bc/policy.pt \
  --output results/new_chunk_bc/evaluation --nominal-only --execute-steps 4
python -m unittest discover -s tests -p test_libero_chunk_bc.py
```

The training JSON records exact episode hashes and losses. NPZ predictions
contain all chunk targets, predictions, and masks. `gripper_audit.json` and
`gripper_around_grasp.png` compare prediction timing against demonstrations.
Checkpoint reload is verified before rollout. Evaluation CSV records raw
predictions, executed actions, within-chunk index, finger positions, object
and EEF positions, and grasp/lift/success diagnostics. These rollout logs are
not training demonstrations and are never fed back into training.
