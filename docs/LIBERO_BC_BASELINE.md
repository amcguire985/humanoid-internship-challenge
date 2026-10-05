# Minimal behavioral-cloning experiment

This experiment uses only `episode_001.h5` (nominal) and `episode_002.h5`
(+2 cm X) from `results/libero_robot_dataset/episodes/`. They contain 2,109 and
2,121 transitions. A third demonstration completed while the earlier chat turn
was interrupted, before collection was stopped; it is preserved but **excluded
from training**. Episode 4 was interrupted and is not an accepted demonstration.
No additional scripted demonstrations are collected for this experiment.

`results/libero_bc_baseline/dataset_audit.json` records shapes, completion,
success, action ranges, and synchronization checks. Both episodes have T actions
and T+1 observations. Training uses `agentview_rgb[:-1]`, `proprio[:-1]`, and
`actions[:]`; terminal observations have no action target. Actions match the
original controller CSV exactly, as do the resulting EEF positions and gripper
positions at observation index t+1. Time spacing is 0.05 seconds. Both cameras
and numeric state are captured from the same simulator state, without stepping
between them. The recorder's ordering is covered by its unit test.

## Model and training

The 162,111-parameter `SmallBC` model has three convolution/ReLU layers with
16, 24, and 32 channels, stride 2, then a 64-dimensional image embedding. It
concatenates that embedding with 18-dimensional normalized proprioception and
uses a 128/64/7 MLP head. Recorded agent-view RGB is downsampled from 128x128 to
64x64 using area averaging and scaled to [0,1]. The 18 state values are arm
joint positions (7), arm velocities (7), finger positions (2), and finger
velocities (2). The separate EEF pose is not a model input.

The target is the exact seven-dimensional LIBERO action. Training minimizes
per-dimension standardized action MSE, with scales floored at 0.001. State scales
are floored at 0.01. Normalizers are estimated from the two training episodes
and saved as model buffers. The single run uses seed 0, Adam at 0.001, batch
size 128, and at most 40 epochs, stopping at normalized MSE below 0.025. It
stopped after 35 epochs. There is no validation-driven tuning, augmentation,
pretrained backbone, phase input, time input, language input, RL, or large VLA.

The loss and action reproduction metrics are measured on the training data,
not held-out data. A low training loss does not establish successful control.
`training_predictions.npz` contains targets and predictions for all 4,230
transitions; `training.png` shows loss and gripper action reproduction.
`checkpoint_validation.json` checks that saved-model inference with the online
image preprocessing matches saved training predictions.

## Closed-loop evaluation

The policy is evaluated once on each of the nominal and +2 cm X reset
configurations, both seen in training. The initial reset, 20 settling actions,
and XY-only perturbation are identical to demonstration preparation. After
that, only the learned policy acts. There is no scripted grasp, transport,
release, feedback correction, or replay fallback. Each action uses the current
agent-view image and current robot proprioception.

The six pose channels are clipped to the demonstrated controller limit of
[-0.5,0.5]; the gripper output is thresholded at zero to -1/open or +1/close.
The evaluation horizon is 2,400 steps (120 seconds at 20 Hz), slightly longer
than either demonstration. Unexpected robot/scene collisions, leaving the
validated EEF workspace, nonfinite state/action, falling below the support,
or actual environment termination stop the rollout. A raw LIBERO success
signal alone does not end it because that predicate can be true while held.

Object acquisition requires bilateral finger contact for five consecutive
steps. Lift requires at least 3 cm rise after acquisition. Task completion
requires acquisition and lift, release, object retention in the configured
placement region, LIBERO success, no grasp, and the EEF at least 8 cm above the
object for 20 consecutive steps. These physical checks are diagnostics only;
they never select an action. Failure phase is a post-hoc milestone label, not
a controller state fed to the model. Raw final and ever-true LIBERO success are
reported separately from task completion.

Evaluation CSVs and sparse screenshots are diagnostic artifacts, not training
demonstrations. No evaluation observations or actions are used to update the
policy. The result answers only whether this small policy reproduces these two
seen starts, not generalization to new placements or task language.

## Reproduce

Use `/home/student/miniconda3/envs/libero/bin/python` with the existing LIBERO,
PyTorch, h5py, NumPy, Pillow, and matplotlib installations. The output directory
may contain audit files but must not already contain a policy checkpoint.
Evaluation refuses to overwrite existing reset results.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/libero_matplotlib \
/home/student/miniconda3/envs/libero/bin/python scripts/train_libero_bc.py \
  --output results/new_bc_baseline

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=2 \
/home/student/miniconda3/envs/libero/bin/python scripts/evaluate_libero_bc.py \
  --checkpoint results/new_bc_baseline/policy.pt \
  --output results/new_bc_baseline/evaluation
```

The default training inputs explicitly name episodes 1 and 2; the trainer never
silently globs other accepted episodes into the experiment. Input file hashes
are saved in `training.json`. The evaluator uses the same rendering settings as
the demonstrations: original cameras/textures/geometry, with shadows,
reflections, and multisample antialiasing disabled for CPU throughput.

Artifacts live in `results/libero_bc_baseline/`: `dataset_audit.json`,
`policy.pt`, `training.json`, `training_predictions.npz`, `training.png`,
`checkpoint_validation.json`, `samples_episode_001/`, `samples_episode_002/`,
and `evaluation/{nominal,perturbed_plus_2cm_x}/`. The evaluation summary and
per-reset metrics document the observed outcome; the result README summarizes
it for review.
