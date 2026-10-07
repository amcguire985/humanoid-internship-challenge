# Ordinary white-mug LIBERO interface validation

The earlier upright experiment is **not a valid orientation baseline**. Its
five completed rollouts failed, issued no positive (close) gripper commands,
and moved the EEF mostly downward. Four nearly identical tiny tilt maxima came
from reset settling. Further constraint experiments are deferred.

## Audit of the previous setup

- Checkpoint: `lerobot/smolvla_base`, revision
  `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` in the supplied Colab run.
  It is a base model intended for task/robot fine-tuning, not an established
  LIBERO/Panda policy.
- The so-called Panda adapter was **observation/action plumbing**, not a separately
  pretrained or newly trained neural adapter. It changed feature metadata to
  18D state and 7D action while retaining the pretrained 32D padded projections.
  No training established semantic compatibility with the new input ordering.
- State was seven joint positions, seven joint velocities, two gripper positions
  and two gripper velocities. This differs from the LIBERO dataset's eight EEF /
  gripper state components.
- Input was one vertically flipped agentview camera, 128x128 RGB, CHW /255.
  Wrist imagery was absent. This differs from the official LIBERO two-camera,
  256x256, 180-degree rotation convention.
- Mean/std were computed from **episode_001 and episode_002 of the local scripted
  black-bowl dataset**, via `DemoDataset.stats()`, not from a checkpoint's training
  data. Action std floors were .001, state std floors .01. For example bowl Z
  action mean/std were -0.001434 / 0.034388, versus the selected LIBERO checkpoint's
  -0.090373 / 0.444729. These are materially different distributions.
- Previous postprocessing clamped pose commands to +/-0.5 and thresholded gripper
  at zero. The sign convention itself was correct: + closes, - opens. Its negative
  gripper outputs were not a demonstrated sign-inversion bug.

This combination cannot establish task competence or constraint obedience.
The incompatible state semantics, missing wrist view, different image geometry,
unrelated normalization and base-model domain mismatch are plausible causes of
failure. Downward drift alone cannot identify which cause dominates. The earlier
trace saved only final actions, so it cannot fully reconstruct raw model output.

## Explicit checkpoint choice

The new runner is `scripts/validate_libero_interface.py`. It explicitly selects
`lerobot/smolvla_libero`, pinned revision
`31d453f7edd78c839a8bbc39744a292686daf0de`.
This is a published LIBERO-trained derivative of SmolVLA. Its `train_config.json`
records `lerobot/libero`, 25,000 steps, and relative control. **We perform no local
training.** It is a candidate for interface validation, not yet a successful
ordinary mug baseline.

The dataset has 40 tasks from Spatial/Object/Goal/Long. Our unchanged task is
`libero_90`, canonical task 72,
`LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate`, language
`put the white mug on the plate`. This single-goal task is outside those standard
40 training tasks; success here is a generalization test and is not promised by
the checkpoint's LIBERO label. Related white-mug tasks in Long are multi-goal.
If input/action diagnostics pass but this task still fails, unseen-task
generalization is a leading remaining hypothesis. We must inspect the new
rollouts before attributing failure to it.

### Metadata correction, with evidence

The checkpoint declares 6D state and three image features, but its serialized
normalization safetensors actually contain **8D state mean/std and 7D action
mean/std**. Both its training dataset metadata and the official LIBERO processor
use eight state elements. The runner rejects unexpected dimensions, explicitly
corrects state feature metadata from 6 to 8 in the model and saved normalizer,
and logs the original config and correction. The learned 32D projection and
weights are unchanged, loaded strictly. This is a metadata correction, not a
trained adapter or a claim that dimensions alone imply competence.

Camera3 is stale inherited metadata: the dataset contains only two images, and
`empty_cameras=0` means the model consumes only present cameras. It is not
filled with a fabricated view. Actual model camera ordering and masks are logged.

## Observation and action contract

| Source | Final policy input |
|---|---|
| `agentview_image` | `observation.images.image` renamed to `camera1` |
| `robot0_eye_in_hand_image` | `observation.images.image2` renamed to `camera2` |
| `robot0_eef_pos` | State elements 0-2, world XYZ metres |
| `robot0_eef_quat` | xyzw converted to axis-angle, state elements 3-5 radians |
| `robot0_gripper_qpos` | State elements 6-7, signed finger positions |

Raw cameras are HWC uint8 **RGB**, 256x256x3. The worker uses the pinned LeRobot
`LiberoProcessorStep` directly: rotate both images 180 degrees, map EEF/gripper
state as above. Images become CHW float32 /255. Saved checkpoint processors
rename keys, add batch dimension, tokenize original language, move to GPU and
normalize with their own tensors. SmolVLA pads/resizes images to 512x512 and
maps pixels from [0,1] to [-1,1]; state is zero-padded to 32 after normalization.
No RGB-to-BGR conversion or ImageNet normalization is added.

The saved pre/postprocessor JSON and safetensors are loaded from the same pinned
checkpoint snapshot as the model. No local dataset is required. Their hashes
and mean/std vectors are recorded. Runtime checks verify normalized state and
denormalized actions against the checkpoint tensors.

Action ordering is `[dx, dy, dz, drot_x, drot_y, drot_z, gripper]`. Model output
is already an action in normalized training coordinates; `raw_policy_output`
and `normalized_action` deliberately contain the **same unpadded 7D vector**.
It is not normalized a second time. The saved postprocessor computes
`denormalized = normalized * std + mean`. Positive normalized gripper values
do not necessarily imply positive denormalized values because of the mean offset.

The final action is clipped only to the environment's actual action bounds
(normally +/-1). No extra +/-0.5 clamp, sign flip, thresholding, frame rotation
or pose scaling is applied. Classic fixed-impedance world-frame delta OSC_POSE
then converts +/-1 to +/-0.05 metres translation and +/-0.5 radians axis-angle.
The live controller is inspected and unexpected bounds/modes are rejected.
These are requested pose deltas, not achieved displacements or joint targets.
Robot dynamics/contact determine the resulting EEF movement.

For Panda, denormalized/final gripper **positive closes, negative opens, zero
holds**. `format_action` integrates opposing finger targets using the sign.
A startup check clones the actual gripper and checks both signs. The worker also
round-trips synthetic open/close targets through the actual saved postprocessor.
Those algebraic checks do not establish successful physical grasping; the
optional standalone simulator probe measures finger aperture under both signs.

## Colab commands

Use the existing validated simulator environment and the already installed
Python 3.12 CUDA SmolVLA environment. Pull the new code after it is committed and
pushed. Do not rerun the previous constrained evaluator or generate bowl stats.

Optional standalone gripper probe (no model download/inference):

```bash
MUJOCO_GL=osmesa /content/micromamba/envs/libero/bin/python scripts/validate_libero_interface.py --gripper-probe-only --output results/libero_interface_gripper_probe
```

First run **one ordinary task**:

```bash
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=2 /content/micromamba/envs/libero/bin/python scripts/validate_libero_interface.py --policy-python /content/micromamba/envs/smolvla/bin/python --num-rollouts 1 --horizon 300 --output results/libero_interface_validation_001
```

If diagnostics show sensible behavior, choose a fresh output folder and run a
small check with `--num-rollouts 3`. The default executes the checkpoint's saved
`n_action_steps=50` before replanning (not the earlier four-action setting).
`--execute-steps` can change it explicitly; the value is recorded. Progress logs
show inference and elapsed time at every replan. Horizon is 300 control steps,
not 300 seconds; inference wall time is independent of simulated time.

The script permits only the audited checkpoint name; a different model requires
an explicit interface audit. It offers no instruction override: every policy
request uses `task.language`. No upright metric or phone-video code is changed.
Outputs refuse to overwrite prior experiments.

## Outputs and report status

- `experiment.json`: task/prompt, pinned checkpoint, original/corrected metadata,
  actual normalization tensors/hashes, action/state/camera conventions.
- `rollout_NNN/controller.json`: observed mode, scaling and gripper sign mapping.
- `rollout_NNN/observation.json`: actual postprocessed model tensor keys,
  dimensions/ranges, raw and normalized state, internal camera ordering/masks,
  resized image shapes and padded state shape.
- `rollout_NNN/trajectory.jsonl`: raw/normalized/denormalized/final action,
  controller-scaled physical delta, EEF displacement, signed finger positions,
  aperture, mug XYZ/height increase, distance, grasp and LIBERO success. Written
  and flushed each step, including partial trajectories after interruption.
- `rollout_NNN/summary.json`: task success, close commands, actual finger closure,
  grasp, minimum EEF-to-mug-root distance, maximum height increase and lift.
- `validation_report.json`: checkpoint/mapping/normalization/gripper contract,
  completed rollout count, close/closure/grasp/lift/success counts.
- `error.json`: interrupted/failed rollout status; not counted as a completion.

Mug lift >=2 cm is a simple diagnostic, not an orientation threshold or a change
to task success. Lift by itself does not prove grasp/transport. Root-body distance
is not fingertip-to-surface distance. Finger closure alone does not prove grasp.
The normal `env.check_success()` remains the only task-success predicate.

**Current new-checkpoint test rollout count: 0.** Close/lift/success counts are
unmeasured. This Windows environment has no LIBERO or CUDA worker. Local tests
cover transport, mappings, normalization dimensions, sign preservation, physical
probe bookkeeping and end-to-end fake-simulator logging. They do not validate
GPU inference, real controller dynamics or task competence. The previous five
base-checkpoint failures remain separate evidence, not results for this model.

## Primary sources inspected

- [SmolVLA base model card](https://huggingface.co/lerobot/smolvla_base)
- [LIBERO-trained checkpoint](https://huggingface.co/lerobot/smolvla_libero)
- [Pinned checkpoint configuration](https://huggingface.co/lerobot/smolvla_libero/blob/31d453f7edd78c839a8bbc39744a292686daf0de/config.json)
- [Pinned training configuration](https://huggingface.co/lerobot/smolvla_libero/blob/31d453f7edd78c839a8bbc39744a292686daf0de/train_config.json)
- [Training dataset metadata](https://huggingface.co/datasets/lerobot/libero/blob/main/meta/info.json)
- [LeRobot LIBERO processor](https://github.com/huggingface/lerobot/blob/8c920c4270460851cedd2737657584586d3dc66f/src/lerobot/processor/env_processor.py)
- [LeRobot model image/action processing](https://github.com/huggingface/lerobot/blob/8c920c4270460851cedd2737657584586d3dc66f/src/lerobot/policies/smolvla/modeling_smolvla.py)
- [Classic Panda gripper](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/grippers/panda_gripper.py)
- [Classic OSC scaling configuration](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/controllers/config/osc_pose.json)
