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

## Requested checkpoint

The runner now selects `HuggingFaceVLA/smolvla_libero`, pinned revision
`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`. Its native configuration has
8D state, two cameras (`image`, `image2`), 7D action, chunk size 50,
`n_action_steps=1` and flow-matching `num_steps=10`. No metadata correction,
Panda adapter, local training or bowl statistics are used. Strictly load the
model and its saved processors from the same snapshot. The checkpoint does not
include a training config or task-coverage manifest; do not infer competence on
our task from the LIBERO name or from another checkpoint's training metadata.

The task stays `libero_90`, canonical index 72,
`LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate`, with original language
`put the white mug on the plate`. Resolve its index by exact name at runtime.

## Observation and action contract

| Source | Final policy input |
|---|---|
| `agentview_image` | `observation.images.image` (first camera) |
| `robot0_eye_in_hand_image` | `observation.images.image2` (second camera) |
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

Run from `/content/humanoid-internship-challenge` after transferring these
changes. The edits are local until committed/pushed; `git pull` alone cannot
retrieve uncommitted changes. On the local checkout, if Git delivery is desired:

```bash
git add scripts/validate_libero_interface.py tests/test_libero_interface.py docs/LIBERO_INTERFACE_VALIDATION.md
git commit -m "Validate requested LIBERO SmolVLA checkpoint with videos"
git push origin HEAD
```

In Colab, use the existing validated micromamba simulator and CUDA policy
 environments. No new LeRobot version is needed. The pinned source is
`8c920c4270460851cedd2737657584586d3dc66f`; worker metadata records actual
LeRobot/Transformers/PyTorch versions. Ensure the policy dependencies and
simulator video encoder are installed:

```bash
%cd /content/humanoid-internship-challenge
!git pull --ff-only
!/content/micromamba/envs/smolvla/bin/python -m pip install -r config/smolvla/requirements-gpu.txt
!/content/micromamba/envs/libero/bin/python -m pip install 'opencv-python-headless==4.10.0.84'
```

Mount Google Drive once, to retain videos outside Git:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Five ordinary-prompt rollouts (two videos per rollout):

```bash
!HUMANOID_VIDEO_ROOT=/content/drive/MyDrive/humanoid_videos MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=2 /content/micromamba/envs/libero/bin/python scripts/validate_libero_interface.py --policy-python /content/micromamba/envs/smolvla/bin/python --num-rollouts 5 --seed 0 --horizon 300 --output results/smolvla_libero_ordinary_5
```

More rollouts, with a fresh directory and reproducible seed:

```bash
!HUMANOID_VIDEO_ROOT=/content/drive/MyDrive/humanoid_videos MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=2 /content/micromamba/envs/libero/bin/python scripts/validate_libero_interface.py --policy-python /content/micromamba/envs/smolvla/bin/python --num-rollouts 10 --seed 100 --horizon 300 --output results/smolvla_libero_ordinary_10
```

Saved execution defaults to one action before replanning, as this checkpoint
specifies, rather than the other checkpoint's 50. `num_steps=10` is the inference
denoising count, not rollout horizon. `--execute-steps` is an explicit experimental
override and is recorded. Five 300-step runs can require substantial GPU time
because each step replans. Seeded initial-state permutations contain no duplicate
states within a run, though separate runs can overlap.

The public checkpoint and tokenizer were accessible without authentication.
The first run downloads model weights plus SmolVLM configuration/tokenizer;
CUDA and internet access are required. If Hub access/rate limits require login,
run `/content/micromamba/envs/smolvla/bin/hf auth login` in a terminal; never
commit a token. No dataset download or local statistics extraction is needed.

Optional standalone physical gripper probe:

```bash
!MUJOCO_GL=osmesa /content/micromamba/envs/libero/bin/python scripts/validate_libero_interface.py --gripper-probe-only --no-video --output results/libero_interface_gripper_probe
```

Stop after ordinary runs. Review videos and diagnostics if all runs fail before
grasp; do not launch the constrained comparison automatically.

## Outputs and report status

- `runtime_sanity.json`: concise input/action/controller/chunk settings.
- `results.md`: requested per-rollout Markdown table; closure means aperture reduced by >1 mm, not merely a close command.
- Videos: `/content/drive/MyDrive/humanoid_videos/processed/smolvla_libero_ordinary_5/rollout_NNN_agentview_image.mp4` and `rollout_NNN_robot0_eye_in_hand_image.mp4`; paths also appear in rollout summaries. Both views are rotated 180 degrees like policy inputs, and encoded at the simulator control frequency.
- `experiment.json`: task/prompt, pinned checkpoint, native metadata,
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
to task success. Lift by itself does not prove grasp/transport. Maximum gravity-relative tilt is retained as a diagnostic only; no orientation conclusions are drawn. Root-body distance
is not fingertip-to-surface distance. Finger closure alone does not prove grasp.
The normal `env.check_success()` remains the only task-success predicate.

**Current new-checkpoint test rollout count: 0.** Close/lift/success counts are
unmeasured. This Windows environment has no LIBERO or CUDA worker. Local tests
cover transport, mappings, normalization dimensions, sign preservation, physical
probe bookkeeping and end-to-end fake-simulator logging. They do not validate
GPU inference, real controller dynamics or task competence. The previous five
base-checkpoint failures remain separate evidence, not results for this model.

## Primary sources inspected

- [Requested model card](https://huggingface.co/HuggingFaceVLA/smolvla_libero)
- [Pinned configuration](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/6721902bc4d61e50a3bfdb11dfb4cb626f05d102/config.json)
- [Pinned preprocessor](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/6721902bc4d61e50a3bfdb11dfb4cb626f05d102/policy_preprocessor.json)
- [Pinned postprocessor](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/6721902bc4d61e50a3bfdb11dfb4cb626f05d102/policy_postprocessor.json)
- [Official LIBERO preprocessing](https://github.com/huggingface/lerobot/blob/8c920c4270460851cedd2737657584586d3dc66f/src/lerobot/processor/env_processor.py)
- [Model image/action processing](https://github.com/huggingface/lerobot/blob/8c920c4270460851cedd2737657584586d3dc66f/src/lerobot/policies/smolvla/modeling_smolvla.py)

## Files controlling the interface

| Setting | Previous baseline | Ordinary validation |
|---|---|---|
| Checkpoint/loading | `scripts/evaluate_upright_mug.py` worker | `scripts/validate_libero_interface.py` constants and worker |
| Library version | `config/smolvla/requirements-gpu.txt` | Same pin; actual versions logged by worker |
| Task/initial states | `evaluate_upright_mug.py` task resolver | Same resolver, original task instruction |
| Observation/camera/state | `evaluate_upright_mug.py`, `prepare_libero_smolvla.py` | `encode_observation`, official `LiberoProcessorStep`, saved processors |
| Normalization | `train_libero_smolvla.py` local bowl stats | Checkpoint processor safetensors, checked by worker |
| Chunk execution | `evaluate_upright_mug.py` | Native config and `evaluate` loop |
| Controller scale/mode | `replay_libero_transport.py::inspect_controller` | Reused by `controller_audit` |
| Gripper mapping | Previous evaluator threshold | Native denormalized sign, checked against live Panda gripper |
| Tilt | `scripts/upright_orientation.py` | Same functions, diagnostic only |

The domain-trained weights, correct EEF state, wrist camera, matching image
geometry and checkpoint statistics remove identified integration mismatches.
They are expected to improve the baseline, but do not guarantee this task's success.
