# Upright white-mug pretrained baseline

**Deferred:** the first five runs failed ordinary task completion. This setup
is not a valid orientation baseline. Start with [ordinary interface validation](LIBERO_INTERFACE_VALIDATION.md) before running more constraint experiments.

This branch evaluates gravity-relative object orientation before any new
post-training. The instruction says the mug contains liquid; LIBERO does not
simulate liquid or spills. Tilt is an evaluation metric, not a new success test.

## Task and coordinate frames

Default suite: `libero_90`, task order 0, canonical zero-based index **72**.
Identifier: `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate`.
Original language: `put the white mug on the plate`.
BDDL goal stays `(On porcelain_mug_1 plate_1)`.
The evaluator resolves the identifier in the installed suite, logs the resulting
index and both instructions, and loads the unchanged BDDL. `--task-index` or
`--task-name` and `--suite` select another task; set `--object-name` accordingly.
It uses distinct benchmark initial states, shuffled by `--seed`, and resets both
simulator and policy RNG/cache for each rollout.

Ground-truth pose comes from `objects_dict['porcelain_mug_1'].root_body` and
MuJoCo `body_xmat` / `body_xquat`. Rotation matrices map local object axes into
the world; saved quaternions use **w,x,y,z**, unlike robosuite observation
quaternions (**x,y,z,w**). The porcelain asset's bottom/top sites are aligned
along local Z. Tilt is acos of the dot product of transformed local +Z with
world +Z, in degrees. Yaw does not affect it. Verify the body name, local +Z,
and approximately zero initial tilt in your installed Colab LIBERO version.
Check that gravity points along world -Z. No first-frame orientation subtraction
is applied. The shared utility also accepts explicitly calibrated local/world
up axes for later human-video comparisons; a tag's +Z is not automatically a
mug's upright axis.

## Policy baseline and limitations

The default is the pretrained `lerobot/smolvla_base` checkpoint, **not** the
repository's fine-tuned bowl checkpoint. No optimizer or training is invoked.
It reuses the existing SmolVLA worker JSON transport and existing LIBERO adapter:
128x128 agentview RGB flipped vertically, 18D Panda joint positions/velocities
and gripper positions/velocities, seven normalized OSC actions. As in the current
evaluator, pose actions are clamped to [-0.5,0.5] and gripper sign becomes +/-1.
The first four actions of each predicted chunk are executed by default.

This adapter and the normalization statistics are part of the baseline, not a
claim that the base policy was pretrained for Panda OSC or LIBERO. Statistics
are borrowed from the existing robot-native bowl dataset; the source file and
hash are logged. Results may show a domain/control mismatch before any language
constraint effect. Do not interpret a failed ordinary task as evidence that the
policy ignores the upright instruction. Changing checkpoint, normalization,
state/image adapter or action processing creates a different baseline.
The currently published `lerobot/smolvla_libero` config is not a drop-in adapter
for this repository's 18D state; do not select it without a separate adapter audit.

The complete constrained prompt is supplied in every inference request:

> Pick up the white mug and place it on the plate. The mug is full of liquid, so keep it upright throughout the entire motion and do not spill it.

Tokenizer capacity is extended without changing weights. Each inference verifies
the processor's actual unmasked token IDs against the complete prompt; truncation
or modification raises an error. Checkpoint revision is resolved and logged;
use `--revision SHA` to reproduce it.

## Run in Colab

Use the existing validated LIBERO simulator Python and separate CUDA policy
Python from [the SmolVLA environment handoff](LIBERO_SMOLVLA.md). Install the
policy dependencies from `config/smolvla/requirements-gpu.txt` into the policy
environment. Simulator needs NumPy, torch, matplotlib and its existing LIBERO /
robosuite / MuJoCo packages, as in the current evaluator. Run from repo root:

```bash
python scripts/evaluate_upright_mug.py --num-rollouts 20 --seed 0 --stats results/libero_smolvla/training/dataset_stats.json --policy-python /path/to/policy/env/bin/python --output results/upright_mug_baseline
```

If both stacks work in the same environment, omit `--policy-python`. If the
statistics file is absent, export the existing robot-native dataset statistics
without training (in the environment that can import the dataset code):

```bash
python -c "import sys,json; from pathlib import Path; sys.path.insert(0,'scripts'); from train_libero_smolvla import DemoDataset; s=DemoDataset(Path('results/libero_smolvla/dataset')).stats(); Path('config/upright_mug_stats.json').write_text(json.dumps({k:{n:v.tolist() for n,v in d.items()} for k,d in s.items()},indent=2))"
python scripts/evaluate_upright_mug.py --num-rollouts 20 --stats config/upright_mug_stats.json --policy-python /path/to/policy/env/bin/python --output results/upright_mug_baseline
```

The dataset must already have been exported using the existing
`prepare_libero_smolvla.py`; this command does not collect new demonstrations.
Useful options: `--horizon 600`, `--execute-steps 4`, `--settle-steps 10`,
`--checkpoint lerobot/smolvla_base`, `--revision SHA`, `--task-index 72`.
Outputs refuse to overwrite an existing directory; choose a fresh directory.
Requesting more rollouts than distinct provided initial states raises an error.

## Outputs and interpretation

- `experiment.json`: task, instructions, resolved model revision, stats/BDDL
  hashes, frame/adapter conventions and experiment settings.
- `rollouts.csv`: one row per completed rollout, seed, initial state index,
  both instructions, LIBERO success, maximum tilt and its timestep/sim time.
- `aggregate.json`: completed rollout count, success rate, mean/median/min/max
  of per-rollout maximum tilt; also printed to the console.
- `rollout_NNN/trajectory.jsonl`: initial, settling and every policy step's
  world quaternion/matrix, tilt, EEF position, executed action and LIBERO success.
- `rollout_NNN/summary.json` and `tilt.png`: per-rollout metrics and headless
  angle trace with its maximum marked.

Timestep 0 is the settled start; negative steps are settling. Initial and
settling orientations contribute to maximum tilt. Policy actions are recorded
with their resulting state at timestep 1 onwards. Sampling is at every control
step endpoint, not every internal MuJoCo physics substep; excursions inside a
control step are not measured. Evaluation stops at normal LIBERO success,
environment termination or the horizon. Success is the unmodified final
`env.check_success()`, independent of tilt. No orientation threshold or RMS metric
is introduced. Trace files are preserved if a rollout fails; partial runs are
not silently counted as completed experiments.

The existing SmolVLA evaluator saves snapshots rather than videos, so this
baseline does not add video capture. Numerical traces and plots are primary.
Generated experiment directories are ignored by Git. This implementation has
local geometry/bookkeeping tests; GPU inference and real rollouts must be
validated in Colab before any baseline outcome is claimed.

```bash
python -m unittest discover -s tests -p test_upright_orientation.py -v
```
