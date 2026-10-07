# Matched original-versus-liquid bowl experiment

This supersedes the earlier constraint-only experiment. Those three failed runs
received no placement instruction and must remain separate evidence.

## Conditions and run count

Exactly **two rollouts**, both on Spatial index 2, seed 0 and initial state index 0:

- A: `pick up the black bowl from table center and place it on the plate`
- B: `pick up the black bowl from table center and place it on the plate. the bowl is full of liquid, do not spill it`

Both use the same pinned checkpoint, installed official environment, original
scene/success predicate, reset/settle sequence, async evaluator, cameras, saved
normalization, controller and action configuration. No training is performed.
The original prompt run needs repeating once because earlier successful baseline
rollouts have no ground-truth object traces for tilt/acceleration. The user has
requested this matched original run. Do not run the previous three-episode cell.

## Implementation and metrics

`scripts/evaluate_bowl_liquid.py` delegates to the pinned official evaluator.
Only policy language is replaced at the checkpoint preprocessor boundary.
Print environment and actual policy language, and check exact unmasked token IDs
for the full task plus constraint. Environment task_description is unchanged.
Both conditions get identical observer hooks inside the official async workers;
original reset/step calls and return values are preserved, with no extra steps.

The target is `akita_black_bowl_1`. Runtime confirms its top-minus-bottom offset
points along local +Z, the opening axis. Ground-truth bowl XYZ, world-from-object
matrix, MuJoCo wxyz quaternion, actual simulation time, gripper aperture/action,
grasp when available, and original LIBERO success are flushed after each official
control step. Timestep 0 is after the unchanged 10 settling steps.

Transport begins at detected grasp plus >=2 cm height increase from settled start.
If unavailable, explicitly label the fallback requiring lift, positive close
command and aperture reduction >1 mm. Release is an open command plus aperture
increase >1 mm relative to transport start without a detected grasp; exclude that
release sample. Otherwise end at success/episode end. Neither approach nor settling
enters the metrics. No transport means unavailable metrics, not zeros.

Gravity-relative tilt uses local +Z transformed into world and acos against world
+Z; yaw does not contribute. Acceleration is the unsmoothed, nonuniform-time,
three-point central second difference of ground-truth bowl positions, with stencils
wholly inside transport. Raw/processed values are identical; endpoints unavailable.
This is a control-rate (20 Hz) estimate, not a physics-substep or continuous peak.

Offline comparison verifies matching actual seed/state index, initial-state hash,
BDDL hash, bowl opening axis, checkpoint hashes, and the two actual prompts before
computing liquid-minus-original tilt and acceleration. Negative deltas mean lower
sampled maxima, but do not establish caution if transport failed or differed.
Plots overlay transport segments at time since settled reset. One pair permits
only a descriptive result; no significance testing. No actual liquid is simulated.

## Colab delivery and exact replacement cell

No environment rebuild is required. Commit/push these changes before git pull:

```bash
git add scripts/evaluate_bowl_liquid.py tests/test_bowl_liquid.py docs/BOWL_LIQUID_EXPERIMENT.md config/smolvla/bowl_liquid_evaluate.bash
git commit -m "Compare matched bowl runs with appended liquid constraint"
git push origin HEAD:upright-mug-orientation
```

With Drive mounted in the existing working runtime, use:

```bash
%%bash
set -euo pipefail
cd /content/humanoid-internship-challenge
git pull --ff-only origin upright-mug-orientation
mountpoint -q /content/drive || { echo "Run the Drive mount cell first."; exit 1; }
export MPLBACKEND=Agg MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=/content/libero-official-config
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=2
unset PYTHONPATH
EVAL_ENV=/content/libero-official-env
CHECKPOINT=/content/smolvla-libero-official-checkpoint
BASELINE=/content/drive/MyDrive/humanoid_results/libero_official_bowl_20261007_204110
RUN_ROOT="/content/drive/MyDrive/humanoid_results/bowl_pair_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"
export RUN_ROOT
"$EVAL_ENV/bin/python" -m pip freeze > "$RUN_ROOT/packages.txt"
# Exactly one original and one appended-constraint episode, both seed/state 0.
for CONDITION in original liquid; do
  "$EVAL_ENV/bin/python" scripts/evaluate_bowl_liquid.py \
    --baseline-root "$BASELINE" --condition "$CONDITION" --episodes 1 \
    --policy.path="$CHECKPOINT" \
    --policy.device=cuda --policy.load_vlm_weights=false \
    --env.type=libero --env.task=libero_spatial --env.task_ids="[2]" \
    --env.control_mode=relative --env.init_states=true --env.hard_reset=true \
    --env.observation_height=256 --env.observation_width=256 \
    --env.max_parallel_tasks=1 --eval.batch_size=1 --eval.n_episodes=1 \
    --seed=0 --output_dir="$RUN_ROOT/$CONDITION" \
    2>&1 | tee "$RUN_ROOT/$CONDITION.log"
done
# Offline comparison only; this does not run another simulation.
"$EVAL_ENV/bin/python" - <<'PY'
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts').resolve()))
from evaluate_bowl_liquid import compare_pair
root=Path(os.environ['RUN_ROOT'])
compare_pair(root/'original', root/'liquid', root)
PY
echo "Results and videos: $RUN_ROOT"
```

## Outputs

`My Drive/humanoid_results/bowl_pair_<timestamp>/` contains:

- `paired_comparison.md`, `paired_comparison.json`: success, maxima and paired deltas.
- `paired_tilt.png`, `paired_acceleration.png`: matched transport overlays.
- `original/` and `liquid/`: experiment manifest, verified actual prompt, official
  `eval_info.json`, and `ground_truth/episode_000/` containing initialization,
  trajectory JSONL, summary JSON, metric CSV and individual plots.
- Videos: `<condition>/videos/libero_spatial_2/eval_episode_0.mp4`.
- `<condition>.log` and `packages.txt`: logs and package versions.

Runtime versions/checkpoint identity must match the successful historical baseline
before starting. All other validated configuration is held fixed. If either run
fails before transport, its metrics and corresponding deltas remain unavailable.
Do not rerun automatically. No physical inference or improvement claim can be made
until the new result files and videos are inspected.
