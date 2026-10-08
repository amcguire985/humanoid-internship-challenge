# Phone-derived transport constraints

Verified `results/bowl_human_reference/reference.csv`, reviewed transport 5.9?16.7 s:
217 samples, 10.8 s. Existing actual-time local quadratic Savitzky?Golay smoothing
(window 7) precedes two actual-time gradients. Exclude five samples at each end;
use P95 of acceleration magnitude, not maximum: **1.7378 m/s?**.
P95 of gravity-relative opening-axis tilt: **18.8393?**. This allowance may
produce no correction for the usual ~13? rim grasp. World vertical is the target
only for excess tilt. Window/P95 are analysis choices, not verified thresholds.
Margins are explicit: acceleration multiplier 1, tilt addition 0?; no extra margin.
Source files remain unchanged. The command recomputes from the Drive reference;
its printed values/hash establish whether that reference matches the local one.

`--transport-mode phone_constraints` is optional (default `trajectory`). Use C
plus the verified phone reference. Constraints are extracted once; there are no
position targets, reference sampling calls, or demonstration-clock dependence in
transport. Policy decides direction/destination. Existing phases, prompts, gains,
placement, gripper gate, slip failure and task initialization remain intact.
Previous A/B/C/D1 trajectory mode stays available.

Acceleration uses OSC affine scaling: translation delta in meters divided by
actual control dt is a commanded velocity proxy. Each change in this velocity is
limited in Euclidean norm to limit ? dt, then mapped back to OSC action units.
Seed it with the last grasp command at transition rather than imposing a stop.
This limits commanded motion, not measured end-effector/bowl acceleration or
physics-substep acceleration. Placement takeover keeps its existing controller
and is outside this limiter. Tilt swing ignores yaw and uses the existing
orientation blend/gain, ramp, correction cap/slew and action bounds.

Files: `scripts/bowl_phone_constraints.py` (extraction/limiter),
`scripts/evaluate_bowl_hybrid.py` (optional integration/provenance),
`tests/test_bowl_phone_constraints.py`, this document, and the pasteable cell below.
No simulation is launched locally. No checkpoint updates or gain tuning.

## One Colab rollout (cell 8 replacement)

Sync branch `upright-mug-orientation` first; mount Drive as before. This cell uses
the same pinned environment/bootstrap as D1. Do not run the D1 cell as well.

```bash
%%bash
set -euo pipefail
cd /content/humanoid-internship-challenge
mountpoint -q /content/drive || { echo "Run cell 8.0 (Drive mount) first."; exit 1; }
export MPLBACKEND=Agg MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=/content/libero-official-config
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=2
unset PYTHONPATH
BASELINE=/content/drive/MyDrive/humanoid_results/libero_official_bowl_20261007_204110
for FILE in packages.txt sanity_manifest.json libero_spatial_2/eval_info.json; do
  [ -f "$BASELINE/$FILE" ] || { echo "Missing original baseline: $BASELINE/$FILE"; exit 1; }
done
for FILE in scripts/bowl_phone_constraints.py config/smolvla/official_sanity_setup.bash config/smolvla/official_sanity_download.bash; do
  [ -f "$FILE" ] || { echo "Missing updated source: $FILE. Sync this checkout before running."; exit 1; }
done
# Cells 1-6 prepare a separate LIBERO-only environment. Keep it intact.
# Prepare the pinned evaluator only on a fresh runtime; neither step runs a rollout.
if [ ! -x /content/libero-official-env/bin/python ]; then
  bash config/smolvla/official_sanity_setup.bash
fi
if [ ! -f /content/smolvla-libero-official-checkpoint/model.safetensors ]; then
  bash config/smolvla/official_sanity_download.bash
fi
/content/libero-official-env/bin/python - <<'PY'
import torch
from lerobot.scripts import lerobot_eval
assert torch.cuda.is_available(), "Select a Colab GPU runtime."
from pathlib import Path
import sys
sys.path.insert(0, 'scripts')
from bowl_phone_constraints import extract
from bowl_human_reference import Reference
extract(Reference.load("/content/drive/MyDrive/humanoid_results/bowl_human_reference/reference.csv"))
assert "phone_constraints" in Path('scripts/evaluate_bowl_hybrid.py').read_text(), "Sync the updated constraints runner."
print('Pinned evaluator available; baseline/checkpoint checks run before simulation.')
PY
RUN_ROOT="$(mktemp -d /content/drive/MyDrive/humanoid_results/bowl_hybrid_phone_constraints_XXXXXXXX)"
printf '%s\n' "$RUN_ROOT/C" > /content/bowl_phone_constraints_last_output.txt
printf '\nResult folder: %s\n' "$(basename "$RUN_ROOT")"
printf 'Google Drive: MyDrive/humanoid_results/%s/C\n' "$(basename "$RUN_ROOT")"
printf 'Full output path: %s\n\n' "$RUN_ROOT/C"
/content/libero-official-env/bin/python scripts/evaluate_bowl_hybrid.py \
  --hybrid-condition C --transport-mode phone_constraints \
  --human-reference /content/drive/MyDrive/humanoid_results/bowl_human_reference/reference.csv \
  --guidance-config config/smolvla/bowl_hybrid.json \
  --placement-config config/smolvla/bowl_placement.json \
  --baseline-root "$BASELINE" \
  --policy.path=/content/smolvla-libero-official-checkpoint \
  --policy.device=cuda --policy.load_vlm_weights=false \
  --env.type=libero --env.task=libero_spatial --env.task_ids="[2]" \
  --env.control_mode=relative --env.init_states=true --env.hard_reset=true \
  --env.observation_height=256 --env.observation_width=256 \
  --env.max_parallel_tasks=1 --eval.batch_size=1 --eval.n_episodes=1 \
  --seed=0 --output_dir="$RUN_ROOT/C" 2>&1 | tee "$RUN_ROOT/C.log"
printf '\nSimulation finished. Result folder: %s\n' "$(basename "$RUN_ROOT")"
printf 'Google Drive: MyDrive/humanoid_results/%s/C\n' "$(basename "$RUN_ROOT")"
printf 'Full output path: %s\n' "$RUN_ROOT/C"
```

Output: `MyDrive/humanoid_results/bowl_hybrid_phone_constraints_<ID>/C/`;
printed before/after simulation. `phone_constraints.json` and `experiment.json`
record extraction, units and provenance. Existing summary, metrics, plots,
videos and phase logs remain. Per-step `trajectory.jsonl` adds proposed/limited
Cartesian velocities, proposed/limited commanded acceleration, limiter fraction,
tilt excess, rotation correction and full action change. Existing measured
bowl acceleration in summary/metrics remains separate. No human XYZ tracking
plot is expected in this mode. Inspect this one rollout before further work.

Local validation: 46 bowl tests passed (4 new constraint tests); existing trajectory-mode regression checks passed. No local simulation was run.
