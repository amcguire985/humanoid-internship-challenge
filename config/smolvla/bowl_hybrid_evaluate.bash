set -euo pipefail
cd /content/humanoid-internship-challenge
git pull --ff-only origin upright-mug-orientation
mountpoint -q /content/drive || { echo "Run the Google Drive mount cell first."; exit 1; }
export MPLBACKEND=Agg MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=/content/libero-official-config
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=2
unset PYTHONPATH
EVAL_ENV=/content/libero-official-env
CHECKPOINT=/content/smolvla-libero-official-checkpoint
BASELINE=/content/drive/MyDrive/humanoid_results/libero_official_bowl_20261007_204110
REFERENCE=/content/drive/MyDrive/humanoid_results/bowl_human_reference/reference.csv
# First integration check: exactly ONE C rollout. A/B are a later separate step.
CONDITIONS="${BOWL_CONDITIONS:-C}"
RUN_ROOT="${BOWL_RUN_ROOT:-/content/drive/MyDrive/humanoid_results/bowl_hybrid_$(date -u +%Y%m%d_%H%M%S)}"
export RUN_ROOT
[ -x "$EVAL_ENV/bin/python" ] || { echo "Restore the known-good official environment first."; exit 1; }
for CONDITION in $CONDITIONS; do
  case "$CONDITION" in A|B|C) ;; *) echo "Use conditions A, B, or C."; exit 1;; esac
  [ ! -e "$RUN_ROOT/$CONDITION" ] || { echo "Condition already exists: $RUN_ROOT/$CONDITION"; exit 1; }
  if [ "$CONDITION" = C ]; then
    "$EVAL_ENV/bin/python" - "$REFERENCE" <<'PY'
import sys
sys.path.insert(0,'scripts')
from bowl_human_reference import Reference
ref=Reference.load(sys.argv[1])
print('Verified human transport duration:',ref.t[-1],'seconds; timing scale 1')
PY
  fi
done
mkdir -p "$RUN_ROOT"
"$EVAL_ENV/bin/python" -m pip freeze > "$RUN_ROOT/packages.txt"
for CONDITION in $CONDITIONS; do
  "$EVAL_ENV/bin/python" scripts/evaluate_bowl_hybrid.py \
    --hybrid-condition "$CONDITION" --human-reference "$REFERENCE" \
    --guidance-config config/smolvla/bowl_hybrid.json \
    --baseline-root "$BASELINE" \
    --policy.path="$CHECKPOINT" --policy.device=cuda --policy.load_vlm_weights=false \
    --env.type=libero --env.task=libero_spatial --env.task_ids="[2]" \
    --env.control_mode=relative --env.init_states=true --env.hard_reset=true \
    --env.observation_height=256 --env.observation_width=256 \
    --env.max_parallel_tasks=1 --eval.batch_size=1 --eval.n_episodes=1 \
    --seed=0 --output_dir="$RUN_ROOT/$CONDITION" \
    2>&1 | tee "$RUN_ROOT/$CONDITION.log"
done
"$EVAL_ENV/bin/python" - <<'PY'
import os,sys
sys.path.insert(0,'scripts')
from evaluate_bowl_hybrid import compare
compare(os.environ['RUN_ROOT'])
PY
printf 'Results, videos, and comparison: %s\n' "$RUN_ROOT"
