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
