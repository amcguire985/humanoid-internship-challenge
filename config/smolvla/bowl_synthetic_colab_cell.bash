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
for FILE in scripts/bowl_synthetic_reference.py config/smolvla/official_sanity_setup.bash config/smolvla/official_sanity_download.bash; do
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
from bowl_synthetic_reference import generate
assert "'D1'" in Path('scripts/evaluate_bowl_hybrid.py').read_text(), "Sync the updated D1 runner."
print('Pinned evaluator available; baseline/checkpoint checks run before simulation.')
PY
RUN_ROOT="$(mktemp -d /content/drive/MyDrive/humanoid_results/bowl_hybrid_synthetic_XXXXXXXX)"
printf '%s\n' "$RUN_ROOT/D1" > /content/bowl_synthetic_last_output.txt
printf '\nResult folder: %s\n' "$(basename "$RUN_ROOT")"
printf 'Google Drive: MyDrive/humanoid_results/%s/D1\n' "$(basename "$RUN_ROOT")"
printf 'Full output path: %s\n\n' "$RUN_ROOT/D1"
/content/libero-official-env/bin/python scripts/evaluate_bowl_hybrid.py \
  --hybrid-condition D1 --synthetic-duration 10.8 \
  --guidance-config config/smolvla/bowl_hybrid.json \
  --placement-config config/smolvla/bowl_placement.json \
  --baseline-root "$BASELINE" \
  --policy.path=/content/smolvla-libero-official-checkpoint \
  --policy.device=cuda --policy.load_vlm_weights=false \
  --env.type=libero --env.task=libero_spatial --env.task_ids="[2]" \
  --env.control_mode=relative --env.init_states=true --env.hard_reset=true \
  --env.observation_height=256 --env.observation_width=256 \
  --env.max_parallel_tasks=1 --eval.batch_size=1 --eval.n_episodes=1 \
  --seed=0 --output_dir="$RUN_ROOT/D1" 2>&1 | tee "$RUN_ROOT/D1.log"
printf '\nSimulation finished. Result folder: %s\n' "$(basename "$RUN_ROOT")"
printf 'Google Drive: MyDrive/humanoid_results/%s/D1\n' "$(basename "$RUN_ROOT")"
printf 'Full output path: %s\n' "$RUN_ROOT/D1"
