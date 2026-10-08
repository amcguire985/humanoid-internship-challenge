# Single synthetic-reference diagnostic

D1 generates a straight minimum-jerk object path at confirmed grasp, saves
`synthetic_reference.csv` with the existing phone CSV/quaternion schema and
metadata sidecar, reloads it using `Reference.load`, and passes it to `Guidance`.
The default is 10.8 seconds at 20 Hz. The start position and constant reference
rotation are the actual confirmed-grasp bowl pose. The target XY comes from the
existing collision-based plate goal. Target Z is the greater of grasp height and
that goal plus the unchanged placement approach height (2.5 cm by default).
This assumes a clear straight route at that height; no obstacle planner or
additional lift is introduced. Simulator poses are oracle endpoint alignment,
not measurements available from phone video.

Transport uses the existing `orientation_enabled=False` option: policy rotational
actions remain active, so constant reference orientation is not a guarantee of
constant actual bowl orientation. The shared placement controller still applies
its existing uprightness correction. Disabled transport orientation also uses
an identity swing when computing the gripper offset, avoiding an unintended
position correction from gravity alignment. All gains, limits, grasp detection,
placement/release criteria, metrics, checkpoint and episode horizon are retained.
Real phone files and settings are unchanged.

The 280-step horizon remains 14 seconds, including grasp and placement. The
shared placement controller may take over before reference completion when its
existing approach criterion holds. A timeout must be inspected alongside
transport tracking and placement phase; duration is never compressed silently.

## Colab: exactly one rollout

Make these updated source files available in `/content/humanoid-internship-challenge`
first. Use the existing known-good environment and checkpoint, and mount Drive.
Run this single cell; it does not call the A/B/C multi-condition wrapper:

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
printf 'Diagnostic outputs: %s\n' "$RUN_ROOT/D1"
```

Outputs retain the existing summary, metrics, phase transitions, tracking plots
and videos. `ground_truth/episode_000/` additionally contains the generated
CSV/metadata and aligned reference. Inspect `summary.json`, `trajectory.jsonl`,
`placement.png` and videos before any subsequent experiment. No D2/E or noise
experiments are implemented. No local simulation success is claimed.

## Files and local checks

- `scripts/bowl_synthetic_reference.py`: generator returning the existing Reference.
- `scripts/evaluate_bowl_hybrid.py`: D1 option and grasp-time generation/load.
- `scripts/bowl_transport_guidance.py`: identity swing for existing disabled orientation mode.
- `tests/test_bowl_synthetic_reference.py`: endpoints, continuity, timing, CSV reload,
  alignment and unchanged transport rotational actions.

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_bowl*.py'`.

Local validation: all 42 bowl tests passed. Local PyTorch is absent, so no LIBERO rollout was launched or validated.

## Compatibility with the supplied notebook

Use `docs/internship_challenge_synthetic.ipynb`, a copy of the supplied notebook.
The cell labelled 8 is the rollout cell (physical notebook cell 16, after the
Drive mount). Earlier initialization cells are unchanged. Cell 8 prepares the
separate pinned evaluator/checkpoint only if absent; it does not use or modify
the Python 3.8 micromamba LIBERO environment. The existing baseline on Drive
is required, and the runner retains its strict package/checkpoint checks.
A fresh setup that differs from baseline versions will stop before simulation;
do not bypass that check or substitute the Python 3.8 interpreter.

Cell 9 now reads the exact D1 output recorded by cell 8, rather than recovering
an old C rollout. No rerun or additional simulation is triggered by cell 9.
The standalone pasteable cell is `config/smolvla/bowl_synthetic_colab_cell.bash`.
Make the changed source and existing setup/download scripts available in Colab
before execution; local uncommitted files are not fetched by `git pull`.
Notebook structure and earlier-cell preservation were checked locally; Colab
execution and CUDA simulation remain untested.
