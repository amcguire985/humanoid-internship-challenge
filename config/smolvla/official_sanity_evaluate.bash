set -euo pipefail
mountpoint -q /content/drive || { echo "Run the Drive mount cell first."; exit 1; }
export MUJOCO_GL=egl LIBERO_CONFIG_PATH=/content/libero-official-config
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=2
unset PYTHONPATH
EVAL_ENV=/content/libero-official-env
RUN_ROOT="/content/drive/MyDrive/humanoid_results/libero_official_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"
export RUN_ROOT
"$EVAL_ENV/bin/python" -m pip freeze > "$RUN_ROOT/packages.txt"
"$EVAL_ENV/bin/python" - <<'PY'
import json, os, hashlib
from pathlib import Path
import importlib.metadata as md
import pyarrow.parquet as pq
from libero.libero import benchmark, get_libero_path
records=[]
for suite_name,index,expected in [
    ('libero_spatial',2,'pick up the black bowl from table center and place it on the plate'),
    ('libero_object',0,'pick up the alphabet soup and place it in the basket')]:
    suite=benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
    task=suite.get_task(index)
    assert task.language==expected,(suite_name,index,task)
    bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    records.append(dict(suite=suite_name,task_index=index,name=task.name,instruction=task.language,
        bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),episodes=3,seed=0,
        initial_state_selection='Official sequential reset indexing; starts at 0, stride 1'))
meta=pq.read_table('/content/libero-official-task-metadata/meta/tasks.parquet').to_pydict()
instructions=meta['__index_level_0__']
assert all(r['instruction'] in instructions for r in records)
assert 'put the white mug on the plate' not in instructions
manifest=dict(checkpoint='HuggingFaceVLA/smolvla_libero',
    checkpoint_revision='6721902bc4d61e50a3bfdb11dfb4cb626f05d102',
    lerobot_revision='8c920c4270460851cedd2737657584586d3dc66f',
    versions={p:md.version(p) for p in ['lerobot','hf-libero','robosuite','mujoco','torch','torchvision','transformers','numpy']},
    dataset_revision='86958911c0f959db2bbbdb107eb3e17c5f9c798e',
    tasks=records,control_mode='relative',image_resolution=[256,256],n_action_steps=1)
Path(os.environ['RUN_ROOT'],'sanity_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
PY

# Exactly two selected tasks, three episodes each: six rollouts total.
for SPEC in libero_spatial:2 libero_object:0; do
  SUITE="${SPEC%:*}"
  TASK_ID="${SPEC#*:}"
  "$EVAL_ENV/bin/lerobot-eval" \
    --policy.path=/content/smolvla-libero-official-checkpoint \
    --policy.device=cuda --policy.load_vlm_weights=false \
    --env.type=libero --env.task="$SUITE" --env.task_ids="[$TASK_ID]" \
    --env.control_mode=relative --env.init_states=true --env.hard_reset=true \
    --env.observation_height=256 --env.observation_width=256 \
    --env.max_parallel_tasks=1 --eval.batch_size=1 --eval.n_episodes=3 \
    --seed=0 --output_dir="$RUN_ROOT/${SUITE}_${TASK_ID}" \
    2>&1 | tee "$RUN_ROOT/${SUITE}_${TASK_ID}.log"
done
echo "Results and videos: $RUN_ROOT"
