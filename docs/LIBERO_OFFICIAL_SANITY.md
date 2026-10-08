# Official SmolVLA / LIBERO sanity check

Status: preparation and source/metadata audit complete; **zero official rollouts
executed here**. This Windows checkout lacks Linux LIBERO and a CUDA policy
runtime. These Colab cells are prepared commands, not a validated installation
or measured competence claim. No custom mug runner or phone-video code changed.

## Checkpoint selection

Keep `HuggingFaceVLA/smolvla_libero` at
`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`. It was last modified 2025-09-17;
`lerobot/smolvla_libero` at `31d453f7edd78c839a8bbc39744a292686daf0de`
was last modified 2026-03-24. Thus the HuggingFaceVLA upload is **not newer**.
Neither model card supplies a convincing LIBERO-specific evaluation recipe or
comparison establishing that one is currently recommended. Current official
LIBERO docs recommend the evaluator and standard suites, not a named SmolVLA
checkpoint. Selection here is based on native compatibility and holding the
checkpoint fixed while changing the evaluation stack.

| Property | HuggingFaceVLA checkpoint (selected) | lerobot checkpoint |
|---|---|---|
| State metadata | 8D | 6D, despite 8D saved statistics |
| Image keys | image, image2 | camera1, camera2, camera3 |
| Actual statistics | 8D state, 7D action mean/std | 8D state, 7D action mean/std |
| Cameras/resolution | Two, RGB 256x256 | Three declared; associated dataset has two |
| Action/chunk/denoising | 7D / 50 / 10 | 7D / 50 / 10 |
| Executed steps per replan | 1 | 50 |
| Dataset provenance | Card says unknown; no train_config.json | train_config.json explicitly says lerobot/libero, 25,000 steps |

Both normalize state/action by mean/std using their own serialized processors.
Use the selected checkpoint unchanged, including its saved n_action_steps=1.
No replacement normalization, camera rename overrides, adapter or orientation
metric is introduced. A native input schema does not establish task competence.

## Official stack and versions

Use official `lerobot-eval` in a single fresh Python 3.12 environment. Pin LeRobot
source `8c920c4270460851cedd2737657584586d3dc66f` (package metadata 0.6.2),
`hf-libero==0.1.4`, and `mujoco==3.3.2`. hf-libero requires `robosuite==1.4.0`
and `robomimic==0.2.0`. Use the packaged Hugging Face LIBERO fork; do not add the
old /content/LIBERO checkout to PYTHONPATH. A separate LIBERO Git clone is not
required. Do not alter the old Python 3.8 simulator environment.

The LeRobot pin accepts Transformers >=5.4,<5.6 and Torch >=2.7,<2.12.
The exact resolved dependency versions cannot be read from the remote Colab
runtime here: `packages.txt` and `sanity_manifest.json` record them before any
rollout. The source pin matches the prior handoff rather than introducing an
unnecessary new LeRobot revision. Install commands let its declared compatible
ranges resolve; save the resulting freeze before comparing runs.

Use Linux, a Colab GPU and `MUJOCO_GL=egl`; keep a separate
`LIBERO_CONFIG_PATH=/content/libero-official-config` to avoid stale asset paths.
The checkpoint, task metadata and tokenizer are public and no login was needed
for inspected files. If authentication is needed due to Hub limits, use
`/content/libero-official-env/bin/hf auth login` interactively. Do not put tokens
in notebook code. LIBERO assets may download on first access. No demonstrations
or policy training are needed.

## Selected tasks (canonical order 0)

| Suite | Index | Task name / original instruction | Planned | Executed | Successes |
|---|---:|---|---:|---:|---:|
| libero_spatial | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate / ?pick up the black bowl from table center and place it on the plate? | 3 | 0 | unmeasured |
| libero_object | 0 | pick_up_the_alphabet_soup_and_place_it_in_the_basket / ?pick up the alphabet soup and place it in the basket? | 3 | 0 | unmeasured |

Both are explicitly present in the inspected task Parquet metadata for both
40-task datasets. This establishes dataset-distribution membership, not the
selected checkpoint's otherwise undocumented training coverage. Two tasks are
the requested minimum; six episodes are enough for an initial sanity check,
not a reliable success-rate estimate. No additional tasks run automatically.

## White-mug status

`LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate`, canonical libero_90 index 72,
is **standard LIBERO but absent from both inspected 40-task datasets**.
Its exact instruction is ?put the white mug on the plate?. The HuggingFaceVLA
checkpoint's exact training manifest is unavailable, so whether it saw this
specific task remains **unclear**, rather than conclusively untrained.
The lerobot checkpoint's published training configuration points to the 40-task
dataset, which excludes this single-goal task.

The closest dataset member is libero_10 index 6:
`LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`.
It shares the scene identifier and mug-to-plate subgoal but has a different BDDL
goal and a longer instruction. It is not the same task or equivalent full
instruction. Scene identifiers alone do not prove identical XML, object poses,
initial-state files or asset versions across installations. The supplied rollout
files do not include asset hashes; exact old/new scene equivalence is unverified.

Other candidates, reported only, are libero_10 index 4 (two mugs onto two plates)
and index 9 (yellow/white mug into microwave plus closing it). For a simpler
container task, Spatial 2 carries an open bowl; it supports gravity-relative
uprightness but differs from a mug. Object 7 (milk into basket) supports transport
and tilt on a container, but is not an open cup. Do not switch the orientation
experiment automatically. The same-scene Long 6 task is the closest mug candidate,
subject to proving nominal competence on that complete task first.

## Official versus custom runner: report differences before changing it

| Component | Official sanity path | Existing custom mug runner |
|---|---|---|
| Task construction | Packaged hf-libero, unmodified standard tasks | Older simulator environment, standard libero_90 BDDL |
| Cameras | Both views, raw RGB; official env processor rotates 180 degrees | Same intended mapping and official LiberoProcessorStep |
| Resolution | Explicit 256x256 matching checkpoint; config default is 360 | 256x256 |
| Rendering | Default simulator rendering, EGL | OSMesa; disables shadows, reflections and multisampling |
| State/normalization | Official 8D EEF/axis-angle/finger processor and saved checkpoint processors | Same intended mapping and saved processors, custom JSON transport |
| Initialization | Fixed benchmark states, sequential state IDs (starts 0, stride 1); hard reset | Seeded permutation (observed IDs 18,23,36); reset + set_init_state |
| Settling | 10 no-op steps, gripper open | 10 no-op steps, gripper open |
| Actions/chunks | Official policy.select_action queue; native n_action_steps=1 | predict_action_chunk then execute 1 and replan, explicit clipping |
| Controller | Packaged relative OSC_POSE, 20 Hz | Audited classic relative OSC_POSE, 20 Hz; library versions differ |
| Horizon | Spatial/Object default 280; Goal 300, Long 520, 90 400 | Mug horizon 300 (shorter than official 90 default) |
| Outputs | Official success metrics and rendered main-camera videos | Custom grasp/lift/closure/tilt traces and two-camera videos |

Potential differences warrant comparison after standard-task results. Do not
interpret them as established causes. No changes to the mug runner are made.

## Decision logic and current answers

1. Genuine LIBERO competence through official evaluator: **not yet measured**.
2. White-mug failure attribution: **unresolved**; custom rendering/runtime,
   initialization and task-distribution differences remain hypotheses.
3. White-mug orientation experiment readiness: **no**, ordinary success missing.
4. Closest dataset-supported alternative: **Long 6**, same-scene mug transport plus
   chocolate-pudding placement. Report as a candidate, do not run it automatically.

If either selected task succeeds, nominal LIBERO competence is demonstrated on
that task; compare the differences above before modifying the custom runner.
If both fail 0/3, there is no reliable baseline: inspect videos, runtime versions,
checkpoint loading and asset paths before more rollouts. Six failures do not
prove global model incapability. Case C remains conditional because selected
checkpoint training provenance is incomplete even though dataset exclusion is
verified. No orientation or custom white-mug runs follow this check.

## Fresh-runtime Colab workflow

Select a GPU runtime. First mount Drive in a Python cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 1. One-time setup (no simulation)

```bash
%%bash
set -euo pipefail
nvidia-smi
mountpoint -q /content/drive || { echo "Run the Drive mount cell first."; exit 1; }
apt-get update -qq
apt-get install -y -qq libegl1 libgl1 libosmesa6 ffmpeg
python -m pip install -q uv
python -m uv venv --python 3.12 --seed --allow-existing /content/libero-official-env
python -m uv pip install --python /content/libero-official-env/bin/python \
  'lerobot[smolvla,libero,evaluation] @ git+https://github.com/huggingface/lerobot.git@8c920c4270460851cedd2737657584586d3dc66f' \
  'hf-libero==0.1.4' 'mujoco==3.3.2'
export LIBERO_CONFIG_PATH=/content/libero-official-config
printf 'n\n' | /content/libero-official-env/bin/python -c 'import libero.libero'
/content/libero-official-env/bin/python -c 'import torch; from lerobot.scripts.lerobot_eval import main; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

### 2. Pinned checkpoint / metadata / asset download (no simulation)

```bash
%%bash
set -euo pipefail
export LIBERO_CONFIG_PATH=/content/libero-official-config
/content/libero-official-env/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('HuggingFaceVLA/smolvla_libero',
    revision='6721902bc4d61e50a3bfdb11dfb4cb626f05d102',
    local_dir='/content/smolvla-libero-official-checkpoint',
    allow_patterns=['*.json','*.safetensors','README.md'])
# Small metadata download only; no demonstration dataset download.
snapshot_download('HuggingFaceVLA/libero', repo_type='dataset',
    revision='86958911c0f959db2bbbdb107eb3e17c5f9c798e',
    local_dir='/content/libero-official-task-metadata',
    allow_patterns=['meta/info.json','meta/tasks.parquet'])
from libero.libero import get_libero_path, get_assets_path
print('BDDL:', get_libero_path('bddl_files'))
print('Initial states:', get_libero_path('init_states'))
from pathlib import Path
assets=Path(get_assets_path())
assert assets.is_dir(), f'LIBERO asset download failed: {assets}'
print('Assets:', assets)
PY
```

### 3. Direct replacement simulation cell: six total episodes

```bash
%%bash
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
```

No Git pull is needed for these self-contained cells; they call the pinned
upstream evaluator directly. A fresh runtime needs no old simulator setup.
Each task is evaluated in a separate command so its completed report survives
failure/interruption in the second task. pipefail preserves evaluator errors.
Task metadata checks and imports do not step a simulator.

### 4. Outputs

Everything is under Drive:
`My Drive/humanoid_results/libero_official_<timestamp>/`.

- `packages.txt`: exact resolved package freeze.
- `sanity_manifest.json`: checkpoint/source revisions, package versions, original
  task names/instructions, seeds, planned episodes, BDDL hashes and documented
  initial-state selection.
- `<suite>_<index>.log`: full official evaluator logs, saved while running.
- `<suite>_<index>/eval_info.json`: official per-task successes and aggregate metrics.
- `<suite>_<index>/videos/<suite>_<index>/eval_episode_N.mp4`: official main-camera videos.

Official aggregation retains success lists but does not retain every episode's
seed/state ID in final eval_info.json. The manifest records seed 0 and the source's
sequential reset-selection rule; this is not extra live instrumentation. Video
filenames correspond to episode order. Confirm the installed source pin before
using that rule. Videos/reports are finalized at evaluator boundaries and may be
missing if a task is interrupted; logs already written remain on Drive.

Provide both eval_info.json files, packages.txt and the task videos after the run.
No measured success table can be produced until those results exist.

## Primary sources

- [Pinned official LIBERO guide](https://github.com/huggingface/lerobot/blob/8c920c4270460851cedd2737657584586d3dc66f/docs/source/libero.mdx)
- [Official evaluator](https://github.com/huggingface/lerobot/blob/8c920c4270460851cedd2737657584586d3dc66f/src/lerobot/scripts/lerobot_eval.py)
- [Official environment/reset implementation](https://github.com/huggingface/lerobot/blob/8c920c4270460851cedd2737657584586d3dc66f/src/lerobot/envs/libero.py)
- [Official environment config (including 360px default)](https://github.com/huggingface/lerobot/blob/8c920c4270460851cedd2737657584586d3dc66f/src/lerobot/envs/configs.py)
- [Selected checkpoint config](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/6721902bc4d61e50a3bfdb11dfb4cb626f05d102/config.json)
- [Selected model card (unknown dataset)](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/6721902bc4d61e50a3bfdb11dfb4cb626f05d102/README.md)
- [Alternative training manifest](https://huggingface.co/lerobot/smolvla_libero/blob/31d453f7edd78c839a8bbc39744a292686daf0de/train_config.json)
- [HuggingFaceVLA task metadata](https://huggingface.co/datasets/HuggingFaceVLA/libero/blob/86958911c0f959db2bbbdb107eb3e17c5f9c798e/meta/tasks.parquet)
- [LeRobot task metadata](https://huggingface.co/datasets/lerobot/libero/blob/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4/meta/tasks.parquet)
- [hf-libero 0.1.4 package](https://pypi.org/project/hf-libero/0.1.4/)
