# First bowl liquid-prompt experiment

## Status and baseline limitation

Official Spatial 2 baseline succeeded 3/3 on 2026-10-07, saved under
`My Drive/humanoid_results/libero_official_bowl_20261007_204110`.
The baseline log explicitly has `eval.recording=False` and official evaluator
`return_episode_data=False`. Only success metrics and videos were persisted;
no ground-truth object pose trajectory was saved. These artifacts cannot recover
gravity-relative bowl tilt or translational acceleration. RGB reconstruction is
not used. Original success is reusable; original dynamics and paired differences
remain unavailable. **Do not automatically rerun Condition A.** No constrained
rollout has been run from this Windows workspace.

## Experimental variable and interpretation

Condition B is exactly `The bowl is full of liquid. Do not spill it.` as specified
in the full request. There is no upright/speed/acceleration instruction. This is a
**replacement** of the task language, so the policy no longer receives the explicit
bowl-to-plate goal. A failure could reflect that omission, not necessarily an
inability to infer liquid-handling behavior. Adding the original instruction to
these sentences would be a different experiment; this runner does not do it.
The environment task description, BDDL, scene, goal and success predicate remain
original Spatial task 2.

## Minimal official-path wrapper

`scripts/evaluate_bowl_liquid.py` calls the same pinned evaluator's `main()`.
The official rollout selects language from `env.call('task_description')`, then
applies environment preprocessing, checkpoint preprocessing/tokenization, and
`policy.select_action`. This revision has no policy-prompt CLI field.

The wrapper delegates all environment preprocessing and saved checkpoint
processing. At the policy preprocessor boundary, it replaces only the batch's
`task` value. It prints both environment and policy instructions and checks exact
unmasked token IDs against the entire specified prompt plus the checkpoint's
newline. A changed or truncated prompt aborts before selecting an action.
`prompt_verification.json` preserves the evidence.

The official environment factory returns the same original environment, with
reset/step/close methods decorated to read simulator fields. Each original reset
and step runs exactly once; results are returned unchanged. No additional env.step
or reset, controller update, render setting, observation transform, action scaling
or success predicate is introduced. Hooks are installed inside async workers by
the decorated factory, so async evaluation remains enabled as in the baseline.
No installed LeRobot files are edited. No phone-video processing/training code changes.

The wrapper checks exact resolved runtime versions against the baseline's
packages.txt and LeRobot's installed source commit. It validates all checkpoint
files against their pinned snapshot_download metadata/ETags. Keep the local
checkpoint's hidden `.cache/huggingface/download` metadata directory. Only the
validated official flags are accepted; overrides of action horizon, episode length,
state selection or async behavior are rejected.

## Pairing

Use one async environment, three episodes, seed 0, original benchmark init states,
hard reset and 10 settling actions, exactly as in the successful run. Official
reset selection starts at state ID 0 and advances by one per episode; seeds are
0,1,2. The baseline manifest recorded this rule rather than actual state hashes.
For B, log actual selected state IDs, seeds, state-byte SHA256 and BDDL SHA256.
Require the observed IDs/seeds to be 0,1,2 before constructing the comparison.
This is matching by the pinned official reset algorithm; baseline state hashes
were not captured, so bit-identical reset verification against A is unavailable.
The checkpoint, package versions, state files and asset cache must remain unchanged.

## Object pose and metrics

The target is `akita_black_bowl_1` from Spatial 2's unchanged BDDL. The other bowl
is a distractor. The asset's top_site is at local [0,0,0.04] and bottom_site at
[0,0,-0.06], establishing local +Z as opening/up. Runtime verifies the object's
top-minus-bottom offset points along +Z. MuJoCo root-body rotation maps object
coordinates to world; quaternion order is w,x,y,z.

The trace contains the first observation after official settling (timestep 0),
then every official action endpoint: simulation time, bowl XYZ, rotation matrix,
quaternion, tilt, signed finger qpos/aperture, action, grasp flag when supported,
and unmodified LIBERO success. Trace files flush every sample for interruption
recovery. Metrics do not derive pose from RGB. Sampling is at control endpoints
(20 Hz), not every physics substep; continuous-time acceleration peaks can be missed.

Transport begins at detected grasp plus bowl-root height increase >=2 cm from
the settled start. If that signal never occurs, explicitly use the fallback:
height increase >=2 cm, positive close action and finger aperture reduced by
>1 mm from settled start. The fallback is a diagnostic and does not prove grasp.
Report whether it was used. Transport ends just before an observed open command,
aperture increase >1 mm relative to transport start, and no detected grasp; if
that release signal is absent, use task success/episode end. Grasp loss alone
is not classified as commanded release. Reset and approach samples are excluded.

Tilt = acos(clamp(world +Z dot (R @ local +Z), -1,1)), degrees. Yaw does not count.
Report the maximum only over transport; retain every sample's tilt in the trace.

Use the true recorded simulation times and unfiltered three-point central second
differences on XYZ wholly within transport. For adjacent spacings h0/h1:
`a_i = 2 * ((p_(i+1)-p_i)/h1 - (p_i-p_(i-1))/h0) / (h0+h1)`.
This is the derivative of adjacent finite-difference interval velocities. Endpoints
are unavailable; no approach/release sample enters the stencil. Report max norm
of valid interior acceleration samples. No smoothing is applied: raw and processed
columns are identical, making the numerical differentiation trace reviewable.
Fewer than three transport samples yields unavailable acceleration. No transport
yields unavailable tilt and acceleration, never zero as evidence of caution.

Tilt and acceleration do not alter task success. No significance testing is used.
A lower maximum alone should not be called useful physical inference if transport
fails or its duration/path differs substantially. The simulator contains no actual
liquid; these are handling proxies, not a measured spill rate.

## Existing-runtime Colab cell

No environment rebuild or dependency update is required. New files to deliver:
`scripts/evaluate_bowl_liquid.py` and its existing dependency
`scripts/upright_orientation.py`. Local changes must be committed/pushed to
`upright-mug-orientation` before the pull below. Example local delivery commands:

```bash
git add scripts/evaluate_bowl_liquid.py tests/test_bowl_liquid.py docs/BOWL_LIQUID_EXPERIMENT.md config/smolvla/bowl_liquid_evaluate.bash
git commit -m "Observe matched bowl liquid prompts around official evaluator"
git push origin HEAD:upright-mug-orientation
```

With Drive mounted, replace the simulation cell with:

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
RUN_ROOT="/content/drive/MyDrive/humanoid_results/bowl_liquid_$(date -u +%Y%m%d_%H%M%S)"
[ -x "$EVAL_ENV/bin/python" ] || { echo "Restore the previously validated official environment first."; exit 1; }
[ -f "$BASELINE/libero_spatial_2/eval_info.json" ] || { echo "Original baseline eval_info.json is missing."; exit 1; }
[ -f "$CHECKPOINT/model.safetensors" ] || { echo "Restore the same pinned checkpoint download first."; exit 1; }
mkdir -p "$RUN_ROOT"
"$EVAL_ENV/bin/python" -m pip freeze > "$RUN_ROOT/packages.txt"
"$EVAL_ENV/bin/python" scripts/evaluate_bowl_liquid.py \
  --baseline-root "$BASELINE" --condition liquid \
  --policy.path="$CHECKPOINT" \
  --policy.device=cuda --policy.load_vlm_weights=false \
  --env.type=libero --env.task=libero_spatial --env.task_ids="[2]" \
  --env.control_mode=relative --env.init_states=true --env.hard_reset=true \
  --env.observation_height=256 --env.observation_width=256 \
  --env.max_parallel_tasks=1 --eval.batch_size=1 --eval.n_episodes=3 \
  --seed=0 --output_dir="$RUN_ROOT/evaluation" \
  2>&1 | tee "$RUN_ROOT/evaluation.log"
echo "Results and videos: $RUN_ROOT"
```

This runs **only three liquid-prompt episodes**. Original baseline is not rerun.
The cell assumes the same running official environment and downloaded snapshot;
a runtime reset needs restoration of that exact package environment first.
Version and checkpoint mismatch checks stop before rollout.

## Outputs

Root: `My Drive/humanoid_results/bowl_liquid_<timestamp>/`.

- `packages.txt`, `evaluation.log`: current runtime and full evaluator output.
- `evaluation/experiment.json`: unchanged baseline identity, actual prompt and design caveats.
- `evaluation/prompt_verification.json`: original environment prompt, actual policy prompt and full token verification.
- `evaluation/eval_info.json`: official task-success outcomes.
- `evaluation/ground_truth/episode_NNN/initialization.json`: actual seed/state index/hash, task hash and opening axis.
- `evaluation/ground_truth/episode_NNN/trajectory.jsonl`: full raw ground-truth trace.
- Same episode folder: `summary.json`, `metrics.csv`, `tilt.png`, `acceleration.png`.
- `evaluation/comparison.md` and `comparison.json`: paired success rows; original tilt/acceleration and their deltas explicitly unavailable.
- `evaluation/videos/libero_spatial_2/eval_episode_N.mp4`: unchanged official main-camera videos.

There are no original/liquid plot overlays because original object-pose traces
are missing. Do not infer original dynamics from video to fill the table. Logs and
raw traces persist during execution; summaries/plots appear after completion.

## What can be answered after B

Compare task success with the three known A successes. Report B transport tilt
and acceleration and visually review the videos. Quantitative reductions relative
to A, paired deltas and the physical behavior gap remain unmeasured until matched
A object-state traces exist. Do not authorize baseline reruns implicitly, conclude
physical inference from the success rate alone, or implement post-training.

## Validation

Nine offline tests cover known quadratic acceleration on uniform/nonuniform time,
exclusion of approach/release samples, no-transport missing values, fallback
labeling, yaw-invariant tilt, actual token verification, truncation rejection, and
preservation of original reset/step return values and task description. These do
not validate live CUDA inference, real async worker transport or physical grasp.

## Primary references

- [Pinned official evaluator language path](https://github.com/huggingface/lerobot/blob/8c920c4270460851cedd2737657584586d3dc66f/src/lerobot/scripts/lerobot_eval.py)
- [Pinned official reset/state selection and factory](https://github.com/huggingface/lerobot/blob/8c920c4270460851cedd2737657584586d3dc66f/src/lerobot/envs/libero.py)
- [Target bowl asset](https://huggingface.co/datasets/lerobot/libero-assets/blob/main/stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml)
