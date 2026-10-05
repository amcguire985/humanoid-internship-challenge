# SmolVLA rapid experiment: GPU handoff

Local status: **dataset prepared and validated; training and evaluation not run**.
No usable CUDA GPU is exposed. The existing LIBERO environment is Python
3.8.13, PyTorch 2.4.1+cu121, CUDA availability false, device count zero.
`nvidia-smi` and `/dev/nvidia*` are absent. Host memory is approximately 20 GiB.
The unrelated default shell Python is 3.14.7; it is not the LIBERO environment.

Official LeRobot main inspected for this task identifies itself as 0.6.2 and
requires Python >=3.12, torch >=2.7,<2.12, torchvision >=0.22,<0.27. Its
SmolVLA extra requires Transformers >=5.4,<5.6 plus accelerate and num2words.
The GPU dependency file pins the exact inspected LeRobot commit. Use a separate
Python 3.12 environment. No packages or model weights were installed/downloaded
locally; only small official source/config files were read for API inspection.

## Dataset mapping

The source is exactly episode_001 and episode_002 HDF5: 2,109 + 2,121 actions,
each with T+1 observations. No new data and no inclusion of episode 3.

| Recorded field | SmolVLA input/target |
|---|---|
| agentview_rgb[t], uint8 HWC 128x128 | observation.images.camera1, float CHW [0,1] |
| proprio[t], 18D | observation.state, same ordering |
| actions[t:t+50], 7D | action, unchanged LIBERO controller units |
| HDF5 language_instruction | task, tokenized by official SmolVLA processor |
| Episode end | action_is_pad mask; never borrow actions from another episode |

Exact language: **pick the akita black bowl between the plate and the ramekin
and place it on the plate**.

Proprioception remains arm positions (7), velocities (7), finger positions (2),
and finger velocities (2). Actions remain six LIBERO OSC commands and the
sign-integrated gripper command (-1 open, +1 close). They are not relabeled as
joint targets. The pretrained base has 32-dimensional padded state/action
capacity, so 18D/7D fit without replacing its learned projection layers. Only
one actual camera is supplied; no wrist observations are fabricated. Model
image preprocessing retains its pretrained configuration (512-square padded
resize); stored source images remain lossless 128-square RGB.

`results/libero_smolvla/dataset/` is a **minimal SmolVLA tensor dataset**, not a
native LeRobot Parquet dataset. It uses memory-mapped NPY arrays and a manifest,
avoiding a modern LeRobot installation solely for local conversion. The GPU
trainer loads these keys directly into the official SmolVLA preprocessing and
policy APIs. It is ready for that runner without another conversion.

The export also retains original timestep, controller step, simulator time,
per-step next success/done flags, exact episode success metadata, and terminal
RGB/state separately. The terminal observation has no fabricated action.
Source hashes are recorded. `validate_libero_smolvla.py` compares 20 samples
against HDF5 including boundaries, grasp onset and release; numeric timeline
and success arrays are also checked.

An optional `--native-from` path uses official `LeRobotDataset.create`,
`add_frame`, `save_episode`, and `finalize` to make an image-based native dataset
without lossy video. It performs a native loader round-trip. **That optional
conversion is prepared but unexecuted**, because LeRobot is not installed here.
The original portable export remains the source of detailed timing metadata.

## Minimal training configuration

Start from `lerobot/smolvla_base` with strict full-checkpoint loading; the Hub
revision is resolved and saved when running. Freeze the vision/language model
with `freeze_vision_encoder=True`, `train_expert_only=True`; train the action
expert, action projections and state projection. No LoRA complexity, no model
from scratch, no visual augmentation. One seed (0), batch 4, AdamW 1e-4,
pretrained 50-action horizon, execute four then replan, ten flow inference steps.

Maximum 200 optimizer updates (800 sampled examples at batch 4), or 15 minutes
of training-loop time. Model download/loading and initial probe are outside
that clock; one update or final probe can extend past the limit. No automatic
retries, batch-size sweep, or second training run. If GPU memory is insufficient,
stop and report it rather than launching an automatic search.

Log flow-matching training loss; every 25 updates inspect eight fixed training
observations with deterministic noise, reporting flow loss, decoded action MSE,
and predicted/demonstrated gripper for the first four actions. After at least
50 updates, stop if both fixed-probe losses improve >=30% from the pretrained
baseline. This is a modest learning-progress heuristic, not a success or
held-out-data claim. Only the final checkpoint is saved. No weighting search.
Mean/std normalization is estimated on the two episodes and reused at inference.

## Commands on a GPU machine

Copy this repo including `results/libero_smolvla/dataset` and the original
simulation assets. Install a CUDA-enabled torch/torchvision combination matching
the driver and LeRobot requirements in a separate Python 3.12 environment, then:

```bash
/path/to/smolvla/bin/python -m pip install -r config/smolvla/requirements-gpu.txt
/path/to/smolvla/bin/python scripts/train_libero_smolvla.py
```

The runner checks CUDA before importing LeRobot, creating training output, or
downloading a checkpoint. Existing training output is never overwritten.

Optional native LeRobot export (not needed by the minimal trainer):

```bash
/path/to/smolvla/bin/python scripts/prepare_libero_smolvla.py \
  --native-from results/libero_smolvla/dataset \
  --output results/libero_smolvla/lerobot_dataset
```

Exactly one nominal evaluation after a completed training run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=2 \
/path/to/libero/bin/python scripts/evaluate_libero_smolvla.py \
  --policy-python /path/to/smolvla/bin/python
```

The legacy LIBERO environment drives the simulator; a separate CUDA worker runs
the fine-tuned model using original 128x128 RGB, 18D state, exact task language,
and saved normalization. It sends chunks over stdin/stdout, executes four
commands, and replans. Uses the existing nominal reset, 2,400-step horizon,
action clipping and gripper threshold, and physical acquisition/lift/placement
checks. Refuses an existing evaluation directory and never runs a perturbation
or second seed. CSV and screenshots retain failures; `evaluation/result.json`
adds starting-pose departure (>=2 cm), approach (>=5 cm distance reduction),
closest EEF distance, close commands, acquisition, lift, placement, LIBERO
success, and first failed milestone. These thresholds are diagnostics, not
inputs or controller overrides.

The GPU-dependent model load, processors, native exporter, and evaluation bridge
are source-reviewed and syntax-checked, **not runtime-validated with LeRobot**.
Do not treat this handoff as a trained policy or measured result.

## Comparison status

| Metric | Simple BC | Chunk BC | SmolVLA |
|---|---|---|---|
| Offline training | Fitted demonstrations | Fitted most actions, missed closing onset | Not run |
| Leaves start / approaches | Remained near start | Remained near start | Not measured |
| Closest EEF distance | See baseline logs | 0.37117 m | Not measured |
| Closes / acquires / lifts | No / no / no | No / no / no | Not measured |
| Transports / places | No | No | Not measured |
| Final LIBERO success | False | False | Not measured |

The pretrained-knowledge experiment remains unanswered until a GPU run is
performed. Absence of hardware is not a negative SmolVLA result. No RL, new
demonstrations, sweeps, repeated evaluations, or performance optimization added.

Sources inspected:
- https://github.com/huggingface/lerobot/blob/8c920c4270460851cedd2737657584586d3dc66f/pyproject.toml
- https://huggingface.co/docs/lerobot/smolvla
- https://huggingface.co/lerobot/smolvla_base/blob/main/config.json
- https://github.com/huggingface/lerobot/blob/8c920c4270460851cedd2737657584586d3dc66f/src/lerobot/policies/smolvla/modeling_smolvla.py
