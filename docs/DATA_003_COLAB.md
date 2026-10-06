# data_003: one rollout per reviewed human demonstration

Windows preparation only: no LIBERO, robosuite, MuJoCo, policy training, new data collection, or physical rollout has run here. The recorder runs in the validated Colab Python 3.8 micromamba environment.

## Editable annotations

Each file in `config/data_003_annotations/` has four user-editable fields in absolute source-video seconds. All four fields are now manually labeled by the user:

| Demo | Human direction | Grasp | Transport start | Transport end | Release | Distance | Max observed lift |
|---|---|---:|---:|---:|---:|---:|---:|
| demo_001 | object -> target | 5.30 | 5.90 | 16.70 | 18.50 | 28.07 cm | 21.59 cm |
| demo_002 | target -> object | 24.50 | 25.40 | 35.50 | 36.50 | 30.34 cm | 12.92 cm |
| demo_003 | object -> target | 41.10 | 42.50 | 53.20 | 55.00 | 30.04 cm | 18.93 cm |

No additional manual timing review is pending. Demo 1's earlier `5:30` entry remains interpreted as 5.30 seconds. Exact requested values are preserved in config; effective execution values snap to the nearest observed video frame. Automatic estimates and old motion brackets remain separate, even when they disagree with manual times. Preparation never overwrites an existing annotation or repairs invalid manual ordering. This manifest requires both manual transport boundaries before a rollout; null values generate review diagnostics without consuming an attempt.

The old automatic transport ends at 11.14, 31.18 and 47.09 seconds truncated the paths. The reviewed intervals use the existing cleaned measurements without further smoothing, reversing, or filling gaps. Samples marked invalid and large source timestamp gaps are rejected.

Each `results/data_003_annotated/demo_NNN/` contains:

- `annotation_diagnostic.png`: object x/y/z and speed versus absolute video time, automatic/manual/effective boundaries.
- `processed_demo.csv`, `transport.csv`, `metadata.json`, `validation.json`: refreshed source input, phases, user timings and source metrics. `transport.csv` is the pickup-to-release export; the retargeter uses the transport_start/end subset of `processed_demo.csv`.
- `retargeting_input.json`: source demo, future episode ID, direction, input/config hashes, effective robot settings and all preflight gates.
- `mapping_preview.json`, `nominal_mapped_transport.csv`: nominal geometry and speed-limited transport references; these are previews, not robot demonstrations.

## Configuration and geometry

`config/data_003_rollouts.json` explicitly selects LIBERO Spatial task 0, seed 0, the same validated bowl-to-plate task, one attempt per source demo, and only baseline episodes 001/002. It lists each demo's source/annotation/prepared paths and episode ID. No robustness sweep or automatic retry is available.

The same task-axis retargeter aligns each demo's own start-to-end axis to the bowl-start-to-plate axis. Demo 002 keeps its target-to-object strategy and chronological sample order. Each selected source path retains its lateral shape and vertical lift residual, using the existing mapping scales. The existing small smooth endpoint correction is retained. Original human contact times describe the source video; robot approach, rim grasp, lift, placement, release and retract timing remains robot-specific scripted behavior.

All three reviewed inputs pass source, timing and nominal geometry checks:

| Demo | Source transport duration | Endpoint correction | Robot transport steps | Transport budget | Simulated transport duration |
|---|---:|---:|---:|---:|---:|
| demo_001 | 10.805 s | 0.81 cm | 4,411 | 5,000 | 220.50 s |
| demo_002 | 10.072 s | 0.48 cm | 4,822 | 5,500 | 241.05 s |
| demo_003 | 10.705 s | 0.84 cm | 7,136 | 8,000 | 356.75 s |

The controller code and shared robot/mapping configs are unchanged. These longer paths need a larger per-demo TRANSPORT timeout than the validated baseline's 2,000 steps. The manifest raises only that timeout, with headroom for live grasp geometry. The 0.02 m/s speed limit, workspace, action clipping, grasp/place sequence and all success/failure checks remain unchanged. The existing resampler slows the whole trajectory based on peak sample speed; do not interpret the durations above as wall-clock runtime. No new human path is shortened to fit a timeout.

Nominal previews cannot certify IK, collision-free carrying, live grasp offset, gravity calibration, or physical success. The live controller still checks these properties. Source world-Z direction and inherited phone calibration remain the documented measurement limitations.

## One Colab command

After pushing this work and pulling `colab/data-003-annotations` into `/content/humanoid-internship-challenge`, run exactly:

```bash
MUJOCO_GL=osmesa /content/micromamba/envs/libero/bin/python /content/humanoid-internship-challenge/scripts/record_transfer_demos.py --config config/data_003_rollouts.json
```

In a notebook code cell, prefix that command with `!`. The script changes to the repository root itself. It refreshes inputs/plots, checks manual annotations and nominal mapping, validates the two baseline HDF5 episodes, then runs at most one physical rollout for each ready demo. No separate preparation or planning command is required. The original video is not needed for this step; see [external video storage](VIDEO_STORAGE.md) if you want to review it.

The environment must already contain the existing processing dependencies (`numpy`, `scipy`, `opencv`, `matplotlib`), `h5py`, LIBERO/robosuite/MuJoCo and the validated assets. Do not install or run simulator dependencies in Windows Python.

## Expected outputs

`results/data_003_robot_rollouts/`:

- `run_readiness.json`: prepared inputs, geometry gates and execution state.
- `attempts/demo_NNN/`: durable one-attempt marker, input snapshot, `status.json`, replay CSV, controller metrics, figures and snapshots. Failure status records the first physical failure phase/reason/step, or a distinct runtime/recording error. Failed HDF5 files remain `episode.partial.h5` here.
- `rejected/demo_NNN.json`: input/preflight failures; these never consume physical attempts.
- `episodes/episode_004.h5`, `episode_005.h5`, `episode_006.h5`: created only for successful, schema-validated episodes from demos 001, 002 and 003 respectively.

An existing attempt directory is never replayed, even after a failure or interruption. Rerunning the command refreshes diagnostics and combines prior accepted episodes, but does not retry an attempted demo.

`results/libero_robot_dataset_multi_demo/episodes/` contains exactly baseline episode_001 and episode_002 plus newly successful episodes. `summary.json` reports successful counts, transitions, each episode's source human demo/direction/hashes, attempts, and rejected inputs. Baseline episode_003 is excluded. At present the two baseline episodes contain 2,109 + 2,121 = 4,230 transitions. After Colab, expect 2 to 5 accepted episodes and 4,230 plus the successful new episode transitions. No SmolVLA training/evaluation is run.

Both old episodes and new recordings use the existing success-only HDF5 schema: N actions and N+1 observations, agent/wrist RGB, proprioception, end-effector pose, phase, language instruction, simulation time and success labels. New episode attributes include human annotation/trajectory hashes, original direction, source paths, task and effective robot/mapping settings. Only completed post-release retained LIBERO successes are accepted into training.

## Windows checks

```powershell
.\.venv\Scripts\python.exe scripts/record_transfer_demos.py --prepare-only
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_transfer_annotations.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_libero_retarget.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_transfer_rollout_preparation.py -v
```

`--prepare-only` never probes/imports the simulator. `--combine-only` validates and combines already recorded HDF5 data without simulation. A normal rollout invocation on Windows is refused before simulator imports.
