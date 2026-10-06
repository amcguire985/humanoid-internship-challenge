# data_003: manual annotations and Colab handoff

Windows preparation is complete. No physical LIBERO rollout or training has run for these demos. Simulation/recording runs only in the validated Google Colab Python 3.8 micromamba runtime; Windows is reserved for video processing, annotations, dataset preparation and tests without LIBERO.

Edit `config/data_003_annotations/demo_001.json`, `demo_002.json` and `demo_003.json`. Times are absolute seconds in `videos/data_003.MOV`. User-supplied manual values are:

| Demo | Direction | Grasp (s) | Release (s) | Distance (cm) | Observed max lift (cm) |
|---|---|---:|---:|---:|---:|
| demo_001 | object -> target | 5.30 | 18.50 | 28.1 | 21.6 |
| demo_002 | target -> object | 24.50 | 36.50 | 30.3 | 12.9 |
| demo_003 | object -> target | 41.10 | 55.00 | 30.0 | 18.9 |

`5:30` in the user's message is interpreted as 5.30 seconds, consistent with this 58-second clip. Manual values are never automatically overwritten. They are snapped to the nearest source frame for execution; the exact requested values and effective snapped values remain in metadata. Null optional `transport_start_time_seconds` / `transport_end_time_seconds` preserve the existing motion-based transport interval. Invalid manual ordering raises an error rather than repairing annotations. Original automatic estimates remain unchanged.

`results/data_003_annotated/demo_NNN/annotation_diagnostic.png` shows x/y/z positions and speed with dashed automatic estimates, solid manual annotations and dotted effective boundaries. `validation.json` reports all positions, lift, distances, timing sources and transport durations. `processed_demo.csv` keeps all original cleaned positions and validity; phase labels reflect the manual values. `transport.csv` contains the manually bounded pickup-through-release interval. Source extracted demos remain unchanged.

All three have complete tracking and valid contact timing order. This passes the source-data checks, including the reverse-direction strategy. It does not certify the selected transport window, IK, controller budget, calibration/gravity direction, or physical task success. The current automatic transport windows end early (demo 1: 11.138s; demo 2: 31.182s; demo 3: 47.088s). Before committing a rollout in Colab, review the optional transport bounds so the window represents the intended carry rather than an early descent. Do not change grasp/release annotations to accommodate an automatic estimate. Geometry-only preview before the Colab-only instruction found excessive endpoint correction for these existing inner windows; Colab must evaluate the final windows against the unchanged controller limits.

Run preparation on Windows after editing annotations:

```powershell
.\.venv\Scripts\python.exe scripts/annotate_transfer_demos.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_transfer_annotations.py -v
```

In Colab, after pulling this commit:

```python
%cd /content/humanoid-internship-challenge
!git pull --ff-only
!MUJOCO_GL=osmesa /content/micromamba/envs/libero/bin/python scripts/annotate_transfer_demos.py
!MUJOCO_GL=osmesa /content/micromamba/envs/libero/bin/python scripts/plan_transfer_rollouts.py
# Review mapping_preview.json and optional transport timing fields before consuming any attempt.
!MUJOCO_GL=osmesa /content/micromamba/envs/libero/bin/python scripts/record_transfer_demos.py
```

The recording wrapper reuses `PickPlace`, `RecordingEnv`, `make_env(seed=0)`, `load_transport` and `retarget_transport`, the existing controller settings and schema validator. It does not change the core controller. Robot approach, grasp, lift, place and release remain scripted; the selected human transport segment is task-axis-aligned to the bowl-start -> plate task. `target -> object` is not discarded and its temporal sample order is preserved. Nominal mapping preflight checks endpoint correction, workspace and speed-limited phase budget before a physical attempt. Live grasp offset checks remain in the validated controller.

Exactly one attempt directory is allowed per human demo. Existing attempt directories are skipped, with no implicit retry. Failed attempts keep diagnostics and partial recordings outside the accepted episode directory, with first physical failure phase/reason when available. Runtime or recording errors are distinguished from physical failure. Only episodes passing the existing completed-sequence, post-release retention, LIBERO success and HDF5 schema checks are accepted.

The combined dataset selects **only the requested baseline episode_001 and episode_002**, even though a third baseline file exists in the repository. New successful episodes will be episode_004 (demo_001), episode_005 (demo_002), and episode_006 (demo_003). Failed/rejected demos contribute no training transitions.

Currently validated baseline dataset:

| Episode | Human source | Transitions |
|---|---|---:|
| episode_001 | test_007_block_only.MOV | 2,109 |
| episode_002 | test_007_block_only.MOV | 2,121 |
| Total | 2 successful episodes | 4,230 |

The local `results/libero_robot_dataset_multi_demo/summary.json` records this inventory. Large duplicate baseline HDF5 copies are not committed; reconstruct them in Colab with:

```python
! /content/micromamba/envs/libero/bin/python scripts/record_transfer_demos.py --combine-only
```

The recorder updates the combined dataset after each attempt, including source-human identity and hashes. No SmolVLA/LeRobot training or evaluation is part of this step.

Validation: six new annotation/direction tests and five existing retargeting mathematics/loader tests passed without LIBERO. Both baseline HDF5 episodes passed the existing episode validator. The core controller and original two episodes are unchanged.
