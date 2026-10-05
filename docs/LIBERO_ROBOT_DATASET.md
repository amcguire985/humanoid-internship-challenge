# Robot-native scripted demonstrations

The dataset lives at `results/libero_robot_dataset/`. The collector is configured to run the existing `PerturbedPickPlace`
controller across **all ten** offsets in
`results/libero_xy_trials_10/preflight.json`. All ten passed geometry/IK checks;
only six had previously completed physical rollouts. This is one task, one
seed, one scene, and ten closely spaced initial bowl positions, not a broad
policy-training corpus. The collector does not train a policy. Collection was stopped at the user's
request: episodes 1–3 completed successfully, episode 4 was interrupted, and
5–10 were not attempted. The subsequent [BC baseline](LIBERO_BC_BASELINE.md)
uses only episodes 1 and 2; episode 3 finished during a chat interruption and
is deliberately excluded from that experiment.

Neither the retargeting code, manipulation code, nor their configurations are
modified. Their hashes must match the saved preflight before collection starts.
The original post-release retention check determines acceptance, not transient
LIBERO success while the bowl is still held. There is one attempt per offset,
with no automatic controller retries. `attempts/*/status.json` explicitly
records success, failure, error, or interruption. Only validated, completed,
successful HDF5 files are atomically moved into `episodes/`. Partial files are
removed on caught failures; an uncatchable termination may leave a `.partial.h5`
in `attempts/`, which must never be loaded for training. Existing dataset output
directories are refused. An initial environment startup/cache failure is
preserved separately in `results/libero_dataset_startup_errors/`; no simulation
or physical attempt began in that launch. A later partial PREGRASP capture was
interrupted to improve CPU rendering throughput; its diagnostic record is kept
in `results/libero_dataset_render_interrupted/` and is excluded from this dataset.

## Episode boundary and synchronization

The identical 20 scene-settling actions and bowl XY perturbation are **reset
preparation**. Collection begins immediately after this preparation and before
PREGRASP, and includes every action through the final RETRACT hold. Thus no
artificial object teleport appears within an episode. Full controller diagnostics
also include the 20 setup steps, so their counts are 20 greater per episode.

An episode contains T actions and T+1 observations. Row `observations/*[t]` is
captured **before** `actions[t]`; row `[t+1]` is its resulting observation. Both
cameras and all numeric observations at a row come from the same simulation
state, with no simulation step between renders. `sim_time` increments by 0.05 s
(20 Hz). The final observation is retained. The wrapper copies the actual action
at the `env.step` boundary. Before acceptance it checks all actions and post-step
success flags against the original controller's independent records.

## Files

```text
results/libero_robot_dataset/
  episodes/episode_NNN.h5       # only complete successful episodes
  attempts/episode_NNN/
    status.json                # acceptance, requested/applied offset, failure reason
    metrics.json               # full original controller success/phase report
    replay.csv                 # original per-step controller diagnostics
    initial_state.npz          # full simulator qpos/qvel after XY reset
    initialization_audit.json   # proof that only bowl XY changed
    *.png, *.csv                # original phase images / retargeting diagnostics
  preflight.json               # copied validated reset protocol and input hashes
  nominal_state.npz            # copied nominal reset state
  provenance.json              # inputs, recorder hash, seed, image resolution
  recording_script_snapshot.py # exact recorder source used for this capture
  summary.json                 # accepted files, counts, shapes, action statistics
  samples/                     # loader-produced observation/action previews
```

Each HDF5 file uses lossless gzip level 1 and one-row chunks. No Python pickle or
video decoder is needed. Numeric state/action values retain float64 precision.

| HDF5 path | Shape / type | Meaning |
|---|---|---|
| `observations/agentview_rgb` | (T+1,128,128,3), uint8 | RGB, top-left origin |
| `observations/eye_in_hand_rgb` | (T+1,128,128,3), uint8 | Panda wrist RGB, if camera exists |
| `observations/joint_pos` | (T+1,7), float64 | Panda arm angles, radians |
| `observations/joint_vel` | (T+1,7), float64 | Panda arm velocities, radians/s |
| `observations/gripper_qpos` | (T+1,2), float64 | Finger joint positions, metres |
| `observations/gripper_qvel` | (T+1,2), float64 | Finger joint velocities, metres/s |
| `observations/proprio` | (T+1,18), float64 | Concatenation of above four state vectors |
| `observations/eef_pose` | (T+1,7), float64 | World xyz metres, quaternion xyzw |
| `observations/sim_time` | (T+1,), float64 | Simulator seconds |
| `observations/libero_success` | (T+1,), bool | Raw LIBERO task predicate at that state |
| `actions` | (T,7), float64 | Actual normalized OSC_POSE + gripper input |
| `timestep` | (T,), int64 | Zero-based dataset timestep |
| `controller_step` | (T,), int64 | Original controller step (starts at 21) |
| `episode_id` | (T,), fixed bytes | Episode identifier repeated per transition |
| `phase` | (T,), S32 | Controller phase when action was issued |
| `rewards` | (T,), float64 | Raw environment reward |
| `environment_done` | (T,), bool | Raw environment done signal, not dataset truncation |

HDF5 attributes: `schema_version`, `episode_id`, `language_instruction`,
`complete`, `episode_success`, `success_metadata_json`, `control_frequency_hz`,
`image_convention`, `eef_pose_convention`, and `proprio_order`.
Language and episode success are constant episode attributes; the loader
broadcasts them into each returned transition rather than duplicating text.
`success_metadata_json` includes completion, release retention, failure details,
controller scaling, and phase boundaries. Episode end is `t == T-1`; do not
infer it from raw `environment_done` or a transient success predicate.

Actions are `[dx,dy,dz,rx,ry,rz,gripper]`, normalized controller inputs, **not**
absolute poses or joint commands. The first six use world-frame translation /
axis-angle delta semantics, with exact input/output scales in
`success_metadata_json.controller`; their scripted clamp is [-0.5,0.5].
Gripper -1 opens and +1 closes using the existing sign-integrated aperture
controller. Gripper position is separately observed, not inferred from command.
The compact numeric observation is 25 values if combining proprio and EEF pose.
Images are directly rendered from MuJoCo and vertically flipped once; no resize,
lossy compression, or subsampling is applied. For CPU rendering throughput,
shadows, reflections, and multisample antialiasing are disabled (recorded in
`render_settings`); original visual meshes, textures, lighting, and camera poses
are retained. These settings affect appearance only, not physics.

## Reproduction and loading

Use the installed LIBERO environment (Python 3.8, h5py, NumPy, Pillow, SciPy,
matplotlib, robosuite, MuJoCo, LIBERO). Writable cache directories are needed in
the restricted workspace:

```bash
NUMBA_CACHE_DIR=/tmp/libero_numba_cache MPLCONFIGDIR=/tmp/libero_matplotlib \
LP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa \
/home/student/miniconda3/envs/libero/bin/python scripts/record_libero_dataset.py \
  --output results/new_robot_dataset

/home/student/miniconda3/envs/libero/bin/python scripts/inspect_libero_dataset.py \
  results/libero_robot_dataset/episodes/episode_001.h5 \
  --samples results/libero_robot_dataset/samples

/home/student/miniconda3/envs/libero/bin/python -m unittest discover \
  -s tests -p 'test_libero_dataset.py' -v
```

`validate_episode(path)` checks dimensions, finite values, IDs, contiguous
indices, time spacing, action bounds, quaternion norms, proprio ordering, terminal
observation, and the full acceptance criteria. `load_transition(h5_file, t)`
returns observation, action, next observation, instruction, ID, timestep, episode
success, and success after the action. `save_samples` saves five paired camera
previews labeled with the corresponding action and a machine-readable JSON file.
Load only paths from `summary.json.episodes`, never from `attempts/`.
