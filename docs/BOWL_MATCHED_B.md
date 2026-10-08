# One frozen matched B versus successful C

No controller code or working C files/configuration are modified. The orchestration
runner archives known working revision 08d37c3d36720498d552711463e0f53992c53386
into a temporary checkout and copies exact settings from the selected successful
C experiment manifest. B bypasses phone constraints using the existing condition
B branch; original grasp instruction and liquid transport instruction remain.
All non-phone guards and placement/release/withdrawal remain identical.

Before simulation, verify C is fully successful, constraint mode/controller
version match, current checkpoint hashes equal C, baseline package versions match,
frozen configuration equals C, and runtime task state/BDDL hashes equal saved C.
Save configuration_diff_before_rollout.json, frozen configs, source hashes and
exact command. Historical C did not record source hashes, so the match to its
source is based on known working revision and manifest algorithm/configuration;
this limitation is logged rather than claiming source hashes were recorded then.

## Colab: one B rollout, no C rerun

Use existing initialization and Drive mount. The pointer below selects the last
phone-constraints run, and the runner refuses it unless fully successful. If
runtime restarted, replace C_OUTPUT with the exact successful .../C Drive path.

```bash
%%bash
set -euo pipefail
cd /content/humanoid-internship-challenge
git fetch origin
git switch upright-mug-orientation
git pull --ff-only origin upright-mug-orientation
export MPLBACKEND=Agg MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=/content/libero-official-config
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=2
unset PYTHONPATH
mountpoint -q /content/drive
C_OUTPUT="$(cat /content/bowl_phone_constraints_last_output.txt)"
/content/libero-official-env/bin/python scripts/run_matched_b.py --c-output "$C_OUTPUT"
```

Output folder prints before rollout: MyDrive/humanoid_results/bowl_matched_B_<ID>/.
One B evaluator invocation, no retries. Failed validation aborts without automatic
rerun. After B, verify checkpoint/settings/placement algorithm/prompt, actual
initialization hashes and OSC scaling/bounds match C, and B logs no phone
constraint or trajectory intervention. Create comparison.md and comparison.json
for success, intact-grasp transport tilt/acceleration, transport duration,
controller completion, grasp loss and failure. B/ contains usual diagnostics and
playback videos. C remains untouched. Share comparison plus B/C traces afterward.

Matched initialization does not force identical transport intervals or grasp
states. One pair is a controlled diagnostic, not reliability evidence. No tuning
or local simulation is performed. Existing CUDA/LIBERO environment is required.

Local validation: three orchestration tests passed using fixtures (not rollout results). Controller files remain unchanged.
