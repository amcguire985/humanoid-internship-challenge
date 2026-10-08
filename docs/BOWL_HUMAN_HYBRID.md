# Human-video-guided bowl hybrid control

This is inference-time hybrid control, not post-training. Model weights are unchanged. The research comparison is B versus C; A is an original-instruction reference. Tilt and acceleration are spill-risk proxies: the scene contains no simulated liquid.

| Condition | Grasp instruction | Transport instruction | Human guidance |
|---|---|---|---|
| A | Original LIBERO instruction | Original | None |
| B | Original | Original + liquid constraint | None |
| C | Original | Original + liquid constraint | Timed object position and orientation |

The exact original instruction is `pick up the black bowl from table center and place it on the plate`. The transport suffix is `. The bowl is full of liquid. Do not spill it.`

## Implementation

The runner wraps `scripts/evaluate_bowl_liquid.py`, retaining its pinned official `lerobot-eval` loop, package/checkpoint hash checks, preprocessing, initialization, and standard success predicate. It reuses the audited bowl pose sampler and `replay_libero_transport.py` world-frame delta OSC inspection/scaling. Existing mug scripts, baseline results, and phone tracking outputs are unchanged.

New modules:

- `calibrate_bowl_reference.py`: recover fixed object-from-tag rotations from co-visible tags; record excluded pose ambiguities, scatter, and graph closure. Object +Z must be the independently verified top-face normal.
- `bowl_human_reference.py`: reconstruct object poses from raw tracking, gravity-align, preserve source time, resample positions linearly and orientations with SLERP, and export pose/velocity/acceleration plus preview.
- `bowl_transport_guidance.py`: confirmed-grasp state machine and bounded blended OSC intervention.
- `evaluate_bowl_hybrid.py`: official-loop hooks, prompt verification, two-camera recording, state diagnostics, transport analysis, and comparisons.

APPROACH becomes GRASP when the model closes the gripper. Three consecutive simulator `_check_grasp` observations with at least 2 cm lift activate TRANSPORT. The next policy inference in B/C uses the appended instruction; `policy.reset()` and both processor resets clear queued/cached actions before tokenization. No simulator reset occurs. The exact transition timestep and token evidence are saved.

C estimates `T_gripper_bowl = inverse(T_world_gripper) @ T_world_bowl` at transition. Human positions are aligned to the current bowl position and plate placement position using a yaw rotation, horizontal length scaling, and an explicitly logged quintic endpoint residual. Vertical shape has scale 1. Gravity-relative tilt is preserved; no time stretching occurs. Desired gripper pose is `T_world_bowl_desired @ inverse(T_gripper_bowl)`.

Human target position advances on original video time, influencing translation speed and acceleration through feedback. Reference velocity/acceleration are exported and logged; this is not a hard acceleration limiter. Orientation targets ramp from the grasp orientation over one second. Normalized action corrections are capped at 0.15 and have a slew limit of 0.6 per second; position/orientation blends default to 0.25/0.35. Near the plate, guidance fades and SmolVLA handles final placement/release. C suppresses an early open command outside the configurable plate neighborhood; this is part of the guidance intervention and is logged.

The rigid-grasp approximation is monitored using relative pose drift. C disables guidance permanently if translation drift exceeds 2.5 cm or rotation drift exceeds 20 degrees. Three consecutive lost-contact samples fail the rollout. Common grasp timeout is 180 control steps; the unchanged official Spatial horizon is 280. These explicit early stopping rules apply to all A/B/C. A normal model release is logged separately. A fallback configuration sets `orientation_enabled=false`; it retains human translation timing but must be reported as a translation-only intervention, not full C. Do not mix fallback C with full C under one experiment label.

Grasp snapshots are diagnostic NPZ files. Restoring MuJoCo state alone does not reliably restore controller goals, actuator bookkeeping, policy queues, and RNG state. The implementation therefore uses matched seed/state 0 instead of claiming shared grasp restoration. `comparison.json` records every grasp-entry pose and timestep so differences are visible.

## Real phone-data preflight and confirmed alignment

The existing processed data_003 demo_001 contains positions but no object orientation. Its metadata explicitly marks gravity and camera calibration unverified. Raw detections retain per-face rotations, enabling the new reconstruction without rerunning video detection. The export uses raw object-center positions, not the prior Savitzky-Golay-filtered positions; this choice preserves acceleration peaks and is logged. Source pose ambiguities are excluded only when a measured alternative remains within 30 degrees of the previous accepted orientation. Missing positions/orientations stop preparation; no gap is silently filled.

Local preflight estimated mounting rotations successfully. Co-visible pairs retained 236/265, 1331/1646, and 274/307 observations; retained 95th-percentile scatter was about 2.8, 5.6, and 4.6 degrees, with graph closure about 1.1 degrees. These are geometric consistency checks, not confirmation of the top face or gravity.

An explicitly **unverified** draft from the real phone trajectory was exported for reviewed transport interval 5.9ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“16.7 seconds: duration 10.8 seconds, timing scale 1. It rejected 32 inconsistent orientation candidates. Assuming ID0 +Z is gravity and ID5 is the top yields approximately 100ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“111 degrees tilt. This assumption is unsuitable for the bowl experiment and must be resolved from the recording/setup. The draft and preview remain under ignored `results/bowl_hybrid_preflight/`; the simulator refuses audit-only or unverified references.

The user confirmed on 2026-10-07 that ID6 is the top face, ID0 lies flat on the table with its printed face upward, and the camera calibration matched. The corrected verified reference preserves the 10.8-second duration and has about 5.69?19.57 degrees tilt. It rejects 32 ambiguous face candidates. The earlier 100?111 degree draft used the wrong top face and is not used. Confirmed mounting/gravity configuration is saved in `config/smolvla/bowl_human_calibration.json`; the preparation cell copies it to Drive. No additional gravity confirmation is required for this recording.

## Exact Colab workflow

New source files must first be made available in the Colab checkout; the scripts have not been published automatically. Mount Drive in a separate Python cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Keep the existing known-good `/content/libero-official-env`, checkpoint, and LIBERO configuration. No package or model version changes are needed.

The confirmed geometry is already saved in the repository. The preparation script copies it to Drive and reconstructs the reference from the existing data_003 tracking files.

This exact cell prepares the human reference and preview, with zero simulations:

```bash
%%bash
bash /content/humanoid-internship-challenge/config/smolvla/bowl_human_prepare.bash
```

Review `reference.png`, its gravity-relative tilt, raw acceleration peaks, and metadata. The measured 10.8-second reference occupies 216 control intervals; grasp time and final placement must fit the unchanged 280-step horizon. No time compression or horizon extension is automatic.

Replace the simulation cell with this for the first integration test: **exactly one C episode**.

```bash
%%bash
bash /content/humanoid-internship-challenge/config/smolvla/bowl_hybrid_evaluate.bash
```

It refuses to simulate without a verified reference. The result directory is printed at completion and all outputs stream directly to Drive. Inspect C before spending two more episodes on A/B.

If C works, set `BOWL_RUN_ROOT` to the exact existing directory printed by the first cell, and run only the missing A/B episodes; C is reused. This deliberately requires the real existing path to avoid rerunning it:

```bash
%%bash
set -euo pipefail
export BOWL_RUN_ROOT=/content/drive/MyDrive/humanoid_results/bowl_hybrid_REPLACE_WITH_EXISTING_TIMESTAMP
export BOWL_CONDITIONS="A B"
[ -f "$BOWL_RUN_ROOT/C/summary.json" ] || { echo "Use the completed C run directory."; exit 1; }
bash /content/humanoid-internship-challenge/config/smolvla/bowl_hybrid_evaluate.bash
```

If calibration is still being resolved, the same runner can independently test B with `BOWL_CONDITIONS=B` (one episode), then reuse that directory for C and A after calibration. A/B do not read a reference file. This verifies the phase/prompt hook without pretending to test human guidance.

## Artifacts and metrics

Each condition saves `summary.json`, official `eval_info.json`, exact prompt/token evidence, transition log, checkpoint/package evidence, and `ground_truth/episode_000/` containing `trajectory.jsonl`, `initialization.json`, `controller.json`, grasp snapshot, agent/wrist videos, `metrics.csv`, tilt/acceleration plots, and desired/actual position/orientation plots. C additionally saves the aligned reference and its preview. JSONL includes proposed policy actions and executed corrected actions; the official evaluator's action tensor represents policy output, so use JSONL for executed-control analysis.

Transport starts at the actual confirmed phase transition. Metrics include the transport interval through release or failure, identify the first possible contact loss, and state whether each peak occurred after that loss. Acceleration uses an unsmoothed nonuniform-time three-point central second difference of simulator bowl positions, only at interior transport samples. Endpoints are unavailable. Sampling is 20 Hz, not physics substeps. Failed grasp/absent transport yields null metrics, never zero. Transport duration, maximum lift, slip, standard success, and early failure reason are independent fields.

Root `comparison.csv`, `comparison.json`, and `comparison.png` compare completed conditions and validate checkpoint hashes, initial-state/BDDL hashes, seed, and controller settings. A one-episode result cannot establish a general causal effect. Historical three-success baseline videos lack pose traces, so they cannot supply these metrics; a new A is justified only for the controlled dynamics comparison. Earlier whole-episode liquid-prompt grasp failure remains a separate observation.

## Validation status

Local tests cover pose transforms, gravity alignment, tilt-preserving endpoint alignment, exact source timing/resampling/export, reference verification, real-contact phase transitions, timeout/release, controller bounds/slew/slip handling, dependence on human orientation and timing, transport-only metrics, and exact prompt/cache switching. Existing bowl tests are also run. The corrected real phone reference was reconstructed and verified against the user-confirmed geometry. Its sampled tilt range is 5.69?19.57 degrees, with unchanged 10.8-second timing.

No CUDA/LIBERO simulation or policy-weight update was performed locally. Actual official async-worker hooks, OSC rim-grasp stability, codecs, successful C placement, and A/B/C comparison still require Colab validation. There are no new experimental performance results yet.

## First Colab integration result and critical fix

The supplied condition C rollout (`bowl_hybrid_20261007_231712`) confirmed grasp/lift at step 62, retained contact, and timed out at step 280 without LIBERO success. Maximum transport tilt was 17.67 degrees and sampled acceleration 1.78 m/s2; duration was 10.9 seconds. These are failure diagnostics, not evidence of improvement over B. Video shows the bowl remained held at the end.

All 15 policy open commands were blocked by guidance, and no sample allowed release. The final target was 5.12 cm away horizontally and 8.83 cm higher than the actual bowl. The original target calculation used asset top/bottom extent sites as if they were physical support surfaces; that was incorrect. The [plate asset XML](https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/master/libero/libero/assets/stable_scanned_objects/plate/plate.xml) separates collision boxes from these extent sites.

`collision_surface_placement_v2` now computes the central plate collision-box top and the upright bowl collision-box bottom. It logs `placement_target.json`, keeps human timing and the 280-step horizon unchanged, and versions the algorithm in the manifest to prevent mixing this repaired intervention with the failed C run. This correction is locally tested but still needs one C retry in Colab. Do not rerun A/B yet.

Two logging/report issues were also fixed: native boolean encoding of the reference-finished flag and creation of the printable Markdown report required by the parent wrapper. These fixes do not change policy actions.

## Orientation diagnosis update

Full-orientation tracking caused unnecessary bowl yaw. Condition C now uses the opening-axis gravity objective and a gradual rotational handoff at release; Condition B remains unchanged. See [the recorded evidence, coordinate audit, code changes, and pending single-rollout validation](BOWL_ROTATION_DIAGNOSIS.md). Earlier full-orientation C results must retain their original algorithm label. Human timed position and acceleration data remain active inputs.
