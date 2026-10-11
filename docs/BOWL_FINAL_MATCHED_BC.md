# Final five-pair B/C evaluation

The batch runner freezes revision `08d37c3d36720498d552711463e0f53992c53386`, the working persistent-withdraw controller. No controller files in this checkout are edited. Its temporary archived source receives only an explicit initial-state selection before the original environment reset and matching reporting assertions. Those adapters and source hashes are recorded in `configuration_diff_before_rollout.json`.

States 1–5 are chosen before results, excluding tuned state 0. No complete tuning-history registry exists; if other states were previously used for tuning, exclude them with `--excluded-states` and explicitly choose five different states with `--states`. The runner refuses overlaps or unavailable indices rather than substituting states. Both conditions use seed 0, initial-state vector hashes, original grasp prompt and identical liquid transport prompt. Order is B/C, C/B, B/C, C/B, B/C. There is one invocation per condition per state, no retries. Infrastructure failures remain unavailable and are retained; controller failures remain measured failures. A matching violation stops the batch, preserving partial progress.

Use the existing Colab runtime, Drive mount, pinned evaluator and checkpoint. First make the new runner and `run_matched_b.py` available in the Colab checkout (these local changes have not been pushed). Replace `C_OUTPUT` if the runtime pointer is absent, with the successful C folder containing `experiment.json` and `summary.json`.

```bash
%%bash
set -euo pipefail
cd /content/humanoid-internship-challenge
export MPLBACKEND=Agg MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=/content/libero-official-config
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=2
unset PYTHONPATH
mountpoint -q /content/drive
C_OUTPUT="$(cat /content/bowl_phone_constraints_last_output.txt)"
/content/libero-official-env/bin/python scripts/run_matched_bc_batch.py \
  --c-output "$C_OUTPUT" \
  --human-reference /content/drive/MyDrive/humanoid_results/bowl_human_reference/reference.csv \
  --states 1 2 3 4 5 --excluded-states 0
```

Exactly ten scheduled rollouts; no dependency reinstall or downloads. Source, checkpoint, package versions, saved controller settings, phone source/metadata hashes and extracted constraint values are checked first. Historical C did not record source hashes; the known working archived revision matches its saved algorithm/configuration. All new B/C episodes share identical frozen source. Actual state, OSC and manifest matching is verified per pair after execution.

Outputs: `MyDrive/humanoid_results/bowl_final_matched_BC_<unique>/`. The full path is printed and saved in `/content/bowl_final_matched_last_output.txt`. Each `state_00N/B` or `C` retains evaluator artifacts and videos; sibling B.log/C.log retain subprocess output. Batch products:

- `rollouts.csv`: ten rows, blank unavailable numeric metrics; grasp success means confirmed GRASP-to-TRANSPORT detection, not merely momentary contact.
- `paired_comparison.csv`: B/C side by side, B-minus-C absolute differences and percentage reductions (blank when B is zero or either metric unavailable).
- `aggregate.json`: task successes, all-scheduled success rates, unavailable outcomes, available metric counts, means/medians, aggregate differences and count of lower-acceleration C pairs.
- `paired_acceleration.png/.svg`, `paired_tilt.png/.svg`.
- `acceleration_overlay.png/.svg`: prespecified first state, even if unfavorable or unsuccessful; unavailable traces omitted with annotation. Time aligned to each condition's actual TRANSPORT entry, not reset.
- `rollout_table.png`, `report.md`, `progress.json`, configuration diff and per-pair matching checks.

Metrics reuse the existing frozen evaluator's intact-grasp TRANSPORT interval, including its PLACE-entry observation boundary. Acceleration is the unsmoothed nonuniform-time central three-point second difference at control-step sampling, excluding post-release motion. No unavailable metric is replaced with zero. Numeric aggregate denominators may differ; inspect available counts and per-pair differences. All-scheduled success rates count only observed true predicates as successes and disclose unavailable outcomes separately.

No local simulations were run. Return the CSVs, aggregate, configuration diff and plots for a final result-based report. Five new pairs remain an exploratory sample; no reliability/generalization claim follows automatically. No tuning is performed after results.
