# Controlled LIBERO XY trials — stopped at user request

Trials 1–5 are fully saved and all five succeeded. They cover the nominal start,
+2 cm X, -2 cm X, +2 cm Y, and -2 cm Y. Each trial directory includes metrics,
per-step CSV/JSONL logs, the retargeted trajectory, initial-state arrays and
invariance audit, phase snapshots, and diagnostic plots.

Trial 6 also completed successfully before cancellation. Trial 7 had started
and was interrupted; its partial logs are preserved and must not be counted as
a completed trial. Trials 8–10 were not run. No trials were retried.

See [summary.csv](summary.csv) for completed-trial metrics and
[preflight.json](preflight.json) for validation of all ten planned offsets.
The planned +4 cm X offset was reduced to +2.50671875 cm due to plate collision,
but that trial was not run. The original ten-trial protocol is documented in
[LIBERO_XY_TRIALS.md](../../docs/LIBERO_XY_TRIALS.md).
