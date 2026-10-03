# Experiment Log

| ID | Question | Method | Metric | Result | Decision |
|---|---|---|---|---|---|
| E01 | Can LIBERO run reliably? | Launch task, reset, step environment | Successful reset / step | Pass | Use LIBERO |
| E02 | Does task-space input move the Panda predictably? | Apply Cartesian commands and inspect EE pose | Change in `robot0_eef_pos` | Pass | Use task-space action interface |
| E03 | Can fiducial markers recover stable metric hand pose? | Fixed world tag + moving hand tag | Position error / jitter | Planned | TBD |
| E04 | Does world-frame normalization reduce camera-motion error? | Compare camera-frame vs world-frame trajectories | Stationary pose variance | Planned | TBD |
| E05 | Can a human-derived trajectory drive the Panda? | Retarget and replay in LIBERO | Tracking error / task progress | Planned | TBD |

## Notes

Only add detail here when an experiment changes the design or produces a useful result.

Example:

**E03**
- 70 mm tag at ~0.7 m distance
- stationary jitter: ...
- known 100 mm motion measured as ...
- decision: ...