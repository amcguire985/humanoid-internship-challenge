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

## 2026-10-04 ? test_007 task-level demonstration

Processed cached test_007 AprilTag observations with world IDs 0/1, object IDs
4?6 and goal ID2, without wrist tracking. `process_task_demo.py` now exports
raw/cleaned trajectories, start/goal-relative coordinates, carrying interval,
threshold phase estimates, normalized progress, metrics and diagnostic plots.
Goal-local offset is explicit in `config/test_007_demo.json`; the initial
[0,0,22.5 mm] offset, +Z height and landscape photo calibration remain unverified.

World/object detection coverage: 100%; goal: 82.7%. All 399 object frames retained;
no spikes rejected or gaps filled. Goal observations have 5.1 mm P95 scatter;
longest goal detection gap is 50 frames (1.70 s endpoint span). Estimated pickup
3.77 s, release/motion end 8.11 s (uncertain quiet bracket). Start-to-goal distance
34.4 cm, final placement error 13.8 cm, signed-axis peak lift 7.9 cm. The run is
flagged for review before retargeting due to geometry/calibration assumptions,
phase uncertainty and final position outside the configured 5 cm goal radius.
40 tests pass, including rotated offset and invalid-gap export regression tests.


### Confirmed circular goal geometry

The user defined a fixed, world-coplanar 12 cm diameter region centred at
[-90, +90, 0] mm in goal-tag coordinates. Coplanar processing now projects
measured goal poses to yaw and XY with world z=0, and locks a pre-pickup
anchor (stationary-goal observations during manipulation are a fallback).
The existing 45 mm cube's desired centre height remains separately 22.5 mm.
Circle membership is based on the projected object centre, with no footprint
or contact claim. Regenerated test_007: final planar centre error 49.7 mm,
inside the 60 mm radius; inferred centre-height error 23.2 mm is reported
separately. The prior outside-goal flag is cleared. Calibration/gravity and
release-confidence flags remain. All 42 tests pass.
