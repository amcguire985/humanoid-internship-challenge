# data_001: seven video-complete transfers

Seven complete transfers were confirmed by video review. Only demo_005 has complete cleaned pickup-to-release coverage; the other six are rejected for retargeting. All remain behind the inherited calibration/height review gate.

| Demo | Direction | Carry interval (s) | Distance (cm) | XY direction (deg) | Maximum observed lift (cm) | Observed path (cm) | Missing carry frames | Longest gap (s) | Decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| [demo_001](demo_001/diagnostic.png) | object -> target | 4.14-11.01 | 31.9 | -125.2 | 14.7 | 35.0 | 116 | 2.20 | Reject: tracking gaps |
| [demo_002](demo_002/diagnostic.png) | target -> object | 17.21-24.51 | 28.6 | 110.8 | 13.8 | 79.4 | 57 | 1.27 | Reject: tracking gaps |
| [demo_003](demo_003/diagnostic.png) | object -> target | 28.71-36.22 | 28.7 | -78.0 | 29.5 | 84.3 | 62 | 1.60 | Reject: tracking gaps |
| [demo_004](demo_004/diagnostic.png) | target -> object | 42.82-48.16 | 39.0 | 37.2 | 14.6 | 64.0 | 24 | 0.53 | Reject: tracking gaps |
| [demo_005](demo_005/diagnostic.png) | object -> target | 52.53-57.46 | 36.6 | -139.8 | 13.7 | 69.0 | 0 | 0.00 | Candidate: geometry/timing review |
| [demo_006](demo_006/diagnostic.png) | target -> object | 63.40-68.00 | 24.5 | 152.4 | 7.6 | 58.8 | 9 | 0.30 | Reject: tracking gaps |
| [demo_007](demo_007/diagnostic.png) | object -> target | 72.93-78.57 | 22.1 | -29.9 | 16.4 | 76.2 | 19 | 0.43 | Reject: tracking gaps |

Distances use robust measured resting endpoints; target-tag anchor is stored separately. Directions are in the existing ID0 world frame. Lift is signed world-Z relative to the measured start; its gravity alignment is unverified. Rejected trajectories have partial paths and observed lift maxima, which can underestimate unseen motion.

Automatic segmentation merged the first three transfers and failed to identify some target endpoints. Video review supplies seven explicit windows/directions. Inner transport phase estimates remain automatic height heuristics and need review; pickup/release are approximate motion brackets, not contact detection. Edit ../../config/data_001_demos.json and rerun the documented extractor.

Outputs preserve missing rows and never connect a path across them. Normalized coordinates preserve those same missing rows. Interpolation is limited to 3 missing frames / 0.15-second endpoint spans. No LIBERO execution or policy training was performed.

Validation: all seven CSV/normalization exports audited for matching validity, missing carry counts, no-gap path sums, and required artifacts. Three new splitter tests and ten existing task-processing tests passed.
