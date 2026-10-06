# data_002 transfer extraction

Seven complete transfers confirmed in video. 0 pass the tracking/segmentation checks; 7 are rejected. Calibration/crop and height-axis validation remain pending.

| Demo | Direction | Pickup-release (s) | Distance (cm) | XY heading (deg) | Observed lift (cm) | Observed path (cm) | Missing carry frames | Longest gap (s) | Decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| [demo_001](demo_001/diagnostic.png) | object -> target | 3.84-12.51 | 22.0 | -110.2 | 12.9 | 50.1 | 31 | 0.57 | Reject |
| [demo_002](demo_002/diagnostic.png) | target -> object | 14.21-21.01 | 25.7 | -51.9 | 12.4 | 32.1 | 125 | 3.94 | Reject |
| [demo_003](demo_003/diagnostic.png) | object -> target | 26.21-34.98 | 27.2 | 140.7 | 11.8 | 28.1 | 143 | 3.70 | Reject |
| [demo_004](demo_004/diagnostic.png) | target -> object | 41.48-51.59 | 22.0 | 87.5 | 22.3 | 37.5 | 123 | 1.40 | Reject |
| [demo_005](demo_005/diagnostic.png) | object -> target | 56.29-65.00 | 23.2 | -102.9 | 3.4 | 7.5 | 243 | 8.10 | Reject |
| [demo_006](demo_006/diagnostic.png) | target -> object | 67.80-77.00 | 49.5 | 41.0 | 18.0 | 33.4 | 167 | 5.34 | Reject |
| [demo_007](demo_007/diagnostic.png) | object -> target | 79.70-90.21 | 43.9 | -134.9 | 16.6 | 82.6 | 93 | 0.77 | Reject |

All lift values are maxima over available observations, not verified maximum physical lift. Gaps can hide the peak; paths include only valid adjacent edges and are partial when tracking is missing. Distances use robust measured resting endpoints; the fixed target-tag anchor is stored separately. XY headings use ID0 world axes.

Automatic segmentation did not recover all seven transfers because of missing observations and pose jitter. Explicit split windows/directions were checked using the 2-second video contact sheets and a denser 1-second placement review. The final transfer includes visible release before the clip ends. Pickup/release annotations are approximate motion/video brackets, not grasp/contact ground truth. Inner transport boundaries retain automatic height heuristics and remain easy to override.

Each demo directory contains metadata.json, processed_demo.csv (authoritative smoothed coordinates/validity), raw_trajectory.csv, cleaned_trajectory.csv (pre-interpolation accepted measurements), transport.csv (pickup-through-release including missing samples), normalized_task_trajectory.csv, and diagnostic.png. Normalization translates the start to the origin, aligns longitudinal XY toward the measured end, uses lateral-left and signed world-Z axes, and divides lengths by planar displacement. No endpoint correction is applied.

Raw tracking: ../data_002_raw. Whole-video diagnostics: full_overview.png, video_contact_sheet_1.jpg, video_contact_sheet_2.jpg, and placement_review.jpg. Per-demo timing overrides and endpoint windows: ../../config/data_002_demos.json.

Reproduce exports:

```powershell
.\.venv\Scripts\python.exe scripts/extract_transfer_demos.py results/data_002_raw --config config/data_002_demos.json --output results/data_002_demos
```

Tracking used the same Standard41h12 configuration as data_001: world 0/1, target 2, object 4/5/6, 60 mm full-pattern stationary tags, 40 mm object tags, 45 mm cube, landscape calibration with center-crop. Cleaning reused existing reprojection/spike rejection, at most 3-frame/0.15-second bounded interpolation and SG7 smoothing within continuous segments. Large gaps remain missing.

Validation: CSV lengths and normalized validity match; missing carry counts and path sums were independently recomputed without crossing gaps; all required artifacts exist. No LIBERO execution or policy training was performed.

demo_001: Unfilled tracking gap during pickup-to-release; reject for retargeting.

demo_002: Unfilled tracking gap during pickup-to-release; reject for retargeting.

demo_003: Unfilled tracking gap during pickup-to-release; reject for retargeting.

demo_004: Unfilled tracking gap during pickup-to-release; reject for retargeting.

demo_005: Unfilled tracking gap during pickup-to-release; reject for retargeting.

demo_006: Unfilled tracking gap during pickup-to-release; reject for retargeting.

demo_007: Unfilled tracking gap during pickup-to-release; reject for retargeting.
