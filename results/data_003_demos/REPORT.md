# data_003 transfer extraction

Three video-complete transfers. 3 pass tracking/segmentation checks; 0 are rejected. Calibration/crop, vertical direction and phase/contact semantics remain pending review.

Raw and cleaned full-video coverage: 1729/1729 (100.0%) raw, 1729/1729 (100.0%) cleaned. Cleaning rejected 0 observations.

| Demo | Direction | Pickup-release (s) | Distance (cm) | XY heading (deg) | Observed max lift (cm) | Path (cm) | Missing carry frames | Decision |
|---|---|---|---:|---:|---:|---:|---:|---|
| [demo_001](demo_001/diagnostic.png) | object -> target | 5.50-19.21 | 28.1 | -121.8 | 21.6 | 147.7 | 0 | Gap-free candidate |
| [demo_002](demo_002/diagnostic.png) | target -> object | 25.61-36.78 | 30.3 | 88.9 | 12.9 | 124.6 | 0 | Gap-free candidate |
| [demo_003](demo_003/diagnostic.png) | object -> target | 42.59-56.59 | 30.0 | -88.9 | 18.9 | 217.9 | 0 | Gap-free candidate |

Distances use robust measured resting start/end positions. The fixed target-tag goal anchor is stored separately. XY headings use the existing ID0 world frame. Maximum lift is measured signed world-Z displacement relative to the resting start, not a verified gravity height. Pose jitter, particularly while lowering/at rest, contributes to path length; full coverage does not establish metric accuracy.

The clip contains three alternating completed transfers including the final release. Automatic stationary segmentation merged them because resting position jitter exceeds the stationary-speed threshold. The saved configuration records explicit video-reviewed splits/directions and resting endpoint medians. Pickup/release are approximate motion/video annotations, not measured contact. Inner transport boundaries retain automatic height heuristics, may omit late lowering/holding, and need review before robot use. Full pickup-through-release trajectories are preserved.

Each demo directory contains metadata.json, processed_demo.csv (authoritative smoothed coordinates with validity/provenance), transport.csv (pickup-through-release), raw_trajectory.csv, cleaned_trajectory.csv (pre-interpolation accepted measurements), normalized_task_trajectory.csv, and diagnostic.png. Normalized coordinates translate the start to the origin, align longitudinal XY with measured task displacement, use lateral-left and signed world-Z axes, and divide lengths by planar task distance. No endpoint correction or invented motion.

Review images: [whole-video positions](full_overview.png), [video contact sheet](video_contact_sheet_1.jpg), [placement/release review](placement_review.jpg). Original [annotated tracking video](../data_003_raw/annotated.mp4). Timing overrides: ../../config/data_003_demos.json.

Reproduce:

```powershell
.\.venv\Scripts\python.exe scripts/extract_transfer_demos.py results/data_003_raw --config config/data_003_demos.json --output results/data_003_demos
```

Unchanged tracker configuration: Standard41h12 world 0/1, target 2, object 4/5/6; 60 mm world/target and 40 mm object full-pattern tags; 45 mm cube; landscape calibration with center-crop. Cleaning uses existing reprojection/spike rejection, bounded short-gap interpolation (3 frames / 0.15-second endpoint span), and SG7 smoothing within continuous segments.

Validation: independent export audit checked artifact presence, matching CSV/normalization validity, missing carry counts and path sums without crossing gaps. No LIBERO execution or policy training. All candidates remain behind the inherited calibration/height review gate.
