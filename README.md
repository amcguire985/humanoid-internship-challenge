# Human Demonstration to Robot Manipulation

## Goal

Test whether a small amount of self-collected human manipulation video can be converted into useful supervision for a simulated Franka Panda.

## Proposed Pipeline

Human video  
- estimate hand / object motion  
- express motion in a stable coordinate frame  
- retarget to Panda end-effector motion  
- generate robot-compatible demonstrations  
- adapt or guide a policy  
- evaluate in LIBERO

## Current Approach

Use a phone camera plus fiducial markers to estimate metric 6-DoF hand motion.

Planned setup:
- fixed marker on the table as a world reference
- marker on the wrist / hand
- calibrated phone camera

The first objective is to recover a smooth 3D hand trajectory and replay an equivalent trajectory in LIBERO.

## Current Status

- LIBERO installed and running
- Panda task-space control tested
- simulated camera observations inspected
- next: validate AprilTag-based pose estimation

## Evaluation

Primary metrics will include:
- pose estimation accuracy / jitter
- trajectory replay quality
- task success rate
- performance with and without human-derived data

## Repository

- `PROJECT.md` — requirements, risks, current plan
- `EXPERIMENTS.md` — experiment log and decisions
- `scripts/` — code
- `results/` — plots, images, videos
## Camera calibration

Install dependencies and run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/calibrate_camera.py --video videos/chessboard_calibration.MOV --frames 15 --seed 42 --orientation portrait
.\.venv\Scripts\python.exe scripts/calibrate_camera.py --orientation landscape
```

The board has **6 by 9 inner corners** (7 by 10 squares).
Square size defaults to **0.023 metres** (2.3 cm). Change it with
`--square-size 0.025`, for example, for 2.5 cm squares.
A uniform change to square size changes the estimated translations' metric
scale; the intrinsic matrix and distortion should remain effectively unchanged.

The portrait result now uses 15 random usable frames from
`videos/chessboard_calibration.MOV`, with seed 42. Sampling proceeds without
replacement and skips frames where the board cannot be detected. Video rotation
metadata is applied to match the tracker; sampled and used frame indices are
recorded in JSON. The landscape result retains the previous photo calibration with its original 3 cm square size.
Photo mode ignores EXIF rotation and calibrates each stored orientation separately:

| Output directory under results/ | Resolution (width x height) | Detected views | RMS reprojection error |
|---|---|---|---|
| camera_calibration_portrait | 1080 x 1920 | 15 usable video frames (18 sampled) | 1.2004 px |
| camera_calibration_landscape | 2048 x 1536 | 5 of 6 | 0.9970 px |

IMG_0041.JPEG was not detected in the landscape group. Each directory contains
`calibration.json`, `calibration.npz`, and annotated corner images.
JSON includes per-view errors, intrinsic uncertainty estimates, and board poses.
Translations are in metres. Distortion order is k1, k2, p1, p2, k3.

Load the landscape result for OpenCV:

```python
import cv2
import numpy as np

calibration = np.load("results/camera_calibration_landscape/calibration.npz")
camera_matrix = calibration["camera_matrix"]
distortion = calibration["distortion_coefficients"]
# frame must use the same pixel orientation and resolution as this calibration.
undistorted = cv2.undistort(frame, camera_matrix, distortion)
# These same arrays can be passed to cv2.solvePnP for marker pose estimation.
```

Use the matrix matching the incoming image orientation and resolution. For a
pure resize, scale the first and second rows of the matrix by the width and
height scale factors respectively; distortion coefficients stay the same.
Cropping or rotation requires additional coordinate adjustments. Phone video
can use a different crop, lens, stabilization, or focus than still photos;
calibrate frames from the intended recording mode when using this for video.

The roughly one-pixel training reprojection errors measure the fit to these
photos, not independent pose accuracy. Inspect the corner overlays and validate
with a known marker distance before relying on metric motion estimates.


## Relative AprilTag tracking

Run the supplied video with ID0 as the bench reference and ID1 as the moving tag:

```powershell
.\.venv\Scripts\python.exe scripts/track_apriltags.py --tag-size 0.078
```

The detected family is AprilTag 25h9. Size is the side length of the **outer
black square**, excluding the white margin, in metres. Both tags default to
7.8 cm; change both with `--tag-size`, or set them independently:

```powershell
.\.venv\Scripts\python.exe scripts/track_apriltags.py --reference-size 0.10 --moving-size 0.08
```

Outputs are in `results/test_001_tracking/`:

- `annotated.mp4`: detected IDs, marker coordinate axes, relative position.
- `trajectory.csv`: one row per decoded frame, decoder timestamp in seconds,
  status, detection flags, pose errors, pose ambiguity gaps, position in metres,
  centre-to-centre distance, and relative Rodrigues rotation vector in radians.
- `trajectory.png`: 3D path with ID0 at the origin plus separate x, y, z position plots against time (cm and seconds), with gaps for unavailable poses.
- `preview.jpg`: first frame with a valid relative pose.
- `summary.json`: settings, effective intrinsics, coverage and limitations.

The initial run tracked 396/570 frames (69.5%) across about 19 seconds. ID0
was detected in 490 frames and ID1 in 426. Relative tracking was unavailable
in 165 frames due to missing detections; another 9 failed the default 3 px
reprojection-error threshold. Blank coordinates are unavailable observations,
not zero position. There is no interpolation, temporal smoothing, or reuse of
an old reference pose: both tags must have usable poses in the same frame.

For each tag, OpenCV estimates a transform from marker coordinates into camera
coordinates using `solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE)`. The tracker
retains the positive-depth solution with the lowest reprojection error, then
computes:

```text
T_ID0_ID1 = inverse(T_camera_ID0) @ T_camera_ID1
position_ID1_in_ID0 = R_camera_ID0.T @ (t_camera_ID1 - t_camera_ID0)
```

The origin is ID0's centre. X points from decoded corner 0 toward corner 1;
Y points from corner 3 toward corner 0; Z points out of the printed face.
The video draws X red, Y green, and Z blue. These axes follow the decoded
OpenCV convention, so they rotate with ID0's printed orientation. They are
not necessarily screen-right/up. Rotation columns are a Rodrigues vector,
not Euler angles. The [OpenCV marker documentation](https://docs.opencv.org/doc/doxygen/html/de/d67/group__objdetect__aruco.html)
describes the AprilTag corner convention.

OpenCV applies the MOV's rotation metadata to decoded frames, producing
1080 x 1920 portrait images. The script therefore uses the **portrait**
video-frame calibration from `chessboard_calibration.MOV` (1080 x 1920).
Matching video dimensions use the measured matrix directly. `--calibration-fit center-crop` is
the default; `resize` assumes an independent width/height resize, and
`strict` requires the calibration resolution to match exactly.
Use `--calibration PATH` to supply a calibration measured from video frames.
For recordings with different dimensions, the selected fit model adjusts the
matrix. Validate metric values with the intended video mode and measured tag sizes. Tag-size errors can create apparent height offsets even when tags
lie on the same surface. Single planar tags can also have ambiguous poses:
a small `ambiguity_gap_px` means the two candidate poses have similar errors.
Keep tags flat and check the raw trajectory before using it as robot supervision.

Other options include `--video`, `--output`, `--family`,
`--reference-id`, `--moving-id`, `--max-reprojection-error`, and
`--no-video`. Run `--help` for the full interface.


## Updated test-video tracking

All three runs use the 15-frame video calibration with 2.3 cm chessboard
squares, 7.8 cm AprilTag sides, and strict matching of 1080 x 1920 frames.

| Video | Tracked frames | Output directory |
|---|---|---|
| test_001_static.MOV | 169/169 (100%) | results/test_001_static_tracking |
| test_002_static.MOV | 166/166 (100%) | results/test_002_static_tracking |
| test_003_static_camera_moving.MOV | 324/405 (80%) | results/test_003_static_camera_moving_tracking |

Each output contains annotated.mp4, trajectory.csv, trajectory.png, preview.jpg,
and summary.json. Run with `--tag-size 0.078 --calibration-fit strict`,
`--video videos/VIDEO.MOV`, and a separate `--output` directory.

## Trajectory postprocessing

Run the offline comparison after tracking:

```powershell
.\.venv\Scripts\python.exe scripts/postprocess_trajectory.py results/test_003_static_camera_moving_tracking/trajectory.csv
```

Outputs go into the run's `postprocessed/` directory: `interpolated.csv`,
`ema_0.03s.csv`, `ema_0.05s.csv`, `ema_0.1s.csv`, `metrics.json`, and
`comparison.png`. Raw tracking files are preserved. Each processed CSV includes
source status, observed/interpolated/missing labels, positions in metres, and
unit quaternions in x,y,z,w order describing ID1 relative to ID0.

Defaults fill at most three missing frames, only between observations no more
than 0.15 seconds apart. Position uses linear interpolation and orientation uses
shortest-path SLERP. This is offline processing: filling requires a future
observation. Long gaps and recording ends remain missing. Translation EMA and
orientation SLERP smoothing use alpha = 1 - exp(-dt/tau), and reset after an
unfilled gap or a timestamp jump larger than the allowed span. Start by assessing
`ema_0.05s.csv`; compare motion delay before selecting it for demonstrations.

Options: `--tau 0.05`, `--max-gap-frames 3`, `--max-gap-span 0.15`, and
`--known-distance 0.28` (example only: supply an actual measured tag-centre
distance). A known distance enables distance RMSE; it is not inferred from the
trajectory. Jitter and mean-shift comparisons use the same originally observed
frames. Static-pose orientation jitter is angular RMS around each rotation mean.
Velocity/acceleration metrics include filled frames and exclude unfilled gaps.
Reported holdout errors compare reconstructed short gaps with withheld raw
measurements, not physical ground truth. Smoother trajectories do not establish
better accuracy; validate bias and motion timing on a moving-hand recording.

Validation: `python -m unittest discover -s tests -v`.

## Hand and object tracking (test_004)

```powershell
.\.venv\Scripts\python.exe scripts/track_apriltags.py --video videos/test_004_hand.MOV --family Standard41h12 --tag-size 0.06 --tag-size-convention full-pattern --reference-id 0 --moving-id 1 --object-id 2 --calibration-fit strict --output results/test_004_hand_tracking
.\.venv\Scripts\python.exe scripts/postprocess_trajectory.py results/test_004_hand_tracking/trajectory.csv --tau 0.05
.\.venv\Scripts\python.exe scripts/postprocess_trajectory.py results/test_004_hand_tracking/object/trajectory.csv --tau 0.05
```

All tags have 6 cm full-pattern sides (confirmed by the user). The detected
family is Standard41h12; its pose border spans 5 of the full pattern's 9 cells,
so the solver uses 0.06 * 5/9 = 0.033333 metres. This excludes surrounding
blank paper. Standard41h12 uses pupil-apriltags; Windows also needs
pupil-pthreads-win (both listed in requirements.txt). Size conventions follow
[AprilTag documentation](https://github.com/AprilRobotics/apriltag) and
[the family definition](https://github.com/AprilRobotics/apriltag/blob/master/tagStandard41h12.c).
ID0 is the world frame, ID1 the
hand, and ID2 the object. Hand outputs remain in the run directory; object CSV
and plot are in `object/`. Both have a `postprocessed/` directory containing
short-gap interpolation and 0.05-second EMA/SLERP outputs. Annotated video shows
all accepted tag axes and reports whether the object is tracked or inferred.

When hand and object are observed together, save
`T_hand_object = inverse(T_camera_hand) @ T_camera_object`.
When the object is absent, estimate
`T_world_object = T_world_hand @ T_hand_object`.
This preserves both relative translation and orientation, including the object
moving around the hand as the hand rotates. It requires a previously measured
hand-object transform and current accepted world and hand observations.
The detector cannot distinguish hand occlusion from other causes of absence;
this run applies the user's rigid-carry assumption to absent ID2 detections.
No carry-duration limit is imposed. Direct observations take priority and update
the offset. A detected but rejected object pose clears the carry offset, as does
an accepted object observation without an accepted hand pose.

Raw object CSV status `inferred_hand` distinguishes inferred poses from `tracked`
observations. Processed CSVs preserve that distinction in `pose_source` and
`source_status`; short-gap fills are labeled `interpolated`. The hand and object
are smoothed separately, so the smoothed pair need not maintain an exactly rigid
offset during filter transients. Motion variance in these dynamic recordings is
not a measure of static jitter or physical accuracy.

## Offline Savitzky-Golay comparison

The earlier EMA command and outputs remain available. The offline alternative is:

```powershell
.\.venv\Scripts\python.exe scripts/filter_trajectory.py results/test_004_hand_tracking/trajectory.csv --butterworth-hz 3
.\.venv\Scripts\python.exe scripts/filter_trajectory.py results/test_004_hand_tracking/object/trajectory.csv --butterworth-hz 3
```

It writes `offline_filtered/` beside the input CSV, with `cleaned.csv`,
`interpolated.csv`, `savgol.csv`, optional `butterworth.csv`, `metrics.json`, and
`comparison.png`. Source status, inferred-hand provenance, outlier flags, and
rejection reasons remain explicit. Positions are metres; quaternion order is
x,y,z,w. The annotated tracker video remains the original pose-estimation output.

Processing order:

1. Reject isolated position/orientation excursions only when both adjacent
   speeds exceed their thresholds and the direct neighbor-to-neighbor speed is
   below the threshold. Defaults are 2 m/s and 720 degrees/s, configurable with
   `--max-speed` and `--max-angular-speed`. This conservative single pass does
   not remove sustained wrong poses, endpoint spikes, or all multi-frame bursts.
   It can reject genuine rapid reversals; inspect the red crosses in the plot.
2. Fill bounded gaps up to three frames and 0.15 seconds between endpoints using
   linear position interpolation and shortest-path SLERP. Rejected samples can
   be filled but retain their outlier flags. Long gaps and ends stay missing.
3. Split at remaining gaps or timestamp jumps; resample each segment at an
   approximately uniform median cadence, filter, and restore original timestamps.
4. Apply seven-sample, order-2 Savitzky-Golay position smoothing (`--window`,
   `--order`). Orientation uses the same polynomial weights on relative rotation
   vectors centered at each sample, then composes the result back into a rotation.
   This avoids blindly smoothing absolute Rodrigues values through the pi wrap.
   Rotation windows spanning 150 degrees or more pass through and are logged.
5. Optional `--butterworth-hz 3` compares an order-2 forward-backward low-pass
   filter. Orientation uses continuous-sign quaternion components and subsequent
   normalization. It is an approximate quaternion smoother; SG uses local
   rotation geometry. Short segments pass through when padding is unavailable.

The plot overlays raw, gap-filled, SG, and optional Butterworth positions and
orientation angle from the initial pose. Red crosses identify rejected poses;
orange spans identify filled gaps. Short segments and rotation-window exceptions
are reported in `filter_passthrough_notes`.

Metrics report path length, peak translational speed, maximum positional and
angular deviation from raw (both including and excluding rejected spikes), and
individual interpolated/unfilled gap frame counts and endpoint spans. Path and
speed never connect across long gaps. `common_edge_path_length_m` compares the
same accepted adjacent raw samples, avoiding apparent improvement from reduced
coverage. Path length over available samples and filled gaps is also included.

Stationary jitter is reported only for explicitly supplied intervals, e.g.
`--stationary 0 5` (repeatable). Use the whole clip only when relative pose is
known to be static. Each interval reports per-axis position standard deviation
and angular RMS about its rotation mean, using the same accepted samples across
methods. No stationary interval is assumed for test_004. These metrics quantify
smoothness and changes to the signal, not absolute accuracy. Check motion timing,
excursions, and endpoint artifacts before using trajectories for robot replay.

Filter APIs: [SciPy Savitzky-Golay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)
and [forward-backward SOS filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).

## Raw multi-tag tracking (test_005)

```powershell
.\.venv\Scripts\python.exe scripts/track_apriltags.py --multi-tag-raw --video videos/test_005_more_tags.MOV --family Standard41h12 --tag-size-convention full-pattern --world-size 0.060 --body-size 0.040 --calibration-fit strict --output results/test_005_more_tags_raw
```

World tags 0-3 are stationary, cube face tags are 4-6, and wrist tags are 7-8.
This command retains the earlier Standard41h12 full-pattern size convention:
60 mm world patterns and 40 mm moving patterns correspond to 33.333 mm and
22.222 mm pose borders. If the supplied sizes measure the pose border instead,
use `--tag-size-convention border`.

`detections.csv` retains every detected instance and its raw camera pose,
reprojection error, ambiguity gap, and world pose when a reference is available.
Individual trajectories and plots are under `object/id4` through `object/id6`
and `hand/id7` through `hand/id8`. They contain one row per frame with blank
coordinates for missing poses. No reprojection-error threshold, temporal filter,
interpolation, or hand-based object inference is applied. Positive-depth PnP
solutions still must exist for a usable pose. An annotated video and summary
are included. Existing results are preserved.

Stationary registration uses the first co-observation of each world-tag pair,
then connects the resulting transform graph to ID0. This fixed spatial
registration is separate from trajectory filtering. Each frame uses the visible
registered world tag with the lowest reprojection error; no old camera pose is
reused. Registration errors, planar ambiguity and reference switches can cause
raw trajectory jumps. Disconnected world tags are reported in the summary.

Cube plots describe individual face-tag centres, not the cube centre. Cube edge
length, mounting offsets, decoded tag rotations and an opposite-face association
are needed for a common cube pose. Repeated IDs cannot identify opposite faces;
simultaneous duplicates remain separate in `detections.csv` and are omitted from
the single-tag trajectory. Wrist tags likewise retain separate frames; curved
printing violates the planar pose model and can cause errors even for detections
with small reprojection residuals.

### 45 mm cube centre

Add `--cube-edge 0.045` to the raw multi-tag command to export
`object/cube_center/trajectory.csv` and `trajectory.png`. ID5 is initially on top,
ID4 initially faces the camera, and ID6 is on the third orthogonal face. For each
current face observation, the centre is 22.5 mm inward along the decoded tag
normal. This assumes each tag is centred and flush on its face. It also works
for opposite faces sharing an ID because their normals point outward locally.

The CSV selects the current face observation with the lowest reprojection error;
it performs no averaging, interpolation, rejection threshold or temporal
filtering. Source ID/instance, candidate count and maximum disagreement between
face-derived centres are logged. Individual face-tag outputs remain unchanged.
Cube orientation is omitted because repeated opposite-face IDs and unspecified
in-plane mounting rotations do not define a unique common orientation.

### Hand centre on a 45 mm cube

Raw multi-tag tracking now also exports `hand/hand_center/trajectory.csv` and
`trajectory.png`. Tags 7 and 8 are treated as centred, flush face tags on a
45 mm wrist cube (`--hand-edge 0.045`, the default). Each observation moves
22.5 mm inward along its outward tag normal. As for the object cube, the current
lowest-reprojection-error candidate supplies the position. Missing frames stay
blank; no temporal filtering, gap filling or averaging is applied. Source tag,
candidate count and cross-face disagreement are logged. These are centre
positions only; a common hand orientation requires mounting rotations. Existing
per-tag trajectories remain available. Curved tags can still bias planar poses
and therefore the centre estimates.

### Centre trajectories: gap filling without smoothing

```powershell
.\.venv\Scripts\python.exe scripts/gap_fill_centers.py results/test_006_slower_raw/hand/hand_center/trajectory.csv --max-gap-frames 10
.\.venv\Scripts\python.exe scripts/gap_fill_centers.py results/test_006_slower_raw/object/cube_center/trajectory.csv --max-gap-frames 10
```

Each centre's `gap_filled_10_frames/` directory contains `interpolated.csv`,
`trajectory.png`, `comparison.png`, and `metrics.json`. Bounded gaps containing
at most 10 missing frames are filled by linear interpolation using decoder
timestamps. The default uses only the frame limit, allowing the full 10-frame
span; `--max-gap-span SECONDS` adds an optional endpoint time cap. Long gaps and
leading/trailing gaps stay blank. Observed positions stay unchanged, and
`source_status` plus `pose_source` distinguish filled samples. No smoothing,
outlier rejection or orientation interpolation is performed. Raw CSVs remain
unchanged; this command reuses detections without rerunning video decoding.

### Gentle centre smoothing comparison

```powershell
.\.venv\Scripts\python.exe scripts/compare_center_smoothing.py results/test_006_slower_raw/hand/hand_center/gap_filled_10_frames/interpolated.csv
.\.venv\Scripts\python.exe scripts/compare_center_smoothing.py results/test_006_slower_raw/object/cube_center/gap_filled_10_frames/interpolated.csv
```

Each input's `gentle_smoothing/` folder contains full-clip `comparison.png`, a
three-second `detail.png`, three filtered CSVs, and `metrics.json`. Four columns
compare the gap-filled baseline, five-frame quadratic SG, seven-frame quadratic
SG, and a causal EMA with a 0.03-second time constant. SG fits local quadratic
polynomials using actual timestamps (equivalent to standard Savitzky-Golay for
uniform timestamps). Short segments pass through when insufficient; endpoint
windows are asymmetric. EMA resets at missing frames. All filters stop across
unfilled gaps or timestamp jumps over three median frame intervals. No new gap
filling or outlier rejection occurs; observed/interpolated provenance is retained.
The detail view centres on the largest available second position difference;
this is a reproducible inspection interval, not an outlier classification.
Metrics measure positional change from the gap-filled baseline, not accuracy.

### Gentle centre smoothing

```powershell
.\.venv\Scripts\python.exe scripts/smooth_centers.py results/test_006_slower_raw/hand/hand_center/gap_filled_10_frames/interpolated.csv
.\.venv\Scripts\python.exe scripts/smooth_centers.py results/test_006_slower_raw/object/cube_center/gap_filled_10_frames/interpolated.csv
```

`gentle_smoothing/` contains 5- and 7-sample quadratic Savitzky-Golay CSVs,
0.03- and 0.05-second EMA CSVs, a combined comparison, individual trajectory
plots and displacement metrics. Start with `savgol_5.csv` for light offline
smoothing. SG uses actual decoder times in a local quadratic fit; interior
5/7-frame windows span approximately 0.13/0.20 seconds at 30 fps. Segment edges
use shifted windows and segments shorter than the window pass through. EMA is
causal smoothing with lag, though the preceding interpolation is offline.
Filters reset at missing samples or adjacent timestamp jumps above 0.15 seconds.
No extra interpolation or outlier rejection occurs. Source labels remain
unchanged; `smoothing_method` records the applied filter. Output positions are
smoothed even when their provenance is observed. Changes from the gap-filled
input are reported in mm, without interpreting them as accuracy or static jitter.

### Interpolation only, up to 10 missing frames (test_006)

```powershell
.\.venv\Scripts\python.exe scripts/postprocess_trajectory.py results/test_006_slower_raw/object/cube_center/trajectory.csv --no-smoothing --max-gap-frames 10 --max-gap-span 0.4 --output results/test_006_slower_raw/object/cube_center/gapfilled_10_frames
.\.venv\Scripts\python.exe scripts/postprocess_trajectory.py results/test_006_slower_raw/hand/hand_center/trajectory.csv --no-smoothing --max-gap-frames 10 --max-gap-span 0.4 --output results/test_006_slower_raw/hand/hand_center/gapfilled_10_frames
```

These position-only outputs contain `interpolated.csv`, `comparison.png` and
`metrics.json`. Bounded gaps of at most 10 missing frames and at most 0.4 seconds
between observed endpoints are filled linearly. No smoothing or outlier rejection
is applied; observed positions remain exactly unchanged. Longer gaps and clip
ends stay blank. `pose_source` distinguishes observed, interpolated and missing
samples; interpolated rows have no source tag identity. Raw inputs are preserved.

### Video with selected SG7 centre overlays

```powershell
.\.venv\Scripts\python.exe scripts/annotate_smoothed_centers.py results/test_006_slower_raw
```

`annotated_sg7_gap10.mp4` preserves raw detected tag outlines and axes and adds
SG7 centre crosses, labelled hand/cube positions, and 15-frame world-space trails
projected using the current camera pose. Thin circles identify gap-filled
samples; unavailable centres are not drawn. The video uses the previously
exported seven-frame quadratic SG results after up-to-ten-frame gap filling.
The original annotated video is preserved. `annotated_sg7_gap10.json` records
frame alignment and overlay counts. Centre overlays do not represent smoothed
object/hand orientation, which remains unspecified by the mounting geometry.

### Annotated SG7 centre video

```powershell
.\.venv\Scripts\python.exe scripts/annotate_smoothed_centers.py results/test_006_slower_raw
```

Writes `sg7_gap_filled_annotated.mp4`, a preview JPEG and overlay metadata JSON.
The video projects the seven-frame SG hand/cube centres through the original
intrinsics and distortion using the current frame's registered world-tag camera
pose. Hand is yellow, cube cyan; filled circles mark observed samples and rings
mark interpolated samples. One-second trails are reprojected into the current
camera and reset at unfilled gaps. The overlay includes world-frame coordinates
and source labels. Frame indexing and source frame count must match the CSVs.
The MP4 is silent and retains the source nominal frame rate; CSV decoder
timestamps drive displayed time and trail duration.

## Learning the code

See [the function guide](docs/FUNCTION_GUIDE.md) for every script function,
coordinate conventions, the current raw-to-SG7-video pipeline, test coverage,
and the cleanup decisions. Position interpolation now has one shared algorithm
in `gap_fill_centers.py`; centre smoothing has one in `smooth_centers.py`.
Earlier CLI interfaces remain supported through small compatibility wrappers.

## Object motion as a task-space description

```powershell
.\.venv\Scripts\python.exe scripts/describe_object_trajectory.py results/test_006_slower_raw/object/cube_center/gap_filled_10_frames/gentle_smoothing/savgol_7.csv
```

Accepts `time,x,y,z` or the existing `time_s,x_m,y_m,z_m` format in seconds
and metres. Outputs in `task_space/`: `trajectory.csv` with
`time,x_rel,y_rel,z_rel,speed,phase`, `metrics.json`, `raw_positions.png`,
`trajectory_3d.png`, and `speed_phases.png`. Input positions are plotted before
normalization; no additional smoothing or interpolation is applied.

Tune `--stationary-speed 0.03`, `--vertical-speed 0.03` (m/s) and
`--min-motion-duration 0.2` (seconds). Default vertical is z; change with
`--vertical-axis x|y|z` if your world frame requires it. Movement sustained for
at least the duration threshold defines approximate task bounds. Inside those
bounds, positive/negative vertical velocity labels lift/lower; other movement
is horizontal transport. Quiet internal samples are stationary during transport.
Pickup/placement remain null when no adjacent quiet observation confirms a
transition. This is a threshold description, not contact/grasp detection.

Unavailable rows are skipped and create segment breaks. Velocity and path never
connect across those breaks or time gaps exceeding three median sample intervals.
Duration covers the entire recording interval. Horizontal displacement is net
start-to-end distance; horizontal path length is also reported. The CSV keeps
original time coordinates so it aligns with the recording. No robot control or
LIBERO coordinate mapping is performed.

The small functions in `describe_object_trajectory.py` are:

- `load_trajectory`: parse either format, validate ordering, and record gaps.
- `normalize_trajectory`: subtract the first valid position.
- `compute_velocity`: timestamp-aware finite differences within each segment.
- `motion_runs`: group consecutive labels without crossing gaps.
- `segment_motion`: apply speed/vertical thresholds and estimate event times.
- `compute_metrics`: calculate net displacement, lift, path, duration and endpoints.
- `plot_results`: render input coordinates, relative 3D path and phase-marked speed.
- `main`: connect the steps, expose tuning options, and export CSV/JSON.


## World and target tracking (test_007)

```powershell
.\.venv\Scripts\python.exe scripts/track_apriltags.py --multi-tag-raw --video videos/test_007_block_only.MOV --calibration results/camera_calibration_landscape/calibration.json --family Standard41h12 --tag-size-convention full-pattern --world-ids 0 1 --target-id 2 --hand-ids --world-size 0.060 --body-size 0.040 --cube-edge 0.045 --calibration-fit center-crop --output results/test_007_block_only_raw
```

IDs 0 and 1 supply the stationary world reference; ID0 remains the origin.
ID2 is the target and never supplies a camera/world reference. Object IDs 4?6
and the 45 mm cube geometry are retained. No wrist tags are requested.
Target trajectory and plot are in `target/id2/`; object face trajectories are
in `object/id4`?`object/id6`, and the object centre is in `object/cube_center/`.
All outputs are raw, with missing observations left blank.

This run assumes the previous 60 mm full-pattern world size also applies to
the target (override with `--target-size`) and retains 40 mm object patterns.
The new video is landscape 1920?1080. It uses the existing landscape still-photo
calibration with a center-crop model, rather than the portrait video calibration.
Validate distances or calibrate the landscape recording mode before treating
these coordinates as accurate metric supervision.


## Task-level human demonstration extraction

The current processing objective is an object/goal-relative task description for
later retargeting. It requires no hand or wrist observations. Run AprilTag tracking
with the command above, then:

```powershell
.\.venv\Scripts\python.exe scripts/process_task_demo.py results/test_007_block_only_raw --config config/test_007_demo.json
```

Outputs in `results/test_007_block_only_raw/task_demo/`:

- `processed_demo.csv`: every source frame, world object centre, positions relative
  to start and goal, speed, phase, normalized transport progress, raw coordinates,
  measurement validity, processed validity, interpolation and rejection flags.
- `raw_trajectory.csv`: unchanged measured positions; `cleaned_trajectory.csv`:
  accepted measurements before interpolation/smoothing.
- `transport.csv`: all frames between approximate pickup and release, including
  invalid rows. Its progress field is time-normalized from `transport_start` to
  `transport_end`, the inner phase between initial lift and final lowering.
- `metadata.json`: positions, displacement, timestamps, inner transport and full
  pickup-to-release metrics, detection rates, retained coverage, missing gaps,
  fixed-goal scatter, review flags, parameters and tracking provenance.
- `trajectory_3d.png`, `position_time.png`, `speed_phases.png`, `top_down.png`.

`process_task_demo.py` reuses `filter_trajectory.reject_spikes` for isolated
position excursions (no object orientation is invented), `fill_position_gaps`
for bounded interpolation, `smooth_positions` for SG smoothing, and existing
velocity/run helpers for segmentation. Defaults reject object-centre samples
above 3 px reprojection error, reject isolated spikes above 2 m/s, fill at most
3 missing frames with at most 0.15 s between endpoints, and apply a seven-sample
quadratic fit. Unfilled gaps and timestamp jumps split filters, velocities and
paths. The raw cached tracking files are preserved. World registration retains
the original tracker's fixed co-observation estimate and current reference choice;
this processing step cannot correct a biased reference or sustained wrong pose.

World axes follow ID0's printed orientation: X corner0?corner1, Y corner3?corner0,
Z out of the tag face. The new preview suggests ID0 is flat on the table, but
there is no gravity calibration. `vertical_axis`, `vertical_sign` and
`vertical_verified` keep height assumptions explicit. The supplied landscape
still-photo calibration is also flagged as unverified for this recording mode.

Edit `config/test_007_demo.json` to change `goal_object_offset`, which is in metres
in **goal-tag coordinates**. Each observation uses
`p_goal_tag + R_world_goal @ goal_object_offset`; a componentwise median forms a
fixed offline goal anchor. The goal is assumed stationary and needs accepted
world-relative poses somewhere in the clip. Missing goal observations are not
synthesized: detection gaps and goal scatter remain in metadata. The goal area is a 12 cm diameter circle, offset `[-0.09,0.09,0]` metres in
goal-tag coordinates. `goal_coplanar: true` projects measured tag poses to world
z=0 with yaw-only rotation, removing spurious tilt/height from the fixed goal.
The anchor uses pre-pickup measurements, with stationary-goal observations during
manipulation as a fallback if pre-pickup detection is unavailable. It is fixed
throughout processing. `goal_center_height: 0.0225` separately specifies the
desired centre height for the existing 45 mm cube. Circle membership tests the
projected object centre, without requiring its whole footprint to fit. Planar
placement error and height error are reported separately; being inside the
circle alone does not establish release or tabletop contact. `goal_offset_verified`
is true for the user-specified surface geometry; gravity and calibration remain
unverified. Both 3D and top-down plots show the circle.

Phase heuristics use sustained speed above `stationary_speed`, signed height
velocity above `vertical_speed`, and stationary brackets of `stationary_duration`.
The initial lift uses the early quarter of the detected motion interval; final
lowering uses its latter half. Goal proximity supports placement confidence;
lowering away from the goal remains labelled but is flagged. Quiet samples
outside the motion interval are pre/post phases. Pickup/release denote motion
transitions, not observed contact. Uncertain estimates remain available with
review flags. Override any bounds explicitly, for example:

```json
"phase_overrides": {
  "pickup_time": 3.8,
  "transport_start": 5.0,
  "transport_end": 8.0,
  "release_time": 9.0
}
```

Overrides must be ordered and within the recording. Setting a bound to null
marks it unavailable. `valid_measurement` describes accepted raw evidence;
`valid_processed` describes available filtered positions. An interpolated sample
may be valid for processing without being a measurement. All distances are
metres and times seconds. Review flags deliberately prevent the demonstration
being labelled usable for retargeting until geometry/calibration, coverage and
phase confidence have been checked. No robot control or learning is included.


## LIBERO object-motion following test

Run `python scripts/replay_libero_transport.py` in the LIBERO environment to
follow the processed test_006 object motion in the exploration scene. Edit
`config/libero_transport.json` for axis permutation, signs and scale. Results,
per-step actions and tracking plots go to `results/libero_transport/`.
See [experiment details](docs/LIBERO_TRAJECTORY.md) for source/frame conventions,
controller scaling, gap handling and interpretation.
