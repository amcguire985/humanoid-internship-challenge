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
.\.venv\Scripts\python.exe scripts/calibrate_camera.py
.\.venv\Scripts\python.exe scripts/calibrate_camera.py --orientation landscape
```

The board has **6 by 9 inner corners** (7 by 10 squares).
Square size defaults to **0.03 metres**. Change it with
`--square-size 0.025`, for example, for 2.5 cm squares.
A uniform change to square size changes the estimated translations' metric
scale; the intrinsic matrix and distortion should remain effectively unchanged.

Photos have two stored orientations, so the script calibrates them separately.
It ignores EXIF rotation and uses stored pixel coordinates throughout:

| Output directory under results/ | Resolution (width x height) | Detected views | RMS reprojection error |
|---|---|---|---|
| camera_calibration_portrait | 1536 x 2048 | 8 of 8 | 1.1472 px |
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
.\.venv\Scripts\python.exe scripts/track_apriltags.py --tag-size 0.10
```

The detected family is AprilTag 25h9. Size is the side length of the **outer
black square**, excluding the white margin, in metres. Both tags default to
10 cm; change both with `--tag-size`, or set them independently:

```powershell
.\.venv\Scripts\python.exe scripts/track_apriltags.py --reference-size 0.10 --moving-size 0.08
```

Outputs are in `results/test_001_tracking/`:

- `annotated.mp4`: detected IDs, marker coordinate axes, relative position.
- `trajectory.csv`: one row per decoded frame, decoder timestamp in seconds,
  status, detection flags, pose errors, pose ambiguity gaps, position in metres,
  centre-to-centre distance, and relative Rodrigues rotation vector in radians.
- `trajectory.png`: single 3D path with ID0 at the origin, with gaps.
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
still-photo calibration, with an assumed centred crop followed by a uniform
resize to match the video aspect ratio. `--calibration-fit center-crop` is
the default; `resize` assumes an independent width/height resize, and
`strict` requires the calibration resolution to match exactly.
Use `--calibration PATH` to supply a calibration measured from video frames.
The assumed crop cannot establish the actual video intrinsics: metric values
remain provisional until validated with the intended video mode and measured
tag sizes. Tag-size errors can create apparent height offsets even when tags
lie on the same surface. Single planar tags can also have ambiguous poses:
a small `ambiguity_gap_px` means the two candidate poses have similar errors.
Keep tags flat and check the raw trajectory before using it as robot supervision.

Other options include `--video`, `--output`, `--family`,
`--reference-id`, `--moving-id`, `--max-reprojection-error`, and
`--no-video`. Run `--help` for the full interface.

