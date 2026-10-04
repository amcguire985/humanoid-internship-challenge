# Function guide

The current pipeline is calibration -> raw tag poses -> world registration -> hand/cube centres -> gap filling -> SG7 smoothing -> annotated video.

CSV positions use metres; plots usually use centimetres. A 4x4 transform maps coordinates between frames: the upper-left 3x3 block rotates, and the last column translates. `T_camera_tag` maps tag coordinates into camera coordinates. Rodrigues rotation vectors are not Euler angles.

For centred, flush tags on a 45 mm cube, the centre is `tag_position - tag_rotation[:, 2] * 0.0225`. This uses the tag's outward normal. Centre outputs contain positions only: the mounting geometry does not yet specify a common body orientation.

## Calibration: scripts/calibrate_camera.py

| Function | What it does |
|---|---|
| `main` | Reads chessboard images or random video frames, detects corners, estimates camera intrinsics and distortion, and saves calibration files and corner previews. |
| `main.video_frames` | Yields randomly selected decoded frames and releases the video capture when sampling ends. |

## Tracking geometry: scripts/track_apriltags.py

| Function | What it does |
|---|---|
| `fit_intrinsics` | Adjusts the camera matrix for strict resolution matching, resizing, or a centred crop. Distortion coefficients stay unchanged. |
| `Standard41Detector.__init__` | Loads the Standard41h12 detector and its Windows pthread DLL when required. |
| `Standard41Detector.detectMarkers` | Detects tag IDs and reorders corners into the OpenCV pose-estimation convention. |
| `marker_points` | Builds the square's four physical corners in metres, in IPPE order. |
| `estimate_pose` | Solves planar PnP, discards solutions behind the camera, and returns the lowest-error pose, reprojection error and ambiguity gap. |
| `relative_transform` | Expresses the moving pose in the reference frame: inverse(reference) times moving. |
| `plot_trajectory` | Saves the 3D path and x/y/z time plots; unavailable positions break the lines. |
| `object_pose` | Earlier single-tag carry model: prefers object observations, otherwise uses a previously measured hand-object offset. Raw multi-tag tracking does not use it. |
| `pose_fields` | Converts a transform into CSV position, Rodrigues rotation vector and distance fields. |
| `main` | Parses tracking options, dispatches multi-tag mode, or runs the earlier single-reference tracker. |

## Multi-tag tracking: scripts/track_multitag_raw.py

| Function | What it does |
|---|---|
| `register_world` | Registers stationary tags to ID0 using first co-visible observations. Disconnected tags remain unavailable. |
| `camera_world` | Selects the lowest-error currently visible registered world tag and obtains world-to-camera pose. No old camera pose is reused. |
| `run` | Decodes footage, records raw detections, registers the world, and exports individual tag and body-centre results. |
| `run.write_csv` | Writes row dictionaries using an explicit column schema. |
| `cube_center` | Moves half a cube edge inward along the face tag's outward z normal. Shared by object and hand geometry. |
| `export_cube_center` | Configures centre export for object IDs 4-6. |
| `export_hand_center` | Configures centre export for wrist IDs 7-8. |
| `export_body_center` | Reuses cached detections, calculates candidate centres, selects the lowest-error face per frame, and writes CSVs, plots and summary settings. |

World registration is a fixed spatial calibration, not temporal smoothing. Its errors and switching reference tags can cause jumps. Centre selection also switches between face observations; candidate disagreement is logged.

## Position gap filling: scripts/gap_fill_centers.py

| Function | What it does |
|---|---|
| `fill_position_gaps` | Shared interpolation algorithm: fills bounded gaps using timestamp-weighted linear positions, subject to frame and optional time limits. Returns positions, provenance labels and gap records. |
| `run` | Reads a centre CSV, fills gaps, preserves observations, and writes interpolated positions, metrics and plots. |
| `main` | Parses the gap-filling CLI; default maximum is ten missing frames. |

Leading/trailing gaps are never extrapolated. Interpolated samples require a future observation. `pose_source` distinguishes observed, interpolated and missing samples.

## Shared smoothing: scripts/smooth_centers.py

| Function | What it does |
|---|---|
| `smooth_positions` | Shared position smoother: causal EMA or local quadratic SG using actual timestamps. Splits at missing samples/time jumps; optionally shrinks windows on short segments. |
| `run` | Writes SG5, SG7, EMA0.03 and EMA0.05 variants, their plots and change metrics. |

SG fits a local quadratic curve and evaluates it at the current timestamp. It uses future samples and asymmetric windows at segment ends. EMA mixes the current observation with the previous filtered result using a time-dependent weight, and can delay motion. Neither method fills remaining gaps. Change metrics measure filter strength, not physical accuracy.

## Side-by-side comparison: scripts/compare_center_smoothing.py

| Function | What it does |
|---|---|
| `smooth_positions` | Compatibility wrapper around the shared smoother, preserving this workflow's window shrinking and cadence-based timestamp-gap policy. |
| `run` | Exports SG5, SG7 and EMA0.03 CSVs plus four-column full-clip and detail plots alongside the gap-filled baseline. |

The detail interval centres on the largest available second position difference; this chooses an inspection interval, not a rejected outlier.

## Annotated video: scripts/annotate_smoothed_centers.py

| Function | What it does |
|---|---|
| `load_rows` | Reads a CSV into row dictionaries. |
| `run` | Loads SG7 centres and current camera poses; writes a new video with projected centres, provenance labels and recent trails. Checks video/CSV frame alignment. |
| `run.project` | Transforms world points into the current camera and projects them through intrinsics/distortion; returns pixel positions and positive-depth flags. |

Historical trail points are reprojected with the current camera pose. Missing centres are not drawn. The overlay shows position, not smoothed body orientation.

## Earlier pose postprocessing: scripts/postprocess_trajectory.py

These commands remain supported because earlier recordings and tests use full poses, including orientations.

| Function | What it does |
|---|---|
| `interpolate_rotation` | Quaternion SLERP between two orientations along the shortest rotation path. |
| `fill_gaps` | Bounded full-pose interpolation: linear position and SLERP orientation. |
| `smooth` | Translation EMA and orientation SLERP smoothing; resets at missing poses/time jumps. |
| `metrics` | Measures coverage, positional/orientation variation, distance error when supplied, and adjacent-frame motion. Static jitter interpretation requires a static recording. |
| `run` | Runs full-pose gap filling, optional smoothing, CSV/plot export and deterministic holdout checks; dispatches centre-only inputs separately. |
| `fill_position_gaps` | Compatibility wrapper around the shared position gap filler, retaining the earlier two-value return and time-limit default. |
| `run_positions` | Preserves the earlier position-only CLI output format while using shared interpolation. |
| `main` | Parses the postprocessing CLI, including `--no-smoothing`. |

## Earlier full-pose filtering: scripts/filter_trajectory.py

| Function | What it does |
|---|---|
| `valid_pose` | Finds rows with finite positions and quaternions. |
| `reject_spikes` | Rejects isolated excursions when both adjacent speeds are excessive but the neighbor-to-neighbor bridge is quiet; returns rejection flags/reasons. |
| `segments` | Splits available samples at missing rows and large timestamp gaps. |
| `filter_segments` | Resamples segments, applies SG or optional zero-phase Butterworth to position/orientation, and restores original timestamps. Short segments can pass through. |
| `gap_records` | Describes missing/rejected runs and their bounding observations. |
| `evaluate` | Computes coverage, path/speed, deviations from raw, and optional stationary-interval jitter. |
| `run` | Coordinates spike rejection, bounded interpolation, filtering, CSV export and comparison metrics. |
| `main` | Parses and validates full-pose filter options. |

## Cleanup decisions

- Removed duplicate position-gap interpolation and duplicate SG/EMA smoothing implementations.
- Kept small compatibility wrappers because existing callers use their interfaces.
- Preserved different workflow policies: the side-by-side comparison shrinks short SG windows and uses a cadence-based time-gap limit; the earlier smoother passes short windows through and defaults to 0.15 seconds.
- Removed a redundant hard-coded centre-summary assignment and corrected stale comments, the tracker description and a corrupted plot title.
- Kept historical full-pose filters and carry logic because they are documented and tested.
- Recordings and generated results were not removed or regenerated.

## Tests

Tests are separate from the production pipeline. Each `test_*` function constructs known inputs and checks behavior named by the method. `MultiTagTests.pose` constructs a synthetic transform for its registration checks.

| Test module | What its functions check |
|---|---|
| `test_tracking_geometry.py` | Pose/size scaling, camera-motion cancellation, crop geometry and strict resolution matching. |
| `test_object_tracking.py` | Detector corner order, rigid carry, reacquisition and required current world/hand observations. |
| `test_multitag_raw.py` | Orthogonal/opposite face centres, chained registration, camera motion, duplicates and disconnected tags. |
| `test_gap_fill_centers.py` | Ten-frame limit, unchanged observations, no endpoint extrapolation, optional time cap and irregular timing. |
| `test_position_gaps.py` | Compatibility interface: ten versus eleven missing frames, endpoint and time-span limits. |
| `test_center_smoothing.py` | Comparison wrapper: quadratic preservation with irregular timing and gap/reset behavior. |
| `test_smooth_centers.py` | Shared smoother: quadratic preservation, gap reset, short-segment passthrough and high-frequency noise reduction. |
| `test_postprocessing.py` | Shortest rotation interpolation, quaternion sign, bounded gaps, EMA reset and no extrapolation. |
| `test_offline_filter.py` | Spike handling, rapid motion preservation, rotation wrapping, SG/Butterworth, gap separation and metric coverage. |

Run all tests with `.venv/Scripts/python.exe -m unittest discover -s tests -v`.

## Task-space analysis: scripts/describe_object_trajectory.py

| Function | What it does |
|---|---|
| `load_trajectory` | Reads either supported position format; skips invalid rows, checks time order and preserves missing-data segment boundaries. |
| `normalize_trajectory` | Subtracts the first valid position without rotating axes. |
| `compute_velocity` | Uses central finite differences internally and one-sided differences at segment endpoints; never differentiates across gaps. |
| `motion_runs` | Collects consecutive phase intervals, splitting at gaps. |
| `segment_motion` | Uses sustained movement to estimate task bounds, then applies speed and signed vertical-velocity thresholds. Quiet internal samples remain pauses. |
| `compute_metrics` | Measures net horizontal displacement, horizontal/3D paths, peak relative height, duration, endpoints and estimated events. |
| `plot_results` | Saves input coordinate plots, relative 3D path and phase-marked speed. |
| `main` | Exposes tuning options and coordinates CSV/JSON/plot export. |

These thresholds are interpretable but can produce short alternating phases near
thresholds. Review event times against the recording. A phase name is not evidence
of contact, grasp success, or a robot action. No LIBERO mapping is applied.
