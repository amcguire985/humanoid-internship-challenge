"""Track AprilTag poses: single-reference or raw multi-tag world/cube/wrist mode."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def fit_intrinsics(matrix, source_size, target_size, mode):
    """Adjust pixel coordinates assuming a resize or a centred crop then resize."""
    sw, sh = source_size
    tw, th = target_size
    k = np.array(matrix, dtype=float, copy=True)
    if mode == "strict":
        if (sw, sh) != (tw, th):
            raise ValueError("Video resolution differs from calibration; select a fit model explicitly")
        return k
    if mode == "resize":
        sx, sy, ox, oy = tw / sw, th / sh, 0, 0
    else:
        sx = sy = max(tw / sw, th / sh)
        ox, oy = (sw * sx - tw) / 2, (sh * sy - th) / 2
    k[0, :] *= sx
    k[1, :] *= sy
    k[0, 2] -= ox
    k[1, 2] -= oy
    return k


class Standard41Detector:
    """Adapt AprilTag 3 corners to OpenCV's canonical top-left-first order."""
    def __init__(self):
        import sys
        if sys.platform == "win32":
            import ctypes
            import pupil_pthreads_win
            self._pthread = ctypes.WinDLL(str(pupil_pthreads_win.dll_path.resolve()))
        from pupil_apriltags import Detector
        self.detector = Detector(families="tagStandard41h12", quad_decimate=1.0, nthreads=2)

    def detectMarkers(self, gray):
        detections = self.detector.detect(gray)
        corners = [d.corners[[1, 0, 3, 2]].astype(np.float32).reshape(1,4,2) for d in detections]
        ids = np.array([[d.tag_id] for d in detections],dtype=np.int32) if detections else None
        return corners, ids, []


def marker_points(side):
    # IPPE_SQUARE order. Axes follow the decoded OpenCV marker corner convention.
    h = side / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def estimate_pose(corners, side, matrix, distortion):
    obj = marker_points(side)
    image = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    count, rotations, translations, _ = cv2.solvePnPGeneric(
        obj, image, matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = []
    if not count:
        return None
    for r, t in zip(rotations, translations):
        rotation = cv2.Rodrigues(r)[0]
        if np.any((rotation @ obj.T + t.reshape(3, 1))[2] <= 0):
            continue
        projected = cv2.projectPoints(obj, r, t, matrix, distortion)[0].reshape(4, 2)
        error = float(np.sqrt(np.mean(np.sum((projected - image) ** 2, axis=1))))
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = t.ravel()
        candidates.append((error, transform, r, t))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    best = candidates[0]
    gap = candidates[1][0] - best[0] if len(candidates) > 1 else None
    return {"error": best[0], "transform": best[1], "rvec": best[2],
            "tvec": best[3], "ambiguity_gap": gap}


def relative_transform(reference, moving):
    result = np.eye(4)
    result[:3, :3] = reference[:3, :3].T @ moving[:3, :3]
    result[:3, 3] = reference[:3, :3].T @ (moving[:3, 3] - reference[:3, 3])
    return result


def plot_trajectory(rows, output, moving_label="ID1", reference_label="ID0"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xyz = np.array([[r.get(f"{axis}_m", np.nan) for axis in "xyz"] for r in rows]) * 100
    valid = np.isfinite(xyz).all(axis=1)
    times = np.array([r["time_s"] for r in rows], dtype=float)
    fig = plt.figure(figsize=(15, 8))
    layout = fig.add_gridspec(3, 2, width_ratios=[1.2, 1])
    space = fig.add_subplot(layout[:, 0], projection="3d")
    space.plot(*(xyz[:, i] for i in range(3)), linewidth=1.2, label=f"{moving_label} trajectory")
    space.scatter(0, 0, 0, color="black", marker="*", s=150, label=f"{reference_label} (origin)")
    if valid.any():
        first, last = np.flatnonzero(valid)[[0, -1]]
        space.scatter(*xyz[first], color="green", s=45, label="Start")
        space.scatter(*xyz[last], color="red", s=45, label="End")
    # Include ID0 and use the same physical scale on all three axes.
    extent = np.vstack((np.zeros((1, 3)), xyz[valid]))
    low, high = extent.min(axis=0), extent.max(axis=0)
    centre = (low + high) / 2
    radius = max(float(np.max(high - low)) * 0.55, 1.0)
    space.set_xlim(centre[0] - radius, centre[0] + radius)
    space.set_ylim(centre[1] - radius, centre[1] + radius)
    space.set_zlim(centre[2] - radius, centre[2] + radius)
    space.set_box_aspect((1, 1, 1))
    space.set(xlabel="x (cm)", ylabel="y (cm)", zlabel="z (cm)",
              title=f"{moving_label} relative to {reference_label}\nCamera calibration")
    space.legend()
    time_axis = None
    for i, (axis, color) in enumerate(zip("xyz", ["tab:red", "tab:green", "tab:blue"])):
        panel = fig.add_subplot(layout[i, 1], sharex=time_axis)
        if time_axis is None:
            time_axis = panel
            panel.set_title(f"{moving_label} position relative to {reference_label} over time")
        panel.plot(times, xyz[:, i], color=color, linewidth=1.2)
        panel.set_ylabel(f"{axis} (cm)")
        panel.grid(True, alpha=0.3)
        if i == 2:
            panel.set_xlabel("Time (s)")
        else:
            panel.tick_params(labelbottom=False)
    fig.tight_layout()
    fig.savefig(output / "trajectory.png", dpi=180)
    plt.close(fig)



def object_pose(reference, hand, object_observation, hand_object, object_detected):
    """Use observations first; rigid carry applies only when the object is absent."""
    if object_observation is not None:
        hand_object = relative_transform(hand, object_observation) if hand is not None else None
    elif object_detected:
        # A detected but rejected pose is not evidence of occlusion.
        hand_object = None
    if reference is None:
        return None, "missing_world", hand_object
    if object_observation is not None:
        return relative_transform(reference, object_observation), "tracked", hand_object
    if not object_detected and hand is not None and hand_object is not None:
        return relative_transform(reference, hand) @ hand_object, "inferred_hand", hand_object
    return None, "missing_object", hand_object


def pose_fields(transform):
    xyz = transform[:3, 3]
    rot = cv2.Rodrigues(transform[:3, :3])[0].ravel()
    return dict(zip(["x_m", "y_m", "z_m", "relative_rx_rad", "relative_ry_rad", "relative_rz_rad"],
                    map(float, [*xyz, *rot])), distance_m=float(np.linalg.norm(xyz)))


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=Path("test_001_static.MOV"))
    p.add_argument("--calibration", type=Path, default=root / "results/camera_calibration_portrait/calibration.json")
    p.add_argument("--tag-size", type=float, default=0.078, help="Outer black-square side in metres")
    p.add_argument("--tag-size-convention", choices=["border", "full-pattern"], default="border", help="Full-pattern supported for Standard41h12 (pose border = 5/9 of pattern)")
    p.add_argument("--reference-size", type=float, help="Override ID0 side in metres")
    p.add_argument("--moving-size", type=float, help="Override ID1 side in metres")
    p.add_argument("--reference-id", type=int, default=0)
    p.add_argument("--moving-id", type=int, default=1)
    p.add_argument("--object-id", type=int, help="Optional object tag; infer occluded poses from its last hand-relative transform")
    p.add_argument("--family", choices=["16h5", "25h9", "36h10", "36h11", "Standard41h12"], default="25h9")
    p.add_argument("--calibration-fit", choices=["center-crop", "resize", "strict"], default="center-crop")
    p.add_argument("--max-reprojection-error", type=float, default=3.0, help="Pose acceptance threshold in pixels")
    p.add_argument("--output", type=Path, default=root / "results/test_001_tracking")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--video-output", type=Path, help="External overlay MP4 path; defaults to HUMANOID_VIDEO_ROOT/processed/<run>/annotated.mp4")
    p.add_argument("--multi-tag-raw", action="store_true", help="Raw configurable world/cube/wrist/target tags; no inference, rejection threshold, filling or filtering")
    p.add_argument("--world-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--cube-ids", type=int, nargs="+", default=[4, 5, 6])
    p.add_argument("--hand-ids", type=int, nargs="*", default=[7, 8])
    p.add_argument("--target-id", type=int, help="Target tag, excluded from world registration")
    p.add_argument("--target-size", type=float, help="Target side using selected convention; defaults to world size")
    p.add_argument("--world-size", type=float, default=0.060, help="World-tag side in metres using selected size convention")
    p.add_argument("--body-size", type=float, default=0.040, help="Cube/wrist tag side in metres using selected size convention")
    p.add_argument("--cube-edge", type=float, help="Cube edge in metres; export raw centre position assuming centred flush face tags")
    p.add_argument("--hand-edge", type=float, default=0.045, help="Wrist cube edge in metres (default 45 mm); assumes centred flush face tags 7-8")
    a = p.parse_args()
    from video_paths import resolve_video, output_video
    try:
        a.video = resolve_video(a.video)
        if not a.no_video:
            a.video_output = output_video(a.output, explicit=a.video_output)
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))
    if a.multi_tag_raw:
        from track_multitag_raw import run
        run(a)
        return
    sizes = {a.reference_id: a.reference_size if a.reference_size is not None else a.tag_size,
             a.moving_id: a.moving_size if a.moving_size is not None else a.tag_size}
    if min(sizes.values()) <= 0 or a.max_reprojection_error <= 0 or a.reference_id == a.moving_id:
        p.error("Sizes/error limit must be positive and IDs must differ")
    if a.object_id is not None:
        if a.object_id in sizes:
            p.error("Object ID must differ from reference and moving IDs")
        sizes[a.object_id] = a.tag_size
    input_sizes = sizes.copy()
    if a.tag_size_convention == "full-pattern":
        if a.family != "Standard41h12":
            p.error("Full-pattern conversion is implemented only for Standard41h12")
        sizes = {key: side * 5 / 9 for key, side in sizes.items()}
    calibration = json.loads(a.calibration.read_text(encoding="utf-8-sig"))
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {a.video}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError("Video has no usable frame rate")
    rotation_meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Video contains no readable frames")
    height, width = frame.shape[:2]
    source = calibration["image_size"]
    if (source[0] > source[1]) != (width > height):
        cap.release()
        raise ValueError("Calibration and decoded video orientations differ; choose matching calibration")
    matrix = fit_intrinsics(calibration["camera_matrix"], source, (width, height), a.calibration_fit)
    distortion = np.array(calibration["distortion_coefficients"], dtype=float)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = Standard41Detector() if a.family == "Standard41h12" else cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, "DICT_APRILTAG_" + a.family)), params)
    a.output.mkdir(parents=True, exist_ok=True)
    writer = None
    if not a.no_video:
        writer = cv2.VideoWriter(str(a.video_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not open MP4 writer; use --no-video")
    rows, index, detections, rejected = [], 0, {key: 0 for key in sizes}, {key: 0 for key in sizes}
    fields = ["frame", "time_s", "status", "reference_detected", "moving_detected",
              "reference_error_px", "moving_error_px", "reference_ambiguity_gap_px",
              "moving_ambiguity_gap_px", "x_m", "y_m", "z_m", "distance_m",
              "relative_rx_rad", "relative_ry_rad", "relative_rz_rad"]
    object_rows, hand_object = [], None
    snapshot_saved = False
    try:
        while ok:
            if frame.shape[:2] != (height, width):
                raise RuntimeError("Video dimensions changed during decoding")
            # Decoder presentation timestamps preserve variable frame timing.
            time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            row = {"frame": index, "time_s": time_s, "status": "missing_tag"}
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            poses = {}
            found = set() if ids is None else set(int(i) for i in ids.ravel())
            row["reference_detected"] = int(a.reference_id in found)
            row["moving_detected"] = int(a.moving_id in found)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for tag_id, side in sizes.items():
                    locations = np.flatnonzero(ids.ravel() == tag_id)
                    if not len(locations):
                        continue
                    detections[tag_id] += 1
                    if len(locations) != 1:
                        if tag_id != a.object_id: row["status"] = "duplicate_id"
                        continue
                    pose = estimate_pose(corners[locations[0]], side, matrix, distortion)
                    label = "reference" if tag_id == a.reference_id else "moving" if tag_id == a.moving_id else "object"
                    if pose is None:
                        rejected[tag_id] += 1
                        if tag_id != a.object_id: row["status"] = "invalid_pose"
                        continue
                    if tag_id != a.object_id:
                        row[label + "_error_px"] = pose["error"]
                        row[label + "_ambiguity_gap_px"] = pose["ambiguity_gap"]
                    if pose["error"] > a.max_reprojection_error:
                        rejected[tag_id] += 1
                        if tag_id != a.object_id: row["status"] = "high_reprojection_error"
                        continue
                    poses[tag_id] = pose
                    cv2.drawFrameAxes(frame, matrix, distortion, pose["rvec"], pose["tvec"], side * 0.6)
            if all(tag_id in poses for tag_id in (a.reference_id, a.moving_id)):
                relative = relative_transform(poses[a.reference_id]["transform"], poses[a.moving_id]["transform"])
                xyz = relative[:3, 3]
                rotation = cv2.Rodrigues(relative[:3, :3])[0].ravel()
                row.update(status="tracked", x_m=float(xyz[0]), y_m=float(xyz[1]), z_m=float(xyz[2]),
                           distance_m=float(np.linalg.norm(xyz)),
                           relative_rx_rad=float(rotation[0]), relative_ry_rad=float(rotation[1]),
                           relative_rz_rad=float(rotation[2]))
                text = "ID1 in ID0: x={:+.1f} y={:+.1f} z={:+.1f} cm".format(*(xyz * 100))
            else:
                text = "Relative pose unavailable: " + row["status"]
            object_status = "disabled"
            if a.object_id is not None:
                transforms = {key: value["transform"] for key, value in poses.items()}
                object_transform, object_status, hand_object = object_pose(
                    transforms.get(a.reference_id), transforms.get(a.moving_id),
                    transforms.get(a.object_id), hand_object, a.object_id in found)
                object_row = {"frame": index, "time_s": time_s, "status": object_status,
                              "reference_detected": int(a.reference_id in found),
                              "moving_detected": int(a.object_id in found)}
                if object_transform is not None:
                    object_row.update(pose_fields(object_transform))
                    if object_status == "inferred_hand":
                        camera_object = transforms[a.reference_id] @ object_transform
                        cv2.drawFrameAxes(frame, matrix, distortion,
                            cv2.Rodrigues(camera_object[:3,:3])[0], camera_object[:3,3], sizes[a.object_id]*0.6)
                object_rows.append(object_row)
            cv2.rectangle(frame, (0, 0), (width, 95), (20, 20, 20), -1)
            cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"{time_s:.2f}s | object: {object_status} | tags {a.tag_size*100:g} cm",
                        (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)
            if not snapshot_saved and row["status"] == "tracked":
                cv2.imwrite(str(a.output / "preview.jpg"), frame)
                snapshot_saved = True
            if writer is not None:
                writer.write(frame)
            rows.append(row)
            index += 1
            if index % 100 == 0:
                print(f"Processed {index} frames", flush=True)
            ok, frame = cap.read()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    with (a.output / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    if a.object_id is not None:
        object_output = a.output / "object"
        object_output.mkdir(parents=True, exist_ok=True)
        with (object_output / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=fields)
            csv_writer.writeheader()
            csv_writer.writerows(object_rows)
        plot_trajectory(object_rows, object_output, f"ID{a.object_id} object", f"ID{a.reference_id}")
    tracked = [r for r in rows if r["status"] == "tracked"]
    summary = {
        "video": a.video.name, "annotated_video": str(a.video_output) if not a.no_video else None, "calibration": str(a.calibration), "opencv_version": cv2.__version__,
        "family": a.family, "tag_sizes_m": sizes, "input_tag_sizes_m": input_sizes, "tag_size_convention": a.tag_size_convention, "reference_id": a.reference_id,
        "moving_id": a.moving_id, "frames_processed": len(rows), "frames_tracked": len(tracked),
        "tracked_fraction": len(tracked) / len(rows), "detections_by_id": detections,
        "pose_rejections_by_id": rejected, "video_size": [width, height], "fps": fps,
        "orientation_metadata_degrees": rotation_meta, "timestamp_source": "decoder CAP_PROP_POS_MSEC",
        "calibration_fit": a.calibration_fit, "effective_camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.tolist(), "max_reprojection_error_px": a.max_reprojection_error,
        "coordinate_frame": "ID0 centre; x from corner0 to corner1; y from corner3 to corner0; z=x cross y (out of printed face). OpenCV decoded corner convention.",
        "relative_transform": "T_ID0_ID1 = inverse(T_camera_ID0) @ T_camera_ID1",
        "limitations": [
            "Metric accuracy depends on matching the calibrated recording mode and measured tag sizes.",
            "Planar pose ambiguity can cause flips; ambiguity gaps are logged (small gap means weakly distinguished poses).",
            "Missing/rejected frames are blank, without interpolation or stale reference reuse.",
            "No temporal filtering is applied; moving paper should be kept flat.",
        ],
    }
    if a.object_id is not None:
        summary["object_id"] = a.object_id
        summary["object_status_counts"] = {status: sum(r["status"] == status for r in object_rows)
                                            for status in sorted({r["status"] for r in object_rows})}
        summary["object_occlusion_assumption"] = "Absent ID2 is assumed rigidly attached to ID1 using the last simultaneous accepted hand/object observation. Requires current world and hand poses. Direct object observations take priority."
    if tracked:
        xyz = np.array([[r[k] for k in ["x_m", "y_m", "z_m"]] for r in tracked])
        summary["position_min_m"] = xyz.min(axis=0).tolist()
        summary["position_max_m"] = xyz.max(axis=0).tolist()
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_trajectory(rows, a.output)
    print(f"Tracked {len(tracked)}/{len(rows)} frames ({summary['tracked_fraction']:.1%}); saved to {a.output}")
    if not tracked:
        raise RuntimeError("No usable relative poses; inspect detections and tag family")


if __name__ == "__main__":
    main()

