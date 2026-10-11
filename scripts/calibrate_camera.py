"""Calibrate from chessboard photos or random video frames; square sizes and translations use metres."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, default=root / "chessboard_calibration")
    p.add_argument("--video", type=Path, help="Sample decoded frames from this video instead of photos")
    p.add_argument("--frames", type=int, default=15, help="Number of random video frames to sample")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible frame selection")
    p.add_argument("--orientation", choices=["portrait", "landscape"], default="portrait", help="Calibrate one stored image orientation at a time")
    p.add_argument("--columns", type=int, default=6, help="Inner corners across board")
    p.add_argument("--rows", type=int, default=9, help="Inner corners down board")
    p.add_argument("--square-size", type=float, default=0.023, help="Square side in metres")
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    if a.square_size <= 0 or min(a.columns, a.rows) < 2:
        p.error("Positive square size and at least two corners per axis required")
    if a.output is None:
        a.output = root / "results" / ("camera_calibration_" + a.orientation)
    if a.frames < 3:
        p.error("At least three video frames required")
    pattern = (a.columns, a.rows)
    obj = np.zeros((a.columns * a.rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:a.columns, 0:a.rows].T.reshape(-1, 2)
    obj *= a.square_size
    video_metadata = None
    if a.video:
        from video_paths import resolve_video
        a.video = resolve_video(a.video)
        cap = cv2.VideoCapture(str(a.video))
        if not cap.isOpened():
            raise IOError(f"Could not open {a.video}")
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if count < a.frames:
            cap.release()
            raise ValueError(f"Video has {count} frames; requested {a.frames}")
        indices = np.random.default_rng(a.seed).permutation(count).tolist()
        video_metadata = {"path": str(a.video), "random_seed": a.seed,
                          "sampled_frame_indices": [], "total_frames": count,
                          "fps": fps, "rotation_metadata_degrees": cap.get(cv2.CAP_PROP_ORIENTATION_META)}
        def video_frames():
            try:
                for index in indices:
                    video_metadata["sampled_frame_indices"].append(index)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = cap.read()
                    if not ok:
                        raise IOError(f"Could not decode frame {index}")
                    yield Path(f"frame_{index:06d}.jpg"), frame
            finally:
                cap.release()
        inputs = video_frames()
        input_count = len(indices)
    else:
        paths = sorted(f for f in a.images.iterdir() if f.suffix.lower() in {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"})
        inputs = ((path, cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)) for path in paths)
        input_count = len(paths)
    objects, points, names, skipped = [], [], [], []
    size = None
    a.output.mkdir(parents=True, exist_ok=True)
    for path, frame in inputs:
        if frame is None:
            skipped.append({"image": path.name, "reason": "Unreadable image"})
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current = (gray.shape[1], gray.shape[0])
        if (current[1] > current[0]) != (a.orientation == "portrait"):
            skipped.append({"image": path.name, "reason": "Other image orientation"})
            continue
        if size is None:
            size = current
        elif current != size:
            raise ValueError(f"{path.name}: resolution {current} differs from {size}")
        found, corners = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not found:
            skipped.append({"image": path.name, "reason": "Board not detected"})
            print(f"Skipped {path.name}", flush=True)
            continue
        objects.append(obj.copy())
        points.append(corners)
        names.append(path.name)
        cv2.drawChessboardCorners(frame, pattern, corners, found)
        if not cv2.imwrite(str(a.output / f"{path.stem}_corners.jpg"), frame):
            raise IOError("Could not save corner overlay")
        print(f"Detected {path.name}", flush=True)
        if a.video and len(names) == a.frames:
            inputs.close()
            input_count = len(video_metadata["sampled_frame_indices"])
            break
    if len(names) < 3:
        raise RuntimeError(f"Only {len(names)} usable views; need at least 3 varied poses")
    rms, matrix, dist, rvecs, tvecs, std, _, errors = cv2.calibrateCameraExtended(objects, points, size, None, None)
    result = {
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": dist.ravel().tolist(),
        "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
        "image_size": list(size),
        "image_orientation": "Decoded video pixels; rotation metadata applied" if a.video else "Stored pixels; EXIF orientation ignored",
        "inner_corners": list(pattern),
        "square_size_m": a.square_size,
        "rms_reprojection_error_px": float(rms),
        "intrinsic_standard_deviations": std.ravel().tolist(),
        "per_image": [
            {"image": name, "rms_reprojection_error_px": float(e[0]),
             "rotation_vector": r.ravel().tolist(), "translation_m": t.ravel().tolist()}
            for name, e, r, t in zip(names, errors, rvecs, tvecs)
        ],
        "skipped_images": skipped,
    }
    if video_metadata is not None:
        video_metadata["used_frame_indices"] = [int(Path(name).stem.split("_")[1]) for name in names]
        result["source_video"] = video_metadata
        if len(names) != a.frames:
            raise RuntimeError(f"Only {len(names)}/{a.frames} sampled frames usable; previous calibration files retained")
    # Remove obsolete overlays only after calibration succeeds.
    for old_overlay in a.output.glob("*_corners.jpg"):
        if old_overlay.name not in {Path(name).stem + "_corners.jpg" for name in names}:
            old_overlay.unlink()
    (a.output / "calibration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez(a.output / "calibration.npz", camera_matrix=matrix, distortion_coefficients=dist,
             image_size=np.array(size), square_size_m=a.square_size,
             rotation_vectors=np.array(rvecs), translation_vectors=np.array(tvecs))
    print(f"Used {len(names)}/{input_count} images; RMS error: {rms:.4f} px")
    print("Camera matrix:\n", matrix)
    print("Distortion:", dist.ravel())

if __name__ == "__main__":
    main()

